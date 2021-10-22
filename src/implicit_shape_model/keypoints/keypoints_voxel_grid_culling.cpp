/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2021, Viktor Seib
 * All rights reserved.
 *
 */

#include "keypoints_voxel_grid_culling.h"

#define PCL_NO_PRECOMPILE
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>
#include <fstream>
#include <boost/algorithm/string.hpp>

namespace ism3d
{
    KeypointsVoxelGridCulling::KeypointsVoxelGridCulling()
    {
        addParameter(m_leafSize, "LeafSize", 0.1f);

        addParameter(m_filter_method_geometry, "FilterMethodGeometry", std::string("None"));
        addParameter(m_filter_type_geometry, "FilterTypeGeometry", std::string("CutOff"));
        addParameter(m_filter_threshold_geometry, "FilterThresholdGeometry", 0.005f);

        addParameter(m_filter_method_color, "FilterMethodColor", std::string("None"));
        addParameter(m_filter_type_color, "FilterTypeColor", std::string("CutOff"));
        addParameter(m_filter_threshold_color, "FilterThresholdColor", 0.02f);

        addParameter(m_max_similar_color_distance, "MaxSimilarColorDistance", 0.05f);
        addParameter(m_filter_cutoff_ratio, "FilterCutoffRatio", 0.5f);

        addParameter(m_disable_filter_in_training, "DisableFilterInTraining", true);

        addParameter(m_refine_position, "RefineKeypointPosition", false);

        // init false, will be set to true in case training phase is running
        m_is_training = false;
    }

    KeypointsVoxelGridCulling::~KeypointsVoxelGridCulling()
    {
    }

    pcl::PointCloud<PointT>::ConstPtr KeypointsVoxelGridCulling::iComputeKeypoints(pcl::PointCloud<PointT>::ConstPtr points,
                                                                            pcl::PointCloud<PointT>::ConstPtr eigenValues,
                                                                            pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                            pcl::PointCloud<PointT>::Ptr pointsWithoutNaNNormals,
                                                                            pcl::PointCloud<PointT>::Ptr eigenValuesWithoutNan, // TODO VS: get rid of it if not needed
                                                                            pcl::PointCloud<pcl::Normal>::Ptr normalsWithoutNaN,
                                                                            pcl::search::Search<PointT>::Ptr search)
    {      
        boost::algorithm::to_lower(m_filter_method_geometry);
        boost::algorithm::to_lower(m_filter_method_color);
        boost::algorithm::to_lower(m_filter_type_geometry);
        boost::algorithm::to_lower(m_filter_type_color);


        // disable in training if desired -> becomes normal voxel grid without culling
        if((m_is_training && m_disable_filter_in_training)
                   || (m_filter_method_geometry == "none" && m_filter_method_color == "none"))
        {
            pcl::VoxelGrid<PointT> voxelGrid;
            voxelGrid.setInputCloud(points);
            voxelGrid.setLeafSize(m_leafSize, m_leafSize, m_leafSize);

            // compute keypoints
            pcl::PointCloud<PointT>::Ptr keypoints(new pcl::PointCloud<PointT>());
            voxelGrid.filter(*keypoints);
            return keypoints;
        }
        else
        {
            // TODO VS: if kpq is not needed for eval, delete it completely or move to own keypoint class
            if(m_filter_method_geometry != "none" && m_filter_method_geometry != "curvature" &&
                    m_filter_method_geometry != "gaussian" && m_filter_method_geometry != "kpq")
            {
                LOG_ERROR("Unsupported keypoint filter method: " << m_filter_method_geometry);
                exit(1);
            }
            if(m_filter_type_geometry != "cutoff" && m_filter_type_geometry != "threshold" && m_filter_type_geometry != "auto")
            {
                LOG_ERROR("Unsupported keypoint filter type: " << m_filter_type_geometry);
                exit(1);
            }
            if(m_filter_method_geometry == "kpq" && m_filter_type_geometry == "auto")
            {
                LOG_ERROR("Filter type " << m_filter_type_geometry << " is not supported with filter method " << m_filter_method_geometry);
                exit(1);
            }

            if(m_filter_method_color != "none" && m_filter_method_color != "colordistance")
            {
                LOG_ERROR("Unsupported keypoint color filter method: " << m_filter_method_color);
                exit(1);
            }
            if(m_filter_type_color != "cutoff" && m_filter_type_color != "threshold")
            {
                LOG_ERROR("Unsupported keypoint color filter type: " << m_filter_type_color);
                exit(1);
            }

            // combine filtered cloud points and filtered normals into one cloud
            pcl::PointCloud<PointNormalT>::Ptr points_with_normals(new pcl::PointCloud<PointNormalT>());
            pcl::concatenateFields(*pointsWithoutNaNNormals, *normalsWithoutNaN, *points_with_normals);

            // compute voxel grid keypoints on cloud with normals
            pcl::VoxelGrid<PointNormalT> voxel_grid;
            voxel_grid.setInputCloud(points_with_normals);
            voxel_grid.setLeafSize(m_leafSize, m_leafSize, m_leafSize);
            pcl::PointCloud<PointNormalT>::Ptr keypoints_with_normals(new pcl::PointCloud<PointNormalT>());
            voxel_grid.filter(*keypoints_with_normals);

            // copy only point information without normals
            pcl::PointCloud<PointT>::Ptr keypoints_without_normals(new pcl::PointCloud<PointT>()); // these keypoints will be filtered here
            pcl::copyPointCloud(*keypoints_with_normals, *keypoints_without_normals);

            // estimate principle curvatures
            pcl::PrincipalCurvaturesEstimation<PointT, pcl::Normal, pcl::PrincipalCurvatures> curv_est;
            curv_est.setSearchSurface(pointsWithoutNaNNormals);
            curv_est.setInputNormals(normalsWithoutNaN);
            curv_est.setSearchMethod(search);
            curv_est.setRadiusSearch(m_leafSize);
            pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures (new pcl::PointCloud<pcl::PrincipalCurvatures>());
            if(m_filter_method_geometry == "gaussian")
            {
                // compute curvatures on sparse cloud only (i.e. only for keypoints)
                curv_est.setInputCloud(keypoints_without_normals);
                curv_est.compute(*principal_curvatures);
            }
            else if(m_filter_method_geometry == "kpq")
            {
                // densely estimate principle curvatures - very slow
                curv_est.setInputCloud(pointsWithoutNaNNormals);
                curv_est.compute(*principal_curvatures);
            }

            LOG_INFO("Number of keypoints before filtering: " << keypoints_without_normals->size());

            auto[geo_scores, color_scores] = getScoresForKeypoints(points_with_normals, keypoints_with_normals, principal_curvatures);
            auto[threshold_geo, threshold_color] = computeThresholds(geo_scores, color_scores);
            m_filter_threshold_geometry = threshold_geo;
            m_filter_threshold_color = threshold_color;

            // prepare resulting keypoints cloud
            pcl::PointCloud<PointT>::Ptr keypoints(new pcl::PointCloud<PointT>());
            for(unsigned idx = 0; idx < keypoints_without_normals->size(); idx++)
            {
                bool geo_passed = true;
                if(m_filter_method_geometry != "none")
                {
                    if(geo_scores[idx] < m_filter_threshold_geometry)
                    {
                        geo_passed = false;
                    }
                }

                bool color_passed = true;
                if(m_filter_method_color != "none")
                {
                    if(color_scores[idx] < m_filter_threshold_color)
                    {
                        color_passed = false;
                    }
                }

                if(geo_passed && color_passed)
                {
                    if(m_refine_position)
                    {
                        PointT keypoint = refineKeypointPosition(geo_scores[idx], color_scores[idx], keypoints_without_normals->at(idx),
                                                                 points_with_normals, principal_curvatures);
                        keypoints->push_back(keypoint);
                    }
                    else
                    {
                        keypoints->push_back(keypoints_without_normals->at(idx));
                    }
                }
            }
            LOG_INFO("Number of keypoints after filtering: " << keypoints->size());
            return keypoints;
        }
    }


    std::tuple<std::vector<float>, std::vector<float>> KeypointsVoxelGridCulling::getScoresForKeypoints(
            const pcl::PointCloud<PointNormalT>::Ptr points_with_normals,
            const pcl::PointCloud<PointNormalT>::Ptr keypoints_with_normals,
            const pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures)
    {
        // init result lists
        std::vector<float> geo_scores(keypoints_with_normals->size(), 0.0f);
        std::vector<float> color_scores(keypoints_with_normals->size(), 0.0f);

        pcl::KdTreeFLANN<PointNormalT> pts_with_normals_tree;
        pts_with_normals_tree.setInputCloud(points_with_normals); // NOTE: search on original cloud
        std::vector<int> point_idxs;
        std::vector<float> point_dists;

        ColorConversion& cc = ColorConversionStatic::getColorConversion();

        for(unsigned idx = 0; idx < keypoints_with_normals->size(); idx++)
        {
            // PCL curvature
            const PointNormalT &reference_point = keypoints_with_normals->at(idx);

            if(m_filter_method_geometry == "curvature")
            {
                geo_scores[idx] = reference_point.curvature;
            }
            if(m_filter_method_geometry == "gaussian")
            {
                // gaussian curvature
                const pcl::PrincipalCurvatures &pc_point = principal_curvatures->at(idx);
                geo_scores[idx] = pc_point.pc1 * pc_point.pc2;
            }

            // group these methods as they need nearest neighbor search
            if(m_filter_method_geometry == "kpq" || m_filter_method_color == "colordistance")
            {
                pts_with_normals_tree.radiusSearch(reference_point, m_leafSize, point_idxs, point_dists);
                if(m_filter_method_geometry == "kpq")
                {
                    // KPQ (keypoint quality)
                    float kpqval = computeKPQ(point_idxs, principal_curvatures);
                    geo_scores[idx] = kpqval;
                }
                if(m_filter_method_color == "colordistance")
                {
                    float color_score = computeColorScore(point_idxs, points_with_normals, reference_point, cc);
                    color_scores[idx] = color_score;
                }
            }
        }

        return std::make_tuple(geo_scores, color_scores);
    }

    std::tuple<float, float> KeypointsVoxelGridCulling::computeThresholds(
            const std::vector<float> &geo_scores_orig,
            const std::vector<float> &color_scores_orig)
    {
        float threshold_geo = std::numeric_limits<float>::min();
        float threshold_color = std::numeric_limits<float>::min();

        // copy original lists for sorting
        std::vector<float> geo_scores;
        std::copy(geo_scores_orig.begin(), geo_scores_orig.end(), std::back_inserter(geo_scores));
        std::vector<float> color_scores;
        std::copy(color_scores_orig.begin(), color_scores_orig.end(), std::back_inserter(color_scores));

        // sort to determine cutoff threshold
        std::sort(geo_scores.begin(), geo_scores.end());
        std::sort(color_scores.begin(), color_scores.end());

        // automatically determine cutoff index
        if(m_filter_method_geometry != "none" && m_filter_type_geometry == "auto")
        {
            // NOTE: used only for "auto" method - if auto proofes useless then delete
            // create histograms
            int hist_size = 100;
            float hist_step = (geo_scores.back() - geo_scores.front()) / hist_size;
            std::vector<int> hist(hist_size, 0);
            for(auto val : geo_scores)
            {
                int curv_bin = (val - geo_scores.front()) / hist_step;
                curv_bin = curv_bin < 0 ? 0 : curv_bin;
                curv_bin = curv_bin >= hist.size() ? hist.size() - 1 : curv_bin;
                hist[curv_bin]++;
            }

            // TODO VS: experimental based on histogramm
            unsigned cutoff_index = 0;
            for(int xxx = 0; xxx < hist.size()-2; xxx++)
            {
                cutoff_index += hist[xxx];
                int diff_1 = hist[xxx+1] - hist[xxx];
                int diff_2 = hist[xxx+2] - hist[xxx+1];
                if(diff_1*2 < diff_2)
                {
                    // set cutoff index to number of points after a high value change in histogram
                    cutoff_index += hist[xxx+1];
                    break;
                }
            }
            threshold_geo = geo_scores.at(cutoff_index);
        }

        // get threshold corresponding to cutoff index
        if(m_filter_method_geometry != "none" && m_filter_type_geometry == "cutoff")
        {
            unsigned cutoff_index = unsigned(m_filter_cutoff_ratio * geo_scores.size());
            threshold_geo = geo_scores.at(cutoff_index);
        }

        if(m_filter_method_color != "none" && m_filter_type_color == "cutoff")
        {
            unsigned cutoff_index = unsigned(m_filter_cutoff_ratio * color_scores.size());
            threshold_color = color_scores.at(cutoff_index);
        }

        return std::make_tuple(threshold_geo, threshold_color);
    }


    // KeyPointQuality as presented by A.Mian et al. in
    // On the Repeatability and Quality of Keypoints for Local Feature-based 3D Object Retrieval from Cluttered Scenes
    // see equations 10 and 11
    // NOTE: for clarification, a more complete equation is presented by F.Tombari et al. in
    // Performance Evaluation of 3D Keypoint DetectorsFederico Tombari
    // see equation 9
    float KeypointsVoxelGridCulling::computeKPQ(const std::vector<int> &pointIdxs,
                     pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principle_curvatures) const
    {
        float max_k1 = std::numeric_limits<float>::min();
        float min_k2 = std::numeric_limits<float>::max();
        float max_K = std::numeric_limits<float>::min();
        float min_K = std::numeric_limits<float>::max();
        float kpq = 0.0f;

        for(int idx : pointIdxs)
        {
            const pcl::PrincipalCurvatures &pc_point = principle_curvatures->at(idx);
            float k_1 = pc_point.pc1;
            float k_2 = pc_point.pc2;
            // K is Gaussian curvature: k1 * k2
            float K = k_1 * k_2;
            if(k_1 > max_k1)
                max_k1 = k_1;
            if(k_2 < min_k2)
                min_k2 = k_2;
            if(K > max_K)
                max_K = K;
            if(K < min_K)
                min_K = K;
            kpq += K;
        }
        int num = pointIdxs.size();
        kpq = (1000.0f / num*num) * kpq + 100.0f*max_K + fabs(100.0f*min_K) + 10*max_k1 + fabs(10*min_k2);

        return kpq;
    }


    float KeypointsVoxelGridCulling::computeColorScore(const std::vector<int> &pointIdxs,
                                                pcl::PointCloud<PointNormalT>::Ptr points_with_normals,
                                                const PointNormalT &ref,
                                                const ColorConversion &cc) const
    {
        // compute reference color
        unsigned char redRef   = ref.r;
        unsigned char greenRef = ref.g;
        unsigned char blueRef  = ref.b;
        float LRef, aRef, bRef;
        cc.RgbToCieLabNormalized(redRef, greenRef, blueRef, LRef, aRef, bRef);

        // compute colors in support
        float threshold = m_max_similar_color_distance;
        int num_distant_color = 0;
        for(int idx : pointIdxs)
        {
            const PointNormalT &point = points_with_normals->at(idx);
            unsigned char red = point.r;
            unsigned char green = point.g;
            unsigned char blue = point.b;
            float L, a, b;
            cc.RgbToCieLabNormalized(red, green, blue, L, a, b);
            float distance = cc.getColorDistance(L, a, b, LRef, aRef, bRef);
            if(distance > threshold)
            {
                num_distant_color++;
            }
        }

        // compute score
        return float(num_distant_color) / pointIdxs.size();
    }


    PointT KeypointsVoxelGridCulling::refineKeypointPosition(
            const float geo_score,
            const float color_score,
            const PointT &keypoint,
            const pcl::PointCloud<PointNormalT>::Ptr points_with_normals,
            const pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr dense_principal_curvatures)
    {
        pcl::KdTreeFLANN<PointNormalT> pts_with_normals_tree;
        pts_with_normals_tree.setInputCloud(points_with_normals);
        std::vector<int> point_idxs;
        std::vector<float> point_dists;
        PointNormalT keyp_normal;
        keyp_normal.x = keypoint.x;
        keyp_normal.y = keypoint.y;
        keyp_normal.z = keypoint.z;
        pts_with_normals_tree.radiusSearch(keyp_normal, m_leafSize*0.5, point_idxs, point_dists);

        float best_geo = geo_score;
        float best_color = color_score;
        int best_index = -1;
        //std::cout << "-------------- found num neighbors: " << point_idxs.size() << std::endl;

        pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr local_principal_curvatures (new pcl::PointCloud<pcl::PrincipalCurvatures>());
        if(m_filter_method_geometry == "gaussian")
        {
            // extract neighbor indices into point cloud
            pcl::PointCloud<PointNormalT>::Ptr nn_cloud_with_normals(new pcl::PointCloud<PointNormalT>());
            pcl::copyPointCloud(*points_with_normals, point_idxs, *nn_cloud_with_normals);
            pcl::PointCloud<PointT>::Ptr nn_cloud_without_normals(new pcl::PointCloud<PointT>());
            pcl::copyPointCloud(*nn_cloud_with_normals, *nn_cloud_without_normals);

            // separate points and normals
            pcl::PointCloud<PointT>::Ptr points_without_normals(new pcl::PointCloud<PointT>());
            pcl::copyPointCloud(*points_with_normals, *points_without_normals);
            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
            pcl::copyPointCloud(*points_with_normals, *normals);

            // estimate principle curvatures
            pcl::PrincipalCurvaturesEstimation<PointT, pcl::Normal, pcl::PrincipalCurvatures> curv_est;
            curv_est.setSearchSurface(points_without_normals);
            curv_est.setInputNormals(normals);
            curv_est.setRadiusSearch(m_leafSize);
            curv_est.setInputCloud(nn_cloud_without_normals);
            curv_est.compute(*local_principal_curvatures);
        }
        if(m_filter_method_geometry == "kpq")
        {
            local_principal_curvatures = dense_principal_curvatures;
        }

        ColorConversion& cc = ColorConversionStatic::getColorConversion();

        // check scores of all neighboring points
        for(unsigned idx = 0; idx < point_idxs.size(); idx++)
        {
            const PointNormalT &nn_point = points_with_normals->at(idx);
            int best_index_geo = -1;
            int best_index_color = -1;
            float last_geo_score = -1;
            float last_color_score = -1;

            if(m_filter_method_geometry == "curvature")
            {
                if(nn_point.curvature > best_geo)
                {
                    best_index_geo = int(idx);
                    last_geo_score = best_geo;
                    best_geo = nn_point.curvature;
                }
            }
            if(m_filter_method_geometry == "gaussian")
            {
                const pcl::PrincipalCurvatures &pc_point = local_principal_curvatures->at(idx);
                if(pc_point.pc1 * pc_point.pc2 > best_geo)
                {
                    best_index_geo = int(idx);
                    last_geo_score = best_geo;
                    best_geo = pc_point.pc1 * pc_point.pc2;
                }
            }
            // group these methods as they need an additional nearest neighbor search
            if(m_filter_method_geometry == "kpq" || m_filter_method_color == "colordistance")
            {
                std::vector<int> point_idxs_kpq;
                std::vector<float> point_dists_kpq;
                pts_with_normals_tree.radiusSearch(nn_point, m_leafSize, point_idxs_kpq, point_dists_kpq);
                if(m_filter_method_geometry == "kpq")
                {
                    float kpqval = computeKPQ(point_idxs, local_principal_curvatures);
                    if(kpqval > best_geo)
                    {
                        best_index_geo = int(idx);
                        last_geo_score = best_geo;
                        best_geo = kpqval;
                    }
                }
                if(m_filter_method_color == "colordistance")
                {
                    float color_score = computeColorScore(point_idxs_kpq, points_with_normals, nn_point, cc);
                    if(color_score > best_color)
                    {
                        best_index_color = int(idx);
                        last_color_score = best_color;
                        best_color = color_score;
                    }
                }
            }
            // consolidate found indices
            if(m_filter_method_geometry != "none" && m_filter_method_color == "none")
            {
                best_index = best_index_geo;
            }
            if(m_filter_method_geometry == "none" && m_filter_method_color != "none")
            {
                best_index = best_index_color;
            }
            if(m_filter_method_geometry != "none" && m_filter_method_color != "none")
            {
                if(best_index_geo != -1 && best_index_color != -1)
                {
                    // both, geometric and color filtering yielded the same index - accept it
                    best_index = int(idx);
                }
                else
                {
                    // geometric and color filtering did not agree on an indes - restore previous best scores
                    if(last_geo_score != -1) best_geo = last_geo_score;
                    if(last_color_score != -1) best_color = last_color_score;
                }
            }
        }

        if(best_index == -1)
        {
            return keypoint;
        }
        else
        {
            const PointNormalT& kp = points_with_normals->at(best_index);
            PointT res(kp.r, kp.g, kp.b);
            res.x = kp.x;
            res.y = kp.y;
            res.z = kp.z;
            return res;
        }
    }

    std::string KeypointsVoxelGridCulling::getTypeStatic()
    {
        return "VoxelGridCulling";
    }

    std::string KeypointsVoxelGridCulling::getType() const
    {
        return KeypointsVoxelGridCulling::getTypeStatic();
    }
}
