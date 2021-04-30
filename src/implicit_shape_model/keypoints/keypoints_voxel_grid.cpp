/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "keypoints_voxel_grid.h"

#define PCL_NO_PRECOMPILE
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <fstream>
#include <boost/algorithm/string.hpp>

namespace ism3d
{
    KeypointsVoxelGrid::KeypointsVoxelGrid()
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
    }

    KeypointsVoxelGrid::~KeypointsVoxelGrid()
    {
    }

    pcl::PointCloud<PointT>::ConstPtr KeypointsVoxelGrid::iComputeKeypoints(pcl::PointCloud<PointT>::ConstPtr points,
                                                                            pcl::PointCloud<PointT>::ConstPtr eigenValues,
                                                                            pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                            pcl::PointCloud<PointT>::Ptr pointsWithoutNaNNormals,
                                                                            pcl::PointCloud<PointT>::Ptr eigenValuesWithoutNan,
                                                                            pcl::PointCloud<pcl::Normal>::Ptr normalsWithoutNaN,
                                                                            pcl::search::Search<PointT>::Ptr search)
    {      
        int knn_kpq = 100;
        // TODO VS disable filtering in training

        boost::algorithm::to_lower(m_filter_method_geometry);
        boost::algorithm::to_lower(m_filter_method_color);
        boost::algorithm::to_lower(m_filter_type_geometry);
        boost::algorithm::to_lower(m_filter_type_color);

        if(m_filter_method_geometry == "none" && m_filter_method_color == "none")
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

            // create cloud containing filtered cloud points and filtered normals
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
            curv_est.setInputCloud(keypoints_without_normals);
            curv_est.setSearchSurface(pointsWithoutNaNNormals);
            curv_est.setInputNormals(normalsWithoutNaN);
            curv_est.setSearchMethod(search);
            curv_est.setRadiusSearch(m_leafSize);
            pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures (new pcl::PointCloud<pcl::PrincipalCurvatures>());
            curv_est.compute(*principal_curvatures);


            pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr dense_principal_curvatures (new pcl::PointCloud<pcl::PrincipalCurvatures>());
            if(m_filter_method_geometry == "kpq")
            {
                // TODO VS: is there a less costly way if I include these into normal computation?
                // densely estimate principle curvatures
                curv_est.setInputCloud(pointsWithoutNaNNormals);
                curv_est.setSearchSurface(pointsWithoutNaNNormals);
                curv_est.setInputNormals(normalsWithoutNaN);
                curv_est.setSearchMethod(search);
                curv_est.setRadiusSearch(m_leafSize);
                curv_est.compute(*dense_principal_curvatures);
            }

            LOG_INFO("Number of keypoints before filtering: " << keypoints_with_normals->size());
            std::vector<float> geo_scores;
            std::vector<float> color_scores;
            std::vector<float> color_scores_unsorted;

            pcl::KdTreeFLANN<PointNormalT> pts_with_normals_tree;
            pts_with_normals_tree.setInputCloud(points_with_normals); // NOTE: search on original cloud
            std::vector<int> point_idxs;
            std::vector<float> point_dists;

            ColorConversion& cc = ColorConversionStatic::getColorConversion();

            for(unsigned idx = 0; idx < keypoints_with_normals->size(); idx++)
            {
                // PCL curvature
                PointNormalT &reference_point = keypoints_with_normals->at(idx);

                if(m_filter_method_geometry == "curvature")
                {
                    geo_scores.push_back(reference_point.curvature);
                }
                if(m_filter_method_geometry == "gaussian")
                {
                    // gaussian curvature
                    const pcl::PrincipalCurvatures &pc_point = principal_curvatures->at(idx);
                    geo_scores.push_back(pc_point.pc1 * pc_point.pc2);
                    // overwrite curvature with current method's value
                    reference_point.curvature = pc_point.pc1 * pc_point.pc2;
                }

                // group these methods as they need nearest neighbor search
                if(m_filter_method_geometry == "kpq" || m_filter_method_color == "colordistance")
                {
                    pts_with_normals_tree.nearestKSearch(reference_point, knn_kpq, point_idxs, point_dists); // TODO VS: switch to radius search?
                    if(m_filter_method_geometry == "kpq")
                    {
                        // KPQ (keypoint quality)
                        float kpqval = computeKPQ(point_idxs, dense_principal_curvatures);
                        geo_scores.push_back(kpqval);
                        // overwrite curvature with current method's value
                        reference_point.curvature = kpqval;
                    }
                    if(m_filter_method_color == "colordistance")
                    {
                        float color_score = computeColorScore(point_idxs, points_with_normals, reference_point, cc);
                        color_scores.push_back(color_score);
                        color_scores_unsorted.push_back(color_score);
                    }
                }
            }

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
                m_filter_threshold_geometry = geo_scores.at(cutoff_index);
            }

            // get threshold corresponding to cutoff index
            if(m_filter_method_geometry != "none" && m_filter_type_geometry == "cutoff")
            {
                unsigned cutoff_index = unsigned(m_filter_cutoff_ratio * geo_scores.size());
                m_filter_threshold_geometry = geo_scores.at(cutoff_index);
            }

            if(m_filter_method_color != "none" && m_filter_type_color == "cutoff")
            {
                unsigned cutoff_index = unsigned(m_filter_cutoff_ratio * color_scores.size());
                m_filter_threshold_color = color_scores.at(cutoff_index);
            }




            pcl::PointCloud<PointT>::Ptr keypoints(new pcl::PointCloud<PointT>());
            for(unsigned idx = 0; idx < keypoints_with_normals->size(); idx++)
            {
                bool geo_passed = true;
                if(m_filter_method_geometry != "none")
                {
                    PointNormalT point = keypoints_with_normals->at(idx);
                    // NOTE: curvature corresponds to chosen geometry type value
                    if(point.curvature < m_filter_threshold_geometry)
                    {
                        geo_passed = false;
                    }
                }

                bool color_passed = true;
                if(m_filter_method_color != "none")
                {
                    if(color_scores_unsorted[idx] < m_filter_threshold_color)
                    {
                        color_passed = false;
                    }
                }

                if(geo_passed && color_passed)
                    keypoints->push_back(keypoints_without_normals->at(idx));
            }
            LOG_INFO("Number of keypoints after filtering: " << keypoints->size());
            return keypoints;
        }
    }

    // KeyPointQuality as presented by A.Mian et al. in
    // On the Repeatability and Quality of Keypoints for Local Feature-based 3D Object Retrieval from Cluttered Scenes
    // see equations 10 and 11
    // NOTE: for clarification a more complete equation is presented by F.Tombari et al. in
    // Performance Evaluation of 3D Keypoint DetectorsFederico Tombari
    // see equation 9
    float KeypointsVoxelGrid::computeKPQ(const std::vector<int> &pointIdxs,
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





    float KeypointsVoxelGrid::computeColorScore(const std::vector<int> &pointIdxs,
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

    std::string KeypointsVoxelGrid::getTypeStatic()
    {
        return "VoxelGrid";
    }

    std::string KeypointsVoxelGrid::getType() const
    {
        return KeypointsVoxelGrid::getTypeStatic();
    }
}
