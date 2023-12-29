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
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>

// TODO VS: rename keypoint_culling to keypoint_selection
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

        addParameter(m_max_similar_color_distance, "MaxSimilarColorDistance", 0.01f);
        addParameter(m_filter_cutoff_ratio, "FilterCutoffRatio", 0.5f); // TODO VS: this param is used as "omit this percentage of keypoints"
                                                                        //    thesis uses it as "keep this percentage of keypoints"  --> change it here

        addParameter(m_disable_filter_in_training, "DisableFilterInTraining", true);
        addParameter(m_combine_filters, "CombineFilters", std::string("RequireCombinedList")); // TODO VS use names from thesis (union, intersection etc)

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

        // TODO VS: find out why this happens
        if(pointsWithoutNaNNormals->size() == 0)
        {
            m_filter_method_geometry = "none";
            m_filter_method_color = "none";
            LOG_WARN("Can not filter keypoints! Proceeding without filtering!");
        }

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
            // TODO VS: move kpq to own keypoint class
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
            LOG_INFO("Initial number of keypoints with normals: " << keypoints_with_normals->size());

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
            // NOTE: the next block is for computing curvature with the same radius as the grid size
            // alternatively curvature can be taken from the already computed normals (usually with lower grid size)
            else if(m_filter_method_geometry == "curvature")
            {
                // need to run new voxel grid to create keypoints without normals (pcl point type)
                // TODO VS: is this necessary? there is keypoints without normals above
                pcl::VoxelGrid<PointT> voxel_grid;
                voxel_grid.setInputCloud(pointsWithoutNaNNormals);
                voxel_grid.setLeafSize(m_leafSize, m_leafSize, m_leafSize);
                pcl::PointCloud<PointT>::Ptr temp_keypoints(new pcl::PointCloud<PointT>());
                voxel_grid.filter(*temp_keypoints);

                pcl::PointCloud<pcl::Normal>::Ptr keypoint_normals(new pcl::PointCloud<pcl::Normal>());
                if (pointsWithoutNaNNormals->isOrganized())
                {
                    pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> normalEst;
                    normalEst.setInputCloud(temp_keypoints);
                    normalEst.setSearchSurface(pointsWithoutNaNNormals);
                    normalEst.setNormalEstimationMethod(normalEst.AVERAGE_3D_GRADIENT);
                    normalEst.setRadiusSearch(m_leafSize);
                    normalEst.setMaxDepthChangeFactor(0.02f);
                    normalEst.setNormalSmoothingSize(10.0f);
                    normalEst.useSensorOriginAsViewPoint(); // flip normals toward scene viewpoint
                    normalEst.compute(*keypoint_normals);
                }
                else
                {
                    pcl::NormalEstimationOMP<PointT, pcl::Normal> normalEst;
                    normalEst.setInputCloud(temp_keypoints);
                    normalEst.setSearchSurface(pointsWithoutNaNNormals);
                    normalEst.setRadiusSearch(m_leafSize);
                    normalEst.setNumberOfThreads(0);
                    normalEst.setViewPoint(0,0,0);
                    normalEst.compute(*keypoint_normals);
                }

                // assign the curvature computed with grid size to the previously computed normals
                for(unsigned nidx = 0; nidx < keypoint_normals->size(); nidx++)
                {
                    keypoints_with_normals->at(nidx).curvature = keypoint_normals->at(nidx).curvature;
                }
            }

            LOG_INFO("Number of keypoints before filtering: " << keypoints_without_normals->size());

            auto[geo_scores, color_scores, combined_scores] = getScoresForKeypoints(points_with_normals, keypoints_with_normals, principal_curvatures);
            auto[threshold_geo, threshold_color, threshold_combined] = computeThresholds(geo_scores, color_scores, combined_scores);

            // prepare resulting keypoints cloud
            pcl::PointCloud<PointT>::Ptr keypoints(new pcl::PointCloud<PointT>());
            for(unsigned idx = 0; idx < keypoints_without_normals->size(); idx++)
            {
                // check geometric quality measure
                bool geo_passed = true;
                if(m_filter_method_geometry != "none")
                {
                    if(geo_scores[idx] < threshold_geo)
                    {
                        geo_passed = false;
                    }
                }

                // check color quality measure
                bool color_passed = true;
                if(m_filter_method_color != "none")
                {
                    if(color_scores[idx] < threshold_color)
                    {
                        color_passed = false;
                    }
                }

                // check combined list of geometric and color quality measure
                bool combined_passed = true;
                if(m_filter_method_geometry != "none" && m_filter_method_color != "none")
                {
                    if(combined_scores[idx] < threshold_combined)
                    {
                        combined_passed = false;
                    }
                }

                // in case of two quality measures selected, decide if keypoint can be accepted
                bool accept_keypoint = false;
                if(m_filter_method_geometry != "none" && m_filter_method_color != "none")
                {
                    if(m_combine_filters == "RequireOne")
                    {
                        accept_keypoint = geo_passed || color_passed;
                    }
                    else if(m_combine_filters == "RequireBoth")
                    {
                        accept_keypoint = geo_passed && color_passed;
                    }
                    else if(m_combine_filters == "RequireCombinedList")
                    {
                        accept_keypoint = combined_passed;
                    }
                }
                else
                {
                    // in case of one quality measures selected: use && since initialization was with true
                    accept_keypoint = geo_passed && color_passed;
                }

                if(accept_keypoint)
                {
                    if(m_refine_position) // TODO VS: part 2 of chapter 18 - change code!!!
                    {
                        PointT keypoint = refineKeypointPosition(geo_scores, color_scores, combined_scores,
                                                                 keypoints_without_normals->at(idx), keypoints_without_normals,
                                                                 geo_passed, color_passed);
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


    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> KeypointsVoxelGridCulling::getScoresForKeypoints(
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

        // compute geo and color scores individually
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

        // compute combined scores
        const auto [gmin, gmax] = std::minmax_element(geo_scores.begin(),geo_scores.end());
        const auto [cmin, cmax] = std::minmax_element(color_scores.begin(),color_scores.end());
        std::vector<float> combined_scores(keypoints_with_normals->size(), 0.0f);
        for(unsigned idx = 0; idx < keypoints_with_normals->size(); idx++)
        {
            float geo_norm = (geo_scores[idx] - *gmin) / *gmax;
            float color_norm = (color_scores[idx] - *cmin) / *cmax;
            combined_scores.at(idx) = geo_norm + color_norm;
        }


        return std::make_tuple(geo_scores, color_scores, combined_scores);
    }

    std::tuple<float, float, float> KeypointsVoxelGridCulling::computeThresholds(
            const std::vector<float> &geo_scores_orig,
            const std::vector<float> &color_scores_orig,
            const std::vector<float> &combined_scores_orig)
    {
        float threshold_geo = std::numeric_limits<float>::min();
        float threshold_color = std::numeric_limits<float>::min();
        float threshold_combined = std::numeric_limits<float>::min();

        // copy original lists for sorting
        std::vector<float> geo_scores;
        std::copy(geo_scores_orig.begin(), geo_scores_orig.end(), std::back_inserter(geo_scores));
        std::vector<float> color_scores;
        std::copy(color_scores_orig.begin(), color_scores_orig.end(), std::back_inserter(color_scores));
        std::vector<float> combined_scores;
        std::copy(combined_scores_orig.begin(), combined_scores_orig.end(), std::back_inserter(combined_scores));

        // sort to determine cutoff threshold
        std::sort(geo_scores.begin(), geo_scores.end());
        std::sort(color_scores.begin(), color_scores.end());
        std::sort(combined_scores.begin(), combined_scores.end());

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

        if(m_filter_method_geometry != "none" && m_filter_method_color != "none"
                && m_filter_type_geometry == "cutoff" && m_filter_type_color == "cutoff")
        {
            unsigned cutoff_index = unsigned(m_filter_cutoff_ratio * combined_scores.size());
            threshold_combined = combined_scores.at(cutoff_index);
        }

        // don't change the thresholds if they are user specified
        if(m_filter_method_geometry != "none" && m_filter_type_geometry == "threshold")
        {
            threshold_geo = m_filter_threshold_geometry;
        }
        if(m_filter_method_color != "none" && m_filter_type_color == "threshold")
        {
            threshold_color = m_filter_threshold_color;
        }

        return std::make_tuple(threshold_geo, threshold_color, threshold_combined);
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
            const std::vector<float> &geo_scores,
            const std::vector<float> &color_scores,
            const std::vector<float> &combined_scores,
            const PointT &keypoint,
            const pcl::PointCloud<PointT>::Ptr &all_keypoints,
            const bool geo_passed,
            const bool color_passed)
    {
        pcl::KdTreeFLANN<PointT> keypoints_tree;
        keypoints_tree.setInputCloud(all_keypoints);
        std::vector<int> point_idxs; // NOTE: result will also contain the keypoint itself
        std::vector<float> point_dists;
        keypoints_tree.radiusSearch(keypoint, m_leafSize*1.5, point_idxs, point_dists);

        int best_index_geo = -1;
        int best_index_color = -1;
        int best_index_combined = -1;
        float best_score_geo = -1;
        float best_score_color = -1;
        float best_score_combined = -1;

        // check scores of all neighboring points
        for(int idx : point_idxs)
        {
            if(m_filter_method_geometry != "none")
            {
                if(geo_scores[idx] > best_score_geo)
                {
                    best_index_geo = idx;
                    best_score_geo = geo_scores[idx];
                }
            }
            if(m_filter_method_color != "none")
            {
                if(color_scores[idx] > best_score_color)
                {
                    best_index_color = idx;
                    best_score_color = color_scores[idx];
                }
            }
            if(m_filter_method_geometry != "none" && m_filter_method_color != "none")
            {
                if(combined_scores[idx] > best_score_combined)
                {
                    best_index_combined = idx;
                    best_score_combined = combined_scores[idx];
                }
            }
        }

        int best_index = -1;
        // consolidate found indices
        if(m_filter_method_geometry == "none")
        {
            best_index = best_index_color;
        }
        else if(m_filter_method_color == "none")
        {
            best_index = best_index_geo;
        }
        else // both are not "none"
        {
            // use both metric, init with combined index
            // also used as fallback for the case that geo and color indices are different
            best_index = best_index_combined; // case: m_combine_filters == "RequireCombinedList"

            if(geo_passed && !color_passed)   // case: m_combine_filters == "RequireOne"
                best_index = best_index_geo;
            if(color_passed && !geo_passed)   // case: m_combine_filters == "RequireOne"
                best_index = best_index_color;
            if(color_passed && geo_passed)    // case: m_combine_filters == "RequireBoth"
            {
                if(best_index_geo == best_index_color)
                {
                    best_index = best_index_geo;
                }
            }
        }

        if(best_index == -1)
        {
            LOG_WARN("Could not refine keypoint location!");
            return keypoint;
        }
        else
        {
            const PointT& kp = all_keypoints->at(best_index);

            PointT result;
            result.x = 0.5 * (kp.x + keypoint.x);
            result.y = 0.5 * (kp.y + keypoint.y);
            result.z = 0.5 * (kp.z + keypoint.z);
            result.r = int(0.5 * (kp.r + keypoint.r));
            result.g = int(0.5 * (kp.g + keypoint.g));
            result.b = int(0.5 * (kp.b + keypoint.b));
            return result;
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
