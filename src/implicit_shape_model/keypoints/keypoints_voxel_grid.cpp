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

namespace ism3d
{
    KeypointsVoxelGrid::KeypointsVoxelGrid()
    {
        addParameter(m_leafSize, "LeafSize", 0.1f);
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
        // TODO VS params
        int knn_kpq = 100;
        float filter_threshold_geometry = 0.005f; // e.g. 0.005 good for method "curvature"
        float filter_cutoff = 0.5f; // value between 0 and 1
        std::string filter_method_geometry = "kpq"; // one of: "curvature", "gaussian", "kpq", "none"
        std::string filter_type_geometry = "cutoff"; // one of: "cutoff", "threshold", "auto"

        std::string filter_method_color = "none"; // one of: "color_distance", "none"
        std::string filter_type_color = "threshold";  // one of: "cutoff", "threshold"
        float filter_threshold_color = 0.02f; // e.g. 0.02 good for color
        // TODO VS extract parameter from color distance method
        // TODO VS disable filtering in training


        if(filter_method_geometry == "none" && filter_method_color == "none")
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
            if(filter_method_geometry != "none" && filter_method_geometry != "curvature" &&
                    filter_method_geometry != "gaussian" && filter_method_geometry != "kpq")
            {
                LOG_ERROR("Unsupported keypoint filter method: " << filter_method_geometry);
                exit(1);
            }
            if(filter_type_geometry != "cutoff" && filter_type_geometry != "threshold" && filter_type_geometry != "auto")
            {
                LOG_ERROR("Unsupported keypoint filter type: " << filter_type_geometry);
                exit(1);
            }

            if(filter_method_geometry == "kpq" && filter_type_geometry == "auto")
            {
                LOG_ERROR("Filter type " << filter_type_geometry << " is not supported with filter method " << filter_method_geometry);
                exit(1);
            }

            if(filter_method_color != "none" && filter_method_color != "color_distance")
            {
                LOG_ERROR("Unsupported keypoint color filter method: " << filter_method_color);
                exit(1);
            }
            if(filter_type_color != "cutoff" && filter_type_color != "threshold")
            {
                LOG_ERROR("Unsupported keypoint color filter type: " << filter_type_color);
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
                // TODO VS: refactor: only search if needed by selected method
                pts_with_normals_tree.nearestKSearch(reference_point, knn_kpq, point_idxs, point_dists);

                if(filter_method_geometry == "curvature")
                {
                    geo_scores.push_back(reference_point.curvature);
                }
                if(filter_method_geometry == "gaussian")
                {
                    // gaussian curvature
                    PointT eig = eigenValuesWithoutNan->at(point_idxs[0]);
                    geo_scores.push_back(eig.x * eig.y);
                    // overwrite curvature with current method's value
                    reference_point.curvature = eig.x * eig.y;
                }
                if(filter_method_geometry == "kpq")
                {
                    // KPQ (keypoint quality)
                    float kpqval = computeKPQ(point_idxs, eigenValuesWithoutNan);
                    geo_scores.push_back(kpqval);
                    // overwrite curvature with current method's value
                    reference_point.curvature = kpqval;
                }
                if(filter_method_color == "color_distance")
                {
                    float color_score = computeColorScore(point_idxs, points_with_normals, reference_point, cc);
                    color_scores.push_back(color_score);
                    color_scores_unsorted.push_back(color_score);
                }
            }

            // sort to determine cutoff threshold
            std::sort(geo_scores.begin(), geo_scores.end());
            std::sort(color_scores.begin(), color_scores.end());

            // automatically determine cutoff index
            if(filter_method_geometry != "none" && filter_type_geometry == "auto")
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
                filter_threshold_geometry = geo_scores.at(cutoff_index);
            }

            // get threshold corresponding to cutoff index
            if(filter_method_geometry != "none" && filter_type_geometry == "cutoff")
            {
                unsigned cutoff_index = unsigned(filter_cutoff * geo_scores.size());
                filter_threshold_geometry = geo_scores.at(cutoff_index);
            }

            if(filter_method_color != "none" && filter_type_color == "cutoff")
            {
                unsigned cutoff_index = unsigned(filter_cutoff * color_scores.size());
                filter_threshold_color = color_scores.at(cutoff_index);
            }


            // copy only point information without normals
            pcl::PointCloud<PointT>::Ptr keypoints_temp(new pcl::PointCloud<PointT>());
            pcl::copyPointCloud(*keypoints_with_normals, *keypoints_temp);

            pcl::PointCloud<PointT>::Ptr keypoints(new pcl::PointCloud<PointT>());
            for(unsigned idx = 0; idx < keypoints_with_normals->size(); idx++)
            {
                bool geo_passed = true;
                if(filter_method_geometry != "none")
                {
                    PointNormalT point = keypoints_with_normals->at(idx);
                    // NOTE: curvature corresponds to chosen geometry type value
                    if(point.curvature < filter_threshold_geometry)
                    {
                        geo_passed = false;
                    }
                }

                bool color_passed = true;
                if(filter_method_color != "none")
                {
                    if(color_scores_unsorted[idx] < filter_threshold_color)
                    {
                        color_passed = false;
                    }
                }

                if(geo_passed && color_passed)
                    keypoints->push_back(keypoints_temp->at(idx));
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
                                         pcl::PointCloud<PointT>::Ptr eigen_values) const
    {
        float max_k1 = std::numeric_limits<float>::min();
        float min_k2 = std::numeric_limits<float>::max();
        float max_K = std::numeric_limits<float>::min();
        float min_K = std::numeric_limits<float>::max();
        float kpq = 0.0f;

        for(int idx : pointIdxs)
        {
            const PointT &eigen = eigen_values->at(idx);
            float k_1 = eigen.x;
            float k_2 = eigen.y;
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
        int num_color_bins = 20;
        float threshold = 1.0f/num_color_bins;
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
