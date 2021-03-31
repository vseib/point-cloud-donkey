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
        bool filter_keypoints = true;
        int knn_kpq = 100;
        float filter_threshold = 0.005f; // e.g. 0.005 good for method "curvature"
        float filter_cutoff = 0.5f; // value between 0 and 1
        std::string method = "curvature"; // one of: "curvature", "gaussian", "kpq"
        std::string type = "cutoff"; // one of: "cutoff", "threshold", "auto"
        // TODO VS: rethink auto

        if(!filter_keypoints)
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
            if(method != "curvature" && method != "gaussian" && method != "kpq")
            {
                LOG_ERROR("Unsupported keypoint filter method: " << method);
                exit(1);
            }
            if(type != "cutoff" && type != "threshold" && type != "auto")
            {
                LOG_ERROR("Unsupported keypoint filter type: " << type);
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



            LOG_INFO("--------- vorher: " << keypoints_with_normals->size());
            std::vector<float> curv;
            std::vector<float> ks;
            std::vector<float> kpq;

            std::vector<float> ks_unsorted;
            std::vector<float> kpq_unsorted;


            pcl::KdTreeFLANN<PointNormalT> pts_with_normals_tree;
            pts_with_normals_tree.setInputCloud(points_with_normals); // NOTE: search on original cloud
            std::vector<int> point_idxs;
            std::vector<float> point_dists;

            for(unsigned idx = 0; idx < keypoints_with_normals->size(); idx++)
            {
                // PCL curvature
                PointNormalT pn1 = keypoints_with_normals->at(idx);
                curv.push_back(pn1.curvature);

                // KPQ (keypoint quality)
                point_idxs.clear();
                point_dists.clear();
                pts_with_normals_tree.nearestKSearch(pn1, knn_kpq, point_idxs, point_dists);
                float kpqval = computeKPQ(point_idxs, eigenValuesWithoutNan);
                kpq.push_back(kpqval);

                // gaussian curvature
                PointT pn2 = eigenValuesWithoutNan->at(point_idxs[0]);
                ks.push_back(pn2.x * pn2.y);

                // TODO VS: temp (?) for later usage
                // idea for later: overwrite curvature by the value value that is currently used
                kpq_unsorted.push_back(kpqval);
                ks_unsorted.push_back(pn2.x * pn2.y);
            }

            // sort to determine cutoff threshold
            std::sort(curv.begin(), curv.end());
            std::sort(ks.begin(), ks.end());
            std::sort(kpq.begin(), kpq.end());




            // create histograms
            int hist_size = 100;
            float hist_step_curv = (curv.back() - curv.front()) / hist_size;
            float hist_step_ks = (ks.back() - ks.front()) / hist_size;
            float hist_step_kpq = (kpq.back() - kpq.front()) / hist_size;
            std::vector<int> curv_hist(hist_size, 0);
            std::vector<int> ks_hist(hist_size, 0);
            std::vector<int> kpq_hist(hist_size, 0);

            for(auto val : curv)
            {
    //            LOG_INFO("val: " << val << "   front: " << curv.front() << "   step: " << hist_step_curv);
                int curv_bin = (val - curv.front()) / hist_step_curv;
    //            LOG_INFO(" bin: " << curv_bin << "  " << curv_hist.size())
                curv_bin = curv_bin < 0 ? 0 : curv_bin;
                curv_bin = curv_bin >= curv_hist.size() ? curv_hist.size() - 1 : curv_bin;
                curv_hist[curv_bin]++;
            }

            for(auto val : ks)
            {
                int ks_bin = (val - ks.front()) / hist_step_ks;
                ks_bin = ks_bin < 0 ? 0 : ks_bin;
                ks_bin = ks_bin >= ks_hist.size() ? ks_hist.size() - 1 : ks_bin;
                ks_hist[ks_bin]++;
            }

            for(auto val : kpq)
            {
                int kpq_bin = (val - kpq.front()) / hist_step_kpq;
                kpq_bin = kpq_bin < 0 ? 0 : kpq_bin;
                kpq_bin = kpq_bin >= kpq_hist.size() ? kpq_hist.size() - 1 : kpq_bin;
                kpq_hist[kpq_bin]++;
            }



//            std::ofstream curv_file;
//            curv_file.open("/home/vseib/Desktop/curv_hist.txt", std::ios_base::app);
//            for(auto val : curv_hist)
//            {
//                curv_file << val << ", ";
//            }
//            curv_file << std::endl;
//            curv_file.close();

//            std::ofstream ks_file;
//            ks_file.open("/home/vseib/Desktop/ks_hist.txt", std::ios_base::app);
//            for(auto val : curv_hist)
//            {
//                ks_file << val << ", ";
//            }
//            ks_file << std::endl;
//            ks_file.close();

//            std::ofstream kpq_file;
//            kpq_file.open("/home/vseib/Desktop/kpq_hist.txt", std::ios_base::app);
//            for(auto val : kpq_hist)
//            {
//                kpq_file << val << ", ";
//            }
//            kpq_file << std::endl;
//            kpq_file.close();



            LOG_INFO("------------------------------------");
            unsigned cutoff_index = unsigned(filter_cutoff * curv.size());

            // will automatically determine cutoff index
            if(type == "auto")
            {
                std::vector<int> hist;
                if(method == "curvature")
                    hist = curv_hist;
                if(method == "gaussian")
                    hist = ks_hist;
                if(method == "kpq")
                    hist = kpq_hist;

                // TODO VS: experimental based on histogramm - check code again, check if compatible with KPQ!
                cutoff_index = 0;
                for(int xxx = 0; xxx < hist.size()-2; xxx++)
                {
                    cutoff_index += hist[xxx];
                    int diff_1 = hist[xxx+1] - hist[xxx];
                    int diff_2 = hist[xxx+2] - hist[xxx+1];
                    if(diff_1*2 < diff_2)
                    {
                        cutoff_index += hist[xxx+1];
                        break;
                    }
                }
            }


            float thresh_curv = curv.at(cutoff_index);
            float thresh_ks = ks.at(cutoff_index);
            float thresh_kpq = kpq.at(cutoff_index);


            if(type == "cutoff")
            {
                if(method == "curvature")
                    filter_threshold = thresh_curv;
                if(method == "gaussian")
                    filter_threshold = thresh_ks;
                if(method == "kpq")
                    filter_threshold = thresh_kpq;
            }

            LOG_INFO(curv.front() << "  " << thresh_curv  << "  " << curv.back());
            LOG_INFO(ks.front() << "  " << thresh_ks << "  " << ks.back());
            LOG_INFO(kpq.front() << "  " << thresh_kpq << "  " << kpq.back());

            // copy only point information without normals
            pcl::PointCloud<PointT>::Ptr keypoints_temp(new pcl::PointCloud<PointT>());
            pcl::copyPointCloud(*keypoints_with_normals, *keypoints_temp);

            pcl::PointCloud<PointT>::Ptr keypoints(new pcl::PointCloud<PointT>());

            for(unsigned idx = 0; idx < keypoints_with_normals->size(); idx++)
            {
                PointNormalT pn1 = keypoints_with_normals->at(idx);

                float compare;
                if(method == "curvature")
                    compare = pn1.curvature;
                if(method == "gaussian")
                    compare = ks_unsorted[idx];
                if(method == "kpq")
                    compare = kpq_unsorted[idx];

                if(compare >= filter_threshold)
                    keypoints->push_back(keypoints_temp->at(idx));
            }

            LOG_INFO("--------- nachher: " << keypoints->size());

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
                                         pcl::PointCloud<PointT>::Ptr eigen_values)
    {
        float max_k1 = std::numeric_limits<float>::min();
        float min_k2 = std::numeric_limits<float>::max();
        float max_K = std::numeric_limits<float>::min();
        float min_K = std::numeric_limits<float>::max();
        float kpq = 0.0f;

        for(int idx : pointIdxs)
        {
            PointT eigen = eigen_values->at(idx);
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


    std::string KeypointsVoxelGrid::getTypeStatic()
    {
        return "VoxelGrid";
    }

    std::string KeypointsVoxelGrid::getType() const
    {
        return KeypointsVoxelGrid::getTypeStatic();
    }
}

