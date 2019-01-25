/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_KEYPOINTS_H
#define ISM3D_KEYPOINTS_H

#include "../utils/json_object.h"
#include "../utils/utils.h"

#define PCL_NO_PRECOMPILE
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/search.h>

namespace ism3d
{
    /**
     * @brief The Keypoints class
     * Works as a functor and computes keypoints on the input point cloud. The resulting
     * keypoints are of the same type as the point cloud, but do not necessarily are
     * an element of the original input cloud.
     */
    class Keypoints
            : public JSONObject
    {
    public:
        virtual ~Keypoints();

        /**
         * @brief Interface function to compute keypoints on the input point cloud.
         * @param points the input point cloud
         * @param normals normals for the input point cloud (normals->size() == points->size())
         * @param pointsWithoutNaNNormals the input point cloud without points with corresponding
         * NaN normals
         * @param normalsWithoutNaN normals without NaN values (normalsWithoutNaN->size() == pointsWithoutNaNNormals->size())
         * @param search the input search tree
         * @return a point cloud containing keypoints
         */
        pcl::PointCloud<PointT>::ConstPtr operator()(pcl::PointCloud<PointT>::ConstPtr points,
                                                     pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                     pcl::PointCloud<PointT>::ConstPtr pointsWithoutNaNNormals,
                                                     pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                     pcl::search::Search<PointT>::Ptr search);

        /**
         * @brief Set the number of threads to use. The derived classes do not need to use it.
         * @param numThreads the number of threads to use
         */
        void setNumThreads(int numThreads);

    protected:
        Keypoints();

        virtual pcl::PointCloud<PointT>::ConstPtr iComputeKeypoints(pcl::PointCloud<PointT>::ConstPtr,
                                                                    pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                                    pcl::PointCloud<PointT>::ConstPtr,
                                                                    pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                                    pcl::search::Search<PointT>::Ptr) = 0;

        int getNumThreads() const;

    private:
        int m_numThreads;
    };
}

#endif // ISM3D_KEYPOINTS_H
