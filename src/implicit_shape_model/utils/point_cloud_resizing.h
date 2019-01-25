/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_POINTCLOUDRESIZING_H
#define ISM3D_POINTCLOUDRESIZING_H

#include "utils.h"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace ism3d
{
    /**
     * @brief The PointCloudResizing class
     * This class allows for fast resizing of an organized point cloud while retaining
     * the organized structure. This way, subsequent PCL algorithms can operate on the
     * organized point cloud much faster.
     * It works by downsampling the input point cloud and computing the mean over
     * neighboring points within a radius of the point cloud resolution of the original
     * cloud.
     * Note however, that the point cloud might contain NaN values which cannot be
     * removed without loosing the organized structure.
     */
    class PointCloudResizing
    {
    public:
        PointCloudResizing();
        ~PointCloudResizing();

        /**
         * @brief Resizes the input point cloud while retaining the organized structure.
         * @param points the input point cloud
         * @return a resized organized point cloud
         */
        pcl::PointCloud<PointT>::ConstPtr operator()(pcl::PointCloud<PointT>::ConstPtr points) const;

        /**
         * @brief Set the resizing factor by which the input cloud resolution will be multiplied.S
         * @param factor the resizing factor (factor >= 1)
         */
        void setResizeFactor(float factor);

    private:
        Eigen::Vector3f getPoint(const Eigen::Matrix3f&, int, int, float) const;
        float getRadius(const Eigen::Matrix3f&, const Eigen::Vector3f&, int, int, int, int) const;

        float m_resizeFactor;
    };
}

#endif // ISM3D_POINTCLOUDRESIZING_H
