/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_KEYPOINTSVOXELGRID_H
#define ISM3D_KEYPOINTSVOXELGRID_H

#include "keypoints.h"
#include "../third_party/pcl_color_conversion/color_conversion.h"

namespace ism3d
{
    /**
     * @brief The KeypointsVoxelGrid class
     * Computes keypoints by uniformly sampling the input point cloud with a low resolution
     * and computing the centroid of points within a voxel.
     */
    class KeypointsVoxelGrid
            : public Keypoints
    {
    public:
        KeypointsVoxelGrid();
        ~KeypointsVoxelGrid();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        pcl::PointCloud<PointT>::ConstPtr iComputeKeypoints(pcl::PointCloud<PointT>::ConstPtr points,
                                                            pcl::PointCloud<PointT>::ConstPtr eigenValues,
                                                            pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                            pcl::PointCloud<PointT>::Ptr pointsWithoutNaNNormals,
                                                            pcl::PointCloud<PointT>::Ptr eigenValuesWithoutNan,
                                                            pcl::PointCloud<pcl::Normal>::Ptr normalsWithoutNaN,
                                                            pcl::search::Search<PointT>::Ptr search);

        float computeKPQ(const std::vector<int> &pointIdxs,
                         pcl::PointCloud<PointT>::Ptr eigen_values) const;

        float computeColorScore(const std::vector<int> &pointIdxs,
                                pcl::PointCloud<PointNormalT>::Ptr points_with_normals,
                                const PointNormalT &ref,
                                const ColorConversion &cc) const;

    private:
        float m_leafSize;
    };
}

#endif // ISM3D_KEYPOINTSVOXELGRID_H
