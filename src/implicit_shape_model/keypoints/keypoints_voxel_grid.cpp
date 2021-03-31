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

        // compute keypoints
        pcl::PointCloud<PointT>::Ptr keypoints(new pcl::PointCloud<PointT>());
        voxelGrid.filter(*keypoints);

        return keypoints;
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

