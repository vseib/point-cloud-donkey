/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "keypoints.h"
#include "keypoints_factory.h"
#include "../utils/utils.h"

namespace ism3d
{
    Keypoints::Keypoints()
        : m_numThreads(1)
    {
    }

    Keypoints::~Keypoints()
    {
    }

    pcl::PointCloud<PointT>::ConstPtr Keypoints::operator()(pcl::PointCloud<PointT>::ConstPtr points,
                                                            pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                            pcl::PointCloud<PointT>::ConstPtr pointsWithoutNaNNormals,
                                                            pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                            pcl::search::Search<PointT>::Ptr search)
    {
        LOG_ASSERT(points->size() == normals->size());
        LOG_ASSERT(pointsWithoutNaNNormals->size() == normalsWithoutNaN->size());

        pcl::PointCloud<PointT>::ConstPtr keypoints = iComputeKeypoints(points, normals,
                                                                        pointsWithoutNaNNormals, normalsWithoutNaN,
                                                                        search);

        LOG_INFO("found " << keypoints->size() << " keypoints");

        return keypoints;
    }

    void Keypoints::setNumThreads(int numThreads)
    {
        m_numThreads = numThreads;
    }

    int Keypoints::getNumThreads() const
    {
        return m_numThreads;
    }
}

