/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "keypoints_harris3d.h"

namespace ism3d
{
    KeypointsHarris3D::KeypointsHarris3D()
    {
        addParameter(m_radius, "Radius", 0.05f);
        addParameter(m_threshold, "Threshold", 0.0001f);
        addParameter(m_nonMaxSupression, "NonMaxSupression", true);
        addParameter(m_refine, "Refine", true);
        addParameter(m_responseMethod, "ResponseMethod", pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI, pcl::Normal>::HARRIS);
    }

    KeypointsHarris3D::~KeypointsHarris3D()
    {
    }

    pcl::PointCloud<PointT>::ConstPtr KeypointsHarris3D::iComputeKeypoints(pcl::PointCloud<PointT>::ConstPtr points,
                                                                           pcl::PointCloud<PointT>::ConstPtr eigenValues,
                                                                           pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                           pcl::PointCloud<PointT>::Ptr pointsWithoutNaNNormals,
                                                                           pcl::PointCloud<PointT>::Ptr eigenValuesWithoutNan,
                                                                           pcl::PointCloud<pcl::Normal>::Ptr normalsWithoutNaN,
                                                                           pcl::search::Search<PointT>::Ptr search)
    {
        pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI, pcl::Normal> harris;

        if (points->isOrganized()) {
            harris.setInputCloud(points);
            harris.setNormals(normals);
        }
        else {
            harris.setInputCloud(pointsWithoutNaNNormals);
            harris.setNormals(normalsWithoutNaN);
        }

        harris.setSearchMethod(search);
        harris.setMethod(m_responseMethod);
        harris.setRadius(m_radius);
        harris.setThreshold(m_threshold);
        harris.setNonMaxSupression(m_nonMaxSupression);
        harris.setRefine(m_refine);
        harris.setNumberOfThreads(getNumThreads());

        // compute keypoints
        pcl::PointCloud<pcl::PointXYZI> keypoints;
        harris.compute(keypoints);

        // discard all information but XYZ
        pcl::PointCloud<PointT>::Ptr outKeypoints(new pcl::PointCloud<PointT>());
        pcl::copyPointCloud(keypoints, *outKeypoints);
        return outKeypoints;
    }

    std::string KeypointsHarris3D::getTypeStatic()
    {
        return "Harris3D";
    }

    std::string KeypointsHarris3D::getType() const
    {
        return KeypointsHarris3D::getTypeStatic();
    }
}
