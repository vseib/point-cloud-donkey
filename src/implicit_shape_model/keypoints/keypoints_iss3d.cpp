/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "keypoints_iss3d.h"

#define PCL_NO_PRECOMPILE
#include <pcl/keypoints/iss_3d.h>

namespace ism3d
{
    KeypointsISS3D::KeypointsISS3D()
    {
        addParameter(m_salientRadius, "SalientRadius", 0.1);
        addParameter(m_nonMaxRadius, "NonMaxRadius", 0.05);
        addParameter(m_gamma21, "Gamma21", 0.975);
        addParameter(m_gamma32, "Gamma32", 0.975);
        addParameter(m_minNeighbors, "MinNeighbors", 5);
    }

    KeypointsISS3D::~KeypointsISS3D()
    {
    }

    pcl::PointCloud<PointT>::ConstPtr KeypointsISS3D::iComputeKeypoints(pcl::PointCloud<PointT>::ConstPtr points,
                                                                        pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                        pcl::PointCloud<PointT>::ConstPtr pointsWithoutNaNNormals,
                                                                        pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                        pcl::search::Search<PointT>::Ptr search)
    {
        //const double normalRadius = 0.06;
        //const double borderRadius = 0.01;

        pcl::ISSKeypoint3D<PointT, PointT> iss;

        if (points->isOrganized()) {
            iss.setInputCloud(points);
            iss.setNormals(normals);
        }
        else {
            iss.setInputCloud(pointsWithoutNaNNormals);
            iss.setNormals(normalsWithoutNaN);
        }

        iss.setSearchMethod(search);

        iss.setSalientRadius(m_salientRadius);
        iss.setNonMaxRadius(m_nonMaxRadius);

        //iss.setNormalRadius(normalRadius);
        //iss.setBorderRadius(borderRadius);

        iss.setThreshold21(m_gamma21);
        iss.setThreshold32(m_gamma32);
        iss.setMinNeighbors(m_minNeighbors);
        iss.setNumberOfThreads(4); // if more than that many threads are used the application crashes "sometimes"
//        iss.setNumberOfThreads(getNumThreads());

        pcl::PointCloud<PointT>::Ptr keypoints(new pcl::PointCloud<PointT>());

        iss.compute(*keypoints);

        return keypoints;
    }

    std::string KeypointsISS3D::getTypeStatic()
    {
        return "ISS3D";
    }

    std::string KeypointsISS3D::getType() const
    {
        return KeypointsISS3D::getTypeStatic();
    }
}
