/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "keypoints_sift3d.h"

namespace ism3d
{
    KeypointsSIFT3D::KeypointsSIFT3D()
    {
        addParameter(m_radius, "Radius", 0.05f);
    }

    KeypointsSIFT3D::~KeypointsSIFT3D()
    {
    }

    pcl::PointCloud<PointT>::ConstPtr KeypointsSIFT3D::iComputeKeypoints(pcl::PointCloud<PointT>::ConstPtr points,
                                                                           pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                           pcl::PointCloud<PointT>::ConstPtr pointsWithoutNaNNormals,
                                                                           pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                           pcl::search::Search<PointT>::Ptr search)
    {
        // create new point cloud and set curvature as intensity (SIFT 3D will then use curvature as "intensity")
        pcl::PointCloud<pcl::PointXYZI>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZI>());

        pcl::SIFTKeypoint<pcl::PointXYZI, pcl::PointXYZI> sift;

        if (points->isOrganized())
        {
            for(int i = 0; i < points->size(); i++)
            {
                pcl::PointXYZI p;
                p.x = points->at(i).x;
                p.y = points->at(i).y;
                p.z = points->at(i).z;
                p.intensity = normals->at(i).curvature;
                newCloud->push_back(p);
            }
            sift.setInputCloud(newCloud);
        }
        else
        {
            for(int i = 0; i < pointsWithoutNaNNormals->size(); i++)
            {
                pcl::PointXYZI p;
                p.x = pointsWithoutNaNNormals->at(i).x;
                p.y = pointsWithoutNaNNormals->at(i).y;
                p.z = pointsWithoutNaNNormals->at(i).z;
                p.intensity = normalsWithoutNaN->at(i).curvature;
                newCloud->push_back(p);
            }
            sift.setInputCloud(newCloud);
        }

        pcl::search::Search<pcl::PointXYZI>::Ptr searchTree;
        if (points->isOrganized())
        {
            searchTree = pcl::search::OrganizedNeighbor<pcl::PointXYZI>::Ptr(new pcl::search::OrganizedNeighbor<pcl::PointXYZI>());
        }
        else
        {
            searchTree = pcl::search::KdTree<pcl::PointXYZI>::Ptr(new pcl::search::KdTree<pcl::PointXYZI>());
        }

        sift.setSearchMethod(searchTree);
        sift.setScales(m_radius, 4, 3);
        sift.setMinimumContrast(0.0);

        // compute keypoints
        pcl::PointCloud<pcl::PointXYZI> keypoints;
        sift.compute(keypoints);

        // discard all information but XYZ
        pcl::PointCloud<PointT>::Ptr outKeypoints(new pcl::PointCloud<PointT>());
        pcl::copyPointCloud(keypoints, *outKeypoints);
        return outKeypoints;
    }

    std::string KeypointsSIFT3D::getTypeStatic()
    {
        return "SIFT3D";
    }

    std::string KeypointsSIFT3D::getType() const
    {
        return KeypointsSIFT3D::getTypeStatic();
    }
}
