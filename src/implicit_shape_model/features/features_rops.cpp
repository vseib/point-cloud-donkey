/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_rops.h"

#define PCL_NO_PRECOMPILE
#include <pcl/features/rops_estimation.h>
#include <pcl/surface/gp3.h>

namespace ism3d
{
    FeaturesROPS::FeaturesROPS()
    {
        addParameter(m_radius, "Radius", 0.1);
    }

    FeaturesROPS::~FeaturesROPS()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesROPS::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                         pcl::search::Search<PointT>::Ptr search)
    {
        const int desc_length = 135;

        // Object for storing the ROPS descriptor for each point.
        pcl::PointCloud<pcl::Histogram<desc_length> >::Ptr descriptors(new pcl::PointCloud<pcl::Histogram<desc_length> >());

        // Object for storing both the points and the normals.
        pcl::PointCloud<PointNormalT>::Ptr cloudNormals(new pcl::PointCloud<PointNormalT>);

        LOG_INFO("starting triangulation");

        // Perform triangulation.
        pcl::concatenateFields(*pointCloudWithoutNaNNormals, *normalsWithoutNaN, *cloudNormals);
        pcl::search::KdTree<PointNormalT>::Ptr kdtree(new pcl::search::KdTree<PointNormalT>);
        kdtree->setInputCloud(cloudNormals);
        pcl::GreedyProjectionTriangulation<PointNormalT> triangulation;
        pcl::PolygonMesh triangles;
        triangulation.setSearchRadius(m_radius);
        triangulation.setMu(2.5);
        triangulation.setMaximumNearestNeighbors(1000);
        triangulation.setMaximumSurfaceAngle(45 * (M_PI/180.0)); // 45 degrees.
        triangulation.setNormalConsistency(false);
        triangulation.setMinimumAngle(10 * (M_PI/180)); // 10 degrees.
        triangulation.setMaximumAngle(120 * (M_PI/180)); // 120 degrees.
        triangulation.setInputCloud(cloudNormals);
        triangulation.setSearchMethod(kdtree);
        triangulation.reconstruct(triangles);
        LOG_INFO("triangulation finished");

        // RoPs estimation object.
        pcl::ROPSEstimation<PointT, pcl::Histogram<desc_length> > rops;
        rops.setInputCloud(keypoints);
        rops.setSearchSurface(pointCloudWithoutNaNNormals);
        rops.setSearchMethod(search);
        rops.setRadiusSearch(m_radius);
        rops.setTriangles(triangles.polygons);
        // Number of partition bins that is used for distribution matrix calculation.
        rops.setNumberOfPartitionBins(5);
        // The greater the number of rotations is, the bigger the resulting descriptor.
        // Make sure to change the histogram size accordingly.
        rops.setNumberOfRotations(3);
        // Support radius that is used to crop the local surface of the point.
        rops.setSupportRadius(m_radius);
        rops.compute(*descriptors);

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(descriptors->size());

        for (int i = 0; i < (int)descriptors->size(); i++)
        {
            ISMFeature& feature = features->at(i);
            const pcl::Histogram<desc_length>& ropsd = descriptors->at(i);

            // store the descriptor
            feature.descriptor.resize(desc_length);
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = ropsd.histogram[j];
        }

        return features;
    }

    std::string FeaturesROPS::getTypeStatic()
    {
        return "RoPS";
    }

    std::string FeaturesROPS::getType() const
    {
        return FeaturesROPS::getTypeStatic();
    }
}
