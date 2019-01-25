/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_usc_global.h"

#define PCL_NO_PRECOMPILE
#include <pcl/features/usc.h>
#include <pcl/common/centroid.h>


namespace ism3d
{
FeaturesUSCGlobal::FeaturesUSCGlobal()
{
}

FeaturesUSCGlobal::~FeaturesUSCGlobal()
{
}

pcl::PointCloud<ISMFeature>::Ptr FeaturesUSCGlobal::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                        pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                        pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                        pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                        pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                        pcl::PointCloud<PointT>::Ptr keypoints,
                                                                        pcl::search::Search<PointT>::Ptr search)
{

    // Object for storing the USC descriptors for each point.
    pcl::PointCloud<pcl::UniqueShapeContext1960>::Ptr descriptors(new pcl::PointCloud<pcl::UniqueShapeContext1960>());

    // USC estimation object.
    pcl::UniqueShapeContext<PointT, pcl::UniqueShapeContext1960, pcl::ReferenceFrame> usc;
    usc.setSearchSurface(pointCloudWithoutNaNNormals);

    // compute cloud radius
    float cloud_radius = getCloudRadius(pointCloudWithoutNaNNormals);
    usc.setRadiusSearch(cloud_radius);

    // for global USC use centroid instead of keypoints
    Eigen::Vector4f centroid4f;
    pcl::compute3DCentroid(*pointCloudWithoutNaNNormals, centroid4f);
    pcl::PointCloud<PointT>::Ptr centroid_cloud(new pcl::PointCloud<PointT>());
    PointT centroid_point;
    centroid_point.x = centroid4f[0];
    centroid_point.y = centroid4f[1];
    centroid_point.z = centroid4f[2];
    centroid_cloud->push_back(centroid_point);
    centroid_cloud->height = 1;
    centroid_cloud->width = 1;
    centroid_cloud->is_dense = false;

    usc.setInputCloud(centroid_cloud);

    // Search radius, to look for neighbors. It will also be the radius of the support sphere.
    usc.setRadiusSearch(cloud_radius);
    // The minimal radius value for the search sphere, to avoid being too sensitive in bins close to the center of the sphere.
    usc.setMinimalRadius(cloud_radius / 10.0);
    //        // Radius used to compute the local point density for the neighbors (the density is the number of points within that radius).
    //        usc.setPointDensityRadius(m_radius / 5.0);

    // Set the radius to compute the Local Reference Frame.
    float lrf_radius = cloud_radius <= 0.1 ? 0.1 : cloud_radius - 0.1;
    usc.setLocalRadius(lrf_radius);


    usc.compute(*descriptors);

    // create descriptor point cloud
    pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
    features->resize(descriptors->size());

    for (int i = 0; i < (int)descriptors->size(); i++)
    {
        ISMFeature& feature = features->at(i);
        feature.globalDescriptorRadius = cloud_radius;
        const pcl::UniqueShapeContext1960& usc = descriptors->at(i);

        // store the descriptor
        feature.descriptor.resize(1960);
        for (int j = 0; j < feature.descriptor.size(); j++)
            feature.descriptor[j] = usc.descriptor[j];
    }

    return features;
}

std::string FeaturesUSCGlobal::getTypeStatic()
{
    return "USC_GLOBAL";
}

std::string FeaturesUSCGlobal::getType() const
{
    return FeaturesUSCGlobal::getTypeStatic();
}
}
