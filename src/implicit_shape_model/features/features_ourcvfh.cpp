/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_ourcvfh.h"

#define PCL_NO_PRECOMPILE
#include <pcl/features/our_cvfh.h>


namespace ism3d
{
FeaturesOURCVFH::FeaturesOURCVFH()
{
}

FeaturesOURCVFH::~FeaturesOURCVFH()
{
}

pcl::PointCloud<ISMFeature>::Ptr FeaturesOURCVFH::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                      pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                      pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                      pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                      pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                      pcl::PointCloud<PointT>::Ptr keypoints,
                                                                      pcl::search::Search<PointT>::Ptr search)
{
    // NOTE: taken from http://robotica.unileon.es/index.php/PCL/OpenNI_tutorial_4:_3D_object_recognition_(descriptors)

    // Object for storing the OUR-CVFH descriptors.
    pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors(new pcl::PointCloud<pcl::VFHSignature308>);

    // OUR-CVFH estimation object.
    pcl::OURCVFHEstimation<PointT, pcl::Normal, pcl::VFHSignature308> ourcvfh;
    ourcvfh.setInputCloud(pointCloudWithoutNaNNormals);
    ourcvfh.setSearchSurface(pointCloudWithoutNaNNormals);
    ourcvfh.setInputNormals(normalsWithoutNaN);
    ourcvfh.setSearchMethod(search);
    ourcvfh.setEPSAngleThreshold(10.0 / 180.0 * M_PI); // 10 degrees, NOTE: was 5 degrees, but this lead to seg-fauls with some point clouds
    ourcvfh.setCurvatureThreshold(1.0);
    ourcvfh.setNormalizeBins(false);
    // Set the minimum axis ratio between the SGURF axes. At the disambiguation phase,
    // this will decide if additional Reference Frames need to be created, if ambiguous.
    ourcvfh.setAxisRatio(0.8);
    ourcvfh.compute(*descriptors);

    float cloud_radius = getCloudRadius(pointCloudWithoutNaNNormals);

    // create descriptor point cloud
    pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
    features->resize(descriptors->size());

    for (int i = 0; i < (int)descriptors->size(); i++)
    {
        ISMFeature& feature = features->at(i);
        feature.globalDescriptorRadius = cloud_radius;
        const pcl::VFHSignature308& ourcvfhd = descriptors->at(i);

        // store the descriptor
        feature.descriptor.resize(308);
        for (int j = 0; j < feature.descriptor.size(); j++)
            feature.descriptor[j] = ourcvfhd.histogram[j];
    }

    normalizeDescriptors(features);

    return features;
}

std::string FeaturesOURCVFH::getTypeStatic()
{
    return "OURCVFH";
}

std::string FeaturesOURCVFH::getType() const
{
    return FeaturesOURCVFH::getTypeStatic();
}
}
