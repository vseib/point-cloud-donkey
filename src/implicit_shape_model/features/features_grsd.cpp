/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_grsd.h"

#define PCL_NO_PRECOMPILE
#include <pcl/features/grsd.h>

namespace ism3d
{
FeaturesGRSD::FeaturesGRSD()
{
    addParameter(m_radius, "Radius", 0.1);
}

FeaturesGRSD::~FeaturesGRSD()
{
}

pcl::PointCloud<ISMFeature>::Ptr FeaturesGRSD::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                   pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                   pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                   pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                   pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                   pcl::PointCloud<PointT>::Ptr keypoints,
                                                                   pcl::search::Search<PointT>::Ptr search)
{
    // Object for storing the GRSD descriptors for each point.
    pcl::PointCloud<pcl::GRSDSignature21>::Ptr descriptors(new pcl::PointCloud<pcl::GRSDSignature21>());

    // GRSD estimation object.
    pcl::GRSDEstimation<PointT, pcl::Normal, pcl::GRSDSignature21> grsd;
    grsd.setInputCloud(pointCloudWithoutNaNNormals);
    grsd.setInputNormals(normalsWithoutNaN);
    grsd.setSearchMethod(search);
    // Search radius, to look for neighbors. Note: the value given here has to be
    // larger than the radius used to estimate the normals.
    grsd.setRadiusSearch(m_radius);

    grsd.compute(*descriptors);

    // compute cloud radius
    float cloud_radius = getCloudRadius(pointCloudWithoutNaNNormals);

    // create descriptor point cloud
    pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
    features->resize(descriptors->size());

    for (int i = 0; i < (int)descriptors->size(); i++)
    {
        ISMFeature& feature = features->at(i);
        feature.globalDescriptorRadius = cloud_radius;
        const pcl::GRSDSignature21& grsdsig = descriptors->at(i);

        // store the descriptor
        feature.descriptor.resize(21);
        for (int j = 0; j < feature.descriptor.size(); j++)
            feature.descriptor[j] = grsdsig.histogram[j];
    }

    normalizeDescriptors(features);

    return features;
}

std::string FeaturesGRSD::getTypeStatic()
{
    return "GRSD";
}

std::string FeaturesGRSD::getType() const
{
    return FeaturesGRSD::getTypeStatic();
}
}
