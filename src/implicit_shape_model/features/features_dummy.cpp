/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_dummy.h"

#define PCL_NO_PRECOMPILE

namespace ism3d
{
    FeaturesDummy::FeaturesDummy()
    {
    }

    FeaturesDummy::~FeaturesDummy()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesDummy::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                       pcl::search::Search<PointT>::Ptr search)
    {

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(1); // global features all have one descriptor

        // store the dummy descriptor
        ISMFeature& feature = features->at(0);
        feature.descriptor.resize(1);
        feature.descriptor[0] = 0;

        return features;
    }

    std::string FeaturesDummy::getTypeStatic()
    {
        return "Dummy";
    }

    std::string FeaturesDummy::getType() const
    {
        return FeaturesDummy::getTypeStatic();
    }
}
