/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_esf.h"

#define PCL_NO_PRECOMPILE
#include <pcl/features/esf.h>


namespace ism3d
{
    FeaturesESF::FeaturesESF()
    {
    }

    FeaturesESF::~FeaturesESF()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesESF::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                       pcl::search::Search<PointT>::Ptr search)
    {

        // Object for storing the ESF descriptor.
        pcl::PointCloud<pcl::ESFSignature640>::Ptr descriptor(new pcl::PointCloud<pcl::ESFSignature640>);

        // ESF estimation object.
        pcl::ESFEstimation<PointT, pcl::ESFSignature640> esf;
        esf.setInputCloud(pointCloudWithoutNaNNormals);

        esf.compute(*descriptor);

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(descriptor->size());

        // compute cloud radius
        float cloud_radius = getCloudRadius(pointCloudWithoutNaNNormals);

        for (int i = 0; i < (int)descriptor->size(); i++)
        {
            ISMFeature& feature = features->at(i);
            feature.globalDescriptorRadius = cloud_radius;
            const pcl::ESFSignature640& esf = descriptor->at(i);

            // store the descriptor
            feature.descriptor.resize(640);
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = esf.histogram[j];
        }

        return features;
    }

    std::string FeaturesESF::getTypeStatic()
    {
        return "ESF";
    }

    std::string FeaturesESF::getType() const
    {
        return FeaturesESF::getTypeStatic();
    }
}
