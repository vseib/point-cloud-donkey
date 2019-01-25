/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_vfh.h"

#define PCL_NO_PRECOMPILE
#include <pcl/features/vfh.h>


namespace ism3d
{
    FeaturesVFH::FeaturesVFH()
    {
    }

    FeaturesVFH::~FeaturesVFH()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesVFH::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                       pcl::search::Search<PointT>::Ptr search)
    {
        // Object for storing the VFH descriptor.
        pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptor(new pcl::PointCloud<pcl::VFHSignature308>);

        // VFH estimation object.
        pcl::VFHEstimation<PointT, pcl::Normal, pcl::VFHSignature308> vfh;
        vfh.setInputCloud(pointCloudWithoutNaNNormals);
        vfh.setSearchSurface(pointCloudWithoutNaNNormals);
        vfh.setInputNormals(normalsWithoutNaN);
        vfh.setSearchMethod(search);
        // Optionally, we can normalize the bins of the resulting histogram, using the total number of points.
        vfh.setNormalizeBins(true);
        // Also, we can normalize the SDC with the maximum size found between
        // the centroid and any of the cluster's points.
        vfh.setNormalizeDistance(false);
        vfh.compute(*descriptor);

        float cloud_radius = getCloudRadius(pointCloudWithoutNaNNormals);

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(descriptor->size());

        for (int i = 0; i < (int)descriptor->size(); i++)
        {
            ISMFeature& feature = features->at(i);
            feature.globalDescriptorRadius = cloud_radius;

            const pcl::VFHSignature308& vfhd = descriptor->at(i);

            // store the descriptor
            feature.descriptor.resize(308);
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = vfhd.histogram[j];
        }

        return features;
    }

    std::string FeaturesVFH::getTypeStatic()
    {
        return "VFH";
    }

    std::string FeaturesVFH::getType() const
    {
        return FeaturesVFH::getTypeStatic();
    }
}
