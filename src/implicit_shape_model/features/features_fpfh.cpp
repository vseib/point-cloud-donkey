/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_fpfh.h"

#define PCL_NO_PRECOMPILE
#include <pcl/features/fpfh_omp.h>

namespace ism3d
{
    FeaturesFPFH::FeaturesFPFH()
    {
        addParameter(m_radius, "Radius", 0.1);
    }

    FeaturesFPFH::~FeaturesFPFH()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesFPFH::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                       pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                       pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                       pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                       pcl::PointCloud<pcl::ReferenceFrame>::Ptr,
                                                                       pcl::PointCloud<PointT>::Ptr keypoints,
                                                                       pcl::search::Search<PointT>::Ptr search)
    {
        pcl::FPFHEstimationOMP<PointT, pcl::Normal, pcl::FPFHSignature33> fpfhEst;

        if (pointCloud->isOrganized()) {
            fpfhEst.setSearchSurface(pointCloud);
            fpfhEst.setInputNormals(normals);
        }
        else {
            fpfhEst.setSearchSurface(pointCloudWithoutNaNNormals);
            fpfhEst.setInputNormals(normalsWithoutNaN);
        }

        fpfhEst.setInputCloud(keypoints);
        fpfhEst.setSearchMethod(search);
        fpfhEst.setNumberOfThreads(getNumThreads());

        // parameters
        fpfhEst.setRadiusSearch(m_radius);

        // compute features
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhFeatures(new pcl::PointCloud<pcl::FPFHSignature33>());
        fpfhEst.compute(*fpfhFeatures);

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(fpfhFeatures->size());

        for (int i = 0; i < (int)fpfhFeatures->size(); i++) {
            ISMFeature& feature = features->at(i);
            const pcl::FPFHSignature33& fpfh = fpfhFeatures->at(i);

            // store actual descriptor
            feature.descriptor.resize(33);
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = fpfh.histogram[j];
        }

        return features;
    }

    std::string FeaturesFPFH::getTypeStatic()
    {
        return "FPFH";
    }

    std::string FeaturesFPFH::getType() const
    {
        return FeaturesFPFH::getTypeStatic();
    }
}
