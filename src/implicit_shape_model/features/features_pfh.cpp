/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_pfh.h"

#define PCL_NO_PRECOMPILE
#include <pcl/features/pfh.h>

namespace ism3d
{
    FeaturesPFH::FeaturesPFH()
    {
        addParameter(m_radius, "Radius", 0.1);
    }

    FeaturesPFH::~FeaturesPFH()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesPFH::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                      pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                      pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                      pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                      pcl::PointCloud<pcl::ReferenceFrame>::Ptr,
                                                                      pcl::PointCloud<PointT>::Ptr keypoints,
                                                                      pcl::search::Search<PointT>::Ptr search)
    {
        pcl::PFHEstimation<PointT, pcl::Normal, pcl::PFHSignature125> pfhEst;

        if (pointCloud->isOrganized()) {
            pfhEst.setSearchSurface(pointCloud);
            pfhEst.setInputNormals(normals);
        }
        else {
            pfhEst.setSearchSurface(pointCloudWithoutNaNNormals);
            pfhEst.setInputNormals(normalsWithoutNaN);
        }

        pfhEst.setInputCloud(keypoints);
        pfhEst.setSearchMethod(search);

        // parameters
        pfhEst.setRadiusSearch(m_radius);

        // compute features
        pcl::PointCloud<pcl::PFHSignature125>::Ptr pfhFeatures(new pcl::PointCloud<pcl::PFHSignature125>());
        pfhEst.compute(*pfhFeatures);

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(pfhFeatures->size());

        for (int i = 0; i < (int)pfhFeatures->size(); i++) {
            ISMFeature& feature = features->at(i);
            const pcl::PFHSignature125& pfh = pfhFeatures->at(i);

            // store actual descriptor
            feature.descriptor.resize(125);
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = pfh.histogram[j];
        }

        return features;
    }

    std::string FeaturesPFH::getTypeStatic()
    {
        return "PFH";
    }

    std::string FeaturesPFH::getType() const
    {
        return FeaturesPFH::getTypeStatic();
    }
}
