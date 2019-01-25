/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_cvfh.h"

#define PCL_NO_PRECOMPILE
#include <pcl/features/cvfh.h>


namespace ism3d
{
    FeaturesCVFH::FeaturesCVFH()
    {
    }

    FeaturesCVFH::~FeaturesCVFH()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesCVFH::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                       pcl::search::Search<PointT>::Ptr search)
    {
        // NOTE: taken from http://robotica.unileon.es/index.php/PCL/OpenNI_tutorial_4:_3D_object_recognition_(descriptors)#VFH

        // Object for storing the CVFH descriptors.
        pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors(new pcl::PointCloud<pcl::VFHSignature308>);

        // CVFH estimation object.
        pcl::CVFHEstimation<PointT, pcl::Normal, pcl::VFHSignature308> cvfh;
        cvfh.setInputCloud(pointCloudWithoutNaNNormals);
        cvfh.setSearchSurface(pointCloudWithoutNaNNormals);
        cvfh.setInputNormals(normalsWithoutNaN);
        cvfh.setSearchMethod(search);
        // Set the maximum allowable deviation of the normals, for the region segmentation step.
        cvfh.setEPSAngleThreshold(10.0 / 180.0 * M_PI); // 10 degrees, NOTE: was 5 degrees, changed because the same change needed to be made in OURCVFH-feature
        // Set the curvature threshold (maximum disparity between curvatures), for the region segmentation step.
        cvfh.setCurvatureThreshold(1.0);
        // Set to true to normalize the bins of the resulting histogram,
        // using the total number of points. Note: enabling it will make CVFH
        // invariant to scale just like VFH, but the authors encourage the opposite.
        cvfh.setNormalizeBins(false);

        cvfh.compute(*descriptors);

        float cloud_radius = getCloudRadius(pointCloudWithoutNaNNormals);

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(descriptors->size());

        for (int i = 0; i < (int)descriptors->size(); i++)
        {
            ISMFeature& feature = features->at(i);
            feature.globalDescriptorRadius = cloud_radius;

            const pcl::VFHSignature308& cvfhd = descriptors->at(i);

            // store the descriptor
            feature.descriptor.resize(308);
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = cvfhd.histogram[j];
        }

        normalizeDescriptors(features);

        return features;
    }

    std::string FeaturesCVFH::getTypeStatic()
    {
        return "CVFH";
    }

    std::string FeaturesCVFH::getType() const
    {
        return FeaturesCVFH::getTypeStatic();
    }
}
