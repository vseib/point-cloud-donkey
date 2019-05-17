/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2019, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_gasd.h"


namespace ism3d
{
    FeaturesGASD::FeaturesGASD()
    {
        addParameter(m_use_color, "GasdWithColor", false);
    }

    FeaturesGASD::~FeaturesGASD()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesGASD::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                       pcl::search::Search<PointT>::Ptr search)
    {
        size_t desc_dims;
        size_t num_desc;
        float* histogram;

        if(m_use_color)
        {
            pcl::PointCloud<pcl::GASDSignature984> descriptor;
            computeDescriptor(descriptor, pointCloudWithoutNaNNormals);
            num_desc = descriptor.size();
            desc_dims = descriptor.at(0).descriptorSize();
            histogram = descriptor.at(0).histogram;
        }
        else
        {
            pcl::PointCloud<pcl::GASDSignature512> descriptor;
            computeDescriptor(descriptor, pointCloudWithoutNaNNormals);
            num_desc = descriptor.size();
            desc_dims = descriptor.at(0).descriptorSize();
            histogram = descriptor.at(0).histogram;
        }

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(num_desc);

        // compute cloud radius
        float cloud_radius = getCloudRadius(pointCloudWithoutNaNNormals);

        for (int i = 0; i < (int) num_desc; i++)
        {
            ISMFeature& feature = features->at(i);
            feature.globalDescriptorRadius = cloud_radius;

            // store the descriptor
            feature.descriptor.resize(desc_dims);
            for (int j = 0; j < desc_dims; j++)
                feature.descriptor[j] = histogram[j];
        }

        return features;

//        // TODO VS: bad code - make it better
//        if(m_use_color)
//        {
//            // Output datasets
//            pcl::PointCloud<pcl::GASDSignature984>::Ptr descriptor(new pcl::PointCloud<pcl::GASDSignature984>());

//            // Create the GASD estimation class, and pass the input dataset to it
//            pcl::GASDColorEstimation<PointT, pcl::GASDSignature984> gasd_est;
//            gasd_est.setInputCloud(pointCloudWithoutNaNNormals);
//            gasd_est.compute(*descriptor);

//            // create descriptor point cloud
//            pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
//            features->resize(descriptor->size());

//            // compute cloud radius
//            float cloud_radius = getCloudRadius(pointCloudWithoutNaNNormals);
//            for (int i = 0; i < (int)descriptor->size(); i++)
//            {
//                ISMFeature& feature = features->at(i);
//                feature.globalDescriptorRadius = cloud_radius;
//                const pcl::GASDSignature984 &gasd = descriptor->at(i);
//                // store the descriptor
//                feature.descriptor.resize(gasd.descriptorSize());
//                for (int j = 0; j < feature.descriptor.size(); j++)
//                    feature.descriptor[j] = gasd.histogram[j];
//            }
//            return features;
//        }
//        else
//        {
//            // Output datasets
//            pcl::PointCloud<pcl::GASDSignature512>::Ptr descriptor(new pcl::PointCloud<pcl::GASDSignature512>());

//            // Create the GASD estimation class, and pass the input dataset to it
//            pcl::GASDColorEstimation<PointT, pcl::GASDSignature512> gasd_est;
//            gasd_est.setColorHistsSize(0);
//            gasd_est.setInputCloud(pointCloudWithoutNaNNormals);
//            gasd_est.compute(*descriptor);

//            // create descriptor point cloud
//            pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
//            features->resize(descriptor->size());

//            // compute cloud radius
//            float cloud_radius = getCloudRadius(pointCloudWithoutNaNNormals);

//            for (int i = 0; i < (int)descriptor->size(); i++)
//            {
//                ISMFeature& feature = features->at(i);
//                feature.globalDescriptorRadius = cloud_radius;
//                const pcl::GASDSignature512 &gasd = descriptor->at(i);
//                // store the descriptor
//                feature.descriptor.resize(gasd.descriptorSize());
//                for (int j = 0; j < feature.descriptor.size(); j++)
//                    feature.descriptor[j] = gasd.histogram[j];
//            }
//            return features;
//        }
    }

    std::string FeaturesGASD::getTypeStatic()
    {
        return "GASD";
    }

    std::string FeaturesGASD::getType() const
    {
        return FeaturesGASD::getTypeStatic();
    }
}
