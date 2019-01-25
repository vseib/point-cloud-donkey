/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_rift.h"

#define PCL_NO_PRECOMPILE
#include <pcl/point_types_conversion.h>
#include <pcl/features/intensity_gradient.h>
#include <pcl/features/rift.h>

namespace ism3d
{
    FeaturesRIFT::FeaturesRIFT()
    {
        addParameter(m_radius, "Radius", 0.1);
    }

    FeaturesRIFT::~FeaturesRIFT()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesRIFT::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                         pcl::search::Search<PointT>::Ptr search)
    {
        // TODO VS: test this with a colored point cloud, without color all features are NAN

        // NOTE: point cloud MUST contain RGB data, otherwise results will be meaningless
        const int rift_distance_bins = 4;
        const int rift_gradient_bins = 8;
        const int rift_size = rift_distance_bins * rift_gradient_bins;

        // Object for storing the point cloud with intensity value.
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudIntensity(new pcl::PointCloud<pcl::PointXYZI>);
        // Object for storing the point cloud with intensity value.
        pcl::PointCloud<pcl::PointXYZI>::Ptr keypointsIntensity(new pcl::PointCloud<pcl::PointXYZI>);

        // Convert the RGB to intensity.
        pcl::PointCloud<PointT> cloud = *pointCloudWithoutNaNNormals;
        pcl::PointCloudXYZRGBtoXYZI(cloud, *cloudIntensity);
        pcl::PointCloud<PointT> keypoints_cloud = *keypoints;
        pcl::PointCloudXYZRGBtoXYZI(keypoints_cloud, *keypointsIntensity);

        // Object for storing the intensity gradients.
        pcl::PointCloud<pcl::IntensityGradient>::Ptr gradients(new pcl::PointCloud<pcl::IntensityGradient>);

        // Compute the intensity gradients.
        pcl::IntensityGradientEstimation <pcl::PointXYZI, pcl::Normal, pcl::IntensityGradient,
            pcl::common::IntensityFieldAccessor<pcl::PointXYZI> > ge;
        ge.setInputCloud(cloudIntensity);
        ge.setInputNormals(normalsWithoutNaN);
        ge.setRadiusSearch(m_radius);
        ge.compute(*gradients);

        pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZI>);

        // Object for storing the RIFT descriptor for each point.
        pcl::PointCloud<pcl::Histogram<rift_size> >::Ptr descriptors(new pcl::PointCloud<pcl::Histogram<rift_size> >());

        // RIFT estimation object
        pcl::RIFTEstimation<pcl::PointXYZI, pcl::IntensityGradient, pcl::Histogram<rift_size> > rift;
        rift.setInputCloud(keypointsIntensity);
        rift.setSearchSurface(cloudIntensity);
        rift.setSearchMethod(kdtree);
        rift.setInputGradient(gradients); // Set the intensity gradients to use.
        rift.setRadiusSearch(m_radius); // Radius, to get all neighbors within.
        rift.setNrDistanceBins(rift_distance_bins); // Set the number of bins to use in the distance dimension.
        rift.setNrGradientBins(rift_gradient_bins); // Set the number of bins to use in the gradient orientation dimension.
        rift.compute(*descriptors);

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(descriptors->size());

        for (int i = 0; i < (int)descriptors->size(); i++)
        {
            ISMFeature& feature = features->at(i);
            const pcl::Histogram<rift_size>& riftd = descriptors->at(i);

            // store the descriptor
            feature.descriptor.resize(rift_size);
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = riftd.histogram[j];
        }

        return features;
    }

    std::string FeaturesRIFT::getTypeStatic()
    {
        return "RIFT";
    }

    std::string FeaturesRIFT::getType() const
    {
        return FeaturesRIFT::getTypeStatic();
    }
}
