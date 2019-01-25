/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_spin_image.h"

#define PCL_NO_PRECOMPILE
#include <pcl/features/spin_image.h>


namespace ism3d
{
    FeaturesSpinImage::FeaturesSpinImage()
    {
        addParameter(m_radius, "Radius", 0.1);
    }

    FeaturesSpinImage::~FeaturesSpinImage()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesSpinImage::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                         pcl::search::Search<PointT>::Ptr search)
    {
        // filtering normals, because for spin images only keypoint normals must be passed
        pcl::PointCloud<pcl::Normal>::Ptr filtered_normals(new pcl::PointCloud<pcl::Normal>());
        filtered_normals->resize(keypoints->size());

        // FIXME: there must be a more efficient way to do this
        for(int i = 0; i < keypoints->size(); i++)
        {
            PointT p = keypoints->at(i);
            for(int j = 0; j < pointCloudWithoutNaNNormals->size(); j++)
            {
                PointT q = pointCloudWithoutNaNNormals->at(j);
                if(p.x == q.x && p.y == q.y && p.z == q.z)
                {
                    filtered_normals->at(i) = normalsWithoutNaN->at(j);
                    break;
                }
            }
        }

        const int desc_length = 153;

        // Object for storing the spin image for each point.
        pcl::PointCloud<pcl::Histogram<desc_length> >::Ptr descriptors(new pcl::PointCloud<pcl::Histogram<desc_length> >());

        // Spin image estimation object.
        pcl::SpinImageEstimation<PointT, pcl::Normal, pcl::Histogram<desc_length> > si;
        si.setInputCloud(keypoints);
        si.setSearchSurface(pointCloudWithoutNaNNormals);
        si.setInputNormals(filtered_normals);
        si.setRadiusSearch(m_radius);
        // Set the resolution of the spin image (the number of bins along one dimension).
        si.setImageWidth(8);

        si.compute(*descriptors);

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(descriptors->size());

        for (int i = 0; i < (int)descriptors->size(); i++)
        {
            ISMFeature& feature = features->at(i);
            const pcl::Histogram<desc_length>& spin = descriptors->at(i);

            // store the descriptor
            feature.descriptor.resize(desc_length);
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = spin.histogram[j];
        }

        return features;
    }

    std::string FeaturesSpinImage::getTypeStatic()
    {
        return "SpinImage";
    }

    std::string FeaturesSpinImage::getType() const
    {
        return FeaturesSpinImage::getTypeStatic();
    }
}
