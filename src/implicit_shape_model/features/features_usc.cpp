/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_usc.h"

#define PCL_NO_PRECOMPILE
#include <pcl/features/usc.h>


namespace ism3d
{
    FeaturesUSC::FeaturesUSC()
    {
        addParameter(m_radius, "Radius", 0.1);
    }

    FeaturesUSC::~FeaturesUSC()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesUSC::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                       pcl::search::Search<PointT>::Ptr search)
    {

        // Object for storing the USC descriptors for each point.
        pcl::PointCloud<pcl::UniqueShapeContext1960>::Ptr descriptors(new pcl::PointCloud<pcl::UniqueShapeContext1960>());

        // USC estimation object.
        pcl::UniqueShapeContext<PointT, pcl::UniqueShapeContext1960, pcl::ReferenceFrame> usc;
        usc.setSearchSurface(pointCloudWithoutNaNNormals);
        usc.setInputCloud(keypoints);

        // Search radius, to look for neighbors. It will also be the radius of the support sphere.
        usc.setRadiusSearch(m_radius);
        // The minimal radius value for the search sphere, to avoid being too sensitive in bins close to the center of the sphere.
        usc.setMinimalRadius(m_radius / 10.0);
//        // Radius used to compute the local point density for the neighbors (the density is the number of points within that radius).
//        usc.setPointDensityRadius(m_radius / 5.0);

        // Set the radius to compute the Local Reference Frame.
        float lrf_radius = m_radius <= 0.1 ? 0.1 : m_radius - 0.1;
        usc.setLocalRadius(lrf_radius);

        usc.compute(*descriptors);

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(descriptors->size());

        for (int i = 0; i < (int)descriptors->size(); i++)
        {
            ISMFeature& feature = features->at(i);
            const pcl::UniqueShapeContext1960& usc = descriptors->at(i);

            // store the descriptor
            feature.descriptor.resize(1960);
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = usc.descriptor[j];
        }

        return features;
    }

    std::string FeaturesUSC::getTypeStatic()
    {
        return "USC";
    }

    std::string FeaturesUSC::getType() const
    {
        return FeaturesUSC::getTypeStatic();
    }
}
