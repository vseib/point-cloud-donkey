/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_3dsc.h"

#define PCL_NO_PRECOMPILE
#include <pcl/features/3dsc.h>


namespace ism3d
{
    Features3DSC::Features3DSC()
    {
        addParameter(m_radius, "Radius", 0.1);
    }

    Features3DSC::~Features3DSC()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr Features3DSC::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                       pcl::search::Search<PointT>::Ptr search)
    {

        // Object for storing the 3DSC descriptors for each point.
        pcl::PointCloud<pcl::ShapeContext1980>::Ptr descriptors(new pcl::PointCloud<pcl::ShapeContext1980>());

        // 3DSC estimation object.
        pcl::ShapeContext3DEstimation<PointT, pcl::Normal, pcl::ShapeContext1980> sc3d;
        sc3d.setInputCloud(keypoints);
        sc3d.setSearchSurface(pointCloudWithoutNaNNormals);
        sc3d.setInputNormals(normalsWithoutNaN);
        sc3d.setSearchMethod(search);
        // Search radius, to look for neighbors. It will also be the radius of the support sphere.
        sc3d.setRadiusSearch(m_radius);
        // The minimal radius value for the search sphere, to avoid being too sensitive in bins close to the center of the sphere.
        sc3d.setMinimalRadius(m_radius / 10.0);
        // Radius used to compute the local point density for the neighbors (the density is the number of points within that radius).
//        sc3d.setPointDensityRadius(0.05 / 5.0);

        sc3d.compute(*descriptors);

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(descriptors->size());

        for (int i = 0; i < (int)descriptors->size(); i++)
        {
            ISMFeature& feature = features->at(i);
            const pcl::ShapeContext1980& sc = descriptors->at(i);

            // store the descriptor
            feature.descriptor.resize(1980);
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = sc.descriptor[j];
        }

        return features;
    }

    std::string Features3DSC::getTypeStatic()
    {
        return "3DSC";
    }

    std::string Features3DSC::getType() const
    {
        return Features3DSC::getTypeStatic();
    }
}
