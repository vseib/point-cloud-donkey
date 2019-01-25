/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_cshot_global.h"

#define PCL_NO_PRECOMPILE
#include <pcl/features/shot_omp.h>
#include <pcl/common/centroid.h>

namespace ism3d
{
    FeaturesCSHOTGlobal::FeaturesCSHOTGlobal()
    {
    }

    FeaturesCSHOTGlobal::~FeaturesCSHOTGlobal()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesCSHOTGlobal::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                       pcl::search::Search<PointT>::Ptr search)
    {

        pcl::SHOTColorEstimationOMP<PointT, pcl::Normal, pcl::SHOT1344> shotEst;

        if (pointCloud->isOrganized()) {
            shotEst.setSearchSurface(pointCloud);
            shotEst.setInputNormals(normals);
        }
        else {
            shotEst.setSearchSurface(pointCloudWithoutNaNNormals);
            shotEst.setInputNormals(normalsWithoutNaN);
        }

        // compute cloud radius
        float cloud_radius = getCloudRadius(pointCloudWithoutNaNNormals);
        shotEst.setRadiusSearch(cloud_radius);

        // for global SHOT use centroid instead of keypoints
        Eigen::Vector4f centroid4f;
        pcl::compute3DCentroid(*pointCloudWithoutNaNNormals, centroid4f);
        pcl::PointCloud<PointT>::Ptr centroid_cloud(new pcl::PointCloud<PointT>());
        PointT centroid_point;
        centroid_point.x = centroid4f[0];
        centroid_point.y = centroid4f[1];
        centroid_point.z = centroid4f[2];
        centroid_cloud->push_back(centroid_point);
        centroid_cloud->height = 1;
        centroid_cloud->width = 1;
        centroid_cloud->is_dense = false;
        shotEst.setInputCloud(centroid_cloud);
        keypoints = centroid_cloud;

        // for global SHOT: compute centroid reference frame
        pcl::PointCloud<pcl::ReferenceFrame>::Ptr centroid_frame(new pcl::PointCloud<pcl::ReferenceFrame>());
        pcl::SHOTLocalReferenceFrameEstimationOMP<PointT, pcl::ReferenceFrame> refEst;
        refEst.setRadiusSearch(cloud_radius);
        refEst.setInputCloud(centroid_cloud);
        refEst.setSearchSurface(pointCloudWithoutNaNNormals);
        refEst.setSearchMethod(search);
        refEst.compute(*centroid_frame);
        shotEst.setInputReferenceFrames(centroid_frame);
        shotEst.setSearchMethod(search);
        referenceFrames = centroid_frame; // forward frame back to caller

        // compute features
        pcl::PointCloud<pcl::SHOT1344>::Ptr shotFeatures(new pcl::PointCloud<pcl::SHOT1344>());
        shotEst.compute(*shotFeatures);

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(shotFeatures->size());

        for (int i = 0; i < (int)shotFeatures->size(); i++)
        {
            ISMFeature& feature = features->at(i);
            feature.globalDescriptorRadius = cloud_radius;
            const pcl::SHOT1344& shot = shotFeatures->at(i);

            // store the descriptor
            feature.descriptor.resize(1344);
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = shot.descriptor[j];
        }

        return features;
    }

    std::string FeaturesCSHOTGlobal::getTypeStatic()
    {
        return "CSHOT_GLOBAL";
    }

    std::string FeaturesCSHOTGlobal::getType() const
    {
        return FeaturesCSHOTGlobal::getTypeStatic();
    }
}
