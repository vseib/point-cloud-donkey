/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_narf.h"

#define PCL_NO_PRECOMPILE
#include <pcl/range_image/range_image_planar.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/features/narf_descriptor.h>

namespace ism3d
{
    FeaturesNARF::FeaturesNARF()
    {
        addParameter(m_radius, "Radius", 0.1);
    }

    FeaturesNARF::~FeaturesNARF()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesNARF::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints_unused,
                                                                         pcl::search::Search<PointT>::Ptr search)
    {

        // TODO VS: test with organized point clouds

        // NOTE: code taken from http://robotica.unileon.es/index.php/PCL/OpenNI_tutorial_4:_3D_object_recognition_(descriptors)#NARF

        // first we need to create a range image from the point cloud
        LOG_INFO("creating a planar projection of the point cloud");

        // Parameters needed by the planar range image object:
        // Image size. Both Kinect and Xtion work at 640x480.
        int imageSizeX = 640;
        int imageSizeY = 480;
        // Center of projection. here, we choose the middle of the image.
        float centerX = 640.0f / 2.0f;
        float centerY = 480.0f / 2.0f;
        // Focal length. The value seen here has been taken from the original depth images.
        // It is safe to use the same value vertically and horizontally.
        float focalLengthX = 525.0f, focalLengthY = focalLengthX;
        // Sensor pose. Thankfully, the cloud includes the data.
        Eigen::Affine3f sensorPose = Eigen::Affine3f(Eigen::Translation3f(pointCloud->sensor_origin_[0],
                                     pointCloud->sensor_origin_[1],
                                     pointCloud->sensor_origin_[2])) *
                                     Eigen::Affine3f(pointCloud->sensor_orientation_);
        // Noise level. If greater than 0, values of neighboring points will be averaged.
        // This would set the search radius (e.g., 0.03 == 3cm).
        float noiseLevel = 0.0f;
        // Minimum range. If set, any point closer to the sensor than this will be ignored.
        float minimumRange = 0.0f;

        // Planar range image object.
        pcl::RangeImagePlanar rangeImage;
        rangeImage.createFromPointCloudWithFixedSize(*pointCloud, imageSizeX, imageSizeY,
                centerX, centerY, focalLengthX, focalLengthY, sensorPose, pcl::RangeImage::CAMERA_FRAME, noiseLevel, minimumRange);

        // Object for storing the keypoints' indices.
        pcl::PointCloud<int>::Ptr keypoints(new pcl::PointCloud<int>);

        // Border extractor object.
        pcl::RangeImageBorderExtractor borderExtractor;
        // Keypoint detection object.
        pcl::NarfKeypoint detector(&borderExtractor);
        detector.setRangeImage(&rangeImage);
        // The support size influences how big the surface of interest will be, when finding keypoints from the border information.
        detector.getParameters().support_size = m_radius;
        detector.compute(*keypoints);

        // Object for storing the NARF descriptors.
        pcl::PointCloud<pcl::Narf36>::Ptr descriptors(new pcl::PointCloud<pcl::Narf36>);

        // The NARF estimator needs the indices in a vector, not a cloud.
        std::vector<int> keypoints_list;
        keypoints_list.resize(keypoints->points.size());
        for (unsigned int i = 0; i < keypoints->size(); ++i)
            keypoints_list[i] = keypoints->points[i];
        // NARF estimation object.
        pcl::NarfDescriptor narf(&rangeImage, &keypoints_list);
        // Support size: choose the same value you used for keypoint extraction.
        narf.getParameters().support_size = m_radius;
        narf.getParameters().rotation_invariant = true;
        narf.compute(*descriptors);

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(descriptors->size());

        for (int i = 0; i < (int)descriptors->size(); i++)
        {
            ISMFeature& feature = features->at(i);
            const pcl::Narf36& narfd = descriptors->at(i);

            // store the descriptor
            feature.descriptor.resize(36);
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = narfd.descriptor[j];
        }

        return features;
    }

    std::string FeaturesNARF::getTypeStatic()
    {
        return "NARF";
    }

    std::string FeaturesNARF::getType() const
    {
        return FeaturesNARF::getTypeStatic();
    }
}
