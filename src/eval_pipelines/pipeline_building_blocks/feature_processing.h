#pragma once

#include <pcl/features/feature.h>
#include "../../implicit_shape_model/utils/ism_feature.h"

using namespace ism3d;

typedef pcl::PointXYZRGB PointT;


class fp // feature processing
{
public:
    inline static int normal_method = 0;
    inline static float normal_radius = 0;
    inline static float keypoint_sampling_radius = 0;
    inline static float reference_frame_radius = 0;
    inline static std::string feature_type = "";
    inline static float feature_radius = 0;
};

void processPointCloud(
        const pcl::PointCloud<PointT>::Ptr cloud,
        pcl::PointCloud<PointT>::Ptr &keypoints_cleaned,
        pcl::PointCloud<ISMFeature>::Ptr &features_cleaned,
        pcl::PointCloud<pcl::Normal>::Ptr &normals_cleaned,
        pcl::PointCloud<pcl::ReferenceFrame>::Ptr &reference_frames_cleaned);

void computeNormals(
        const pcl::PointCloud<PointT>::Ptr cloud,
        const pcl::search::Search<PointT>::Ptr searchTree,
        pcl::PointCloud<pcl::Normal>::Ptr& normals);

void filterNormals(
        const pcl::PointCloud<pcl::Normal>::Ptr normals,
        const pcl::PointCloud<PointT>::Ptr cloud,
        pcl::PointCloud<pcl::Normal>::Ptr &normals_without_nan,
        pcl::PointCloud<PointT>::Ptr &cloud_without_nan);

void computeKeypoints(
        const pcl::PointCloud<PointT>::Ptr cloud,
        pcl::PointCloud<PointT>::Ptr &keypoints);

void computeReferenceFrames(
        const pcl::PointCloud<PointT>::Ptr cloud,
        const pcl::PointCloud<PointT>::Ptr keypoints,
        const pcl::search::Search<PointT>::Ptr searchTree,
        pcl::PointCloud<PointT>::Ptr &keypoints_clean,
        pcl::PointCloud<pcl::ReferenceFrame>::Ptr &reference_frames);

void computeDescriptors(
        const pcl::PointCloud<PointT>::Ptr cloud,
        const pcl::PointCloud<pcl::Normal>::Ptr normals,
        const pcl::PointCloud<PointT>::Ptr keypoints,
        const pcl::search::Search<PointT>::Ptr searchTree,
        const pcl::PointCloud<pcl::ReferenceFrame>::Ptr reference_frames,
        pcl::PointCloud<ISMFeature>::Ptr &features);

void removeNanDescriptors(
        const pcl::PointCloud<ISMFeature>::Ptr features,
        pcl::PointCloud<ISMFeature>::Ptr &features_cleaned);

