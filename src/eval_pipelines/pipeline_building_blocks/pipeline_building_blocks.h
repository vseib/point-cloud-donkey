#pragma once

#include <flann/flann.hpp>
#include <pcl/features/feature.h>
#include <pcl/recognition/cg/hough_3d.h>
#include "../../implicit_shape_model/utils/ism_feature.h"

using namespace ism3d;

typedef pcl::PointXYZRGB PointT;

pcl::CorrespondencesPtr findNnCorrespondences(
        const pcl::PointCloud<ISMFeature>::Ptr& scene_features,
        const float matching_threshold,
        const flann::Index<flann::L2<float>> &index);

void remapIndicesToLocalCloud(
        const pcl::CorrespondencesPtr object_scene_corrs,
        const pcl::PointCloud<ISMFeature>::Ptr all_features,
        const std::vector<Eigen::Vector3f> all_center_vectors,
        pcl::PointCloud<PointT>::Ptr &object_keypoints,
        pcl::PointCloud<ISMFeature>::Ptr &object_features,
        std::vector<Eigen::Vector3f> &object_center_vectors,
        pcl::PointCloud<pcl::ReferenceFrame>::Ptr &object_lrf);

std::vector<Eigen::Vector3f> prepareCenterVotes(
        const pcl::CorrespondencesPtr object_scene_corrs,
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        const std::vector<Eigen::Vector3f> object_center_vectors);

void castVotesAndFindMaxima(
        const pcl::CorrespondencesPtr object_scene_corrs,
        const std::vector<Eigen::Vector3f> &votelist,
        const float relative_threshold,
        const bool use_distance_weight,
        std::vector<double> &maxima,
        std::vector<std::vector<int>> &vote_indices,
        std::shared_ptr<pcl::recognition::HoughSpace3D> &hough_space);

void generateClassificationHypotheses(
        const pcl::CorrespondencesPtr object_scene_corrs,
        const std::vector<std::vector<int>> &vote_indices,
        const pcl::PointCloud<ISMFeature>::Ptr codebook_features,
        std::vector<std::pair<unsigned, float>> &results);

void generateHypothesesWithAbsoluteOrientation(
        const pcl::CorrespondencesPtr object_scene_corrs,
        const std::vector<std::vector<int>> &vote_indices,
        const pcl::PointCloud<PointT>::Ptr scene_keypoints,
        const pcl::PointCloud<PointT>::Ptr object_keypoints,
        const float inlier_threshold,
        const bool refine_model,
        const bool use_hv,
        std::vector<Eigen::Matrix4f> &transformations,
        std::vector<pcl::Correspondences> &model_instances);

void findClassAndPositionFromCluster(
        const pcl::Correspondences &filtered_corrs,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        const std::vector<Eigen::Vector3f> &object_center_vectors,
        const int num_classes,
        unsigned &resulting_class,
        int &resulting_num_votes,
        Eigen::Vector3f &resulting_position);
