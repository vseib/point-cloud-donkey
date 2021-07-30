#pragma once

#include <flann/flann.hpp>
#include <pcl/features/feature.h>
#include <pcl/recognition/cg/hough_3d.h>
#include "../../implicit_shape_model/utils/ism_feature.h"

using namespace ism3d;

typedef pcl::PointXYZRGB PointT;

pcl::CorrespondencesPtr findNnCorrespondences(
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        const float matching_threshold,
        const flann::Index<flann::L2<float>> &index);

pcl::CorrespondencesPtr findNnCorrespondences(
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
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

void clusterCorrespondences(
        const pcl::CorrespondencesPtr object_scene_corrs,
        const pcl::PointCloud<PointT>::Ptr scene_keypoints,
        const pcl::PointCloud<PointT>::Ptr object_keypoints,
        const pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_lrf,
        const pcl::PointCloud<pcl::ReferenceFrame>::Ptr object_lrf,
        const bool use_distance_weight,
        const float bin_size,
        const float corr_threshold,
        const float lrf_radius,
        const bool use_hough,
        const bool recognize,
        std::vector<pcl::Correspondences> &clustered_corrs,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &transformations);

void generateClassificationHypotheses(
        const pcl::CorrespondencesPtr object_scene_corrs,
        const std::vector<std::vector<int>> &vote_indices,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        std::vector<std::pair<unsigned, float>> &results);

void generateClassificationHypotheses(
        const std::vector<pcl::Correspondences> &clustered_corrs,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        std::vector<std::pair<unsigned, float>> &results);

void generateHypothesesWithAbsoluteOrientation(
        const pcl::CorrespondencesPtr object_scene_corrs,
        const std::vector<std::vector<int>> &vote_indices,
        const pcl::PointCloud<PointT>::Ptr scene_keypoints,
        const pcl::PointCloud<PointT>::Ptr object_keypoints,
        const float inlier_threshold,
        const bool refine_model,
        const bool separate_voting_spaces,
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

void findClassAndPositionFromTransformedObjectKeypoints(
        const pcl::Correspondences &filtered_corrs,
        const Eigen::Matrix4f &transformation,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        const pcl::PointCloud<PointT>::Ptr object_keypoints,
        const std::vector<Eigen::Vector3f> &object_center_vectors,
        const int num_classes,
        unsigned &resulting_class,
        int &resulting_num_votes,
        Eigen::Vector3f &resulting_position);

void findClassAndPointsFromCorrespondences(
        const pcl::Correspondences &corrs,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        unsigned &res_class,
        int &res_num_votes,
        pcl::PointCloud<PointT>::Ptr scene_points);

void findPositionFromCluster(
        const pcl::Correspondences &filtered_corrs,
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        const std::vector<Eigen::Vector3f> &object_center_vectors,
        Eigen::Vector3f &resulting_position);

void findPositionFromTransformedObjectKeypoints(
        const pcl::Correspondences &filtered_corrs,
        const Eigen::Matrix4f &transformation,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        const pcl::PointCloud<PointT>::Ptr object_keypoints,
        const std::vector<Eigen::Vector3f> &object_center_vectors,
        Eigen::Vector3f &resulting_position);

void generateCloudsFromTransformations(
        const std::vector<pcl::Correspondences> clustered_corrs,
        const std::vector<Eigen::Matrix4f> transformations,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        std::vector<pcl::PointCloud<PointT>::ConstPtr> &instances);

void alignCloudsWithICP(
        const float icp_max_iterations,
        const float icp_correspondence_distance,
        const pcl::PointCloud<PointT>::Ptr scene_keypoints,
        const std::vector<pcl::PointCloud<PointT>::ConstPtr> &instances,
        std::vector<pcl::PointCloud<PointT>::ConstPtr> &registered_instances,
        std::vector<Eigen::Matrix4f> &final_transformations);

void runGlobalHV(
        const pcl::PointCloud<PointT>::Ptr scene_cloud,
        std::vector<pcl::PointCloud<PointT>::ConstPtr> &registered_instances,
        const float inlier_threshold,
        const float occlusion_threshold,
        const float regularizer,
        const float clutter_regularizer,
        const float radius_clutter,
        const bool detect_clutter,
        const float normal_radius,
        std::vector<bool> &hypotheses_mask);

void performSelfAdaptedHoughVoting(
        const pcl::CorrespondencesPtr &object_scene_corrs,
        const pcl::PointCloud<PointT>::Ptr object_keypoints,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        const pcl::PointCloud<pcl::ReferenceFrame>::Ptr object_lrf,
        const pcl::PointCloud<PointT>::Ptr scene_keypoints,
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        const pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_lrf,
        const bool use_distance_weight,
        const int initial_bin_number,
        float initial_matching_threshold,
        float rel_threshold,
        std::vector<double> &maxima,
        std::vector<std::vector<int>> &vote_indices,
        pcl::CorrespondencesPtr &model_scene_corrs_filtered,
        float &found_bin_size);

void prepareSelfAdaptedVoting(
        const pcl::CorrespondencesPtr &object_scene_corrs_filtered,
        const pcl::PointCloud<PointT>::Ptr object_keypoints,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        const pcl::PointCloud<pcl::ReferenceFrame>::Ptr object_lrf,
        const pcl::PointCloud<PointT>::Ptr scene_keypoints,
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        const pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_lrf,
        std::vector<std::pair<float,float>> &votes,
        std::pair<float,float> &rmse_E_min_max,
        std::pair<float,float> &rmse_T_min_max);

void getMetricsAndInlierPoints(
        const std::vector<pcl::PointCloud<PointT>::ConstPtr> registered_instances,
        const pcl::PointCloud<PointT>::Ptr scene_cloud,
        const float threshold,
        std::vector<pcl::PointCloud<PointT>::ConstPtr> &inlier_points_of_instances,
        std::vector<float> &fs_metrics,
        std::vector<float> &mr_metrics);
