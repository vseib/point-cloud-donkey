#ifndef GLOBAL_HV_H
#define GLOBAL_HV_H

#include <vector>
#include <string>
#include <flann/flann.hpp>
#include <pcl/features/feature.h>
#include <pcl/recognition/cg/hough_3d.h>
#include "../../implicit_shape_model/utils/ism_feature.h"
#include "../../implicit_shape_model/voting/voting_maximum.h"

using namespace ism3d;

typedef pcl::PointXYZRGB PointT;

class SelfAdaptHGHV
{

public:

    SelfAdaptHGHV(std::string dataset, float bin=-1, float th=-1);

    virtual ~SelfAdaptHGHV()
    {
    }

    void train(const std::vector<std::string> &filenames,
               const std::vector<unsigned> &class_labels,
               const std::vector<unsigned> &instance_labels,
               const std::string &output_file);

    std::vector<std::pair<unsigned, float>> classify(const std::string &filename);

    std::vector<ism3d::VotingMaximum> detect(const std::string &filename);

    bool loadModel(std::string &filename);

    void setLabels(std::map<unsigned, std::string> &class_labels,
                   std::map<unsigned, std::string> &instance_labels,
                   std::map<unsigned, unsigned> &instance_to_class_map)
    {
        m_class_labels = class_labels;
        m_instance_labels = instance_labels;
        m_instance_to_class_map = instance_to_class_map;
    }

    std::map<unsigned, std::string> getClassLabels()
    {
        return m_class_labels;
    }

    std::map<unsigned, std::string> getInstanceLabels()
    {
        return m_instance_labels;
    }

    std::map<unsigned, unsigned> getInstanceClassMap()
    {
        return m_instance_to_class_map;
    }

private:

    pcl::PointCloud<ISMFeature>::Ptr processPointCloud(pcl::PointCloud<PointT>::Ptr cloud);

    void computeNormals(pcl::PointCloud<PointT>::Ptr cloud,
                        pcl::PointCloud<pcl::Normal>::Ptr &normals,
                        pcl::search::Search<PointT>::Ptr searchTree) const;

    void filterNormals(pcl::PointCloud<pcl::Normal>::Ptr &normals,
                       pcl::PointCloud<pcl::Normal>::Ptr &normals_without_nan,
                       pcl::PointCloud<PointT>::Ptr &cloud,
                       pcl::PointCloud<PointT>::Ptr &cloud_without_nan) const;

    void computeKeypoints(pcl::PointCloud<PointT>::Ptr &keypoints,
                          pcl::PointCloud<PointT>::Ptr &cloud) const;

    void computeReferenceFrames(pcl::PointCloud<pcl::ReferenceFrame>::Ptr &reference_frames,
                                       pcl::PointCloud<PointT>::Ptr &keypoints,
                                       pcl::PointCloud<PointT>::Ptr &cloud,
                                       pcl::search::Search<PointT>::Ptr &searchTree) const;

    void computeDescriptors(pcl::PointCloud<PointT>::Ptr &cloud,
                                   pcl::PointCloud<pcl::Normal>::Ptr &normals,
                                   pcl::PointCloud<PointT>::Ptr &keypoints,
                                   pcl::search::Search<PointT>::Ptr &searchTree,
                                   pcl::PointCloud<pcl::ReferenceFrame>::Ptr &reference_frames,
                                   pcl::PointCloud<ISMFeature>::Ptr &features) const;

    void removeNanDescriptors(pcl::PointCloud<ISMFeature>::Ptr &features,
                                     pcl::PointCloud<ISMFeature>::Ptr &features_cleaned) const;

    flann::Matrix<float> createFlannDataset() const;

    std::vector<std::pair<unsigned, float>> classifyObject(const pcl::PointCloud<ISMFeature>::Ptr& scene_features);

    std::tuple<std::vector<std::pair<unsigned, float> >, std::vector<Eigen::Vector3f> >
                                            findObjects(const pcl::PointCloud<ISMFeature>::Ptr& scene_features,
                                                        const pcl::PointCloud<PointT>::Ptr cloud);
    void prepare_voting(
            std::vector<std::pair<float,float>> &votes,
            std::pair<float,float> &rmse_E_min_max,
            std::pair<float,float> &rmse_T_min_max,
            const pcl::CorrespondencesPtr &model_scene_corrs_filtered,
            const pcl::PointCloud<PointT>::Ptr object_keypoints,
            const pcl::PointCloud<ISMFeature>::Ptr object_features,
            const pcl::PointCloud<pcl::ReferenceFrame>::Ptr object_lrf) const;

    std::vector<pcl::Correspondences> getCorrespondeceClustersFromMaxima(
            const std::vector<double> &maxima,
            const std::vector<std::vector<int>> &voteIndices,
            const pcl::CorrespondencesPtr &model_scene_corrs_filtered) const;

    std::vector<std::pair<unsigned, float>> getResultsFromMaxima(
                const std::vector<double> &maxima,
                const std::vector<std::vector<int>> &voteIndices,
                const pcl::CorrespondencesPtr &model_scene_corrs_filtered,
                const pcl::PointCloud<ISMFeature>::Ptr object_features) const;

    void performSelfAdaptedHoughVoting(std::vector<double> &maxima,
                                       std::vector<std::vector<int>> &voteIndices,
                                       pcl::CorrespondencesPtr &model_scene_corrs_filtered,
                                       const pcl::CorrespondencesPtr &model_scene_corrs,
                                       const pcl::PointCloud<PointT>::Ptr object_keypoints,
                                       const pcl::PointCloud<ISMFeature>::Ptr object_features,
                                       const pcl::PointCloud<pcl::ReferenceFrame>::Ptr object_lrf);

    void findClassAndPositionFromCluster(
            const pcl::Correspondences &filtered_corrs,
            const pcl::PointCloud<ISMFeature>::Ptr object_features,
            const pcl::PointCloud<ISMFeature>::Ptr scene_features,
            unsigned &resulting_class,
            int &resulting_num_votes,
            Eigen::Vector3f &resulting_position) const;

    bool saveModelToFile(std::string &filename,
                         std::map<unsigned, pcl::PointCloud<ISMFeature>::Ptr> &all_features,
                         std::map<unsigned, std::vector<Eigen::Vector3f>> &all_vectors) const;

    bool loadModelFromFile(std::string& filename);

    pcl::CorrespondencesPtr findNnCorrespondences(const pcl::PointCloud<ISMFeature>::Ptr& scene_features) const;

    std::map<unsigned, std::string> m_class_labels;
    std::map<unsigned, std::string> m_instance_labels;
    std::map<unsigned, unsigned> m_instance_to_class_map;

    std::shared_ptr<pcl::recognition::HoughSpace3D> m_hough_space; // using 3d since there is no 2d hough in pcl

    float m_normal_radius;
    float m_reference_frame_radius;
    float m_feature_radius;
    float m_keypoint_sampling_radius;
    int m_k_search; // TODO remove
    int m_normal_method;
    std::string m_feature_type;
    float m_corr_threshold;
    Eigen::Vector3d m_min_coord;
    Eigen::Vector3d m_max_coord;
    Eigen::Vector3d m_bin_size;

    // TODO VS check these params
    int m_icp_max_iter;
    float m_icp_corr_distance;

    int m_number_of_classes;
    std::vector<unsigned> m_class_lookup;
    pcl::PointCloud<ISMFeature>::Ptr m_features;
    pcl::PointCloud<PointT>::Ptr m_scene_keypoints;
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr m_scene_lrf;
    std::vector<Eigen::Vector3f> m_center_vectors;

    flann::Index<flann::L2<float>> m_flann_index;
};

#endif // GLOBAL_HV_H
