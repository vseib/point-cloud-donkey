#ifndef GLOBAL_HV_H
#define GLOBAL_HV_H

#include <vector>
#include <string>
#include <flann/flann.hpp>
#include <pcl/features/feature.h>
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

    flann::Matrix<float> createFlannDataset() const;

    void classifyObject(
            const pcl::PointCloud<ISMFeature>::Ptr scene_features,
            const pcl::PointCloud<PointT>::Ptr scene_keypoints,
            const pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_lrf,
            std::vector<std::pair<unsigned, float>> &results);

    void findObjects(
            const pcl::PointCloud<PointT>::Ptr scene_cloud,
            const pcl::PointCloud<ISMFeature>::Ptr scene_features,
            const pcl::PointCloud<PointT>::Ptr scene_keypoints,
            const pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_lrf,
            const bool use_hv,
            std::vector<std::pair<unsigned, float>> &results,
            std::vector<Eigen::Vector3f> &positions);

    std::vector<pcl::Correspondences> getCorrespondeceClustersFromMaxima(
            const std::vector<double> &maxima,
            const std::vector<std::vector<int>> &voteIndices,
            const pcl::CorrespondencesPtr &model_scene_corrs_filtered) const;

    std::vector<std::pair<unsigned, float>> getResultsFromMaxima(
                const std::vector<double> &maxima,
                const std::vector<std::vector<int>> &voteIndices,
                const pcl::CorrespondencesPtr &model_scene_corrs_filtered,
                const pcl::PointCloud<ISMFeature>::Ptr object_features) const;

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

    std::map<unsigned, std::string> m_class_labels;
    std::map<unsigned, std::string> m_instance_labels;
    std::map<unsigned, unsigned> m_instance_to_class_map;

    float m_corr_threshold;

    // TODO VS check these params
    int m_icp_max_iter;
    float m_icp_corr_distance;

    int m_number_of_classes;
    std::vector<unsigned> m_class_lookup;
    pcl::PointCloud<ISMFeature>::Ptr m_features;
    std::vector<Eigen::Vector3f> m_center_vectors;

    flann::Index<flann::L2<float>> m_flann_index;
};

#endif // GLOBAL_HV_H
