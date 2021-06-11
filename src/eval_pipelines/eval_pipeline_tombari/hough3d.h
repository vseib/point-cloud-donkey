#ifndef HOUGH3D_H
#define HOUGH3D_H

#include <vector>
#include <string>
#include <functional>
#include <flann/flann.hpp>
#include <pcl/features/feature.h>
#include <pcl/recognition/cg/hough_3d.h>
#include "../../implicit_shape_model/utils/ism_feature.h"
#include "../../implicit_shape_model/voting/voting_maximum.h"

using namespace ism3d;

typedef pcl::PointXYZRGB PointT;

class Hough3d
{

public:
    Hough3d(std::string dataset);

    virtual ~Hough3d()
    {
    }

    void train(const std::vector<std::string> &filenames,
               const std::vector<unsigned> &class_labels,
               const std::vector<unsigned> &instance_labels,
               const std::string &output_file) const;

    std::vector<std::pair<unsigned, float>> classify(const std::string &filename, bool useSingleVotingSpace) const;

    std::vector<ism3d::VotingMaximum> detect(const std::string &filename, bool useHypothesisVerification) const;

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

    pcl::PointCloud<ISMFeature>::Ptr processPointCloud(pcl::PointCloud<PointT>::Ptr cloud) const;

    void computeNormals(pcl::PointCloud<PointT>::Ptr cloud,
                        pcl::PointCloud<pcl::Normal>::Ptr& normals,
                        pcl::search::Search<PointT>::Ptr searchTree) const;

    void filterNormals(pcl::PointCloud<pcl::Normal>::Ptr normals,
                       pcl::PointCloud<pcl::Normal>::Ptr &normals_without_nan,
                       pcl::PointCloud<PointT>::Ptr cloud,
                       pcl::PointCloud<PointT>::Ptr &cloud_without_nan) const;

    void computeKeypoints(pcl::PointCloud<PointT>::Ptr &keypoints,
                          pcl::PointCloud<PointT>::Ptr cloud) const;

    void computeReferenceFrames(pcl::PointCloud<pcl::ReferenceFrame>::Ptr &reference_frames,
                                       pcl::PointCloud<PointT>::Ptr keypoints,
                                       pcl::PointCloud<PointT>::Ptr cloud,
                                       pcl::search::Search<PointT>::Ptr searchTree) const;

    void computeDescriptors(pcl::PointCloud<PointT>::Ptr cloud,
                                   pcl::PointCloud<pcl::Normal>::Ptr normals,
                                   pcl::PointCloud<PointT>::Ptr keypoints,
                                   pcl::search::Search<PointT>::Ptr searchTree,
                                   pcl::PointCloud<pcl::ReferenceFrame>::Ptr reference_frames,
                                   pcl::PointCloud<ISMFeature>::Ptr &features) const;

    void removeNanDescriptors(pcl::PointCloud<ISMFeature>::Ptr features,
                                     pcl::PointCloud<ISMFeature>::Ptr &features_cleaned) const;

    flann::Matrix<float> createFlannDataset();

    std::tuple<std::vector<std::pair<unsigned, float> >, std::vector<Eigen::Vector3f> >
                                            findObjects(const pcl::PointCloud<ISMFeature>::Ptr& scene_features,
                                                        const pcl::PointCloud<PointT>::Ptr scene_cloud,
                                                        const bool use_hv) const;

    std::vector<std::pair<unsigned, float>> classifyObjectsWithSeparateVotingSpaces(const pcl::PointCloud<ISMFeature>::Ptr& scene_features) const;
    std::vector<std::pair<unsigned, float>> classifyObjectsWithUnifiedVotingSpaces(const pcl::PointCloud<ISMFeature>::Ptr& scene_features) const;

    bool saveModelToFile(std::string &filename,
                         std::map<unsigned, pcl::PointCloud<ISMFeature>::Ptr> &all_features,
                         std::map<unsigned, std::vector<Eigen::Vector3f>> &all_vectors) const;

    bool loadModelFromFile(std::string& filename);


    template<typename T>
    static bool containsValue(const std::vector<T> &vec, const T &val)
    {
        for(T elem : vec)
            if (elem == val) return true;

        return false;
    }

    std::map<unsigned, std::string> m_class_labels;
    std::map<unsigned, std::string> m_instance_labels;
    std::map<unsigned, unsigned> m_instance_to_class_map;

    Eigen::Vector3d m_min_coord;
    Eigen::Vector3d m_max_coord;
    Eigen::Vector3d m_bin_size;
    std::shared_ptr<pcl::recognition::HoughSpace3D> m_hough_space;

    float m_normal_radius;
    float m_reference_frame_radius;
    float m_feature_radius;
    float m_keypoint_sampling_radius;
    int m_k_search;
    int m_normal_method;
    std::string m_feature_type;

    int m_number_of_classes;
    std::vector<unsigned> m_class_lookup;
    pcl::PointCloud<ISMFeature>::Ptr m_features;
    std::vector<Eigen::Vector3f> m_center_vectors;

    flann::Index<flann::L2<float>> m_flann_index;
};

#endif // HOUGH3D_H
