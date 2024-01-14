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
    Hough3d(std::string dataset, float bin = -1, float th = -1, float count = -1, float count2 = -1);

    virtual ~Hough3d()
    {
    }

    void train(const std::vector<std::string> &filenames,
               const std::vector<unsigned> &class_labels,
               const std::vector<unsigned> &instance_labels,
               const std::string &output_file);

    std::vector<std::pair<unsigned, float>> classify(const std::string &filename, bool useSingleVotingSpace);

    std::vector<ism3d::VotingMaximum> detect(const std::string &filename, bool useHypothesisVerification);

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

    std::map<unsigned, float> getDetectionThresholds()
    {
        return m_class_radii;
    }

private:

    flann::Matrix<float> createFlannDataset();


    void classifyObjectsWithSeparateVotingSpaces(
            const pcl::PointCloud<ISMFeature>::Ptr scene_features,
            std::vector<std::pair<unsigned, float>> &results);

    void classifyObjectsWithSingleVotingSpace(
            const pcl::PointCloud<ISMFeature>::Ptr scene_features,
            std::vector<std::pair<unsigned, float>> &results);

    void findObjects(
            const pcl::PointCloud<ISMFeature>::Ptr& scene_features,
            const pcl::PointCloud<PointT>::Ptr scene_keypoints,
            const bool use_hv,
            std::vector<std::pair<unsigned, float>> &results,
            std::vector<Eigen::Vector3f> &positions);

    bool saveModelToFile(std::string &filename,
                         std::map<unsigned, pcl::PointCloud<ISMFeature>::Ptr> &all_features,
                         std::map<unsigned, std::vector<Eigen::Vector3f>> &all_vectors,
                         std::map<unsigned, std::vector<Utils::BoundingBox>> &all_bounding_boxes) const;

    bool loadModelFromFile(std::string& filename);

    std::map<unsigned, std::string> m_class_labels;
    std::map<unsigned, std::string> m_instance_labels;
    std::map<unsigned, unsigned> m_instance_to_class_map;

    Eigen::Vector3d m_min_coord;
    Eigen::Vector3d m_max_coord;
    Eigen::Vector3d m_bin_size;
    std::shared_ptr<pcl::recognition::HoughSpace3D> m_hough_space;

    float m_th;
    float m_count;
    float m_count2;

    bool m_use_mvbb;
    int m_number_of_classes;
    float m_cutoff_distance_z;
    float m_inlier_th; // ransac inlier threshold
    std::vector<unsigned> m_class_lookup;
    pcl::PointCloud<ISMFeature>::Ptr m_features; // codebook
    pcl::PointCloud<PointT>::Ptr m_keypoints;
    std::vector<Eigen::Vector3f> m_center_vectors;
    std::map<unsigned, float> m_class_radii;
    flann::Index<flann::L2<float>> m_flann_index;
};

#endif // HOUGH3D_H
