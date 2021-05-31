#ifndef LNBNN_H
#define LNBNN_H

#include <vector>
#include <string>
#include <flann/flann.hpp>
#include <pcl/features/feature.h>

#include "../../implicit_shape_model/utils/ism_feature.h"

using namespace ism3d;

typedef pcl::PointXYZRGB PointT;

class Lnbnn
{

public:

    Lnbnn();

    virtual ~Lnbnn()
    {
    }

    void train(const std::vector<std::string> &filenames, const std::vector<std::string> &labels, const std::string &output_file) const;

    std::vector<std::pair<unsigned, float>> classify(const std::string &filename) const;

    bool loadModel(std::string &filename);

private:

    pcl::PointCloud<ISMFeature>::Ptr processPointCloud(pcl::PointCloud<PointT>::Ptr &cloud) const;

    void computeNormals(pcl::PointCloud<PointT>::Ptr &cloud,
                        pcl::PointCloud<pcl::Normal>::Ptr& normals,
                        pcl::search::Search<PointT>::Ptr &searchTree) const;

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

    flann::Matrix<float> createFlannDataset();

    std::vector<std::pair<unsigned, float>> accumulateClassDistances(const pcl::PointCloud<ISMFeature>::Ptr& features) const;

    bool saveModelToFile(std::string &filename, std::map<unsigned, pcl::PointCloud<ISMFeature>::Ptr> &all_features) const;

    bool loadModelFromFile(std::string& filename);


    template<typename T>
    static bool containsValue(const std::vector<T> &vec, const T &val)
    {
        for(T elem : vec)
            if (elem == val) return true;

        return false;
    }

    float m_normal_radius;
    float m_reference_frame_radius;
    float m_feature_radius;
    float m_keypoint_sampling_radius;
    int m_k_search;
    bool m_rgbd_camera_data;

    int m_number_of_classes;
    std::vector<unsigned> m_class_lookup;
    pcl::PointCloud<ISMFeature>::Ptr m_features;

    flann::Index<flann::L2<float>> m_flann_index;
};

#endif // LNBNN_H
