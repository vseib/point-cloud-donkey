#ifndef PCL_ORCG_H
#define PCL_ORCG_H

#include <vector>
#include <string>
#include <flann/flann.hpp>
#include <pcl/features/feature.h>

#include "../../implicit_shape_model/utils/ism_feature.h"

using namespace ism3d;

typedef pcl::PointXYZRGB PointT;

class Orcg
{

public:

    Orcg();

    virtual ~Orcg()
    {
    }

    bool prepareScene(const std::string &filename_scene);

    std::vector<std::pair<unsigned, float>> findObjectInScene(const std::string &filename_model) const;

private:

    pcl::PointCloud<ISMFeature>::Ptr processPointCloud(pcl::PointCloud<PointT>::Ptr &cloud,
                                                       pcl::PointCloud<PointT>::Ptr &keyp) const;

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

    flann::Matrix<float> createFlannDataset(const pcl::PointCloud<ISMFeature>::Ptr &features) const;

    pcl::CorrespondencesPtr findNnCorrespondences(const pcl::PointCloud<ISMFeature>::Ptr& features) const;

    float m_normal_radius;
    float m_reference_frame_radius;
    float m_feature_radius;
    float m_keypoint_sampling_radius;
    int m_k_search;

    float m_cg_size;
    float m_cg_thresh;
    bool m_use_hough;
    int m_icp_max_iter;
    float m_icp_corr_distance;

    int m_number_of_classes;

    pcl::PointCloud<PointT>::Ptr m_scene;
    pcl::PointCloud<PointT>::Ptr m_scene_keypoints;
    pcl::PointCloud<ISMFeature>::Ptr m_scene_features;
};

#endif // PCL_ORCG_H
