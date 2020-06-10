/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2020, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_GLOBALCLASSIFIER_H
#define ISM3D_GLOBALCLASSIFIER_H

#include <vector>
#include <Eigen/Core>

#include "../features/features.h"
#include "../utils/ism_feature.h"
#include "../utils/flann_helper.h"
// TODO VS move this class to that folder later
#include "../classifier/custom_SVM.h"
#include "voting_maximum.h"

#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>
#include <pcl/recognition/cg/hough_3d.h>

// to use SVM
#include <opencv2/ml/ml.hpp>

namespace ism3d
{
    // represents an accumulated global result for a single class
    struct GlobalResultAccu
    {
        GlobalResultAccu(unsigned num_occurences, float score)
        {
            this->num_occurences = num_occurences;
            this->score_sum = score;
        }

        unsigned num_occurences;
        float score_sum;
    };

    /**
     * @brief The GlobalClassifier class
     * TODO VS add description
     */
    class GlobalClassifier
    {

    public:
        GlobalClassifier(Features* global_descriptor,
                         std::string method,
                         int k_global);
        virtual ~GlobalClassifier();

        // segment isolated object from point cloud
        void segmentROI(const pcl::PointCloud<PointT>::ConstPtr &points,
                              const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
                              const ism3d::VotingMaximum &maximum,
                              pcl::PointCloud<PointT>::Ptr &segmented_points,
                              pcl::PointCloud<pcl::Normal>::Ptr &segmented_normals);

        void classify(const pcl::PointCloud<PointT>::ConstPtr &points,
                      const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
                      VotingMaximum &maximum);

        void computeAverageRadii(std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> &global_features);

        void loadSVMModels(std::string &svm_path);

        void setFlannHelper(std::shared_ptr<FlannHelper> fh)
        {
            m_flann_helper = fh;
        }

        void setLoadedFeatures(pcl::PointCloud<ISMFeature>::Ptr loaded_features)
        {
            m_global_features = loaded_features;
        }

        void setDistanceType(std::string type)
        {
            m_distance_type = type;
        }

        void enableSingleObjectMode()
        {
            m_single_object_mode = true;
        }


    private:
        void insertGlobalResult(std::map<unsigned, GlobalResultAccu> &max_global_voting,
                                unsigned found_class,
                                float score) const;

        pcl::PointCloud<ISMFeature>::ConstPtr computeGlobalFeatures(const pcl::PointCloud<PointT>::ConstPtr points,
                                                                    const pcl::PointCloud<pcl::Normal>::ConstPtr normals);

        bool m_index_created;
        bool m_single_object_mode;
        bool m_svm_error;
        std::string m_global_feature_method;
        std::string m_distance_type;
        int m_k_global_features;

        Features* m_feature_algorithm; // object to compute features on input clouds
        pcl::PointCloud<ISMFeature>::Ptr m_global_features; // features obtained during training

        CustomSVM m_svm;
        std::vector<std::string> m_svm_files;
        std::shared_ptr<FlannHelper> m_flann_helper;
        std::map<unsigned, float> m_average_radii;
    };
}

#endif // ISM3D_GLOBALCLASSIFIER_H
