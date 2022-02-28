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
#include "../voting/voting_maximum.h"
#include "custom_SVM.h"

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
        GlobalResultAccu(unsigned num_occurences, float score, unsigned instance_id)
        {
            this->num_occurences = num_occurences;
            this->score_sum = score;
            insertInstanceLabel(instance_id, score);
        }

        void insertInstanceLabel(unsigned instance_id, float score)
        {
            if(instance_ids.find(instance_id) != instance_ids.end())
            {
                // found
                std::pair<int, float> &prev = instance_ids.at(instance_id);
                prev.first++;
                prev.second += score;
            }
            else
            {
                // not found
                instance_ids.insert({instance_id, {1,score}});
            }
        }

        unsigned num_occurences;
        float score_sum;
        // id --> pair of num_occurences and score_sum
        std::map<unsigned, std::pair<int, float>> instance_ids;
    };

    /**
     * @brief The GlobalClassifier class
     * This class wrapps a classifier for global feature descriptors. The classifier can be KNN or SVM.
     * Instance labels are only available by calling the KNN classifier (which is done also if SVM is selected).
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

        void setMergeParams(const float min_svm_score, const float rate_limit, const float weight_factor)
        {
            m_min_svm_score = min_svm_score;
            m_rate_limit = rate_limit;
            m_weight_factor = weight_factor;
        }

        void mergeGlobalAndLocalHypotheses(const int merge_function,
                                           std::vector<VotingMaximum> &maxima);

        void setFlannHelper(std::shared_ptr<FlannHelper> fh)
        {
            m_flann_helper = fh;
        }

        void setLoadedFeatures(pcl::PointCloud<ISMFeature>::Ptr loaded_features)
        {
            m_global_features = loaded_features;
        }

        void setGlobalRadii(std::map<unsigned, float> &global_radii)
        {
            m_average_radii = global_radii;
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

        void classifyWithKNN(pcl::PointCloud<ISMFeature>::ConstPtr global_features,
                             VotingMaximum &maximum);

        void classifyWithSVM(pcl::PointCloud<ISMFeature>::ConstPtr global_features,
                             VotingMaximum &maximum);

        void insertGlobalResult(std::map<unsigned, GlobalResultAccu> &max_global_voting,
                                unsigned found_class, unsigned instance_id,
                                float score) const;

        pcl::PointCloud<ISMFeature>::ConstPtr computeGlobalFeatures(const pcl::PointCloud<PointT>::ConstPtr points,
                                                                    const pcl::PointCloud<pcl::Normal>::ConstPtr normals);

        void useHighRankedGlobalHypothesis(std::vector<VotingMaximum> &maxima);

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

        float m_min_svm_score;
        float m_rate_limit;
        float m_weight_factor;
    };
}

#endif // ISM3D_GLOBALCLASSIFIER_H
