/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_VOTING_H
#define ISM3D_VOTING_H

#include <vector>
#include <map>
#include <Eigen/Core>

#include "voting_maximum.h"

#include "../features/features.h"
#include "../utils/utils.h"
#include "../utils/json_object.h"
#include "../utils/ism_feature.h"
#include "../classifier/global_classifier.h"

namespace ism3d
{
    /**
     * @brief The Voting class
     * Provides an abstract interface for the voting process in which object occurrences will
     * be detected. Before detection, votes are cast into the voting space, representing object
     * hypotheses. During detection, the sum of all votes will be analyzed in order to find the
     * most likely object positions for each class.
     */
    class Voting
            : public JSONObject
    {

    public:
        virtual ~Voting();


        /**
         * @brief cast a vote into the hough space
         * @param position the vote position
         * @param weight the vote weight
         * @param classId the class id for the vote
         * @param instanceId the instance id for the vote
         * @param keypoint the keypoint position from which the vote originated
         * @param boundingBox the bounding box for the object that casted the vote
         * @param codeword the codeword the vote belongs to
         */
        void vote(Eigen::Vector3f position,
                  float weight,
                  unsigned classId, unsigned instanceId,
                  const Eigen::Vector3f& keypoint,
                  const Utils::BoundingBox& boundingBox,
                  const std::shared_ptr<Codeword> &codeword);

        /**
         * @brief find maxima in the hough voting space in order to identify object occurrences
         * @param points used to calculate global features
         * @param normals used to calculate global features
         * @return a list of maxima representing found object occurrences
         */
        std::vector<VotingMaximum> findMaxima(pcl::PointCloud<PointT>::ConstPtr &points, pcl::PointCloud<pcl::Normal>::ConstPtr &normals);

        /**
         * @brief printMaxima Prints the resulting maxima to std::out, nicely formatted
         * @param maxima All maxima to print
         */
        void printMaxima(const std::vector<VotingMaximum> &maxima);

        /**
         * @brief get all votes
         * @return a map of votes, the key represents the class id
         */
        const std::map<unsigned, std::vector<Vote> >& getVotes() const;

        /**
         * @brief calculate average bounding box dimensions and object radii per class during training to be used as hints object sizes during recognition
         * @param boundingBoxes bounding boxes of trained objects
         * @param object_radii object radii of trained objects
         */
        void forwardBoxesAndRadii(const std::map<unsigned, std::vector<Utils::BoundingBox>> &boundingBoxes,
                                  const std::map<unsigned, std::vector<float>> &object_radii);

        /**
         * @brief forwardGlobalFeatures set a map <class id, list of feature clouds> to be stored during training and
         *        used during detection in NON-single-object-mode
         * @param globalFeatures map with global features
         */
        void forwardGlobalFeatures(std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &globalFeatures);

        /** @brief set pointer to global features descriptor for detection (set while loading the dataset)
         *  @param glob - pointer to the descriptor object
         */
        void setGlobalFeatureDescriptor(Features* glob)
        {
            m_global_feature_descriptor = glob;
        }

        /**
         * @brief clear data
         */
        virtual void clear();

        // set when FLANN index for local features is created in ImplicitShapeModel.cpp
        void setDistanceType(std::string type)
        {
            if(m_use_global_features)
                m_global_classifier->setDistanceType(type);
        }

        void setSVMPath(std::string path)
        {
            m_svm_path = path;
        }

        bool isUsingGlobalFeatures()
        {
            return m_use_global_features;
        }

        std::map<unsigned, std::pair<float, float>> getObjectClassDimensions()
        {
            return m_dimensions_map;
        }

    protected:
        Voting();

        virtual void iFindMaxima(pcl::PointCloud<PointT>::ConstPtr&,
                                 std::vector<Vote>&,
                                 std::vector<Eigen::Vector3f>&,
                                 std::vector<double>&,
                                 std::vector<std::vector<unsigned>>&,
                                 std::vector<std::vector<Vote>>& votes_per_cluster,
                                 unsigned) = 0;

        void iSaveData(boost::archive::binary_oarchive &oa) const;
        bool iLoadData(boost::archive::binary_iarchive &ia);

        // maps class ids to a vector of global features, number of objects per class = number of global features per class
        std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> m_global_features; // used only during training

        bool m_single_object_mode;
        std::string m_max_filter_type;
        std::string m_max_type_param;

    private:

        std::tuple<std::vector<Eigen::Vector3f>, std::vector<std::vector<Vote>>> filterVotesWithRansac(
                const std::vector<Eigen::Vector3f> &clusters,
                const std::vector<std::vector<Vote>> &cluster_votes,
                const float inlier_threshold) const;

        static bool sortMaxima(const VotingMaximum&, const VotingMaximum&);

        void normalizeWeights(std::vector<VotingMaximum> &maxima);

        void softmaxWeights(std::vector<VotingMaximum> &maxima);

        std::map<unsigned, std::vector<Vote>> m_votes;

        float m_minThreshold;   // retrieve all maxima above the weight threshold
        int m_minVotesThreshold; // retrieve all maxima above the vote threshold
        int m_bestK;            // additionally retrieve only the k best maxima
        bool m_averageRotation;

        bool m_use_global_features;

        std::shared_ptr<GlobalClassifier> m_global_classifier;
        // these values are passed to the global classifier ...
        // ... when model is loaded
        int m_k_global_features;
        std::string m_global_feature_method;
        std::string m_svm_path; // path in config to svm models
        Features* m_global_feature_descriptor;
        // ... when maxima are detected
        int m_merge_function;
        float m_min_svm_score;
        float m_rate_limit;
        float m_weight_factor;

        // these values are passed to the maxima handler
        std::string m_radiusType;
        float m_radiusFactor;
        std::map<unsigned, std::pair<float, float>> m_dimensions_map;
        std::map<unsigned, std::pair<float, float>> m_variance_map;

        bool m_vote_filtering_with_ransac;
        bool m_refine_model;
        float m_inlier_threshold;
        std::string m_inlier_threshold_type;

        // NOTE: only for debug // not used anymore?
        static float state_gt;
        static float state_true;
        static float state_false;
    };
}

#endif // ISM3D_VOTING_H
