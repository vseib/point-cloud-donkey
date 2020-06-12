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

#include "../features/features.h"
#include "../utils/utils.h"
#include "../utils/json_object.h"
#include "../utils/ism_feature.h"
#include "global_classifier.h"

// TODO VS X: clean up
// - there are multiple global feature members
// - there are two setter functions for global features used in ism.cpp
// - this class is for voting, but is now additionally used for global features and maxima merging
// - pointer to global descriptor calculation is set here


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
         * @brief The Vote struct
         * The internal vote representation
         */
        struct Vote
        {
            Eigen::Vector3f position;
            float weight;
            unsigned classId;
            unsigned instanceId;
            Eigen::Vector3f keypoint;       // associated keypoint position
            Utils::BoundingBox boundingBox; // associated bounding box
            int codewordId;                 // codeword the vote belongs to
        };

        /**
         * @brief cast a vote into the hough space
         * @param position the vote position
         * @param weight the vote weight
         * @param classId the class id for the vote
         * @param instanceId the instance id for the vote
         * @param keypoint the keypoint position from which the vote originated
         * @param boundingBox the bounding box for the object that casted the vote
         * @param codewordId the id of the codeword the vote belongs to
         */
        void vote(Eigen::Vector3f position,
                  float weight,
                  unsigned classId, unsigned instanceId,
                  const Eigen::Vector3f& keypoint,
                  const Utils::BoundingBox& boundingBox, int codewordId);

        /**
         * @brief find maxima in the hough voting space in order to identify object occurrences
         * @param points used to calculate global features
         * @param normals used to calculate global features
         * @return a list of maxima representing found object occurrences
         */
        std::vector<VotingMaximum> findMaxima(pcl::PointCloud<PointT>::ConstPtr &points, pcl::PointCloud<pcl::Normal>::ConstPtr &normals);

        /**
         * @brief get all votes
         * @return a map of votes, the key represents the class id
         */
        const std::map<unsigned, std::vector<Voting::Vote> >& getVotes() const;

        /**
         * @brief calculate average bounding box dimensions during training to be used as hints for bin size and bandwidth during recognition
         * @param boundingBoxes bounding boxes of trained objects
         */
        void determineAverageBoundingBoxDimensions(const std::map<unsigned, std::vector<Utils::BoundingBox> > &boundingBoxes);

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
            m_global_classifier->setDistanceType(type);
        }

        void setSVMPath(std::string path)
        {
            m_svm_path = path;
        }

    protected:
        Voting();

        virtual void iFindMaxima(pcl::PointCloud<PointT>::ConstPtr&,
                                 const std::vector<Voting::Vote>&,
                                 std::vector<Eigen::Vector3f>&,
                                 std::vector<double>&,
                                 std::vector<std::vector<unsigned>>&,
                                 std::vector<std::vector<int>>&,
                                 std::vector<std::vector<float>>&,
                                 unsigned) = 0;

        void iSaveData(boost::archive::binary_oarchive &oa) const;
        bool iLoadData(boost::archive::binary_iarchive &ia);

        // maps class ids to a vector of global features, number of objects per class = number of global features per class
        std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> m_global_features; // used only during training

        bool m_single_object_mode;
        std::string m_max_filter_type;
        std::string m_max_type_param;

    private:

        static bool sortMaxima(const VotingMaximum&, const VotingMaximum&);

        void normalizeWeights(std::vector<VotingMaximum> &maxima);

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

        // NOTE: only for debug
        static float state_gt;
        static float state_true;
        static float state_false;
    };
}

#endif // ISM3D_VOTING_H
