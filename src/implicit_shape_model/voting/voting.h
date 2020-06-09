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
#include <boost/shared_ptr.hpp>

#include "../features/features.h"
#include "../classifier/custom_SVM.h"
#include "../utils/utils.h"
#include "../utils/json_object.h"
#include "../utils/ism_feature.h"
#include "../utils/flann_helper.h"

// to use SVM
#include <opencv2/ml/ml.hpp>

#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/filters/extract_indices.h>

// TODO VS X: clean up
// - there are multiple global feature members
// - there are two setter functions for global features used in ism.cpp
// - this class is for voting, but is now additionally used for global features and maxima merging
// - pointer to global descriptor calculation is set here


namespace ism3d
{
    /**
     * @brief The VotingMaximum struct
     * A voting maximum represents a found object occurrence and is characterized
     * by a position and a bounding box. The weight defines the certainty of the
     * detection and the class id is the class type of the object.
     * Voting and maxima detection is performed for each class individually.
     */
    struct VotingMaximum
    {
        VotingMaximum() {
            position = Eigen::Vector3f(0, 0, 0);
            weight = 0;
            classId = -1;
            boundingBox.size = Eigen::Vector3f(0,0,0);
            boundingBox.position = position;
            boundingBox.rotQuat = boost::math::quaternion<float>(1, 0, 0, 0);
            globalHypothesis = {-1, -1};
            currentClassHypothesis = {-1, -1};
        }

        Eigen::Vector3f position;
        float weight;
        unsigned classId;
        Utils::BoundingBox boundingBox;
        std::vector<int> voteIndices;

        // TODO VS temp for testing: as soon as it is final include it in maxima merging: look for all occurences of TEMP FIX THIS!
        std::pair<int, float> globalHypothesis; // pair of: class_id and score
        std::pair<int, float> currentClassHypothesis; // pair of: this class_id and score
    };

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

    enum class SingleObjectMaxType
    {
        DEFAULT,    // default means no special treatment
        COMPLETE_VOTING_SPACE,
        BANDWIDTH,
        MODEL_RADIUS
    };


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

        /**
         * @brief setGlobalFeatures set a PointCloud with global featues to be used during detection in single-object-mode
         * @param globalFeatures
         */
        void setGlobalFeatures(pcl::PointCloud<ISMFeature>::Ptr &globalFeatures);

        /** @brief set pointer to global features descriptor for detection (set while loading the dataset)
         *
         */
        void setGlobalFeatureDescriptor(Features* glob)
        {
            m_globalFeatureDescriptor = glob;
        }

        /**
         * @brief clear data
         */
        virtual void clear();

        // set when FLANN index for local features is created in ImplicitShapeModel.cpp
        void setDistanceType(std::string dist)
        {
            m_distanceType = dist;
        }

        void setSVMPath(std::string path)
        {
            m_svm_path = path;
        }

    protected:
        Voting();

        void verifyMaxHypothesisWithGlobalFeatures(const pcl::PointCloud<PointT>::ConstPtr &points,
                                                   const pcl::PointCloud<pcl::Normal>::ConstPtr &normals, VotingMaximum &maximum);

        void classifyGlobalFeatures(const pcl::PointCloud<ISMFeature>::ConstPtr global_features, VotingMaximum &maximum);


        virtual void iFindMaxima(pcl::PointCloud<PointT>::ConstPtr&,
                                 const std::vector<Voting::Vote>&,
                                 std::vector<Eigen::Vector3f>&,
                                 std::vector<double>&,
                                 std::vector<std::vector<int> >&,
                                 std::vector<std::vector<float> >&,
                                 unsigned) = 0;

        float getSearchDistForClass(const unsigned class_id) const;

        void iSaveData(boost::archive::binary_oarchive &oa) const;
        bool iLoadData(boost::archive::binary_iarchive &ia);

        float m_radius;              // holds the bin size or the bandwith

        std::string m_radiusType; // take value from config or used learned average bounding box dimensions
        float m_radiusFactor; // factor for radius, in case radius type is NOT Config

        // maps class ids to average pairs of two longest bounding box dimensions <first radius, second radius>
        std::map<unsigned, std::pair<float, float> > m_id_bb_dimensions_map;
        // maps class ids to average pairs of two longest bounding box dimensions variances <first variance, second variance>
        std::map<unsigned, std::pair<float, float> > m_id_bb_variance_map;

        bool m_use_global_features;
        Features* m_globalFeatureDescriptor;

        CustomSVM m_svm; // for global feature classification
        bool m_svm_error;
        std::vector<std::string> m_svm_files;

        // maps class ids to a vector of global features, number of objects per class = number of global features per class
        std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > m_global_features; // used only during training

        // all global features as a cloud
        pcl::PointCloud<ISMFeature>::Ptr m_global_features_single_object; // used during detection in single-object-mode
        pcl::PointCloud<ISMFeature>::Ptr m_all_global_features_cloud; // used only during detection in NON-single-object-mode

        // average radii of each class from training
        std::map<unsigned, float> m_average_radii;

        std::string m_global_feature_method;
        int m_k_global_features;
        bool m_single_object_mode;
        int m_global_feature_influence_type;
        float m_global_param_min_svm_score;
        float m_global_param_rate_limit;
        float m_global_param_weight_factor;

        std::string m_max_filter_type;
        SingleObjectMaxType m_max_type;
        std::string m_max_type_param;

        // used to create FLANN index in FLANN helper
        std::string m_distanceType;

    private:

        std::vector<VotingMaximum> filterMaxima(const std::vector<VotingMaximum> &maxima, bool merge = false) const;
        std::vector<VotingMaximum> mergeAndFilterMaxima(const std::vector<VotingMaximum> &maxima) const;

        VotingMaximum mergeMaxima(const std::vector<VotingMaximum> &max_list) const;

        void insertGlobalResult(std::map<unsigned, GlobalResultAccu> &max_global_voting, unsigned found_class, float score);

        static bool sortMaxima(const VotingMaximum&, const VotingMaximum&);

        void normalizeWeights(std::vector<VotingMaximum> &maxima);

        std::map<unsigned, std::vector<Vote> > m_votes;

        float m_minThreshold;   // retrieve all maxima above the weight threshold
        int m_minVotesThreshold; // retrieve all maxima above the vote threshold
        int m_bestK;            // additionally retrieve only the k best maxima
        bool m_averageRotation;

        std::shared_ptr<FlannHelper> m_flann_helper;
        bool m_index_created;

        // NOTE: only for debug
        static float state_gt;
        static float state_true;
        static float state_false;
    };
}

#endif // ISM3D_VOTING_H
