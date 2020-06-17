/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_CODEWORDDISTRIBUTION_H
#define ISM3D_CODEWORDDISTRIBUTION_H

#include <Eigen/Core>
#include <pcl/point_types.h>

#include "../utils/utils.h"
#include "../utils/json_object.h"

namespace ism3d
{
    class Distance;
    class Codeword;
    class ISMFeature;
    class BoundingBox;
    class Voting;

    /**
     * @brief The CodewordDistribution class
     * The codeword distribution contains a codeword and a list of activation vectors. These specify the locations
     * on which a codeword can be found on a model. Also, additional information is stored. A codeword distribution
     * entry is created in the activation process during training. If a codeword is activated several times, more
     * activation vectors are added to the distribution.
     * During detection, all possible activation vectors for an activated codeword are cast into a weighted voting space.
     */
    class CodewordDistribution
            : public JSONObject
    {
    public:
        CodewordDistribution();
        ~CodewordDistribution();

        /**
         * @brief Add a codeword to the distribution.
         * @param codeword the codeword
         * @param feature the feature that activated the codeword
         * @param boundingBox the bounding box for the training model
         */
        void addCodeword(const std::shared_ptr<Codeword>& codeword,
                         const ISMFeature& feature,
                         const Utils::BoundingBox& boundingBox);

        /**
         * @brief Cast all votes for this codeword distribution into the voting space.
         * @param feature the feature that activated the codeword
         * @param distance the distance measure
         * @param classSigmas a map of sigmas for each class
         * @param useClassWeight true to use statistical weights
         * @param useVoteWeight true to use center weights
         * @param useMatchingWeight true to use matching weights
         * @param useCodewordWeight true to use codeword weights
         * @param voting the voting space
         * @param distribution codeword distribution with all codewords and votes
         */
        void castVotes(const ISMFeature& feature,
                       const Distance* distance,
                       const std::map<unsigned, float>& classSigmas,
                       bool useClassWeight,
                       bool useVoteWeight,
                       bool useMatchingWeight,
                       bool useCodewordWeight,
                       Voting& voting, const std::map<int, std::shared_ptr<CodewordDistribution> > &distribution) const;

        /**
         * @brief Compute learned weights.
         */
        void computeWeights();

        /**
         * @brief Add another distribution with the same associated codeword.
         * @param distribution the distribution to add
         */
        void addDistribution(std::shared_ptr<CodewordDistribution> distribution);

        /**
         * @brief set distribution-specific class weights
         * @param classWeights the class weights for this entry
         */
        void setClassWeights(std::map<unsigned, float> classWeights);

        /**
         * @brief Get the codeword for the distribution.
         * @return the codeword
         */
        const std::shared_ptr<Codeword>& getCodeword() const;

        /**
         * @brief Get the codeword id for the distribution.
         * @return the codeword id
         */
        int getCodewordId() const;

        /**
         * @brief Get the votes for this codeword relative to the feature position.
         * @return the votes
         */
        const std::vector<Eigen::Vector3f>& getVotes() const;

        /**
         * @brief Get the original votes for this codeword. They are given in absolute positions
         * and should only be used for display purposes. They are not saved with the distribution.
         * @return the original votes
         */
        const std::vector<Eigen::Vector3f>& getOriginalVotes() const;

        /**
         * @brief Get the learned weights for each vote vector.
         * @return the learned weights
         */
        const std::vector<float>& getWeights() const;

        /**
         * @brief Get class ids for each vote vector.
         * @return the list of class ids
         */
        const std::vector<unsigned>& getClassIds() const;

        /**
         * @brief Get instance ids for each vote vector.
         * @return the list of instance ids
         */
        const std::vector<unsigned>& getInstanceIds() const;

        /**
         * @brief Check if a class id has a corresponding vote.
         * @param classId the class id to check
         * @return true if the class id has a corresponding vote
         */
        bool hasClassId(unsigned classId) const;

        /**
         * @brief Get a list of distinct class ids, i.e. a list without duplicates.
         * @return the list of distinct class ids
         */
        std::vector<unsigned> getDistinctClassIds() const;

        /**
         * @brief Get the bounding boxes for each vote vector.
         * @return the list of bounding boxes.
         */
        const std::vector<Utils::BoundingBox>& getBoundingBoxes() const;

        /**
         * @brief Get the number of votes contained in the distribution.
         * @return the number of votes
         */
        int getNumVotes() const;

        /**
         * @brief Get the number of votes for the given class id.
         * @param classId the class id
         * @return the number of votes for the given class id
         */
        int getNumVotesForClass(unsigned classId) const;

    protected:

        void iSaveData(boost::archive::binary_oarchive &oa) const;
        bool iLoadData(boost::archive::binary_iarchive &ia);

    private:
        void castVote(const Eigen::Vector3f&,
                      const pcl::ReferenceFrame&,
                      const ISMFeature&,
                      Utils::BoundingBox,
                      float weight,
                      unsigned classId, unsigned instanceId,
                      Voting&, int codewordId) const;

        // saved with the distribution
        std::shared_ptr<Codeword> m_codeword;             // the associated codeword
        std::vector<Eigen::Vector3f> m_votes;               // vote vectors per codeword
        std::vector<float> m_weights;                       // weights for vote vectors
        std::vector<unsigned> m_class_ids;                   // class ids for vote vectors
        std::vector<unsigned> m_instance_ids;               // instance ids for vote vectors
        std::vector<Utils::BoundingBox> m_boundingBoxes;    // bounding boxes for vote vectors
        std::map<unsigned, float> m_classWeights;

        // not saved with the distribution, only needed during training
        std::vector<Eigen::Vector3f> m_originalVotes;       // contains votes before any transformation
        std::vector<ISMFeature> m_features;
        std::vector<Eigen::Vector3f> m_modelCenters;
    };
}

#endif // ISM3D_CODEWORDDISTRIBUTION_H
