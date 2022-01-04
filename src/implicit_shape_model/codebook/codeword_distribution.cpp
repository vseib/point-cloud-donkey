/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2020, Viktor Seib
 * All rights reserved.
 *
 */

#include "codeword_distribution.h"
#include "codeword_distribution_factory.h"

#include "../codebook/codeword.h"
#include "../utils/ism_feature.h"
#include "../voting/voting.h"
#include "../utils/utils.h"
#include "../utils/distance.h"
#include "../utils/exception.h"

namespace ism3d
{
    inline float gaussDist(float sigmaSqr, float dist)
    {
        return (1 / sqrt(2*M_PI*sigmaSqr)) * exp(-pow(dist, 2) / (2 * sigmaSqr));
    }


    CodewordDistribution::CodewordDistribution()
    {
    }

    CodewordDistribution::~CodewordDistribution()
    {
    }

    void CodewordDistribution::addCodeword(const std::shared_ptr<Codeword>& codeword,
                                           const ISMFeature& feature,
                                           const Utils::BoundingBox& boundingBox)
    {
        if (!codeword.get())
            throw RuntimeException("invalid codeword");

        if (!m_codeword.get())
            m_codeword = codeword;
        else if (m_codeword->getId() != codeword->getId())
            throw RuntimeException("codeword ids not matching");

        // get activation position, relative to object center
        Eigen::Vector3f keyPos(feature.x, feature.y, feature.z);
        Eigen::Vector3f center = boundingBox.position; // TODO VS: try swapping bbox for cloud centroid
        Eigen::Vector3f vote = center - keyPos;
        m_originalVotes.push_back(vote);

        vote = Utils::rotateInto(vote, feature.referenceFrame);

        // update activation distribution
        m_votes.push_back(vote);
        m_class_ids.push_back(feature.classId);
        m_instance_ids.push_back(feature.instanceId);

        m_features.push_back(feature);
        m_modelCenters.push_back(center);

        // transform bounding box coordinate system into reference frame
        Utils::BoundingBox newBox = boundingBox;
        boost::math::quaternion<float> rotQuat;
        Utils::getRotQuaternion(feature.referenceFrame, rotQuat);
        newBox.rotQuat = newBox.rotQuat * boost::math::conj(rotQuat);
        m_boundingBoxes.push_back(newBox);
    }

    void CodewordDistribution::castVotes(const ISMFeature& feature,
                                         const Distance* distance,
                                         const std::map<unsigned, float>& classSigmas, //NOTE: sigmas were stored as sigma^2 (variance)
                                         bool useClassWeight,
                                         bool useVoteWeight,
                                         bool useMatchingWeight,
                                         bool useCodewordWeight,
                                         Voting& voting,
                                         const std::map<int, std::shared_ptr<CodewordDistribution> > &distribution) const
    {
        LOG_ASSERT(m_votes.size() == m_weights.size());
        LOG_ASSERT(m_votes.size() == m_boundingBoxes.size());
        LOG_ASSERT(m_votes.size() == m_class_ids.size());

        float dist = (*distance)(m_codeword->getData(), feature.descriptor);

        // compute vote position
        for (int i = 0; i < (int)m_votes.size(); i++)
        {
            const Eigen::Vector3f& vote = m_votes[i];
            unsigned classId = m_class_ids[i];
            unsigned instanceId = m_instance_ids[i];

            // find weight for class
            std::map<unsigned, float>::const_iterator it = m_classWeights.find(classId);
            float classWeight = 1.0f;
            if (it != m_classWeights.end())
                classWeight = it->second;
            else
                LOG_WARN("no class weight found for class " << classId);

            // find sigma for class
            it = classSigmas.find(classId);
            float classSigma = 0;
            if (it != classSigmas.end()) {
                classSigma = it->second;
            }
            else
            {
                LOG_WARN("no sigma found for class " << classId);
                classSigma = 1;
            }

            // compute likelihood weight to replace old matching weight
            float matching_weight = gaussDist(classSigma, dist); // NOTE: classSigma is actually class variance, so no square needed

            // learned weight per vote
            float voteWeight = m_weights[i];

            // compute vote weight
            float weight = 1.0f;
            // TODO VS rethink all weights and maybe remove some: remove codeword weight
            weight = useClassWeight ? weight * classWeight : weight;
            weight = useVoteWeight ? weight * voteWeight : weight;
            weight = useMatchingWeight ? weight * matching_weight : weight;
            weight = useCodewordWeight ? weight * m_codeword->getWeight() : weight;

            // TODO VS: rethink this if-clause
            if(abs(dist) > 2*classSigma)
            {
                //LOG_ERROR("------------ discarding vote due to big distance, dist: " << dist << ", sigma: " << classSigma);
                continue;
            }

            if (weight < std::numeric_limits<float>::epsilon())
                continue;

            // cast vote into voting space
            castVote(vote, feature.referenceFrame, feature, m_boundingBoxes[i],
                     weight, classId, instanceId, voting, m_codeword);
        }
    }

    void CodewordDistribution::castVote(const Eigen::Vector3f& vote,
                                        const pcl::ReferenceFrame& refFrame,
                                        const ISMFeature& feature,
                                        Utils::BoundingBox boundingBox,
                                        float weight,
                                        unsigned classId,
                                        unsigned instanceId,
                                        Voting& voting,
                                        std::shared_ptr<Codeword> codeword) const
    {
        Eigen::Vector3f keyPos(feature.x, feature.y, feature.z);

        // transform center position using reference frame
        Eigen::Vector3f center = keyPos + Utils::rotateBack(vote, refFrame);

        // transform bounding box coordinate system from reference frame back into world coordinate system
        boost::math::quaternion<float> rotQuat;
        Utils::getRotQuaternion(refFrame, rotQuat);
        boundingBox.rotQuat = boundingBox.rotQuat * rotQuat;

        voting.vote(center, weight, classId, instanceId, keyPos, boundingBox, codeword);
    }


    // TODO VS: check what this method is used for
    void CodewordDistribution::computeWeights()
    {
        /*Eigen::Vector3f pos(1, 1, 1);
        Eigen::Vector3f size(3, 2, 3);
        Eigen::Vector3f center(2, 1, 1);

        Eigen::Vector3f halfSize = size * 0.5f;
        Eigen::Vector3f dirToCenter = center - pos;
        Eigen::Vector3f normalizedDir(dirToCenter[0] / halfSize[0],
                dirToCenter[1] / halfSize[1],
                dirToCenter[2] / halfSize[2]);
        Eigen::Vector3f newCenter = pos + normalizedDir;

        return;*/

        const float sigma = 0.5f;

        LOG_ASSERT(m_votes.size() == m_features.size());
        LOG_ASSERT(m_votes.size() == m_modelCenters.size());

        // get all the votes for this codeword
        m_weights.resize(m_votes.size());
        for (int i = 0; i < (int)m_votes.size(); i++)
        {
            const Eigen::Vector3f& vote = m_votes[i];

            // compute a list of weights for each vote and activating keypoint
            std::vector<float> listOfWeights;
            for (int j = 0; j < (int)m_features.size(); j++)
            {
                const ISMFeature& feature = m_features[j];
                const Utils::BoundingBox& boundingBox = m_boundingBoxes[j];

                // transform vote back into world coordinate system
                Eigen::Vector3f keyPos(feature.x, feature.y, feature.z);
                Eigen::Vector3f center = keyPos + Utils::rotateBack(vote, feature.referenceFrame);

                // compute a normalized center with respect to the bounding box
                Eigen::Vector3f boxHalfSize = boundingBox.size;// * 0.5f;
                Eigen::Vector3f dirToCenter = center - m_modelCenters[i];
                Eigen::Vector3f normalizedDirToCenter(dirToCenter[0] / boxHalfSize[0],
                        dirToCenter[1] / boxHalfSize[1],
                        dirToCenter[2] / boxHalfSize[2]);

                // get the distance from the vote in relation to the keypoint position to the model center
                float dist = dirToCenter.norm();
                float normalizedDist = normalizedDirToCenter.norm();

                // NOTE: Sigma specifies the variance between a measured object center and the actual object
                // center in a normalized way with respect to the bounding box's dimensions. A sigma of 0.2 means
                // that the probability decreases from 1 to 0.36 if the measured center is farther away than
                // 20 % of the bounding box's size.

                // compute gaussian distribution (weight \in {0, 1})
                float weight = exp((-1 * (dist * dist)) / (sigma * sigma));

                listOfWeights.push_back(weight);
            }

            LOG_ASSERT(listOfWeights.size() > 0);
            LOG_ASSERT(listOfWeights.size() == m_features.size());

            // compute median over weights
            float median = 0;
            sort(listOfWeights.begin(), listOfWeights.end());
            if (listOfWeights.size() % 2 == 0)
                median = (listOfWeights[listOfWeights.size() / 2 - 1] + listOfWeights[listOfWeights.size() / 2]) / 2;
            else
                median = listOfWeights[listOfWeights.size() / 2];

            m_weights[i] = median;
        }
    }

    void CodewordDistribution::addDistribution(std::shared_ptr<CodewordDistribution> distribution)
    {
        if (distribution->getCodewordId() != getCodewordId()) {
            LOG_ERROR("codewords do not match");
            return;
        }

        m_votes.insert(m_votes.end(), distribution->m_votes.begin(), distribution->m_votes.end());
        m_class_ids.insert(m_class_ids.end(), distribution->m_class_ids.begin(), distribution->m_class_ids.end());
        m_instance_ids.insert(m_instance_ids.end(), distribution->m_instance_ids.begin(), distribution->m_instance_ids.end());
        m_boundingBoxes.insert(m_boundingBoxes.end(), distribution->m_boundingBoxes.begin(), distribution->m_boundingBoxes.end());
        m_originalVotes.insert(m_originalVotes.end(), distribution->m_originalVotes.begin(), distribution->m_originalVotes.end());
        m_features.insert(m_features.end(), distribution->m_features.begin(), distribution->m_features.end());
        m_modelCenters.insert(m_modelCenters.end(), distribution->m_modelCenters.begin(), distribution->m_modelCenters.end());

        // recompute weights
        m_weights.clear();
        computeWeights();
    }

    void CodewordDistribution::setClassWeights(std::map<unsigned, float> classWeights)
    {
        m_classWeights = classWeights;
    }

    const std::shared_ptr<Codeword>& CodewordDistribution::getCodeword() const
    {
        return m_codeword;
    }

    int CodewordDistribution::getCodewordId() const
    {
        return m_codeword->getId();
    }

    const std::vector<Eigen::Vector3f>& CodewordDistribution::getVotes() const
    {
        return m_votes;
    }

    const std::vector<Eigen::Vector3f>& CodewordDistribution::getOriginalVotes() const
    {
        if (m_originalVotes.size() != 0)
            return m_originalVotes;
        else
            return m_votes;
    }

    const std::vector<float>& CodewordDistribution::getWeights() const
    {
        return m_weights;
    }

    const std::vector<unsigned>& CodewordDistribution::getClassIds() const
    {
        return m_class_ids;
    }

    const std::vector<unsigned>& CodewordDistribution::getInstanceIds() const
    {
        return m_instance_ids;
    }

    bool CodewordDistribution::hasClassId(unsigned classId) const
    {
        for (int i = 0; i < (int)m_class_ids.size(); i++) {
            unsigned id = m_class_ids[i];
            if (id == classId)
                return true;
        }
        return false;
    }

    std::vector<unsigned> CodewordDistribution::getDistinctClassIds() const
    {
        std::vector<unsigned> classIds;
        for (int i = 0; i < (int)m_class_ids.size(); i++) {
            unsigned classId = m_class_ids[i];
            if (std::find(classIds.begin(), classIds.end(), classId) == classIds.end())
                classIds.push_back(classId);
        }
        return classIds;
    }

    const std::vector<Utils::BoundingBox>& CodewordDistribution::getBoundingBoxes() const
    {
        return m_boundingBoxes;
    }

    int CodewordDistribution::getNumVotes() const
    {
        return m_votes.size();
    }

    int CodewordDistribution::getNumVotesForClass(unsigned classId) const
    {
        int numVotes = 0;
        for (int i = 0; i < (int)m_class_ids.size(); i++) {
            if (m_class_ids[i] == classId)
                numVotes++;
        }
        return numVotes;
    }

    void CodewordDistribution::iSaveData(boost::archive::binary_oarchive &oa) const
    {
        m_codeword->saveData(oa);

        int votes_size = m_votes.size();
        oa << votes_size;
        for (int i = 0; i < (int)m_votes.size(); i++)
        {
            oa << m_votes[i][0];
            oa << m_votes[i][1];
            oa << m_votes[i][2];
        }

        oa << m_weights;
        oa << m_class_ids;
        oa << m_instance_ids;

        int class_weights_size = m_classWeights.size();
        oa << class_weights_size;
        for (std::map<unsigned, float>::const_iterator it = m_classWeights.begin(); it != m_classWeights.end(); it++)
        {
            int classId = it->first;
            float weight = it->second;
            oa << classId;
            oa << weight;
        }

        int bounding_box_size = m_boundingBoxes.size();
        oa << bounding_box_size;
        for (int i = 0; i < (int)m_boundingBoxes.size(); i++)
        {
            const Utils::BoundingBox& bbox = m_boundingBoxes[i];
            float quat1 = bbox.rotQuat.R_component_1();
            float quat2 = bbox.rotQuat.R_component_2();
            float quat3 = bbox.rotQuat.R_component_3();
            float quat4 = bbox.rotQuat.R_component_4();
            oa << quat1;
            oa << quat2;
            oa << quat3;
            oa << quat4;
            oa << bbox.size[0];
            oa << bbox.size[1];
            oa << bbox.size[2];
        }
    }

    bool CodewordDistribution::iLoadData(boost::archive::binary_iarchive &ia)
    {
        m_codeword = std::shared_ptr<Codeword>(new Codeword());

        if(!m_codeword.get())
        {
            LOG_ERROR("Could not create codeword!");
            return false;
        }

        if(!m_codeword->loadData(ia))
        {
            LOG_ERROR("Could not read codeword!");
            return false;
        }

        m_votes.clear();
        m_weights.clear();
        m_class_ids.clear();
        m_instance_ids.clear();
        m_classWeights.clear();
        m_boundingBoxes.clear();

        int votes_size;
        ia >> votes_size;
        for (int i = 0; i < votes_size; i++)
        {
            float x, y, z;
            ia >> x;
            ia >> y;
            ia >> z;
            Eigen::Vector3f vector(x,y,z);
            m_votes.push_back(vector);
        }

        ia >> m_weights;
        ia >> m_class_ids;
        ia >> m_instance_ids;

        int weights_size;
        ia >> weights_size;
        for (int i = 0; i < weights_size; i++)
        {
            int classId;
            float weight;
            ia >> classId;
            ia >> weight;
            m_classWeights[classId] = weight;
        }

        int bb_size;
        ia >> bb_size;
        for (int i = 0; i < bb_size; i++)
        {
            Utils::BoundingBox bbox;
            float quat1, quat2, quat3, quat4;
            float x, y, z;
            ia >> quat1;
            ia >> quat2;
            ia >> quat3;
            ia >> quat4;
            ia >> x;
            ia >> y;
            ia >> z;
            bbox.rotQuat = boost::math::quaternion<float>(quat1, quat2, quat3, quat4);
            bbox.size = Eigen::Vector3f(x,y,z);
            m_boundingBoxes.push_back(bbox);
        }

        return true;
    }
}
