/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "codebook.h"
#include "codebook_factory.h"
#include "../utils/utils.h"
#include "../utils/distance.h"

#include <random>

#include "codeword_distribution.h"
#include "../activation_strategy/activation_strategy.h"
#include "../activation_strategy/activation_strategy_knn.h"
#include "../activation_strategy/activation_strategy_inn.h"
#include "../activation_strategy/activation_strategy_threshold.h"

namespace ism3d
{
Codebook::Codebook()
    : m_activationStrategy(0)
{
    m_activationStrategy = new ActivationStrategyKNN();

    addParameter(m_useClassWeight, "UseClassWeight", false);
    addParameter(m_useVoteWeight, "UseVoteWeight", false);
    addParameter(m_useMatchingWeight, "UseMatchingWeight", false);
    addParameter(m_useCodewordWeight, "UseCodewordWeight", false);

    addParameter(m_use_partial_shot, "UsePartialShot", false);
    addParameter(m_partial_shot_type, "PartialShotType", std::string("front"));

    addParameter(m_use_random_codebook, "UseRandomCodebook", false);
    addParameter(m_random_codebook_factor, "RandomCodebookFactor", 1.0f);
}

Codebook::~Codebook()
{
    delete m_activationStrategy;
    m_distribution.clear();
}

template
void Codebook::activate<flann::L2<float> >(const std::vector<std::shared_ptr<Codeword>> &codewords,
const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> &features,
const std::map<unsigned, std::vector<Utils::BoundingBox>> &boundingBoxes,
const Distance* distance, flann::Index<flann::L2<float>> &index,
const bool flann_exact_match);

template
void Codebook::activate<flann::ChiSquareDistance<float> >(const std::vector<std::shared_ptr<Codeword>> &codewords,
const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> &features,
const std::map<unsigned, std::vector<Utils::BoundingBox>> &boundingBoxes,
const Distance* distance, flann::Index<flann::ChiSquareDistance<float>> &index,
const bool flann_exact_match);

template<typename T>
void Codebook::activate(const std::vector<std::shared_ptr<Codeword>> &codewords,
                        const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> &features,
                        const std::map<unsigned, std::vector<Utils::BoundingBox>> &boundingBoxes,
                        const Distance* distance, flann::Index<T> &index, const bool flann_exact_match)
{
    LOG_ASSERT(features.size() == boundingBoxes.size());

    // for every class
    LOG_INFO("Starting step 1");
    for (std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> >::const_iterator it = features.begin();
         it != features.end(); it++)
    {
        unsigned classId = it->first;
        LOG_INFO("Processing class " << (classId+1) << " of " << features.size());

        const std::vector<pcl::PointCloud<ISMFeature>::Ptr>& classModelFeatures = it->second;

        std::map<unsigned, std::vector<Utils::BoundingBox> >::const_iterator bbIt = boundingBoxes.find(classId);
        if (bbIt == boundingBoxes.end()) {
            LOG_ERROR("no matching bounding box found for class id " << classId);
            continue;
        }

        const std::vector<Utils::BoundingBox>& boundingBoxesClass = bbIt->second;
        if (boundingBoxesClass.size() != classModelFeatures.size()) {
            LOG_ERROR("unequal number of bounding boxes and objects for class id " << classId);
            continue;
        }

        // these vectors are used to accumulate processed data for variance computation per class
        std::vector<ISMFeature> allModelFeatures;
        std::vector<std::shared_ptr<Codeword>> allActivatedCodewords;

        // NOTE: using ALL features and codewords with large datasets will take a very long time and use
        //       huge amounts of memory; in most cases, using partial computation will be enough, this is
        //       controlled by max_elements
        // count features of this class
        int num_features = 0;
        for (int i = 0; i < (int)classModelFeatures.size(); i++)
        {
            num_features += classModelFeatures[i]->size();
        }
        int max_elements = sqrt(num_features);

        // for every model in the class
        for (int i = 0; i < (int)classModelFeatures.size(); i++)
        {
            const Utils::BoundingBox& boundingBox = boundingBoxesClass[i];
            const pcl::PointCloud<ISMFeature>::Ptr modelFeatures = classModelFeatures[i];

            // for every feature in the model
            for (int j = 0; j < (int)modelFeatures->size(); j++)
            {
                const ISMFeature& feature = modelFeatures->at(j);
                std::vector<std::shared_ptr<Codeword>> activatedCodewords;

                // activate codeword with the current feature
                if(m_activationStrategy->getType() == "KNN")
                {
                    ActivationStrategyKNN* asknn = dynamic_cast<ActivationStrategyKNN*>(m_activationStrategy);
                    activatedCodewords = asknn->activateKNN(feature, codewords, index, flann_exact_match);
                }
                else if(m_activationStrategy->getType() == "INN")
                {
                    ActivationStrategyINN* asinn = dynamic_cast<ActivationStrategyINN*>(m_activationStrategy);
                    activatedCodewords = asinn->activateINN(feature, codewords, index, flann_exact_match);
                }
                else
                {
                    activatedCodewords = m_activationStrategy->operate(feature, codewords, distance);
                }

                // for every activated codeword
                for (int k = 0; k < (int)activatedCodewords.size(); k++)
                {
                    const std::shared_ptr<Codeword>& codeword = activatedCodewords[k];

                    // if the codeword has not yet been activated, create a new distribution entry
                    if (m_distribution.find(codeword->getId()) == m_distribution.end())
                    {
                        m_distribution[codeword->getId()] =
                                std::shared_ptr<CodewordDistribution>(new CodewordDistribution);
                    }

                    // add the codeword to the distribution
                    m_distribution[codeword->getId()]->addCodeword(codeword, feature, boundingBox);
                }
                if(allActivatedCodewords.size() < max_elements)
                    allActivatedCodewords.insert(allActivatedCodewords.end(), activatedCodewords.begin(), activatedCodewords.end());
            }
            if(allModelFeatures.size() < max_elements)
                allModelFeatures.insert(allModelFeatures.end(), modelFeatures->begin(), modelFeatures->end());
        }

        // compute the mean distance between all class-specific features and their activated codewords
        float sum = 0;
        std::vector<float> distances;
        for (const ISMFeature& feature : allModelFeatures)
        {
            for(const std::shared_ptr<Codeword>& codeword : allActivatedCodewords)
            {
                float d = (*distance)(feature.descriptor, codeword->getData());
                sum += d;
                distances.push_back(d);
            }
        }

        // mean of distances between all features and all activated codewords inside the class
        int num = allModelFeatures.size()*allActivatedCodewords.size();
        float mean = sum / num;

        // compute the corresponding class-specific variance
        float variance = 0;
        for (const float &dist : distances)
        {
            float diff = dist - mean;
            variance += diff * diff;
        }
        variance /= num - 1;

        // store class-specific variance
        m_classSigmas[classId] = variance;
    }

    // distribution was changed, fill list with codewords
    m_codewords.clear();
    for (distribution_t::const_iterator it = m_distribution.begin(); it != m_distribution.end(); it++)
        m_codewords.push_back(it->second->getCodeword());

    // remove all codewords that have more than 1 vote in KNN activation with k = 1 or in INN
    bool clean_up = false;
    if(m_activationStrategy->getType() == "KNN")
    {
        ActivationStrategyKNN* asknn = dynamic_cast<ActivationStrategyKNN*>(m_activationStrategy);
        if(asknn->getK() == 1) clean_up = true;
    }
    else if(m_activationStrategy->getType() == "INN")
    {
        clean_up = true;
    }

    if(clean_up)
    {
        distribution_t clean_distribution;
        for (distribution_t::const_iterator it = m_distribution.begin(); it != m_distribution.end(); it++)
        {
            if(it->second->getNumVotes() == 1)
            {
                clean_distribution.insert({it->first, it->second});
            }
        }
        m_distribution = clean_distribution;
    }

    LOG_INFO("Starting step 2");
    // compute weights for each codeword (center weights)
    for (distribution_t::const_iterator it = m_distribution.begin(); it != m_distribution.end(); it++) {
        //#pragma omp task
        it->second->computeWeights();
    }

    LOG_INFO("Starting step 3");
    // get a map containing the number of votes for a class for a codeword
    std::map<unsigned, std::map<int, int> > numVotesForClassForCodeword;

    for (distribution_t::const_iterator it = m_distribution.begin(); it != m_distribution.end(); it++) {
        const std::shared_ptr<CodewordDistribution>& entry = it->second;

        std::vector<unsigned> classIds = entry->getDistinctClassIds();
        for (int j = 0; j < (int)classIds.size(); j++)
        {
            unsigned classId = classIds[j];
            int numVotes = entry->getNumVotesForClass(classId);

            if (numVotesForClassForCodeword.find(classId) == numVotesForClassForCodeword.end())
                numVotesForClassForCodeword[classId][entry->getCodewordId()] = numVotes;
            else {
                std::map<int, int>& votesPerCodeword = numVotesForClassForCodeword[classId];
                if (votesPerCodeword.find(entry->getCodewordId()) == votesPerCodeword.end())
                    votesPerCodeword[entry->getCodewordId()] = numVotes;
                else
                    votesPerCodeword[entry->getCodewordId()] += numVotes;
            }
        }
    }

    LOG_INFO("Starting step 4");
    // get the number of features from which a class was learned
    std::map<unsigned, int> numFeaturesForClass;    // number of features from which class ci was learned
    for (std::map<unsigned, std::map<int, int> >::const_iterator it = numVotesForClassForCodeword.begin();
         it != numVotesForClassForCodeword.end(); it++)
    {
        unsigned classId = it->first;
        numFeaturesForClass[classId] = getNumOfFeaturesForClass(classId);
    }

    LOG_INFO("Starting step 5");
    // compute a mapping of codeword id to
    // (number of votes in that codeword divided by number of features in a class)
    // summed over all classes that use that codeword
    std::map<int, float> sum;
    for (std::map<unsigned, std::map<int, int> >::const_iterator it = numVotesForClassForCodeword.begin();
         it != numVotesForClassForCodeword.end(); it++)
    {
        unsigned classId = it->first;
        const std::map<int, int>& votesPerCodeword = it->second;
        int numFeatures = numFeaturesForClass[classId];

        for (std::map<int, int>::const_iterator it2 = votesPerCodeword.begin(); it2 != votesPerCodeword.end(); it2++)
        {
            int codewordId = it2->first;
            int numVotes = it2->second;

            if (sum.find(codewordId) == sum.end())
                sum[codewordId] = numVotes / (float)numFeatures;
            else
                sum[codewordId] += numVotes / (float)numFeatures;
        }
    }

    LOG_INFO("Starting step 6");
    // create a mapping of class id to all activated codewords of this class
    std::map<unsigned, std::vector<int> > codewordsPerClass;
    for (distribution_t::const_iterator it = m_distribution.begin(); it != m_distribution.end(); it++)
    {
        const std::shared_ptr<CodewordDistribution>& entry = it->second;
        const std::shared_ptr<Codeword>& codeword = entry->getCodeword();

        // get the number of words voting for a class
        const std::vector<unsigned>& classIds = entry->getClassIds();
        for (int k = 0; k < (int)classIds.size(); k++)
        {
            unsigned classId = classIds[k];

            std::map<unsigned, std::vector<int> >::const_iterator it = codewordsPerClass.find(classId);
            if (it == codewordsPerClass.end()) {
                codewordsPerClass[classId].push_back(codeword->getId());
            }
            else {
                const std::vector<int>& list = it->second;
                if (std::find(list.begin(), list.end(), codeword->getId()) == list.end())
                    codewordsPerClass[classId].push_back(codeword->getId());
            }
        }
    }

    LOG_INFO("Starting step 7");
    // number of words that vote for a class
    for (std::map<unsigned, std::vector<int> >::const_iterator it = codewordsPerClass.begin(); it != codewordsPerClass.end(); it++) {
        unsigned classId = it->first;
        m_term1[classId] = 1.0f / (float)it->second.size();
    }

    LOG_INFO("Starting step 8");
    // computed weights relating votes per codeword (m_term2) and votes per codeword and number of featues per class (m_term3)
    for (distribution_t::const_iterator it = m_distribution.begin(); it != m_distribution.end(); it++)
    {
        const std::shared_ptr<CodewordDistribution>& entry = it->second;
        m_term2[entry->getCodewordId()] = 1.0f / (float)entry->getNumVotes();

        std::vector<unsigned> classes = entry->getDistinctClassIds();
        for (int i = 0; i < (int)classes.size(); i++)
        {
            unsigned classId = classes[i];
            int numFeatures = numFeaturesForClass.find(classId)->second;
            int numVotes = numVotesForClassForCodeword.find(classId)->second.find(entry->getCodewordId())->second;

            m_term3[classId] = (numVotes / (float)numFeatures) / sum[entry->getCodewordId()];
        }
    }

    LOG_INFO("Starting step 9");
    // set class weights for distribution entry
    for (distribution_t::const_iterator it = m_distribution.begin(); it != m_distribution.end(); it++)
    {
        const std::shared_ptr<CodewordDistribution>& entry = it->second;

        // create learned class weights
        std::map<unsigned, float> classWeights;
        std::vector<unsigned> classes = entry->getDistinctClassIds();
        std::map<int, float>::const_iterator it2 = m_term2.find(entry->getCodewordId());
        for (int i = 0; i < (int)classes.size(); i++)
        {
            unsigned classId = classes[i];

            std::map<unsigned, float>::const_iterator it1 = m_term1.find(classId);
            std::map<unsigned, float>::const_iterator it3 = m_term3.find(classId);

            classWeights[classId] = it1->second * it2->second * it3->second;
        }

        // set class weights
        entry->setClassWeights(classWeights);
    }

    LOG_INFO("Size of distribution at the end of training: " << m_distribution.size());
}


template
void Codebook::castVotes<flann::L2<float>>(pcl::PointCloud<ISMFeature>::ConstPtr features,
    const Distance* distance, Voting& voting, flann::Index<flann::L2<float>> &index, const bool flann_exact_match) const;

template
void Codebook::castVotes<flann::ChiSquareDistance<float>>(pcl::PointCloud<ISMFeature>::ConstPtr features,
    const Distance* distance, Voting& voting, flann::Index<flann::ChiSquareDistance<float>> &index, const bool flann_exact_match) const;

template<typename T>
void Codebook::castVotes(pcl::PointCloud<ISMFeature>::ConstPtr features,
                         const Distance* distance, Voting& voting,
                         flann::Index<T> &index, const bool flann_exact_match) const
{
    pcl::PointCloud<ISMFeature>::Ptr features_new(new pcl::PointCloud<ISMFeature>());
    pcl::copyPointCloud(*features, *features_new);

    castVotes(features_new, distance, voting, index, flann_exact_match);
}


template
void Codebook::castVotes<flann::L2<float> >(pcl::PointCloud<ISMFeature>::Ptr features,
const Distance* distance, Voting& voting,
flann::Index<flann::L2<float> > &index,
const bool flann_exact_match) const;

template
void Codebook::castVotes<flann::ChiSquareDistance<float> >(pcl::PointCloud<ISMFeature>::Ptr features,
const Distance* distance, Voting& voting,
flann::Index<flann::ChiSquareDistance<float> > &index,
const bool flann_exact_match) const;

template<typename T>
void Codebook::castVotes(pcl::PointCloud<ISMFeature>::Ptr features,
                         const Distance* distance, Voting& voting, flann::Index<T> &index, const bool flann_exact_match) const
{
    // NOTE: refer to http://www.matheboard.de/archive/30610/thread.html

    if (isEmpty())
        return;

    // reduce descriptor for partial shot
    if(m_use_partial_shot)
    {
        int hist_size = 11;
        std::string desc_type = "SHOT";
        int shot_length = 352;
        int cshot_length = 1344;

        // create signature mask in case partial shot is used
        std::vector<bool> signature_mask = getSignatureMask();

        for (int i = 0; i < (int)features->size(); i++)
        {
            const ISMFeature& feature = features->at(i);
            if(feature.descriptor.size() > shot_length)
            {
                desc_type = "CSHOT";
            }

            // prepare partial descriptor
            std::vector<float> partial_descriptor;

            // handle SHOT or geometrical part of CSHOT
            for (int j = 0; j < shot_length; j++)
            {
                int signature_index = (j/hist_size);
                if(signature_mask.at(signature_index))
                {
                    partial_descriptor.push_back(feature.descriptor[j]);
                    //std::cout << j << " ";
                }
                else
                {
                    //std::cout << std::endl << "not using signature index: " << signature_index << std::endl;
                    j += (hist_size-1); // add 1 less because loop also adds 1
                }
            }

            if(desc_type == "CSHOT")
            {
                // handle color part of cshot
                hist_size = 31;
                for (int j = shot_length; j < feature.descriptor.size(); j++)
                {
                    int signature_index = ((j-shot_length)/hist_size);
                    if(signature_mask.at(signature_index))
                    {
                        partial_descriptor.push_back(feature.descriptor[j]);
                    }
                    else
                    {
                        //std::cout << std::endl << "not using signature index: " << signature_index << std::endl;
                        j += (hist_size-1); // add 1 less because loop also adds 1
                    }
                }
            }

            features->at(i).descriptor = partial_descriptor;
            //std::cout << "descriptor has size: " << feature.descriptor.size() << std::endl;
        }
    }

    std::vector<std::shared_ptr<CodewordDistribution> > activatedEntries;
    std::map<unsigned, std::vector<int> > featureIndices;

    const std::vector<std::shared_ptr<Codeword>> codewords = getCodewords();


#pragma omp parallel for
    for (int i = 0; i < (int)features->size(); i++)
    {
        const ISMFeature& feature = features->at(i);

        // activate codeword with keypoint
        std::vector<std::shared_ptr<Codeword> > activatedCodewords;
        if(m_activationStrategy->getType() == "KNN")
        {
            ActivationStrategyKNN* asknn = dynamic_cast<ActivationStrategyKNN*>(m_activationStrategy);
            activatedCodewords = asknn->activateKNN(feature, codewords, index, flann_exact_match);
        }
        else if(m_activationStrategy->getType() == "INN")
        {
            ActivationStrategyINN* asinn = dynamic_cast<ActivationStrategyINN*>(m_activationStrategy);
            activatedCodewords = asinn->activateINN(feature, codewords, index, flann_exact_match);
        }
        else
        {
            activatedCodewords = m_activationStrategy->operate(feature, codewords, distance);
        }

        // cast votes for each activated codeword
        for (int j = 0; j < (int)activatedCodewords.size(); j++)
        {
            const std::shared_ptr<Codeword>& codeword = activatedCodewords[j];

            // m_distribution maps a codeword id to its corresponding vote distribution
            if (m_distribution.find(codeword->getId()) == m_distribution.end())
            {
                LOG_WARN("codeword not found in distribution, skipping");
                continue;
            }

            std::shared_ptr<CodewordDistribution> entry = m_distribution.find(codeword->getId())->second;

            if (!entry.get()) {
                LOG_ERROR("invalid entry, skipping");
                continue;
            }

#pragma omp critical
            {
                featureIndices[entry->getCodewordId()].push_back(i); // maps an activated codeword id to the activating feature index
                if(featureIndices[entry->getCodewordId()].size() == 1) // avoids dublicate activated entries
                {
                    activatedEntries.push_back(entry);
                }
            }
        }
    }

    // actually cast votes
#pragma omp parallel for
    for (int i = 0; i < (int)activatedEntries.size(); i++)
    {
        std::shared_ptr<CodewordDistribution> entry = activatedEntries[i];
        std::shared_ptr<Codeword> codeword = entry->getCodeword();
        std::vector<int> featInd = featureIndices[codeword->getId()];

        for (int j = 0; j < (int)featInd.size(); j++)
        {
            const ISMFeature& feature = features->at(featInd[j]);
            entry->castVotes(feature, distance, m_classSigmas, m_useClassWeight, m_useVoteWeight,
                             m_useMatchingWeight, m_useCodewordWeight, voting, m_distribution);
        }
    }
}


void Codebook::addDistribution(std::shared_ptr<CodewordDistribution> distribution)
{
    std::shared_ptr<CodewordDistribution> distr = getDistributionById(distribution->getCodewordId());
    if (distr.get())
        distr->addDistribution(distribution);
    else
        m_distribution[distribution->getCodewordId()] = distribution;
}

bool Codebook::removeDistribution(int id)
{
    for (distribution_t::iterator it = m_distribution.begin(); it != m_distribution.end(); it++) {
        if (it->first == id)
        {
            m_distribution.erase(it);
            return true;
        }
    }
    LOG_WARN("could not remove codeword with id " << id);
    return false;
}

bool Codebook::removeDistributions(const std::vector<int>& ids)
{
    bool success = true;
    for (int i = 0; i < (int)ids.size(); i++) {
        if (!removeDistribution(ids[i]))
            success = false;
    }

    // distribution was changed, fill list with codewords
    m_codewords.clear();
    for (distribution_t::const_iterator it = m_distribution.begin(); it != m_distribution.end(); it++)
        m_codewords.push_back(it->second->getCodeword());

    return success;
}

std::shared_ptr<Codeword> Codebook::getCodewordById(int id) const
{
    std::shared_ptr<CodewordDistribution> distribution = getDistributionById(id);
    if (distribution.get())
        return distribution->getCodeword();
    return std::shared_ptr<Codeword>();
}

std::shared_ptr<CodewordDistribution> Codebook::getDistributionById(int id) const
{
    distribution_t::const_iterator it = m_distribution.find(id);
    if (it != m_distribution.end())
        return it->second;
    return std::shared_ptr<CodewordDistribution>();
}

std::vector<std::shared_ptr<Codeword>> Codebook::getCodewords(std::string warn) const
{
    if(m_use_partial_shot)
    {
        return m_partial_codewords;
    }
    else
    {
        if(warn != "") LOG_WARN(warn);
        return m_codewords;
    }
}

std::vector<std::shared_ptr<Codeword>> Codebook::getCompleteCodewords() const
{
    return m_codewords;
}

std::vector<std::shared_ptr<Codeword>> Codebook::getPartialCodewords() const
{
    std::string warn = "Warning: partial codewords not available! Using complete codewords.";
    return getCodewords(warn);
}

const std::map<int, std::shared_ptr<CodewordDistribution> >& Codebook::getDistribution() const
{
    return m_distribution;
}

bool Codebook::containsCodewordWithId(int id) const
{
    if (m_distribution.find(id) == m_distribution.end())
        return false;
    return true;
}

int Codebook::getNumCodewordsForClass(unsigned classId) const
{
    int numCodewords = 0;

    for (distribution_t::const_iterator it = m_distribution.begin(); it != m_distribution.end(); it++) {
        const std::shared_ptr<CodewordDistribution>& entry = it->second;
        std::vector<unsigned> classIds = entry->getDistinctClassIds();
        for (int i = 0; i < (int)classIds.size(); i++)
            if (classIds[i] == classId) {
                numCodewords++;
                break;
            }
    }

    return numCodewords;
}

int Codebook::getNumOfFeaturesForClass(unsigned classId) const
{
    int numFeatures = 0;
    for (distribution_t::const_iterator it = m_distribution.begin(); it != m_distribution.end(); it++)
    {
        const std::vector<unsigned>& classes = it->second->getClassIds();

        for (int j = 0; j < (int)classes.size(); j++) {
            if (classes[j] == classId)
                numFeatures++;
        }
    }

    return numFeatures;
}

std::vector<unsigned> Codebook::getClasses() const
{
    std::map<unsigned, int> classes;

    for (distribution_t::const_iterator it = m_distribution.begin(); it != m_distribution.end(); it++) {
        const std::shared_ptr<CodewordDistribution>& distribution = it->second;
        std::vector<unsigned> classIds = distribution->getDistinctClassIds();
        for (int i = 0; i < (int)classIds.size(); i++)
            classes[classIds[i]]++;
    }

    std::vector<unsigned> result;
    for (std::map<unsigned, int>::const_iterator it = classes.begin(); it != classes.end(); it++)
        result.push_back(it->first);

    return result;
}

bool Codebook::isEmpty() const
{
    return m_distribution.empty();
}

void Codebook::clear()
{
    m_distribution.clear();
}

Json::Value Codebook::iChildConfigsToJson() const
{
    Json::Value children(Json::objectValue);

    children["ActivationStrategy"] = m_activationStrategy->configToJson();

    return children;
}

bool Codebook::iChildConfigsFromJson(const Json::Value& children)
{
    const Json::Value* activationStrategy = &children["ActivationStrategy"];

    if (activationStrategy->isNull() || !activationStrategy->isObject())
        return false;

    // NOTE: as m_distribution is a map and the respective keys are only stored as data, it
    // will be initialized later in iDataFromJson

    // clear
    delete m_activationStrategy;

    // create child objects
    m_activationStrategy = Factory<ActivationStrategy>::create(*activationStrategy);

    if (!m_activationStrategy)
        return false;

    return true;
}

void Codebook::iSaveData(boost::archive::binary_oarchive &oa) const
{
    int distribution_size = m_distribution.size();
    oa << distribution_size;
    for (std::map<int, std::shared_ptr<CodewordDistribution> >::const_iterator it = m_distribution.begin();
         it != m_distribution.end(); it++)
    {
        const std::shared_ptr<CodewordDistribution>& entry = it->second;
        entry->saveData(oa);
    }

    int class_sigmas_size = m_classSigmas.size();
    oa << class_sigmas_size;
    for (std::map<unsigned, float>::const_iterator it = m_classSigmas.begin(); it != m_classSigmas.end(); it++) {
        int classId = it->first;
        float sigma = it->second;
        oa << classId;
        oa << sigma;
    }

    m_activationStrategy->saveData(oa);
}

bool Codebook::iLoadData(boost::archive::binary_iarchive &ia)
{
    // clear
    m_distribution.clear();

    // NOTE: this is only for debug to write out the indices of randomly selected codewords; check absolute path below
    // NOTE: to write out indices selected by a feature weighting algorithm check the same debug flag in implicit_shape_model.cpp
    bool debug_flag_write_out = false && m_use_random_codebook;
    // NOTE: this is only for debug to read in the indices of selected codewords; check absolute path below
    bool debug_flag_read_in = false && !m_use_random_codebook;

    std::ofstream ofs;

    if(debug_flag_write_out)
    {
        std::string path = "/home/vseib/Desktop/cwids/out.txt";
        LOG_INFO("Writing out file with randomly selected codeword indices to " << path);
        ofs.open(path, std::ofstream::out);
    }

    std::vector<int> indices_to_use;
    if(debug_flag_read_in)
    {
        std::string path = "/home/vseib/Desktop/cwids/best.txt";
        LOG_INFO("Reading in file with provided codeword indices from " << path);
        std::ifstream ifs;
        ifs.open(path, std::ifstream::in);
        std::string line;
        while(std::getline(ifs, line))
        {
            std::istringstream iss(line);
            int a;
            if (!(iss >> a)) { break; }
            indices_to_use.push_back(a);
        }
        ifs.close();
    }

    // create distribution
    int distribution_size;
    ia >> distribution_size;
    LOG_INFO("Loading codebook with size: " << distribution_size);

    for(int i = 0; i < distribution_size; i++)
    {
        std::shared_ptr<CodewordDistribution> entry(new CodewordDistribution());
        if(!entry.get())
        {
            LOG_ERROR("Could not create codeword distribution with index " << i << "!");
            return false;
        }

        if(!entry->loadData(ia))
        {
            LOG_ERROR("Could not read codeword distribution with index " << i << "!");
            return false;
        }

        // for random codebook: skip features while loading
        if(m_use_random_codebook)
        {
            std::random_device rd;
            std::mt19937 mt(rd());
            std::uniform_int_distribution<int> dist(0, distribution_size);
            if(dist(mt) > m_random_codebook_factor * distribution_size)
                continue;
        }

        if(debug_flag_read_in)
        {
            int id = entry->getCodewordId();
            bool found = false;
            for(int xxx = 0; xxx < indices_to_use.size(); xxx++)
            {
                if(indices_to_use.at(xxx) == id)
                {
                    found = true;
                    break;
                }
            }
            if(found) m_distribution[entry->getCodewordId()] = entry;
        }
        else // this is the normal case
        {
            m_distribution[entry->getCodewordId()] = entry;
        }

        if(debug_flag_write_out) ofs << entry->getCodewordId() << std::endl;
    }
    if(m_use_random_codebook) LOG_INFO("Reduced codebook size: " << m_distribution.size());
    if(debug_flag_read_in) LOG_INFO("Loaded codebook size: " << m_distribution.size());
    if(debug_flag_write_out) ofs.close();

    // fill list with codewords
    m_codewords.clear();
    for (distribution_t::const_iterator it = m_distribution.begin(); it != m_distribution.end(); it++)
        m_codewords.push_back(it->second->getCodeword());

    m_codeword_dim = m_codewords.at(0)->getData().size();

    // fill list with partial codewords
    if(m_use_partial_shot)
    {
        int hist_size = 11;
        std::string desc_type = "SHOT";
        int shot_length = 352;
        int cshot_length = 1344;

        LOG_INFO("creating partial shot descriptors");
        for (distribution_t::const_iterator it = m_distribution.begin(); it != m_distribution.end(); it++)
        {
            std::shared_ptr<Codeword> cw = it->second->getCodeword();
            std::vector<float> descriptor = cw->getData();

            // create signature mask
            std::vector<bool> signature_mask = getSignatureMask();

            if(descriptor.size() > shot_length)
            {
                desc_type = "CSHOT";
            }

            // prepare partial descriptor
            std::vector<float> partial_descriptor;
            if(desc_type == "SHOT")
                partial_descriptor.reserve(shot_length/2);
            else if(desc_type == "CSHOT")
                partial_descriptor.reserve(cshot_length/2);

            // handle SHOT or geometrical part of CSHOT
            for (int j = 0; j < shot_length; j++)
            {
                int signature_index = (j/hist_size);
                if(signature_mask.at(signature_index))
                {
                    partial_descriptor.push_back(descriptor[j]);
                    //std::cout << j << " ";
                }
                else
                {
                    //std::cout << std::endl << "not using signature index: " << signature_index << std::endl;
                    j += (hist_size-1); // add 1 less because loop also adds 1
                }
            }

            if(desc_type == "CSHOT")
            {
                // handle color part of cshot
                hist_size = 31;
                for (int j = shot_length; j < descriptor.size(); j++)
                {
                    int signature_index = ((j-shot_length)/hist_size);
                    if(signature_mask.at(signature_index))
                    {
                        partial_descriptor.push_back(descriptor[j]);
                        //std::cout << j << " ";
                    }
                    else
                    {
                        //std::cout << std::endl << "not using signature index: " << signature_index << std::endl;
                        j += (hist_size-1); // add 1 less because loop also adds 1
                    }
                }
            }

            cw->setData(partial_descriptor, cw->getNumFeatures(), cw->getWeight());
            m_codeword_dim = partial_descriptor.size();
            m_partial_codewords.push_back(cw);
        }
    }

    // fill class sigmas
    int class_sigmas_size;
    ia >> class_sigmas_size;
    m_classSigmas.clear();
    for (int i = 0; i < class_sigmas_size; i++)
    {
        int classId;
        float sigma;
        ia >> classId;
        ia >> sigma;
        m_classSigmas[classId] = sigma;
    }

    m_activationStrategy->loadData(ia);

    return true;
}

std::vector<bool> Codebook::getSignatureMask() const
{
    // SHOT descriptor has 32 signature bins, each bin holds a histogram of length 11
    // CSHOT descriptor has 32 signature bins, each bin holds a histogram of length 31

    std::vector<bool> result(32, false);
    if(m_partial_shot_type == "front" || m_partial_shot_type == "dense_x")
    {
        // bins with indices 8 to 23 correspond to the frontal part
        for(int i = 8; i <= 23; i++)
            result[i] = true;
    }
    else if(m_partial_shot_type == "back" || m_partial_shot_type == "sparse_x")
    {
        // bins with indices 0 to 7 and 24 to 31 correspond to the rear part
        for(int i = 0; i <= 7; i++)
            result[i] = true;
        for(int i = 24; i <= 31; i++)
            result[i] = true;
    }
    else if(m_partial_shot_type == "left" || m_partial_shot_type == "positive_y")
    {
        // bins with indices 16 to 31 correspond to the left part
        for(int i = 16; i <= 31; i++)
            result[i] = true;
    }
    else if(m_partial_shot_type == "right" || m_partial_shot_type == "negative_y")
    {
        // bins with indices 0 to 15 correspond to the right part
        for(int i = 0; i <= 15; i++)
            result[i] = true;
    }
    else if(m_partial_shot_type == "top" || m_partial_shot_type == "dense_z")
    {
        // all odd bins correspond to the top part
        for(int i = 0; i <= 31; i++)
            if(i % 2 == 1) result[i] = true;
    }
    else if(m_partial_shot_type == "bottom" || m_partial_shot_type == "sparse_z")
    {
        // all even bins correspond to the bottom part
        for(int i = 0; i <= 31; i++)
            if(i % 2 == 0) result[i] = true;
    }
    else if(m_partial_shot_type == "dense_x_or_z")
    {
        // this is actually "top" and "front" combined

        // bins with indices 8 to 23 correspond to the frontal part
        for(int i = 8; i <= 23; i++)
            result[i] = true;

        // all odd bins correspond to top part
        for(int i = 0; i <= 31; i++)
            if(i % 2 == 1) result[i] = true;
    }
    else if(m_partial_shot_type == "dense_x_and_z")
    {
        // this is actually "top" logical AND "front"

        // bins with indices 8 to 23 correspond to the frontal part
        // and odd bins correspond to upper part
        for(int i = 8; i <= 23; i++)
            if(i % 2 == 1) result[i] = true;
    }
    else if(m_partial_shot_type == "front_turn_left")
    {
        // take some from front and some from left
        for(int i = 12; i <= 27; i++)
            result[i] = true;
    }
    else if(m_partial_shot_type == "front_turn_right")
    {
        // take some from front and some from left
        for(int i = 4; i <= 19; i++)
            result[i] = true;
    }
    else
    {
        LOG_WARN("Unknown partial shot type: " << m_partial_shot_type << "! Using complete descriptor!");
        return std::vector<bool>(32,true);
    }

    return result;
}
}


