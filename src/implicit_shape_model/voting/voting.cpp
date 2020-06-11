/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "voting.h"
#include "voting_factory.h"
#include "maxima_handler.h"
#include "../codebook/codeword_distribution.h"

#include <fstream>
#include <sstream>

#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/common/centroid.h>

namespace ism3d
{

Voting::Voting()
{
    addParameter(m_minThreshold, "MinThreshold", 0.0f);
    addParameter(m_minVotesThreshold, "MinVotesThreshold", 1);
    addParameter(m_bestK, "BestK", -1);
    addParameter(m_averageRotation, "AverageRotation", false);
    addParameter(m_radiusType, "BinOrBandwidthType", std::string("Config"));
    addParameter(m_radiusFactor, "BinOrBandwidthFactor", 1.0f);
    addParameter(m_max_filter_type, "MaxFilterType", std::string("None"));
    addParameter(m_max_type_param, "SingleObjectMaxType", std::string("Default"));
    addParameter(m_single_object_mode, "SingleObjectMode", false);

    addParameter(m_use_global_features, "UseGlobalFeatures", false);
    addParameter(m_global_feature_method, "GlobalFeaturesStrategy", std::string("KNN"));
    addParameter(m_global_feature_influence_type, "GlobalFeatureInfluenceType", 3);
    addParameter(m_k_global_features, "GlobalFeaturesK", 1);
    addParameter(m_global_param_min_svm_score, "GlobalParamMinSvmScore", 0.70f);
    addParameter(m_global_param_rate_limit, "GlobalParamRateLimit", 0.60f);
    addParameter(m_global_param_weight_factor, "GlobalParamWeightFactor", 1.5f);
}

Voting::~Voting()
{
    m_votes.clear();
}

void Voting::vote(Eigen::Vector3f position, float weight, unsigned classId, unsigned instanceId,
                  const Eigen::Vector3f& keypoint, const Utils::BoundingBox& boundingBox, int codewordId)
{
    // add the vote
    Vote newVote;
    newVote.position = position; // position of object center the vote votes for
    newVote.weight = weight;
    newVote.classId = classId;
    newVote.instanceId = instanceId;
    newVote.keypoint = keypoint;
    newVote.boundingBox = boundingBox;
    newVote.codewordId = codewordId;

#pragma omp critical
    {
        m_votes[classId].push_back(newVote);
    }
}

std::vector<VotingMaximum> Voting::findMaxima(pcl::PointCloud<PointT>::ConstPtr &points,
                                              pcl::PointCloud<pcl::Normal>::ConstPtr &normals)
{
    // TODO VS try to move these into maxima_handler.cpp
    // set max type based on parameter value
    if(m_max_type_param == "None" || m_max_type_param == "Default")
        m_max_type = SingleObjectMaxType::DEFAULT;
    else if(m_max_type_param == "BandwidthVotes")
        m_max_type = SingleObjectMaxType::BANDWIDTH;
    else if(m_max_type_param == "VotingSpaceVotes")
        m_max_type = SingleObjectMaxType::COMPLETE_VOTING_SPACE;
    else if(m_max_type_param == "ModelRadiusVotes")
        m_max_type = SingleObjectMaxType::MODEL_RADIUS;

    // forward values to helper function
    MaximaHandler::setRadiusType(m_radiusType);
    MaximaHandler::setRadiusFactor(m_radiusFactor);

    if (m_votes.size() == 0)
        return std::vector<VotingMaximum>();

    std::vector<VotingMaximum> maxima;

    // find votes for each class individually
    // iterate over map that assigns each class id with a list of votes
    for (std::map<unsigned, std::vector<Voting::Vote> >::const_iterator it = m_votes.begin();
         it != m_votes.end(); it++)
    {
        unsigned classId = it->first;
        const std::vector<Voting::Vote>& votes = it->second; // all votes for this class

        std::vector<Eigen::Vector3f> clusters;  // positions of maxima
        std::vector<double> maximaValues;       // weights of maxima
        std::vector<std::vector<unsigned>> instanceIds; // list of instance ids for each maximum
        std::vector<std::vector<int>> voteIndices; // list of indices of all votes for each maximum
        std::vector<std::vector<float>> reweightedVotes; // reweighted votes, a list for each maximum

        // process the algorithm to find maxima on the votes of the current class
        iFindMaxima(points, votes, clusters, maximaValues, instanceIds, voteIndices, reweightedVotes, classId);

        LOG_ASSERT(clusters.size() == maximaValues.size());
        LOG_ASSERT(clusters.size() == voteIndices.size());

        // TODO VS: look here for bounding box filtering (i.e. remove outliers) (to determine an orientation during detection)
        // also use m_id_bb_dimensions_map and m_id_bb_variances_map

        // iterate through all found maxima for current class ID
        #pragma omp parallel for
        for (int i = 0; i < (int)clusters.size(); i++)
        {
            if (maximaValues[i] < m_minThreshold || voteIndices.at(i).size() < m_minVotesThreshold)
                continue;

            const std::vector<int>& clusterVotes = voteIndices[i];
            const std::vector<float>& reweightedClusterVotes = reweightedVotes[i];
            if (clusterVotes.size() == 0)
                continue;

            VotingMaximum maximum;
            maximum.classId = classId;
            maximum.instanceIds = instanceIds[i];
            maximum.position = clusters[i];
            maximum.weight = maximaValues[i];
            maximum.voteIndices = voteIndices[i];

            std::vector<boost::math::quaternion<float>> quats;
            std::vector<float> weights;

            // compute weighted maximum values
            float maxWeight = 0;
            maximum.boundingBox.size = Eigen::Vector3f(0, 0, 0);
            for (int j = 0; j < (int)clusterVotes.size(); j++)
            {
                int id = clusterVotes[j];
                const Voting::Vote& vote = votes[id];
                float newWeight = reweightedClusterVotes[j];

                boost::math::quaternion<float> rotQuat = vote.boundingBox.rotQuat;
                quats.push_back(rotQuat);
                weights.push_back(newWeight);

                maximum.boundingBox.size += newWeight * vote.boundingBox.size;
                maxWeight += newWeight;
            }

            // weights should sum up to one
            for (int j = 0; j < (int)weights.size(); j++)
                weights[j] /= maxWeight;

            maximum.boundingBox.position = maximum.position;
            maximum.boundingBox.size /= maxWeight;

            // compute interpolation between quaternions
            if (m_averageRotation)
            {
                boost::math::quaternion<float> rotQuat;
                Utils::quatWeightedAverage(quats, weights, rotQuat);
                maximum.boundingBox.rotQuat = rotQuat;
            }

            // in non-single object mode: extract points around each maxima region and compute global features
            if(m_use_global_features && !m_single_object_mode)
            {
                pcl::PointCloud<PointT>::Ptr segmented_points(new pcl::PointCloud<PointT>());
                pcl::PointCloud<pcl::Normal>::Ptr segmented_normals(new pcl::PointCloud<pcl::Normal>());

                m_global_classifier->segmentROI(points, normals, maximum, segmented_points, segmented_normals);
                m_global_classifier->classify(segmented_points, segmented_normals, maximum);
            }

            #pragma omp critical
            {
                maxima.push_back(maximum);
            }
        }
    }

    // in single object mode: compute global features on the whole cloud once
    if(m_use_global_features && m_single_object_mode)
    {
        VotingMaximum global_result;
        // TODO VS: global classification does not use instance labels so far
        m_global_classifier->classify(points, normals, global_result);

        // add global result to all maxima if in single object mode
        for(VotingMaximum &max : maxima)
            max.globalHypothesis = global_result.globalHypothesis;

        // if no maxima found in single object mode, use global hypothesis and fill in values
        if(maxima.size() == 0)
        {
            global_result.classId = global_result.globalHypothesis.first;
            global_result.weight = global_result.globalHypothesis.second;
            global_result.instanceIds = global_result.instanceIds;
            Eigen::Vector4d centroid;
            pcl::compute3DCentroid(*points, centroid);
            global_result.position = Eigen::Vector3f(centroid.x(), centroid.y(), centroid.z());
            global_result.boundingBox = Utils::computeMVBB<PointT>(points);
            maxima.push_back(global_result);
        }
    }

    // filter maxima if not in single object mode
    std::vector<VotingMaximum> filtered_maxima = maxima; // init for the case that no filtering type is selected
    if(!m_single_object_mode)
    {
        // TODO VS add instances to maxima merging
        // TODO VS add global result to maxima merging
        filtered_maxima = MaximaHandler::filterMaxima(m_max_filter_type, maxima);
    }
    maxima = filtered_maxima;

    // sort maxima
    std::sort(maxima.begin(), maxima.end(), Voting::sortMaxima);
    // apply normalization: turn weights to probabilities
    normalizeWeights(maxima);

    // add global features to result classification
    if(m_use_global_features) // here we have a sorted list of local maxima, all maxima have a global feature result
    {
        // NOTE: types 1, 2 and 3 are for single object mode only // TODO VS add warning
        if(m_global_feature_influence_type == 1 || m_global_feature_influence_type == 2)
        {
            // type 1: blind belief in good scores
            // type 2: belief in good scores if global class is among the top classes
            if(maxima.at(0).globalHypothesis.second > m_global_param_min_svm_score)
            {
                if(m_global_feature_influence_type == 1)
                    maxima.at(0).classId = maxima.at(0).globalHypothesis.first;
                else // TODO VS X: else branch is same code as type 3 -- refactor
                {
                    float top_weight = maxima.at(0).weight;
                    int global_class = maxima.at(0).globalHypothesis.first;

                    // check if global class is among the top classes
                    for(int i = 0; i < maxima.size(); i++)
                    {
                        float cur_weight = maxima.at(i).weight;
                        int cur_class = maxima.at(i).classId;

                        if(cur_weight >= top_weight * m_global_param_rate_limit && cur_class == global_class)
                        {
                            maxima.at(0).classId = maxima.at(0).globalHypothesis.first;
                            break;
                        }
                        else if(cur_weight < top_weight * m_global_param_rate_limit)
                        {
                            break;
                        }
                    }
                }
            }
        }
        else if(m_global_feature_influence_type == 3)
        {
            // type 3: take global class if it is among the top classes
            float top_weight = maxima.at(0).weight;
            int global_class = maxima.at(0).globalHypothesis.first;

            // check if global class is among the top classes
            for(int i = 0; i < maxima.size(); i++)
            {
                float cur_weight = maxima.at(i).weight;
                int cur_class = maxima.at(i).classId;

                if(cur_weight >= top_weight * m_global_param_rate_limit && cur_class == global_class)
                {
                    maxima.at(0).classId = maxima.at(0).globalHypothesis.first;
                    break;
                }
                else if(cur_weight < top_weight * m_global_param_rate_limit)
                {
                    break;
                }
            }
        }
        // TODO VS: for NON single object mode include maximum.currentClassHypothesis
        else if(m_global_feature_influence_type == 4)
        {
            // type 4: upweight consistent results by fixed factor
            for(VotingMaximum &max : maxima)
            {
                if(max.classId == max.globalHypothesis.first)
                    max.weight *= m_global_param_weight_factor;
            }
        }
        else if(m_global_feature_influence_type == 5)
        {
            // type 5: upweight consistent results depending on weight
            for(VotingMaximum &max : maxima)
            {
                if(max.classId == max.globalHypothesis.first)
                    max.weight *= 1 + max.globalHypothesis.second;
            }
        }
        else if(m_global_feature_influence_type == 6)
        {
            // type 6: apply intermediate T-conorm: S(a,b) = a+b-ab
            for(VotingMaximum &max : maxima)
            {
                if(max.classId == max.globalHypothesis.first)
                {
                    float w1 = max.weight;
                    float w2 = max.globalHypothesis.second;
                    max.weight = w1+w2 - w1*w2;
                }
            }
        }

        // sort maxima and normalize again - global features might have changed weights
        std::sort(maxima.begin(), maxima.end(), Voting::sortMaxima);
        normalizeWeights(maxima);
    }

    // only keep the best k maxima, if specified
    if (m_bestK > 0 && maxima.size() >= m_bestK)
        maxima.erase(maxima.begin() + m_bestK, maxima.end());

    for (int i = 0; i < (int)maxima.size(); i++)
    {
        const VotingMaximum& max = maxima[i];
        std::ostringstream ostr;
        for(auto elem : max.instanceIds)
        {
            ostr << elem << " ";
        }
        ostr << "\b"; // move cursor one position back (to overwrite last space)
        LOG_INFO("maximum " << i << ", class: " << max.classId <<
                 ", instances: " << ostr.str() <<
                 ", weight: " << max.weight <<
                 ", glob: (" << max.globalHypothesis.first << ", " << max.globalHypothesis.second << ")" <<
                 ", this: (" << max.currentClassHypothesis.first << ", " << max.currentClassHypothesis.second << ")" <<
                 ", num votes: " << max.voteIndices.size());
    }
    return maxima;
}


bool Voting::sortMaxima(const VotingMaximum& maxA, const VotingMaximum& maxB)
{
    return maxA.weight > maxB.weight;
}

const std::map<unsigned, std::vector<Voting::Vote> >& Voting::getVotes() const
{
    return m_votes;
}

void Voting::clear()
{
    m_votes.clear();
}

void Voting::determineAverageBoundingBoxDimensions(const std::map<unsigned, std::vector<Utils::BoundingBox> > &boundingBoxes)
{
    m_dimensions_map.clear();
    m_variance_map.clear();

    for(auto it : boundingBoxes)
    {
        unsigned classId = it.first;
        float max_accu = 0;
        float max_accuSqr = 0;
        float med_accu = 0;
        float med_accuSqr = 0;

        // check each bounding box of this class id
        for(auto box : it.second)
        {
            float max = box.size.maxCoeff();
            float min = box.size.minCoeff();
            // find the other value
            float med = box.size[0];
            for(int i = 1; i < 3; i++)
            {
                if(med == max || med == min)
                {
                    med = box.size[i];
                }
            }

            // use "radius" of bb dimensions, i.e. half of the sizes
            max_accu += max/2;
            med_accu += med/2;
            max_accuSqr += ((max/2)*(max/2));
            med_accuSqr += ((med/2)*(med/2));
        }

        // compute average
        max_accu /= it.second.size();
        med_accu /= it.second.size();
        max_accuSqr /= it.second.size();
        med_accuSqr /= it.second.size();

        // compute variance
        float max_var = max_accuSqr - (max_accu*max_accu);
        float med_var = med_accuSqr - (med_accu*med_accu);
        m_dimensions_map.insert({classId, {max_accu, med_accu}});
        m_variance_map.insert({classId, {max_var, med_var}});
    }
}

void Voting::normalizeWeights(std::vector<VotingMaximum> &maxima)
{
    float sum = 0;
    for(const VotingMaximum &max : maxima)
    {
        sum += max.weight;
    }

    for(VotingMaximum &max : maxima)
    {
        max.weight /= sum;
    }
}

void Voting::forwardGlobalFeatures(std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &globalFeatures)
{
    m_global_features = globalFeatures;
}

void Voting::iSaveData(boost::archive::binary_oarchive &oa) const
{
    // fill in bounding box information
    int bb_dims_size = m_dimensions_map.size();
    oa << bb_dims_size;
    for(auto it : m_dimensions_map)
    {
        int classId = it.first;
        float firstDim = it.second.first;
        float secondDim = it.second.second;
        oa << classId;
        oa << firstDim;
        oa << secondDim;
    }

    int bb_vars_size = m_variance_map.size();
    oa << bb_vars_size;
    for(auto it : m_variance_map)
    {
        int classId = it.first;
        float firstVar = it.second.first;
        float secondVar = it.second.second;
        oa << classId;
        oa << firstVar;
        oa << secondVar;
    }

    // fill in global features
    int glob_feat_size = m_global_features.size();
    oa << glob_feat_size;
    for(auto it : m_global_features)
    {
        int classId = it.first; // descriptor type is same for all features, only store it once
        oa << classId;

        int cloud_size = it.second.size();
        oa << cloud_size;
        for(auto feat_cloud : it.second) // iterate over each vector element (point cloud) of one class
        {
            int feat_size = feat_cloud->points.size();
            oa << feat_size;
            for(auto feat : feat_cloud->points) // iterate over each descriptor (point in the cloud)
            {
                // save reference frame
                for(unsigned i = 0; i < 9; i++)
                {
                    oa << feat.referenceFrame.rf[i];
                }
                // save descriptor
                oa << feat.descriptor;
                oa << feat.globalDescriptorRadius;
            }
        }
    }
}

bool Voting::iLoadData(boost::archive::binary_iarchive &ia)
{
    // read bounding box data
    m_dimensions_map.clear();
    m_variance_map.clear();

    int bb_dims_size;
    ia >> bb_dims_size;
    for(int i = 0; i < bb_dims_size; i++)
    {
        unsigned classId;
        float firstDim;
        float secondDim;
        ia >> classId;
        ia >> firstDim;
        ia >> secondDim;
        m_dimensions_map.insert({classId, {firstDim, secondDim}});
    }

    int bb_vars_size;
    ia >> bb_vars_size;
    for(int i = 0; i < bb_vars_size; i++)
    {
        unsigned classId;
        float firstVar;
        float secondVar;
        ia >> classId;
        ia >> firstVar;
        ia >> secondVar;
        m_variance_map.insert({classId, {firstVar, secondVar}});
    }

    MaximaHandler::setBoundingBoxMaps(m_dimensions_map, m_variance_map);

    // read global features
    if(m_use_global_features)
    {
        // load all global features from training into a single cloud
        pcl::PointCloud<ISMFeature>::Ptr global_features_cloud(new pcl::PointCloud<ISMFeature>());
        std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> global_features_map;
        int descriptor_length;

        int global_feat_size;
        ia >> global_feat_size;
        for(int i = 0; i < global_feat_size; i++)
        {
            unsigned classId;
            ia >> classId;

            std::vector<pcl::PointCloud<ISMFeature>::Ptr> cloud_vector;

            int cloud_size;
            ia >> cloud_size;
            for(int j = 0; j < cloud_size; j++)
            {
                pcl::PointCloud<ISMFeature>::Ptr temp_cloud(new pcl::PointCloud<ISMFeature>());

                int feat_size;
                ia >> feat_size;
                for(int k = 0; k < feat_size; k++)
                {
                    pcl::ReferenceFrame referenceFrame;
                    for(int i_ref = 0; i_ref < 9; i_ref++)
                    {
                        float ref;
                        ia >> ref;
                        referenceFrame.rf[i_ref] = ref;
                    }

                    std::vector<float> descriptor;
                    ia >> descriptor;
                    float radius;
                    ia >> radius;

                    // TODO VS load and store instance label in global features
                    ISMFeature ism_feature;
                    ism_feature.referenceFrame = referenceFrame;
                    ism_feature.descriptor = descriptor;
                    ism_feature.globalDescriptorRadius =  radius;
                    ism_feature.classId = classId;
                    temp_cloud->push_back(ism_feature);
                    global_features_cloud->push_back(ism_feature);
                    descriptor_length = ism_feature.descriptor.size(); // are all the same just overwrite
                }
                temp_cloud->height = 1;
                temp_cloud->width = temp_cloud->size();
                temp_cloud->is_dense = false;
                cloud_vector.push_back(temp_cloud);
            }
            global_features_map.insert({classId, cloud_vector});
        }

        // create flann index
        // TODO VS simplify constructor call, e.g. first argument is descriptor_length
        std::shared_ptr<FlannHelper> fh = std::make_shared<FlannHelper>(
                    global_features_cloud->at(0).descriptor.size(), global_features_cloud->size());
        fh->createDataset(global_features_cloud);
        // NOTE: index will be build when the first object is recognized - otherwise parameters are not initialized from config, but with default values
        //fh->buildIndex(m_distanceType, m_num_kd_trees);

        m_global_classifier = std::make_shared<GlobalClassifier>(
                    m_global_feature_descriptor,
                    m_global_feature_method,
                    m_k_global_features);
        m_global_classifier->setFlannHelper(fh);
        m_global_classifier->setLoadedFeatures(global_features_cloud);
        m_global_classifier->computeAverageRadii(global_features_map);
        m_global_classifier->loadSVMModels(m_svm_path);

        if(m_single_object_mode)
        {
            m_global_classifier->enableSingleObjectMode();
        }
    }
    return true;
}

}
