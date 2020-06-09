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
#include "../codebook/codeword_distribution.h"

#include <fstream>
#include <pcl/common/centroid.h>

namespace ism3d
{

std::string exec(const char *cmd); // defined at the end of file

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

    addParameter(m_use_global_features, "UseGlobalFeatures", false);
    addParameter(m_global_feature_method, "GlobalFeaturesStrategy", std::string("KNN"));
    addParameter(m_global_feature_influence_type, "GlobalFeatureInfluenceType", 3);
    addParameter(m_k_global_features, "GlobalFeaturesK", 1);
    addParameter(m_global_param_min_svm_score, "GlobalParamMinSvmScore", 0.70f);
    addParameter(m_global_param_rate_limit, "GlobalParamRateLimit", 0.60f);
    addParameter(m_global_param_weight_factor, "GlobalParamWeightFactor", 1.5f);

    m_index_created = false;
    m_svm_error = false;
    m_single_object_mode = false;
}

Voting::~Voting()
{
    m_votes.clear();

    // delete files that were unpacked for recognition
    if(m_svm_files.size() > 1)
    {
        for(std::string s : m_svm_files)
        {
            int ret = std::system(("rm " + s).c_str());
        }
    }
}

void Voting::vote(Eigen::Vector3f position, float weight, unsigned classId,
                  const Eigen::Vector3f& keypoint, const Utils::BoundingBox& boundingBox, int codewordId)
{
    // add the vote
    Vote newVote;
    newVote.position = position; // position of object center the vote votes for
    newVote.weight = weight;
    newVote.classId = classId;
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
    // set max type based on parameter value
    if(m_max_type_param == "None" || m_max_type_param == "Default")
        m_max_type = SingleObjectMaxType::DEFAULT;
    else if(m_max_type_param == "BandwidthVotes")
        m_max_type = SingleObjectMaxType::BANDWIDTH;
    else if(m_max_type_param == "VotingSpaceVotes")
        m_max_type = SingleObjectMaxType::COMPLETE_VOTING_SPACE;
    else if(m_max_type_param == "ModelRadiusVotes")
        m_max_type = SingleObjectMaxType::MODEL_RADIUS;

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
        std::vector<std::vector<int> > voteIndices; // list of indices of all votes for each maximum
        std::vector<std::vector<float> > reweightedVotes; // reweighted votes, a list for each maximum

        // process the algorithm to find maxima on the votes of the current class
        iFindMaxima(points, votes, clusters, maximaValues, voteIndices, reweightedVotes, classId);

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
            maximum.position = clusters[i];
            maximum.weight = maximaValues[i];
            maximum.voteIndices = voteIndices[i];

            std::vector<boost::math::quaternion<float> > quats;
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

            // in non-single object mode: extract points around maxima region and compute global features
            // do this for each maximum
            if(m_use_global_features && !m_single_object_mode)
            {
                verifyMaxHypothesisWithGlobalFeatures(points, normals, maximum);
            }

            #pragma omp critical
            {
                maxima.push_back(maximum);
            }
        }
    }

    // in single object mode: classify global features instead of maxima points
    // do this only once
    if(m_use_global_features && m_single_object_mode)
    {
        VotingMaximum global_result;
        classifyGlobalFeatures(m_global_features_single_object, global_result);

        // add global result to all maxima if in single object mode
        for(VotingMaximum &max : maxima)
            max.globalHypothesis = global_result.globalHypothesis;

        // if no maxima found in single object mode, use global hypothesis and fill in values
        if(maxima.size() == 0)
        {
            global_result.classId = global_result.globalHypothesis.first;
            global_result.weight = global_result.globalHypothesis.second;
            Eigen::Vector4d centroid;
            pcl::compute3DCentroid(*points, centroid);
            global_result.position = Eigen::Vector3f(centroid.x(), centroid.y(), centroid.z());
            global_result.boundingBox = Utils::computeMVBB<PointT>(points);
            maxima.push_back(global_result);
        }
    }

    // TODO VS: globale features bis hierhin geprüft
    // * derzeit gibt es zwei global results: globalHypothesis und bestThisClassId
    // ---> refactorn, damit es nur noch eins gibt


    // filter maxima
    std::vector<VotingMaximum> filtered_maxima = maxima; // init for the case that no filtering type is selected

    if(m_single_object_mode)
    {
        // TODO VS refactoring - moved this to voting_mean_shift.cpp
//        pcl::PointCloud<PointNormalT>::Ptr pointsWithNormals(new pcl::PointCloud<PointNormalT>());
//        pcl::concatenateFields(*points, *normals, *pointsWithNormals);

//        // vote based single maxima computation
//        if(m_single_object_max_type_str == "VotingSpaceVotes")
//            filtered_maxima  = computeSingleMaxPerClass(pointsWithNormals, SingleObjectMaxType::COMPLETE_VOTING_SPACE);
//        if(m_single_object_max_type_str == "BandwidthVotes")
//            filtered_maxima  = computeSingleMaxPerClass(pointsWithNormals, SingleObjectMaxType::BANDWIDTH);
//        if(m_single_object_max_type_str == "ModelRadiusVotes")
//            filtered_maxima  = computeSingleMaxPerClass(pointsWithNormals, SingleObjectMaxType::MODEL_RADIUS);

        // in single object mode global results are the same for all maxima
        if(maxima.size() > 0)
        {
            for(VotingMaximum &m : filtered_maxima)
            {
                m.globalHypothesis.first = maxima[0].globalHypothesis.first;
                m.globalHypothesis.second = maxima[0].globalHypothesis.second;
            }
        }
    }

    // TODO VS: fix global features if not in single object mode - add global result merging
    if(!m_single_object_mode)
    {
        if(m_max_filter_type == "Simple") // search in bandwith radius and keep only maximum with the highest weight
            filtered_maxima = filterMaxima(maxima);
        if(m_max_filter_type == "Merge")  // search in bandwith radius, merge maxima of same class and keep only maximum with the highest weight
            filtered_maxima = mergeAndFilterMaxima(maxima);
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
        LOG_INFO("maximum " << i << ", class: " << max.classId << ", weight: " << max.weight <<
                 ", glob: (" << max.globalHypothesis.first << ", " << max.globalHypothesis.second << ")" <<
                 ", this: (" << max.currentClassHypothesis.first << ", " << max.currentClassHypothesis.second << ")" <<
                 ", num votes: " << max.voteIndices.size());
    }
    return maxima;
}

std::vector<VotingMaximum> Voting::mergeAndFilterMaxima(const std::vector<VotingMaximum> &maxima) const
{
    return filterMaxima(maxima, true);
}

std::vector<VotingMaximum> Voting::filterMaxima(const std::vector<VotingMaximum> &maxima, bool merge) const
{
    // find and merge maxima of different classes that are closer than bandwith or bin_size
    std::vector<VotingMaximum> close_maxima;
    std::vector<VotingMaximum> filtered_maxima;
    std::vector<bool> dirty_list(maxima.size(), false);

    for(unsigned i = 0; i < maxima.size(); i++)
    {
        if(dirty_list.at(i))
            continue;

        // set adaptive search distance depending on config and class id
        float search_dist = getSearchDistForClass(maxima.at(i).classId);

        // check distance to other maxima
        for(unsigned j = i+1; j < maxima.size(); j++)
        {
            if(dirty_list.at(j))
                continue;

            float dist = (maxima.at(j).position - maxima.at(i).position).norm();
            float other_search_dist = getSearchDistForClass(maxima.at(j).classId);
            // only subsume maxima of classes with a smaller or equal search dist
            if(dist < search_dist && other_search_dist <= search_dist)
            {
                close_maxima.push_back(maxima.at(j));
                dirty_list.at(j) = true;
            }
        }

        // if some neighbors found, also add itself
        if(close_maxima.size() > 0)
        {
            close_maxima.push_back(maxima.at(i));
        }

        // merge close maxima of same classes before filtering
        if(merge && close_maxima.size() > 1) // > 1 because the maximum itself was added
        {
            std::vector<VotingMaximum> merged_maxima(maxima.size());
            std::map<unsigned, std::vector<VotingMaximum>> same_class_ids; // maps a class id to a list of close maxima with that id

            // create max list
            for(VotingMaximum m : close_maxima)
            {
                unsigned class_id = m.classId;
                if(same_class_ids.find(class_id) == same_class_ids.end())
                {
                    same_class_ids.insert({class_id, {m}});
                }
                else
                {
                    same_class_ids.at(class_id).push_back(m);
                }
            }
            // merge maxima of same classes
            for(auto it : same_class_ids)
            {
                VotingMaximum max = mergeMaxima(it.second);
                merged_maxima.push_back(max);
            }
            close_maxima = merged_maxima;
        }

        // if a close maximum was found, leave only the one with the highest weight
        if(close_maxima.size() > 1) // > 1 because the maximum itself was added
        {
            VotingMaximum best_max;
            for(VotingMaximum m : close_maxima)
            {
                if(m.weight > best_max.weight)
                {
                    best_max = m;
                }
            }
            filtered_maxima.push_back(best_max);
        }
        else
        {
            filtered_maxima.push_back(maxima.at(i));
        }
        close_maxima.clear();
    }
    return filtered_maxima;
}


VotingMaximum Voting::mergeMaxima(const std::vector<VotingMaximum> &max_list) const
{
    VotingMaximum result;
    for(VotingMaximum m : max_list)
    {
        // NOTE: position and bounding box must be handled before changing weight!
        result.position = result.position * result.weight + m.position * m.weight;
        result.position /= (result.weight + m.weight);
        result.boundingBox.position = result.position;
        result.boundingBox.size = result.boundingBox.size * result.weight + m.boundingBox.size * m.weight;
        result.boundingBox.size /= (result.weight + m.weight);
        boost::math::quaternion<float> rotQuat;
        Utils::quatWeightedAverage({result.boundingBox.rotQuat, m.boundingBox.rotQuat}, {result.weight, m.weight}, rotQuat);
        result.boundingBox.rotQuat = rotQuat;

        result.classId = m.classId;
        result.weight += m.weight;
        result.voteIndices.insert(result.voteIndices.end(), m.voteIndices.begin(), m.voteIndices.end());

        // TEMP FIX THIS! -- should be some kind of average
        result.globalHypothesis = m.globalHypothesis;
        result.currentClassHypothesis = m.currentClassHypothesis;
    }
    return result;
}


float Voting::getSearchDistForClass(const unsigned class_id) const
{
    // NOTE: m_radius is assigned in derived classes and is related either to the bandwidth or bin size
    float search_dist = 0;
    if(m_radiusType == "Config")
        search_dist = m_radius;
    if(m_radiusType == "FirstDim")
        search_dist = m_id_bb_dimensions_map.at(class_id).first * m_radiusFactor;
    if(m_radiusType == "SecondDim")
        search_dist = m_id_bb_dimensions_map.at(class_id).second * m_radiusFactor;
    return search_dist;
}


void Voting::verifyMaxHypothesisWithGlobalFeatures(const pcl::PointCloud<PointT>::ConstPtr &points,
                                                   const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
                                                   VotingMaximum &maximum)
{
    // used to extract a portion of the input cloud to estimage a global feature
    pcl::KdTreeFLANN<PointT> input_points_kdtree;
    if(m_use_global_features)
    {
        input_points_kdtree.setInputCloud(points);
    }

    // first segment region cloud from input with typical radius for this class id
    pcl::PointCloud<PointT>::Ptr segmented_points(new pcl::PointCloud<PointT>());
    pcl::PointCloud<pcl::Normal>::Ptr segmented_normals(new pcl::PointCloud<pcl::Normal>());
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    float radius = m_average_radii.at(maximum.classId);
    PointT query;
    query.x = maximum.position.x();
    query.y = maximum.position.y();
    query.z = maximum.position.z();

    if(input_points_kdtree.radiusSearch(query, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
    {
        // segment points
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud(points);
        pcl::PointIndices::Ptr indices (new pcl::PointIndices());
        indices->indices = pointIdxRadiusSearch;
        extract.setIndices(indices);
        extract.filter(*segmented_points);
        // segment normals
        pcl::ExtractIndices<pcl::Normal> extract_normals;
        extract_normals.setInputCloud(normals);
        extract_normals.setIndices(indices); // use same indices
        extract_normals.filter(*segmented_normals);
    }
    else
    {
        LOG_WARN("Error during nearest neighbor search.");
    }

    // compute global feature on segmented points
    pcl::PointCloud<PointT>::ConstPtr dummy_keypoints(new pcl::PointCloud<PointT>());
    pcl::search::Search<PointT>::Ptr search = pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>());
    pcl::PointCloud<ISMFeature>::ConstPtr global_features =
            (*m_globalFeatureDescriptor)(segmented_points, segmented_normals, segmented_points, segmented_normals, dummy_keypoints, search);

    classifyGlobalFeatures(global_features, maximum);
}

void Voting::classifyGlobalFeatures(const pcl::PointCloud<ISMFeature>::ConstPtr global_features, VotingMaximum &maximum)
{
    // if no SVM data available defaul to KNN
    if(m_svm_error) m_global_feature_method = "KNN";

    // process current global features according to some strategy
    if(!m_index_created)
    {
        LOG_INFO("creating flann index for global features");
        m_flann_helper->buildIndex(m_distanceType, 1);
        m_index_created = true;
    }

    if(m_global_feature_method == "KNN")
    {
        LOG_INFO("starting global classification with knn");
        std::map<unsigned, GlobalResultAccu> max_global_voting; // maps class id to struct with number of occurences and score
        int num_all_entries = 0;

        // find nearest neighbors to current global features in learned data
        // NOTE: some global features produce more than 1 descriptor per object, hence the loop
        for(ISMFeature query_feature : global_features->points)
        {
            // insert the query point
            flann::Matrix<float> query(new float[query_feature.descriptor.size()], 1, query_feature.descriptor.size());
            for(int i = 0; i < query_feature.descriptor.size(); i++)
            {
                query[0][i] = query_feature.descriptor.at(i);
            }

            // search
            std::vector<std::vector<int> > indices;
            std::vector<std::vector<float> > distances;
            if(m_flann_helper->getDistType() == "Euclidean")
            {
                m_flann_helper->getIndexL2()->knnSearch(query, indices, distances, m_k_global_features, flann::SearchParams(-1));
            }
            else if(m_flann_helper->getDistType() == "ChiSquared")
            {
                m_flann_helper->getIndexChi()->knnSearch(query, indices, distances, m_k_global_features, flann::SearchParams(-1));
            }
            else if(m_flann_helper->getDistType() == "Hellinger")
            {
                m_flann_helper->getIndexHel()->knnSearch(query, indices, distances, m_k_global_features, flann::SearchParams(-1));
            }
            else if(m_flann_helper->getDistType() == "HistIntersection")
            {
                m_flann_helper->getIndexHist()->knnSearch(query, indices, distances, m_k_global_features, flann::SearchParams(-1));
            }
            delete[] query.ptr();

            // classic KNN approach
            num_all_entries += indices[0].size(); // NOTE: is not necessaraly k, because only (k-x) might have been found
            // loop over results
            for(int i = 0; i < indices[0].size(); i++)
            {
                // insert result
                ISMFeature temp = m_all_global_features_cloud->at(indices[0].at(i));
                float dist_squared = distances[0].at(i);
                const float sigma = 0.1;
                const float denom = 2 * sigma * sigma;
                float score = std::exp(-dist_squared/denom);
                insertGlobalResult(max_global_voting, temp.classId, score);
            }
        }

        std::pair<unsigned, float> global_result = {maximum.classId, 0}; // pair of class id and score
        // determine score based on all votes
        unsigned max_occurences = 0;
        if(m_single_object_mode)
        {
            // find class with most occurences
            for(auto it : max_global_voting)
            {
                if(it.second.num_occurences > max_occurences)
                {
                    max_occurences = it.second.num_occurences;
                    global_result.first = it.first;
                }
            }
            // compute score for best class (NOTE: determining best class based on score did not work well)
            GlobalResultAccu gra = max_global_voting.at(global_result.first);
            global_result.second = gra.score_sum / gra.num_occurences;
        }
        else // determine score based on current class
        {
            if(max_global_voting.find(maximum.classId) != max_global_voting.end())
            {
                GlobalResultAccu gra = max_global_voting.at(maximum.classId);
                global_result.second = gra.num_occurences > 0 ? gra.score_sum / gra.num_occurences : 0;
            }
        }

        // assign global result
        maximum.globalHypothesis = global_result;
        maximum.currentClassHypothesis = global_result;
    }
    else if(m_global_feature_method == "SVM")
    {
        LOG_INFO("starting global classification with svm");
        CustomSVM::SVMResponse svm_response;
        std::vector<CustomSVM::SVMResponse> all_responses; // in case one object has multiple global features
        // NOTE: some global features produce more than 1 descriptor per object, hence the loop
        for(ISMFeature query_feature : global_features->points)
        {
            // convert to SVM data format
            std::vector<float> data_raw = query_feature.descriptor;
            float data[data_raw.size()];
            for(unsigned i = 0; i < data_raw.size(); i++)
            {
                data[i] = data_raw.at(i);
            }
            cv::Mat svm_input_data(1, data_raw.size(), CV_32FC1, data);

            CustomSVM::SVMResponse temp_response = m_svm.predictUnifyScore(svm_input_data, m_svm_files);
            all_responses.push_back(temp_response);
        }

        // check if several responses are available
        if(all_responses.size() > 1)
        {
            std::map<unsigned, GlobalResultAccu> global_result_per_class;
            for(const CustomSVM::SVMResponse &resp : all_responses)
            {
                insertGlobalResult(global_result_per_class, (unsigned) resp.label, resp.score);
            }

            int best_class = 0;
            int best_occurences = 0;
            for(const auto it : global_result_per_class)
            {
                // NOTE: what if there are 2 equal classes?
                if(it.second.num_occurences > best_occurences) // find highest number of occurences
                {
                    best_occurences = it.second.num_occurences;
                    best_class = it.first;
                }
            }

            // find best class in list of responses with "best" (highest) score
            float best_score = -999999;
            CustomSVM::SVMResponse best_response = all_responses.at(0); // init with first value
            for(int i = 0; i < all_responses.size(); i++)
            {
                if(all_responses.at(i).label == best_class)
                {
                    if(all_responses.at(i).score > best_score)
                    {
                        best_score = all_responses.at(i).score;
                        best_response = all_responses.at(i);
                    }
                }
            }
            svm_response = best_response;
        }
        else if(all_responses.size() == 1)
        {
            svm_response = all_responses.at(0);
        }

        // assign global result
        float cur_score = m_single_object_mode ? 0 : svm_response.all_scores.at(maximum.classId);
        maximum.globalHypothesis = {svm_response.label, svm_response.score};
        maximum.currentClassHypothesis = {maximum.classId, cur_score};
    }
}

void Voting::insertGlobalResult(std::map<unsigned, GlobalResultAccu> &max_global_voting,
                                unsigned found_class,
                                float score)
{
    if(max_global_voting.find(found_class) != max_global_voting.end())
    {
        // found
        GlobalResultAccu &prev = max_global_voting.at(found_class);
        prev.num_occurences++;
        prev.score_sum += score;
    }
    else
    {
        // not found
        max_global_voting.insert({found_class, {1,score} });
    }
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
    m_id_bb_dimensions_map.clear();
    m_id_bb_variance_map.clear();

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
        m_id_bb_dimensions_map.insert({classId, {max_accu, med_accu}});
        m_id_bb_variance_map.insert({classId, {max_var, med_var}});
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

void Voting::setGlobalFeatures(pcl::PointCloud<ISMFeature>::Ptr &globalFeatures)
{
    m_global_features_single_object = globalFeatures;
    m_single_object_mode = true;
}

void Voting::forwardGlobalFeatures(std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &globalFeatures)
{
    m_global_features = globalFeatures;
}

void Voting::iSaveData(boost::archive::binary_oarchive &oa) const
{
    // fill in bounding box information
    int bb_dims_size = m_id_bb_dimensions_map.size();
    oa << bb_dims_size;
    for(auto it : m_id_bb_dimensions_map)
    {
        int classId = it.first;
        float firstDim = it.second.first;
        float secondDim = it.second.second;
        oa << classId;
        oa << firstDim;
        oa << secondDim;
    }

    int bb_vars_size = m_id_bb_variance_map.size();
    oa << bb_vars_size;
    for(auto it : m_id_bb_variance_map)
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
    // fill the data
    m_id_bb_dimensions_map.clear();
    m_id_bb_variance_map.clear();

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
        m_id_bb_dimensions_map.insert({classId, {firstDim, secondDim}});
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
        m_id_bb_variance_map.insert({classId, {firstVar, secondVar}});
    }

    // read global features
    if(m_use_global_features)
    {
        // accumulate all global features into a single cloud
        m_all_global_features_cloud = pcl::PointCloud<ISMFeature>::Ptr(new pcl::PointCloud<ISMFeature>());

        m_global_features.clear();
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
                pcl::PointCloud<ISMFeature>::Ptr feature_cloud(new pcl::PointCloud<ISMFeature>());

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

                    ISMFeature ism_feature;
                    ism_feature.referenceFrame = referenceFrame;
                    ism_feature.descriptor = descriptor;
                    ism_feature.globalDescriptorRadius =  radius;
                    ism_feature.classId = classId;
                    feature_cloud->push_back(ism_feature);
                    m_all_global_features_cloud->push_back(ism_feature);
                    descriptor_length = ism_feature.descriptor.size(); // are all the same just overwrite
                }
                feature_cloud->height = 1;
                feature_cloud->width = feature_cloud->size();
                feature_cloud->is_dense = false;
                cloud_vector.push_back(feature_cloud);
            }
            m_global_features.insert({classId, cloud_vector});
        }

        // create flann index
        m_flann_helper = std::make_shared<FlannHelper>(m_all_global_features_cloud->at(0).descriptor.size(), m_all_global_features_cloud->size());
        m_flann_helper->createDataset(m_all_global_features_cloud);
        // NOTE: index will be build when the first object is recognized - otherwise parameters are not initialized from config, but with default values
        //m_flann_helper->buildIndex(m_distanceType, m_num_kd_trees);

        // compute average radii
        for(auto it = m_global_features.begin(); it != m_global_features.end(); it++)
        {
            float avg_radius = 0;
            int num_points = 0;
            unsigned classID = it->first;

            std::vector<pcl::PointCloud<ISMFeature>::Ptr> cloud_vector = it->second;
            for(auto cloud : cloud_vector)
            {
                for(ISMFeature ism_feature : cloud->points)
                {
                    avg_radius += ism_feature.globalDescriptorRadius;
                    num_points += 1;
                }
            }
            m_average_radii.insert({classID, avg_radius / num_points});
        }

        // NOTE: not needed anymore
        m_global_features.clear();

        // load SVM for global features
        // NOTE: if SVM works better than nearest neighbor, all of the above with global features can be removed ... except the radius
        if(m_svm_path != "")
        {
            // get path and check for errors
            boost::filesystem::path path(m_svm_path);
            boost::filesystem::path p_comp = boost::filesystem::complete(path);

            if(boost::filesystem::exists(p_comp) && boost::filesystem::is_regular_file(p_comp))
            {
                m_svm_files.clear();
                // check if multiple svm files are available (i.e. 1 vs all svm)
                if(m_svm_path.find("tar") != std::string::npos)
                {
                    // show the content of the tar file
                    std::string result = exec(("tar -tf " + p_comp.string()).c_str());
                    // split the string and add to list
                    std::stringstream sstr;
                    sstr.str(result);
                    std::string item;
                    while (std::getline(sstr, item, '\n'))
                    {
                        boost::filesystem::path paths(item);
                        boost::filesystem::path ppp = boost::filesystem::complete(paths);
                        m_svm_files.push_back(ppp.string());
                    }
                    // unzip tar file
                    int ret = std::system(("tar -xzf " + p_comp.string()).c_str());
                    sleep(2);
                }
                else
                {
                    // only one file: standard OpenCV SVM (i.e. pairwise 1 vs 1 svm)
                    m_svm_files.push_back(p_comp.string());
                }
            }
            else
            {
                LOG_ERROR("SVM file not valid or missing!");
                m_svm_error = true;
            }
        }
        else
        {
            LOG_ERROR("SVM path is NULL!");
            m_svm_error = true;
        }
    }
    return true;
}


// NOTE: from http://stackoverflow.com/questions/478898/how-to-execute-a-command-and-get-output-of-command-within-c-using-posix
std::string exec(const char* cmd)
{
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != NULL)
            result += buffer.data();
    }
    return result;
}

}
