/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2020, Viktor Seib
 * All rights reserved.
 *
 */

#include "feature_ranking.h"
#include "ranking_factory.h"
#include "../utils/utils.h"
#include "../utils/distance.h"
#include "../utils/debug_utils.h"

#include <fstream>

namespace ism3d
{
FeatureRanking::FeatureRanking()
    : m_numThreads(0)
{
    addParameter(m_k_search, "KSearch", 10);
    addParameter(m_dist_thresh, "DistanceThreshold", 0.1f);
    addParameter(m_factor, "Factor", 0.75f);
    addParameter(m_extract_list, "ExtractFromList", std::string("invalid"));
    addParameter(m_extract_offset, "ExtractOffset", 0.0f);
}

FeatureRanking::~FeatureRanking()
{
}

// TODO VS: in all subclasses: whenever flann::<distance-type> is used, make it automatically the same as in config (either L2 or Chi-Squared)

std::tuple<std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>>, pcl::PointCloud<ISMFeature>::Ptr>
FeatureRanking::operator()(std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> &features, int num_kd_trees, bool flann_exact_match)
{
    m_num_kd_trees = num_kd_trees;
    m_flann_exact_match = flann_exact_match;
    int num_input_features = countFeatures(features);

    // variables to hold the result
    std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> features_reduced;
    pcl::PointCloud<ISMFeature>::Ptr all_features_reduced(new pcl::PointCloud<ISMFeature>());

    // init values
    features_reduced.clear();
    all_features_reduced->clear();

    // compute scores
    std::map<unsigned, std::vector<float>> scores = iComputeScores(features);
    std::map<unsigned, std::vector<std::pair<int, float>>> index_score_map;

    if(getType() != "Uniform")
    {
        // rank with scores
        index_score_map = rankFeaturesWithScores(scores);
        scores = extractSubsetFromRankedList(index_score_map, scores);

        // debug output: enable / disable inside method
        DebugUtils::writeOutForDebug(index_score_map, getType());
    }

    std::map<unsigned, std::vector<std::vector<float>>> feature_scores = unflatten_list(scores, features);

    // NOTE: this is only for debug to write out the indices selected by a feature ranking algorithm; check absolute path below
    // NOTE: to write out additional information about selected features by a feature ranking algorithm check same flag farther down in this file
    // NOTE: to write out indices randomly selected during recognition, check the same debug flag in codebook.cpp
    // NOTE: to write out the actual feature vectors see below the next for loop
    bool debug_flag_write_out = false;
    int temp_idx = 0;
    std::ofstream ofs;
    if(debug_flag_write_out) ofs.open("/home/vseib/Desktop/cwids/selected_idxs.txt", std::ofstream::out);

    // remove features with score zero
    for(auto it : feature_scores)
    {
        unsigned class_id = it.first;
        std::vector<std::vector<float> > scores_doublelist = it.second;
        features_reduced.insert({class_id, std::vector<pcl::PointCloud<ISMFeature>::Ptr>()});
        for(int i = 0; i < scores_doublelist.size(); i++)
        {
            pcl::PointCloud<ISMFeature>::Ptr cloud(new pcl::PointCloud<ISMFeature>());
            features_reduced.at(class_id).push_back(cloud);
        }

        for(int first_idx = 0; first_idx < scores_doublelist.size(); first_idx++)
        {
            for(int second_idx = 0; second_idx < scores_doublelist.at(first_idx).size(); second_idx++)
            {
                float score = scores_doublelist.at(first_idx).at(second_idx);
                if(score != 0)
                {
                    ISMFeature current_feature = features.at(class_id).at(first_idx)->at(second_idx);
                    features_reduced.at(class_id).at(first_idx)->push_back(current_feature);
                    all_features_reduced->push_back(current_feature);

                    if(debug_flag_write_out) ofs << temp_idx << std::endl;
                }
                if(debug_flag_write_out) temp_idx++;
            }
        }
    }
    if(debug_flag_write_out) ofs.close();

    // NOTE: for debug only - write features to file before and after reduction
    //DebugUtils::writeToFile(features, "features_all");
    //DebugUtils::writeToFile(features_reduced, "features_reduced");

    int num_output_features = countFeatures(features_reduced);
    LOG_INFO("input features: " << num_input_features << ", output features: " << num_output_features << ", output ratio: "
              << (((float)num_output_features)/(num_input_features)));

    return std::make_tuple(features_reduced, all_features_reduced);
}


std::map<unsigned, std::vector<std::pair<int, float>>> FeatureRanking::rankFeaturesWithScores(std::map<unsigned, std::vector<float>> &temp_scores)
{
    // for each class: list of pairs (first: index in class list, second: num of activations)
    std::map<unsigned, std::vector<std::pair<int, float>>> index_score_map;
    for(int i = 0; i < temp_scores.size(); i++)
    {
        index_score_map.insert({i, std::vector<std::pair<int, float>>()});
        for(int j = 0; j < temp_scores.at(i).size(); j++)
        {
            float score = temp_scores.at(i).at(j);
            index_score_map.at(i).push_back({j,score});
        }
    }

    if(m_extract_list != "invalid")
    {
        LOG_WARN("Config parameter \"ExtractFromList\" is deprecated. Use \"ExtractOffset\" instead!");
        LOG_WARN("For now, the given value for \"ExtractFromList\" will be automatically converted.");

        // NOTE: m_extract_list specifies where the ranked features should be extracted from the scored list
        // now using m_extract_offset to use a fine-grained extraction
        if(m_extract_list == "front")
            m_extract_offset = 0.0f;
        if(m_extract_list == "center" || m_extract_list == "middle")
            m_extract_offset = 0.5 * (1-m_factor);
        if(m_extract_list == "back")
            m_extract_offset = 1.0 - m_factor;
    }

    // sort count index map
    for(auto &i : index_score_map)
    {
        //if(m_extract_list == "front")
            std::sort(i.second.begin(), i.second.end(), [](const std::pair<int,float> &a, const std::pair<int,float> &b){return a.second < b.second;});
        //else
        //    std::sort(i.second.begin(), i.second.end(), [](const std::pair<int,float> &a, const std::pair<int,float> &b){return a.second > b.second;});
    }
    return index_score_map;
}


std::map<unsigned, std::vector<float>> FeatureRanking::extractSubsetFromRankedList(
                                           const std::map<unsigned, std::vector<std::pair<int, float>>> &index_score_map,
                                           const std::map<unsigned, std::vector<float>> &scores)
{
    // init scores map
    std::map<unsigned, std::vector<float>> scores_clean; // this will hold a subset of the ranked scores
    for(int i = 0; i < scores.size(); i++)
    {
        scores_clean.insert({i, std::vector<float>(scores.at(i).size(), 0)});
    }

    // assign scores subset
    for(int class_idx = 0; class_idx < index_score_map.size(); class_idx++)
    {
        float min_index; // inclusive
        float max_index; // exclusive
        min_index = index_score_map.at(class_idx).size() * m_extract_offset;
        max_index = index_score_map.at(class_idx).size() * (m_factor + m_extract_offset);
        // sanitize values
        if(min_index < 0)
            min_index = 0;
        if(max_index > index_score_map.at(class_idx).size())
            max_index = index_score_map.at(class_idx).size();

        for(int j = 0; j < index_score_map.at(class_idx).size(); j++)
        {
            std::pair<int, float> temp = index_score_map.at(class_idx).at(j);
            if(j >= min_index && j < max_index)
            {
                scores_clean.at(class_idx).at(temp.first) = 1;
            }
        }
    }
    return scores_clean;
}


std::map<unsigned, std::vector<std::vector<float>>> FeatureRanking::unflatten_list(std::map<unsigned, std::vector<float> > &scores_flat_list,
                                                                                const FeatureMapT &features)
{
    // unflatten scores list
    std::map<unsigned, std::vector<std::vector<float> > > scores;
    for(int class_id = 0; class_id < features.size(); class_id++)
    {
        int object_index = 0;
        scores.insert({class_id, std::vector<std::vector<float>>(features.at(class_id).size())});
        for(int i = 0; i < scores_flat_list.at(class_id).size(); i++)
        {
            if(scores.at(class_id).at(object_index).size() == features.at(class_id).at(object_index)->size())
            {
                do
                {
                    object_index++;
                }
                while(features.at(class_id).at(object_index)->size() == 0);
            }

            float next_score = scores_flat_list.at(class_id).at(i);
            scores.at(class_id).at(object_index).push_back(next_score);
        }
    }
    return scores;
}


void FeatureRanking::setNumThreads(int numThreads)
{
    m_numThreads = numThreads;
}

int FeatureRanking::getNumThreads() const
{
    return m_numThreads;
}

int FeatureRanking::countFeatures(const FeatureMapT &features)
{
    int num = 0;
    for(auto it : features)
    {
        num += countFeatures(it.second);
    }
    return num;
}

int FeatureRanking::countFeatures(const std::vector<pcl::PointCloud<ISMFeature>::Ptr> &features)
{
    int num = 0;
    for(auto it : features)
    {
        num += it->size();
    }
    return num;
}

pcl::PointCloud<ISMFeature>::Ptr FeatureRanking::createCloudWithClassIds(const FeatureRanking::FeatureMapT &features,
                                                                    bool filter, unsigned filter_class, bool inverse)
{
    pcl::PointCloud<ISMFeature>::Ptr result(new pcl::PointCloud<ISMFeature>());
    for(auto it : features) // loop over pairs in map
    {
        for(auto it2 : it.second) // loop over vector elements
        {
            for(ISMFeature f : it2->points) // loop over points
            {
                f.classId = it.first;
                if(filter)
                {
                    if(inverse && filter_class != it.first)
                    {
                        result->push_back(f);
                    }
                    else if(!inverse && filter_class == it.first)
                    {
                        result->push_back(f);
                    }
                }
                else
                    result->push_back(f);
            }
        }
    }
    result->height = 1;
    result->width = result->points.size();
    result->is_dense = false;
    return result;
}

std::vector<int> FeatureRanking::findSimilarFeatures(pcl::KdTreeFLANN<ISMFeature> &kdtree, ISMFeature &feature)
{
    std::vector<int> pointIdNNSearch(m_k_search);
    std::vector<float> pointNNSquaredDistance(m_k_search);
    if(!kdtree.nearestKSearch(feature, m_k_search, pointIdNNSearch, pointNNSquaredDistance) > 0)
        LOG_WARN("Error during nearest neighbor search.");

    // use distance threshold to define equal features
    std::vector<int> result;
    for(int i = 0; i < pointIdNNSearch.size(); i++)
    {
        if(pointNNSquaredDistance.at(i) < m_dist_thresh)
            result.push_back(pointIdNNSearch.at(i));
    }
    return result;
}

flann::Matrix<float> FeatureRanking::createFlannDataset(pcl::PointCloud<ISMFeature>::Ptr &class_features)
{
    // create a dataset with all features for matching / activation
    int descriptor_size = class_features->at(0).descriptor.size();
    flann::Matrix<float> dataset(new float[class_features->size() * descriptor_size],
            class_features->size(), descriptor_size);

    // build dataset
    for(int i = 0; i < class_features->size(); i++)
    {
        ISMFeature ism_feat = class_features->at(i);
        std::vector<float> descriptor = ism_feat.descriptor;
        for(int j = 0; j < (int)descriptor.size(); j++)
        {
            dataset[i][j] = descriptor.at(j);
        }
    }

    return dataset;
}


std::vector<float> FeatureRanking::findNeighborsDistances(flann::Index<flann::ChiSquareDistance<float> > &index, flann::Matrix<float> &query)
{
    std::vector<std::vector<int> > indices;
    std::vector<std::vector<float> > distances;
    flann::SearchParams params = m_flann_exact_match ? flann::SearchParams(-1) : flann::SearchParams(128);
    index.knnSearch(query, indices, distances, m_k_search, params);
    return distances.at(0);
}


std::vector<int> FeatureRanking::findSimilarFeaturesFlann(flann::Index<flann::ChiSquareDistance<float> > &index, flann::Matrix<float> &query)
{
    std::vector<std::vector<int> > indices;
    std::vector<std::vector<float> > distances;
    flann::SearchParams params = m_flann_exact_match ? flann::SearchParams(-1) : flann::SearchParams(128);
    index.knnSearch(query, indices, distances, m_k_search, params);
    // use distance threshold to define equal features
    std::vector<int> result;
    if(indices.size() > 0)
    {
        for(int i = 0; i < indices.at(0).size(); i++)
        {
            if(distances.at(0).at(i) < m_dist_thresh)
                result.push_back(indices.at(0).at(i));
        }
    }
    return result;
}

}
