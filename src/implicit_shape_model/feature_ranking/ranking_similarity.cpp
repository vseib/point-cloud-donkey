/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2021, Viktor Seib
 * All rights reserved.
 *
 */

#include "ranking_similarity.h"

namespace ism3d
{
RankingSimilarity::RankingSimilarity()
{
}

RankingSimilarity::~RankingSimilarity()
{
}

std::map<unsigned, std::vector<float> > RankingSimilarity::iComputeScores(
        const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &features)
{
    LOG_INFO("starting similarity ranking");

    std::map<unsigned, std::vector<float>> temp_scores;
    // determine class index offsets
    for(unsigned i = 0; i < features.size(); i++)
    {
        // init score list
        pcl::PointCloud<ISMFeature>::Ptr current_class_features = createCloudWithClassIds(features, true, i, false);
        temp_scores.insert({i, std::vector<float>(current_class_features->size(), 0)});
    }

    // loop over all classes
    for(unsigned class_id = 0; class_id < features.size(); class_id++)
    {
        // insert list for the current class
        pcl::PointCloud<ISMFeature>::Ptr current_class_features = createCloudWithClassIds(features, true, class_id, false);
        pcl::PointCloud<ISMFeature>::Ptr other_classes_features = createCloudWithClassIds(features, true, class_id, true);

        // create flann indices for current class and the rest
        flann::Matrix<float> dataset_current = createFlannDataset(current_class_features);
        flann::Index<flann::L2<float>> index_current(dataset_current, flann::KDTreeIndexParams(m_num_kd_trees));
        index_current.buildIndex();
        flann::Matrix<float> dataset_other = createFlannDataset(other_classes_features);
        flann::Index<flann::L2<float>> index_other(dataset_other, flann::KDTreeIndexParams(m_num_kd_trees));
        index_other.buildIndex();

        // use each feature from current class as query
        for(ISMFeature query_feature : current_class_features->points)
        {
            // insert the query point
            flann::Matrix<float> query(new float[query_feature.descriptor.size()], 1, query_feature.descriptor.size());
            for(int j = 0; j < query_feature.descriptor.size(); j++)
            {
                query[0][j] = query_feature.descriptor.at(j);
            }

            float best_distance_own_class;
            int query_idx;

            // handle this classes features
            {
                // query current class
                std::vector<std::vector<int>> indices_raw;
                std::vector<std::vector<float>> distances_raw;
                flann::SearchParams params = m_flann_exact_match ? flann::SearchParams(-1) : flann::SearchParams(128);
                index_current.knnSearch(query, indices_raw, distances_raw, m_k_search, params);
                // simplify indexing for the following steps
                std::vector<int> indices = indices_raw.at(0);
                std::vector<float> distances = distances_raw.at(0);

                best_distance_own_class = distances[1]; // index 0 is the query itself
                query_idx = indices[0];

                // upweight neighbors that have higher distances to the query
                for(int idx = 0; idx < indices.size(); idx++)
                {
                    int feat_idx = indices.at(idx);
                    float distance = distances.at(idx);

                    // update score of neighbors
                    float score = getScore(distance);
                    temp_scores.at(class_id).at(feat_idx) += score;
                }
            }

            // handle other classes features
            {
                // query current class
                std::vector<std::vector<int>> indices_raw;
                std::vector<std::vector<float>> distances_raw;
                flann::SearchParams params = m_flann_exact_match ? flann::SearchParams(-1) : flann::SearchParams(128);
                index_other.knnSearch(query, indices_raw, distances_raw, 1, params);

                // downweight query if it is too similar to other classes features
                float distance_ratio = best_distance_own_class / distances_raw[0][0];
                float penalty = getScore(distance_ratio, true);
                temp_scores.at(class_id).at(query_idx) += penalty * m_k_search;
            }

            // delete flann pointer
            delete[] query.ptr();
        }
    }

    return temp_scores;
}

float RankingSimilarity::getScore(float distance, bool as_penalty)
{
    // penalty is negative and has higher gain (function has more impact)
    float gain = as_penalty ? -2.5f : 1.0f;

    // sigmoid shifted down by 0.5 and stretched to be in range [0, 1] for positive distances
    return 2.0f * (1.0f / (1.0f + exp(-distance*gain)) - 0.5f);
}

std::string RankingSimilarity::getTypeStatic()
{
    return "Similarity";
}

std::string RankingSimilarity::getType() const
{
    return RankingSimilarity::getTypeStatic();
}
}
