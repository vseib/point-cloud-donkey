/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "ranking_knn_activation.h"

namespace ism3d
{
RankingKnnActivation::RankingKnnActivation()
{
    addParameter(m_use_feature_position, "UseFeaturePosition", false);
    addParameter(m_score_increment_type, "ScoreIncrementType", 0);
}

RankingKnnActivation::~RankingKnnActivation()
{
}

std::map<unsigned, std::vector<float> > RankingKnnActivation::iComputeScores(
        const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> &features)
{
    std::map<unsigned, std::vector<float>> temp_scores;
    std::map<unsigned, int> class_index_offsets;

    pcl::PointCloud<ISMFeature>::Ptr all_features = createCloudWithClassIds(features, false);

    LOG_INFO("starting knn activation ranking");
    // determine class index offsets
    class_index_offsets.insert({0,0});
    for(int i = 0; i < features.size(); i++)
    {
        pcl::PointCloud<ISMFeature>::Ptr current_class_features = createCloudWithClassIds(features, true, i, false);
        int cur_offset = class_index_offsets.at(i);
        // set offset for next class
        if(i != features.size() - 1)
            class_index_offsets.insert({i+1, cur_offset+current_class_features->size()});
        // init activated count
        temp_scores.insert({i, std::vector<float>(current_class_features->size(), 0)});
    }

    // create flann index
    flann::Matrix<float> dataset = createFlannDataset(all_features);
    flann::Index<flann::ChiSquareDistance<float> > index(dataset, flann::KDTreeIndexParams(m_num_kd_trees));
    index.buildIndex();

    // find activated features
    for(int i = 0; i < features.size(); i++)
    {
        pcl::PointCloud<ISMFeature>::Ptr current_class_features = createCloudWithClassIds(features, true, i, false);

        for(ISMFeature query_feature : current_class_features->points)
        {
            // insert the query point
            flann::Matrix<float> query(new float[query_feature.descriptor.size()], 1, query_feature.descriptor.size());
            for(int j = 0; j < query_feature.descriptor.size(); j++)
            {
                query[0][j] = query_feature.descriptor.at(j);
            }

            std::vector<std::vector<int> > indices;
            std::vector<std::vector<float> > distances;
            flann::SearchParams params = m_flann_exact_match ? flann::SearchParams(-1) : flann::SearchParams(128);
            index.knnSearch(query, indices, distances, m_k_search+1, params);
            std::vector<int> similar_features = indices.at(0);

            float query_center_dist = query_feature.centerDist;

            for(int idx = 0; idx < similar_features.size()-1; idx++)
            {
                int feat_idx = similar_features.at(idx);
                unsigned class_id = all_features->at(feat_idx).classId;
                int offset = class_index_offsets.at(class_id);

                // compute distance-dependent score
                float feat_center_dist = all_features->at(feat_idx).centerDist;
                float score_dist_rate = exp(std::fabs(feat_center_dist - query_center_dist));
                score_dist_rate = m_use_feature_position ? score_dist_rate : 1; // if position is not used, set score to 1

                float current_dist = distances.at(0).at(idx);

                // NOTE: this is for backward compatibility: overwrite type 0 by 1
                m_score_increment_type = m_score_increment_type == 0 ? 1 : m_score_increment_type;

                // validate increment type
                if(m_score_increment_type > 3 || m_score_increment_type < 1)
                {
                    LOG_WARN("Invalid score increment type: " << m_score_increment_type << "! Using type 1 instead.");
                    m_score_increment_type = 1;
                }

                // update depending on increment type
                // accessing feature at idx - offset, since flann index contains all classes
                if(m_score_increment_type == 1)
                    temp_scores.at(class_id).at(feat_idx - offset) += score_dist_rate; // just count
                else if(m_score_increment_type == 2)
                    temp_scores.at(class_id).at(feat_idx - offset) += score_dist_rate / (current_dist + 1); // add score proportional to distance
                else if(m_score_increment_type == 3)
                    temp_scores.at(class_id).at(feat_idx - offset) += score_dist_rate * exp(current_dist);
            }

            // delete flann pointer
            delete[] query.ptr();
        }
    }
    // delete flann pointers
    delete[] dataset.ptr();

    return temp_scores;
}


std::string RankingKnnActivation::getTypeStatic()
{
    return "KNNActivation";
}

std::string RankingKnnActivation::getType() const
{
    return RankingKnnActivation::getTypeStatic();
}
}
