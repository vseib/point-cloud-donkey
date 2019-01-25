/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "ranking_incremental.h"

namespace ism3d
{
RankingIncremental::RankingIncremental()
{
}

RankingIncremental::~RankingIncremental()
{
}

std::map<unsigned, std::vector<float> > RankingIncremental::iComputeScores(
        const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &features)
{
    std::map<unsigned, std::vector<float> > temp_scores;
    std::map<unsigned, int> class_index_offsets;

    pcl::PointCloud<ISMFeature>::Ptr all_features = createCloudWithClassIds(features, false);

    LOG_INFO("starting incremental ranking");
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
    flann::Index<flann::L2<float> > index(dataset, flann::KDTreeIndexParams(m_num_kd_trees));
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

            // prepare results
            std::vector<std::vector<int> > indices;
            std::vector<std::vector<float> > distances;
            flann::SearchParams params = m_flann_exact_match ? flann::SearchParams(-1) : flann::SearchParams(128);
            index.knnSearch(query, indices, distances, m_k_search + 1, params);

            if(distances.size() > 0 && distances.at(0).size() > 0)
            {
                for(int dist_idx = 0; dist_idx < distances[0].size()-1; dist_idx++)
                {
                    int feat_idx = indices[0][dist_idx];
                    unsigned class_id = all_features->at(feat_idx).classId;
                    int offset = class_index_offsets.at(class_id);
                    float dist_b = distances[0][dist_idx+1];
                    temp_scores.at(class_id).at(feat_idx - offset) += distances[0][dist_idx] - dist_b;
                }
            }
            // delete flann pointer
            delete[] query.ptr();
        }
    }
    // delete flann pointers
    delete[] dataset.ptr();

    return temp_scores;
}


std::string RankingIncremental::getTypeStatic()
{
    return "Incremental";
}

std::string RankingIncremental::getType() const
{
    return RankingIncremental::getTypeStatic();
}
}
