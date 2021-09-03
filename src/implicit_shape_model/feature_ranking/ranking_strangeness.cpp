/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "ranking_strangeness.h"

namespace ism3d
{
RankingStrangeness::RankingStrangeness()
{
}

RankingStrangeness::~RankingStrangeness()
{
}

std::map<unsigned, std::vector<float> > RankingStrangeness::iComputeScores(
        const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &features)
{
    std::map<unsigned, std::vector<float> > temp_scores;
    std::map<unsigned, int> class_index_offsets;

    LOG_INFO("starting strangeness ranking");
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

    // create a separate flann index for each class
    std::vector<flann::Matrix<float>> datasets;
    std::vector<flann::Index<flann::ChiSquareDistance<float>>> flann_indices;
    for(int i = 0; i < features.size(); i++)
    {
        pcl::PointCloud<ISMFeature>::Ptr current_class_features = createCloudWithClassIds(features, true, i, false);

        flann::Matrix<float> dataset_current = createFlannDataset(current_class_features);
        datasets.push_back(dataset_current);
        flann::Index<flann::ChiSquareDistance<float> > index_current(dataset_current, flann::KDTreeIndexParams(m_num_kd_trees));
        index_current.buildIndex();
        flann_indices.push_back(index_current);
    }

    // loop over features
    for(int i = 0; i < features.size(); i++)
    {
        pcl::PointCloud<ISMFeature>::Ptr current_class_features = createCloudWithClassIds(features, true, i, false);

        for(int feat_idx = 0; feat_idx < current_class_features->size(); feat_idx++)
        {
            ISMFeature query_feature = current_class_features->at(feat_idx);
            // insert the query point
            flann::Matrix<float> query(new float[query_feature.descriptor.size()], 1, query_feature.descriptor.size());
            for(int j = 0; j < query_feature.descriptor.size(); j++)
            {
                query[0][j] = query_feature.descriptor.at(j);
            }

            std::vector<float> all_dist_sums;
            for(int j = 0; j < flann_indices.size(); j++)
            {
                std::vector<float> dist_list = findNeighborsDistances(flann_indices.at(j), query);
                float sum = 0;
                for(int k = 0; k < dist_list.size(); k++)
                {
                    sum += dist_list.at(k);
                }
                all_dist_sums.push_back(sum);
            }
            // compute score according to Eq.2 of the paper (see comments in header)
            float score = all_dist_sums.at(i);
            all_dist_sums.at(i) = -1; //replace by smallest dummy
            std::sort(all_dist_sums.begin(), all_dist_sums.end());
            score /= all_dist_sums.at(1); // take second value, since the first value is dummy
            temp_scores.at(i).at(feat_idx) = score;

            // delete flann pointer
            delete[] query.ptr();
        }
    }

    // delete flann pointers
    for(int i = 0; i < datasets.size(); i++)
    {
        delete[] datasets.at(i).ptr();
    }

    return temp_scores;
}

std::string RankingStrangeness::getTypeStatic()
{
    return "Strangeness";
}

std::string RankingStrangeness::getType() const
{
    return RankingStrangeness::getTypeStatic();
}
}
