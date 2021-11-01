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
    addParameter(m_update_type, "UpdateType", std::string("dist"));
    addParameter(m_intra_pos, "IntraPosition", std::string("front"));
    addParameter(m_inter_pos, "InterPosition", std::string("front"));
}

RankingSimilarity::~RankingSimilarity()
{
}

std::map<unsigned, std::vector<float> > RankingSimilarity::iComputeScores(
        const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &features)
{
    LOG_INFO("starting similarity ranking");

    // separately ranking intra and iter class score, then combining both scores
    std::map<unsigned, std::vector<float>> intra_class_scores;
    std::map<unsigned, std::vector<float>> inter_class_scores;
    std::map<unsigned, std::vector<float>> temp_scores;

    std::map<unsigned, int> class_index_offsets;

    // determine class index offsets
    class_index_offsets.insert({0,0});
    for(unsigned i = 0; i < features.size(); i++)
    {
        pcl::PointCloud<ISMFeature>::Ptr current_class_features = createCloudWithClassIds(features, true, i, false);
        int cur_offset = class_index_offsets.at(i);
        // set offset for next class
        if(i != features.size() - 1)
            class_index_offsets.insert({i+1, cur_offset+current_class_features->size()});
        // init activated count
        intra_class_scores.insert({i, std::vector<float>(current_class_features->size(), 0)});
        inter_class_scores.insert({i, std::vector<float>(current_class_features->size(), 0)});
        temp_scores.insert({i, std::vector<float>(current_class_features->size(), 0)});
    }

    // TODO VS: collection of temp params
    //std::string update_type = "count";
    //std::string update_type = "dist";
    //std::string update_type = "score";
    std::string update_type = m_update_type;


    // loop over all classes
    for(unsigned class_id = 0; class_id < features.size(); class_id++)
    {
        // insert list for the current class
        pcl::PointCloud<ISMFeature>::Ptr current_class_features = createCloudWithClassIds(features, true, class_id, false);
        pcl::PointCloud<ISMFeature>::Ptr other_classes_features = createCloudWithClassIds(features, true, class_id, true);

        // create flann indices for current class and the rest
        flann::Matrix<float> dataset_current = createFlannDataset(current_class_features);
        flann::Index<flann::ChiSquareDistance<float>> index_current(dataset_current, flann::KDTreeIndexParams(m_num_kd_trees));
        index_current.buildIndex();
        flann::Matrix<float> dataset_other = createFlannDataset(other_classes_features);
        flann::Index<flann::ChiSquareDistance<float>> index_other(dataset_other, flann::KDTreeIndexParams(m_num_kd_trees));
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
            int query_class_size = current_class_features->size();

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
                    float update_value = 0;

                    // update value is a "penalty": smaller values for rare features, higher values for common features
                    if(update_type == "count")
                        update_value = 1;
                    else if(update_type == "score")
                        update_value = -score;
                    else if(update_type == "dist")
                        update_value = -distance;

                    intra_class_scores.at(class_id).at(feat_idx) += update_value;
                }
            }

            // handle other classes features
            {
                // query current class
                std::vector<std::vector<int>> indices_raw;
                std::vector<std::vector<float>> distances_raw;
                flann::SearchParams params = m_flann_exact_match ? flann::SearchParams(-1) : flann::SearchParams(128);
                index_other.knnSearch(query, indices_raw, distances_raw, 100, params); // TODO VS: try replacing 100 by (num-classes-1)*m_k_search
                // simplify indexing for the following steps
                std::vector<int> indices = indices_raw.at(0);
                std::vector<float> distances = distances_raw.at(0);

                // downweight neighbors from other classes
                for(int idx = 0; idx < indices.size(); idx++)
                {
                    int feat_idx = indices.at(idx);
                    float distance = distances.at(idx);
                    unsigned class_id_other = other_classes_features->at(feat_idx).classId;
                    int offset = class_index_offsets.at(class_id_other);

                    // update score of neighbors
                    float score = getScore(distance);
                    float update_value = 0;

                    // update value is a "penalty": smaller values for separated features, higher values for close inter-class features
                    if(update_type == "count")
                        update_value = 1;
                    else if(update_type == "score")
                        update_value = -score;
                    else if(update_type == "dist")
                        update_value = -distance;

                    if(class_id_other < class_id)
                        inter_class_scores.at(class_id_other).at(feat_idx - offset) += update_value;
                    else
                        inter_class_scores.at(class_id_other).at(feat_idx + query_class_size - offset) += update_value;
                }
            }

            // delete flann pointer
            delete[] query.ptr();
        }
    }

    // combine both score lists
    // assumption for intra-class: if sorted ascendingly, "good" features are in the center of the list (not too strange, not too common)
    // assumption for inter-class: if sorted ascendingly, "good" features are in the beginning of the list (not too close to other classes)

    {
        std::cout << "############ intra ############" << std::endl;
        for(unsigned class_id = 0; class_id < features.size(); class_id++)
        {
            std::cout << "------------ class - " << class_id << std::endl;
            std::vector<float> list = intra_class_scores[class_id];
            std::sort(list.begin(), list.end());
            float min = list.front();
            float max = list.back();
            float median = list.at(std::floor(list.size()/2));

            // in case some features have zero score so far, remap zero to "best" value
            // (only necessary for "score" and "dist" types
            if(update_type == "score" || update_type == "dist")
            {
                for(unsigned i = 0; i < intra_class_scores[class_id].size(); i++)
                {
                    if(intra_class_scores[class_id][i] == 0)
                    {
                        intra_class_scores[class_id][i] = min * 1.1f; // TODO VS: tune this value? --> must be smaller than min, but how much "advantage" to give?
                    }
                }
            }
            list = intra_class_scores[class_id];
            std::sort(list.begin(), list.end());
            min = list.front();
            max = list.back();
            median = list.at(std::floor(list.size()/2));


            // transform scores
            for(unsigned i = 0; i < intra_class_scores[class_id].size(); i++)
            {
                if(m_intra_pos == "front" || m_intra_pos == "back")
                {
                    // favors the front of the list
                    if(min < 0)
                    {
                        intra_class_scores[class_id][i] =
                                (intra_class_scores[class_id][i]+std::fabs(min)) / (max+std::fabs(min));
                    }
                    else
                    {
                        intra_class_scores[class_id][i] = intra_class_scores[class_id][i]/max;
                    }
                }
                if(m_intra_pos == "back") // this second step is needed for "back"
                {
                    // favors the back
                    intra_class_scores[class_id][i] = std::fabs(intra_class_scores[class_id][i] - 1);
                }

                if(m_intra_pos == "center")
                {
                // favors the center of the list
                // with median: slightly shifted to the back
                // with 0.5: more centered
                //median = median > 0 ? 0.5f : -0.5f;
                    intra_class_scores[class_id][i] = std::fabs((intra_class_scores[class_id][i]-median) / (max-median));
                }

                if(m_intra_pos == "center2")
                {
                // favors the center of the list
                // with median: slightly shifted to the back
                // with 0.5: more centered
                    median = median > 0 ? 0.5f : -0.5f;
                    intra_class_scores[class_id][i] = std::fabs((intra_class_scores[class_id][i]-median) / (max-median));
                }

            }
        }

        std::cout << "############ inter ############" << std::endl;
        for(unsigned class_id = 0; class_id < features.size(); class_id++)
        {
            std::cout << "------------ class - " << class_id << std::endl;
            std::vector<float> list = inter_class_scores[class_id];
            std::sort(list.begin(), list.end());
            float min = list.front();
            float max = list.back();
            float median = list.at(std::floor(list.size()/2));

            // in case some features have zero score so far, remap zero to "best" value
            // (only necessary for "score" and "dist" types
            if(update_type == "score" || update_type == "dist")
            {
                for(unsigned i = 0; i < inter_class_scores[class_id].size(); i++)
                {
                    if(inter_class_scores[class_id][i] == 0)
                    {
                        inter_class_scores[class_id][i] = min * 1.1f; // TODO VS: tune this value? --> must be smaller than min, but how much "advantage" to give?
                    }
                }
            }
            list = inter_class_scores[class_id];
            std::sort(list.begin(), list.end());
            min = list.front();
            max = list.back();
            median = list.at(std::floor(list.size()/2));

            // transform scores
            for(unsigned i = 0; i < inter_class_scores[class_id].size(); i++)
            {
                if(m_inter_pos == "front" || m_inter_pos == "back")
                {
                    // favors the front of the list
                    if(min < 0)
                    {
                        inter_class_scores[class_id][i] =
                                (inter_class_scores[class_id][i]+std::fabs(min)) / (max+std::fabs(min));
                    }
                    else
                    {
                        inter_class_scores[class_id][i] = inter_class_scores[class_id][i]/max;
                    }
                }
                if(m_inter_pos == "back") // this second step is needed for "back"
                {
                    // favors the back
                    inter_class_scores[class_id][i] = std::fabs(inter_class_scores[class_id][i] - 1);
                }

                if(m_inter_pos == "center")
                {
                // favors the center of the list
                // with median: slightly shifted to the back
                // with 0.5: more centered
                //median = median > 0 ? 0.5f : -0.5f;
                    inter_class_scores[class_id][i] = std::fabs((inter_class_scores[class_id][i]-median) / (max-median));
                }

                if(m_inter_pos == "center2")
                {
                // favors the center of the list
                // with median: slightly shifted to the back
                // with 0.5: more centered
                    median = median > 0 ? 0.5f : -0.5f;
                    inter_class_scores[class_id][i] = std::fabs((inter_class_scores[class_id][i]-median) / (max-median));
                }
            }
        }

        // merging
        std::cout << "\n\n\n\n ######## merging scores " << std::endl;
        for(unsigned class_id = 0; class_id < features.size(); class_id++)
        {
            for(unsigned index = 0; index < temp_scores[class_id].size(); index++)
            {
                temp_scores[class_id][index] = inter_class_scores[class_id][index] + intra_class_scores[class_id][index];
            }
        }
    }

    return temp_scores;
}

float RankingSimilarity::getScore(float distance, bool as_penalty)
{
    // penalty is negative and has higher gain (function has more impact)
    float gain = as_penalty ? -1.0f : 1.0f;

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
