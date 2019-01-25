/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "ranking_naive_bayes.h"

namespace ism3d
{
RankingNaiveBayes::RankingNaiveBayes()
{
}

RankingNaiveBayes::~RankingNaiveBayes()
{
}

std::map<unsigned, std::vector<float> > RankingNaiveBayes::iComputeScores(const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &features)
{
    std::map<unsigned, std::vector<float> > temp_scores;

    LOG_INFO("starting naive bayes ranking");
    for(int i = 0; i < features.size(); i++)
    {
        // insert list for the current class
        pcl::PointCloud<ISMFeature>::Ptr current_class_features = createCloudWithClassIds(features, true, i, false);
        pcl::PointCloud<ISMFeature>::Ptr other_classes_features = createCloudWithClassIds(features, true, i, true);
        temp_scores.insert({i, std::vector<float>(current_class_features->size(), 0)});

        // create flann indices for current class and the rest
        flann::Matrix<float> dataset_current = createFlannDataset(current_class_features);
        flann::Index<flann::L2<float> > index_current(dataset_current, flann::KDTreeIndexParams(m_num_kd_trees));
        index_current.buildIndex();
        flann::Matrix<float> dataset_other = createFlannDataset(other_classes_features);
        flann::Index<flann::L2<float> > index_other(dataset_other, flann::KDTreeIndexParams(m_num_kd_trees));
        index_other.buildIndex();

//        // create kd trees for current class and the rest
//        ISMFeaturePointRepresentation ism_rep(current_class_features->at(0).descriptor.size());
//        pcl::KdTreeFLANN<ISMFeature> kdtree_current;
//        kdtree_current.setPointRepresentation(boost::make_shared<const ISMFeaturePointRepresentation>(ism_rep));
//        kdtree_current.setInputCloud(current_class_features);
//        pcl::KdTreeFLANN<ISMFeature> kdtree_other;
//        kdtree_other.setPointRepresentation(boost::make_shared<const ISMFeaturePointRepresentation>(ism_rep));
//        kdtree_other.setInputCloud(other_classes_features);

        for(int feat_idx = 0; feat_idx < current_class_features->size(); feat_idx++)
        {
            ISMFeature query_feature = current_class_features->at(feat_idx);
            // insert the query point
            flann::Matrix<float> query(new float[query_feature.descriptor.size()], 1, query_feature.descriptor.size());
            for(int j = 0; j < query_feature.descriptor.size(); j++)
            {
                query[0][j] = query_feature.descriptor.at(j);
            }

//            std::vector<int> same_feature_indices_neg = findSimilarFeatures(kdtree_other, query_feature);
//            std::vector<int> same_feature_indices_pos = findSimilarFeatures(kdtree_current, query_feature);
            std::vector<int> same_feature_indices_neg = findSimilarFeaturesFlann(index_other, query);
            std::vector<int> same_feature_indices_pos = findSimilarFeaturesFlann(index_current, query);
            float num_pos = (float) same_feature_indices_pos.size();
            float num_neg = (float) same_feature_indices_neg.size();
            float num_current = (float) current_class_features->size();
            float num_other = (float) other_classes_features->size();

            float pos_prob = num_pos / num_current;
            float neg_prob = num_neg / num_other;
            // float score = pos_prob / (pos_prob+neg_prob);
            float score = pos_prob / ((num_pos + num_neg) / (num_current + num_other));
            temp_scores.at(i).at(feat_idx) = score;

            // delete flann pointer
            delete[] query.ptr();
        }

        // delete flann pointers
        delete[] dataset_current.ptr();
        delete[] dataset_other.ptr();
    }

    return temp_scores;
}


std::string RankingNaiveBayes::getTypeStatic()
{
    return "NaiveBayes";
}

std::string RankingNaiveBayes::getType() const
{
    return RankingNaiveBayes::getTypeStatic();
}
}
