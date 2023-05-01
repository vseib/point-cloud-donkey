/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_FEATURE_RANKING_H
#define ISM3D_FEATURE_RANKING_H

#include "../utils/json_object.h"
#include "../utils/utils.h"
#include "../utils/ism_feature.h"

#define PCL_NO_PRECOMPILE
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/search.h>
#include <pcl/kdtree/kdtree_flann.h>

namespace ism3d
{
    /**
     * @brief The FeatureRanking class
     * Works as a functor and computes scores for feature descriptors during training.
     */
    class FeatureRanking
            : public JSONObject
    {

    typedef std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> FeatureMapT;
    typedef std::map<unsigned, std::vector<std::pair<int, float>>> ScoringMapT;

    public:
        virtual ~FeatureRanking();

        /**
         * @brief Interface function to compute scores on descriptors in the input point cloud.
         * @param features map assigning each class an input point cloud with computed features
         * @param num_kd_trees number of flann kdtrees to use for index
         * @param flann_exact_match if true flann will find exact nearest neighbor
         * @return a map with feature scores representing the structure of the input features map
         */
    std::tuple<std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>>,
            pcl::PointCloud<ISMFeature>::Ptr>
    operator()(
            std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> &features,
            int num_kd_trees = 4, bool flann_exact_match = false);

        /**
         * @brief Set the number of threads to use. The derived classes do not need to use it.
         * @param numThreads the number of threads to use
         */
        void setNumThreads(int numThread);

    protected:
        FeatureRanking();

        std::map<unsigned, std::vector<std::pair<int, float>>> rankFeaturesWithScores(
                std::map<unsigned, std::vector<float>> &temp_scores);

        std::map<unsigned, std::vector<float>> extractSubsetFromRankedList(
                const std::map<unsigned, std::vector<std::pair<int, float>>> &index_score_map,
                const std::map<unsigned, std::vector<float>> &scores);

        std::map<unsigned, std::vector<std::vector<float>>> unflatten_list(
                std::map<unsigned, std::vector<float>> &scores_flat_list,
                const FeatureMapT &features);

        int getNumThreads() const;

        virtual std::map<unsigned, std::vector<float> > iComputeScores(
                const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &f) = 0;

        int countFeatures(const FeatureMapT &features);
        int countFeatures(const std::vector<pcl::PointCloud<ISMFeature>::Ptr> &features);

        // puts all features into one cloud, if desired only returns one specific class (filter_class)
        // or all others (inverse)
        pcl::PointCloud<ISMFeature>::Ptr createCloudWithClassIds
            (const FeatureRanking::FeatureMapT &features,
             bool filter = false, unsigned filter_class = 0, bool inverse = false);

        flann::Matrix<float> createFlannDataset(pcl::PointCloud<ISMFeature>::Ptr &class_features);

        std::vector<float> findNeighborsDistances(flann::Index<flann::ChiSquareDistance<float> > &index, flann::Matrix<float> &query);

        std::vector<int> findSimilarFeaturesFlann(flann::Index<flann::ChiSquareDistance<float> > &index, flann::Matrix<float> &query);

        std::vector<int> findSimilarFeatures(pcl::KdTreeFLANN<ISMFeature> &kdtree, ISMFeature &feature);

        int m_k_search;
        float m_dist_thresh;
        float m_factor;
        int m_num_kd_trees;
        bool m_flann_exact_match;

        std::string m_extract_list;
        float m_extract_offset;

        template<typename T>
        std::vector<float> accumulateClassDistances(const pcl::PointCloud<ISMFeature>::Ptr& features,
                                                              const flann::Index<T> &index, const bool exact_match,
                                                              const std::vector<unsigned> class_look_up,
                                                              const int num_classes) const
        {
            std::vector<float> class_distances(num_classes, 0);
            int k_search = 11;

            // loop over all features extracted from the input model
            for(int fe = 0; fe < features->size(); fe++)
            {
                // insert the query point
                ISMFeature feature = features->at(fe);
                flann::Matrix<float> query(new float[feature.descriptor.size()], 1, feature.descriptor.size());
                for(int i = 0; i < feature.descriptor.size(); i++)
                {
                    query[0][i] = feature.descriptor.at(i);
                }

                // prepare results
                std::vector<std::vector<int> > indices;
                std::vector<std::vector<float> > distances;
                flann::SearchParams params = exact_match ? flann::SearchParams(-1) : flann::SearchParams(128);
                index.knnSearch(query, indices, distances, k_search, params);

                delete[] query.ptr();

                // background distance
                float dist_b = 0;
                if(distances.size() > 0 && distances.at(0).size() > 1)
                {
                    dist_b = distances.at(0).back(); // get last element
                }

                std::vector<unsigned> used_classes;
                if(distances.size() > 0 && distances.at(0).size() > 0)
                {
                    for(int i = 0; i < distances[0].size()-1; i++)
                    {
                        unsigned class_idx = class_look_up[indices[0][i]];
                        if(!Utils::containsValue(used_classes, class_idx))
                        {
                            class_distances.at(class_idx) += distances[0][i] - dist_b;
                            used_classes.push_back(class_idx);
                        }
                    }
                }
            }

            return class_distances;
        }

    private:

        int m_numThreads;
    };
}

#endif // ISM3D_FEATURE_RANKING_H
