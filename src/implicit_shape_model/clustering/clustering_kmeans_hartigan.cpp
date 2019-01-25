/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "clustering_kmeans_hartigan.h"

namespace ism3d
{
    ClusteringKMeansHartigan::ClusteringKMeansHartigan()
    {
        addParameter(m_maxK, "MaxK", 10);
    }

    ClusteringKMeansHartigan::~ClusteringKMeansHartigan()
    {
    }

    void ClusteringKMeansHartigan::process(pcl::PointCloud<ISMFeature>::ConstPtr features)
    {
        cluster(features, 1);

        //int maxK = features->size() - 1;
        int maxK = m_maxK;

        std::vector<std::vector<std::vector<float> > > centers(maxK);
        std::vector<std::vector<int> > indices(maxK);
        std::vector<float> dispersions(maxK);

        // iterative clustering
        for (int i = 0; i < maxK; i++) {
            cluster(features, i + 1);

            centers[i] = m_centers;
            indices[i] = m_indices;
            dispersions[i] = withinClusterSumOfSquares(m_centers, features);
        }

        // compute hartigan's index
        int bestK = 0;
        float maxValue = 0;
        std::vector<float> hartigansIndex(maxK - 1);
        for (int i = 0; i < (int)hartigansIndex.size(); i++) {
            int numClusters = i + 1;
            float factor = (features->size() - numClusters - 1);
            float index = ((dispersions[i] / dispersions[i + 1]) - 1) * factor;

            hartigansIndex[i] = index;

            if (index > maxValue) {
                maxValue = index;
                bestK = i + 1;
            }
        }

        LOG_INFO("best value for k: " << bestK);

        // choose clustering which matches the best determined number for k
        m_centers = centers[bestK - 1];
        m_indices = indices[bestK - 1];
    }

    float ClusteringKMeansHartigan::withinClusterSumOfSquares(const std::vector<std::vector<float> >& clusters,
                                                              pcl::PointCloud<ISMFeature>::ConstPtr data)
    {
        float compactness = 0.0f;
        for (unsigned int i_point = 0; i_point < (int)data->size(); i_point++)
        {
            const ISMFeature& sample = data->at(i_point);

            int k_best = 0;
            float min_dist = std::numeric_limits<float>::max ();

            for (int i_cluster = 0; i_cluster < (int)clusters.size(); i_cluster++)
            {
                const std::vector<float>& cluster = clusters[i_cluster];
                float dist = m_dist(sample.descriptor, cluster);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    k_best = i_cluster;
                }
            }
            compactness += min_dist;
        }
        return compactness;
    }

    std::string ClusteringKMeansHartigan::getTypeStatic()
    {
        return "KMeansHartigan";
    }

    std::string ClusteringKMeansHartigan::getType() const
    {
        return ClusteringKMeansHartigan::getTypeStatic();
    }
}
