/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "clustering_agglomerative.h"
#include "../utils/ism_feature.h"
#include "../utils/distance.h"

namespace ism3d
{
    ClusteringAgglomerative::ClusteringAgglomerative()
    {
        addParameter(m_threshold, "Threshold", 1.2f);
    }

    ClusteringAgglomerative::~ClusteringAgglomerative()
    {
    }

    void ClusteringAgglomerative::process(pcl::PointCloud<ISMFeature>::ConstPtr features)
    {
        if (features->size() == 0)
            return;

        // clusters
        std::list<t_cluster_center> clusters;

        // init each entry as cluster
        for (int i = 0; i < (int)features->size(); i++) {
            const ISMFeature& feature = features->at(i);
            t_center data = feature.descriptor;
            t_descriptor_indices indices;
            indices.push_back(i);
            clusters.push_back(t_cluster_center(data, indices));
        }

        // perform agglomerative cluster until threshold is reached
        float similarity = std::numeric_limits<float>::infinity();
        do {
            float minDist = std::numeric_limits<float>::infinity();
            std::list<t_cluster_center>::iterator minCluster1 = clusters.end();
            std::list<t_cluster_center>::iterator minCluster2 = clusters.end();

            // compute the two nearest clusters
            for (std::list<t_cluster_center>::iterator it1 = clusters.begin();
                 it1 != clusters.end(); it1++) {
                const t_cluster_center& cluster1 = *it1;

                for (std::list<t_cluster_center>::iterator it2 = clusters.begin();
                     it2 != clusters.end(); it2++) {
                    if (it1 == it2)
                        continue;   // don't compare the same clusters

                    const t_cluster_center& cluster2 = *it2;

                    float distance = clusterDistance(cluster1, cluster2, features);
                    //float distance = clusterDistance(cluster1, cluster2);
                    if (distance < minDist) {
                        minDist = distance;
                        minCluster1 = it1;
                        minCluster2 = it2;
                    }
                }
            }

            similarity = 1.0f / minDist;
            if (similarity < m_threshold)
                break;

            // merge clusters
            if (minCluster1 != clusters.end() && minCluster2 != clusters.end())
                merge(clusters, minCluster1, minCluster2);
        } while (clusters.size() > 1);

        int i = 0;
        m_centers.resize(clusters.size());
        m_indices.resize(features->size());
        for (std::list<t_cluster_center>::const_iterator it = clusters.begin();
             it != clusters.end(); it++, i++) {
            // assign cluster centers
            m_centers[i] = it->first;

            // assign cluster indices to descriptors
            const t_descriptor_indices& indices = it->second;
            for (int j = 0; j < indices.size(); j++)
                m_indices[indices[j]] = i;
        }
    }

    float ClusteringAgglomerative::clusterDistance(const t_cluster_center& center1,
                                                   const t_cluster_center& center2,
                                                   pcl::PointCloud<ISMFeature>::ConstPtr features)
    {
        const t_descriptor_indices& indices1 = center1.second;
        const t_descriptor_indices& indices2 = center2.second;

        float dist = 0;
        for (int i = 0; i < (int)indices1.size(); i++) {
            const ISMFeature& feat1 = features->at(indices1[i]);
            for (int j = 0; j < (int)indices2.size(); j++) {
                const ISMFeature& feat2 = features->at(indices2[j]);
                dist += getDistance()(feat1.descriptor, feat2.descriptor);
            }
        }

        if (indices1.size() == 0 || indices2.size() == 0)
            return std::numeric_limits<float>::infinity();

        dist /= (indices1.size() * indices2.size());

        return dist;
    }

    float ClusteringAgglomerative::clusterDistance(const t_cluster_center& center1,
                                                   const t_cluster_center& center2)
    {
        return getDistance()(center1.first, center2.first);
    }

    float ClusteringAgglomerative::clusterSimilarity(const t_cluster_center& center1,
                                                   const t_cluster_center& center2,
                                                   pcl::PointCloud<ISMFeature>::ConstPtr features)
    {
        // normalized cross correlation does not work for two objects when their mean (cluster center) is
        // initially equal to the first associated descriptor. In this case, the numerator is zero and so is
        // the ncc measure.

        const t_descriptor_indices& indices1 = center1.second;
        const t_descriptor_indices& indices2 = center2.second;

        float similarity = 0;
        for (int i = 0; i < (int)indices1.size(); i++) {
            const ISMFeature& feat1 = features->at(indices1[i]);
            for (int j = 0; j < (int)indices2.size(); j++) {
                const ISMFeature& feat2 = features->at(indices2[j]);
                // normalized cross correlation measure for descriptors
                float sum = 0;
                float normalize1 = 0;
                float normalize2 = 0;
                for (int k = 0; k < (int)feat1.descriptor.size(); k++) {
                    float value1 = (feat1.descriptor[k] - center1.first[k]);
                    float value2 = (feat2.descriptor[k] - center2.first[k]);

                    sum += value1 * value2;
                    normalize1 += value1 * value1;
                    normalize2 += value2 * value2;
                }

                float NCC = 0;
                if (sum > 0 || (normalize1 > 0 && normalize2 > 0))
                    NCC = sum / (sqrtf(normalize1 * normalize2));

                similarity += NCC;
            }
        }

        if (indices1.size() == 0 || indices2.size() == 0)
            return std::numeric_limits<float>::infinity();

        similarity /= (indices1.size() * indices2.size());
        return similarity;
    }

    void ClusteringAgglomerative::merge(std::list<t_cluster_center>& clusters,
                                        const std::list<t_cluster_center>::iterator& cluster1,
                                        const std::list<t_cluster_center>::iterator& cluster2)
    {
        // merge cluster2 into cluster1 and remove cluster2

        t_center& center1 = cluster1->first;
        const t_center& center2 = cluster2->first;

        // update the cluster center
        for (int i = 0; i < (int)center1.size(); i++) {
            center1[i] += center2[i];
            center1[i] /= 2.0f;
        }

        // merge descriptor ids
        t_descriptor_indices& cluster1Indices = cluster1->second;
        const t_descriptor_indices& cluster2Indices = cluster2->second;
        cluster1Indices.insert(cluster1Indices.end(), cluster2Indices.begin(), cluster2Indices.end());

        // remove second cluster
        clusters.erase(cluster2);
    }

    std::string ClusteringAgglomerative::getTypeStatic()
    {
        return "Agglomerative";
    }

    std::string ClusteringAgglomerative::getType() const
    {
        return ClusteringAgglomerative::getTypeStatic();
    }
}
