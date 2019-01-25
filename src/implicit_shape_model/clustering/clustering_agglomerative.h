/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_CLUSTERINGAGGLOMERATIVE_H
#define ISM3D_CLUSTERINGAGGLOMERATIVE_H

#include "clustering.h"

#include <list>

namespace ism3d
{
    /**
     * @brief The ClusteringAgglomerative class
     * Performs an agglomerative clustering on the input data. The algorithm starts with the input features
     * as individual clusters and iteratively merges the two most similar clusters as long as their similarity
     * is above a threshold. The number of clusters is only determined by the threshold.
     */
    class ClusteringAgglomerative
            : public Clustering
    {
    public:
        ClusteringAgglomerative();
        ~ClusteringAgglomerative();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        void process(pcl::PointCloud<ISMFeature>::ConstPtr);

    private:
        typedef std::vector<float> t_center;
        typedef std::vector<int> t_descriptor_indices;
        typedef std::pair<t_center, t_descriptor_indices> t_cluster_center;

        float clusterDistance(const t_cluster_center&,
                              const t_cluster_center&,
                              pcl::PointCloud<ISMFeature>::ConstPtr);
        float clusterDistance(const t_cluster_center&,
                              const t_cluster_center&);
        float clusterSimilarity(const t_cluster_center&,
                                const t_cluster_center&,
                                pcl::PointCloud<ISMFeature>::ConstPtr);
        void merge(std::list<t_cluster_center>&,
                   const std::list<t_cluster_center>::iterator&,
                   const std::list<t_cluster_center>::iterator&);

        float m_threshold;
    };
}

#endif // ISM3D_CLUSTERINGAGGLOMERATIVE_H
