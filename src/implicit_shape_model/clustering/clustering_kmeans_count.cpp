/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "clustering_kmeans_count.h"

namespace ism3d
{
    ClusteringKMeansCount::ClusteringKMeansCount()
    {
        addParameter(m_clusterCount, "ClusterCount", 10);
    }

    ClusteringKMeansCount::~ClusteringKMeansCount()
    {
    }

    void ClusteringKMeansCount::process(pcl::PointCloud<ISMFeature>::ConstPtr features)
    {
        cluster(features, m_clusterCount);
    }

    std::string ClusteringKMeansCount::getTypeStatic()
    {
        return "KMeansCount";
    }

    std::string ClusteringKMeansCount::getType() const
    {
        return ClusteringKMeansCount::getTypeStatic();
    }
}
