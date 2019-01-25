/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */


#include "clustering_kmeans_factor.h"

namespace ism3d
{
    ClusteringKMeansFactor::ClusteringKMeansFactor()
    {
        addParameter(m_clusterFactor, "ClusterFactor", 0.2f);
    }

    ClusteringKMeansFactor::~ClusteringKMeansFactor()
    {
    }

    void ClusteringKMeansFactor::process(pcl::PointCloud<ISMFeature>::ConstPtr features)
    {
        int clusterCount = 1;
        if (m_clusterFactor > 1) {
            LOG_WARN("cluster count factor has to be in range [0, 1], setting to 0.5");
            m_clusterFactor = 0.5f;
        }

        clusterCount = (int)round(features->size() * m_clusterFactor);

        cluster(features, clusterCount);
    }

    std::string ClusteringKMeansFactor::getTypeStatic()
    {
        return "KMeansFactor";
    }

    std::string ClusteringKMeansFactor::getType() const
    {
        return ClusteringKMeansFactor::getTypeStatic();
    }
}
