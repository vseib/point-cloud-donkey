/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "clustering_kmeans_thumb_rule.h"

namespace ism3d
{
    ClusteringKMeansThumbRule::ClusteringKMeansThumbRule()
    {
    }

    ClusteringKMeansThumbRule::~ClusteringKMeansThumbRule()
    {
    }

    void ClusteringKMeansThumbRule::process(pcl::PointCloud<ISMFeature>::ConstPtr features)
    {
        // see: Kanti Mardia, Multivariate Analysis, p. 365
        float value = sqrtf(features->size() / 2.0f);
        int clusterCount = (int)round(value);

        cluster(features, clusterCount);
    }

    std::string ClusteringKMeansThumbRule::getTypeStatic()
    {
        return "KMeansThumbRule";
    }

    std::string ClusteringKMeansThumbRule::getType() const
    {
        return ClusteringKMeansThumbRule::getTypeStatic();
    }
}
