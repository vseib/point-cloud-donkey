/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_CLUSTERINGFACTORY_H
#define ISM3D_CLUSTERINGFACTORY_H

#include "../utils/factory.h"
#include "clustering_agglomerative.h"
#include "clustering_kmeans_count.h"
#include "clustering_kmeans_factor.h"
#include "clustering_kmeans_thumb_rule.h"
#include "clustering_kmeans_hartigan.h"
#include "clustering_none.h"

namespace ism3d
{
    template <>
    Clustering* Factory<Clustering>::createByType(const std::string& type)
    {
        if (type == ClusteringKMeansCount::getTypeStatic())
            return new ClusteringKMeansCount();
        else if (type == ClusteringKMeansFactor::getTypeStatic())
            return new ClusteringKMeansFactor();
        else if (type == ClusteringKMeansThumbRule::getTypeStatic())
            return new ClusteringKMeansThumbRule();
        else if (type == ClusteringKMeansHartigan::getTypeStatic())
            return new ClusteringKMeansHartigan();
        else if (type == ClusteringAgglomerative::getTypeStatic())
            return new ClusteringAgglomerative();
        else if (type == ClusteringNone::getTypeStatic())
            return new ClusteringNone();
        else
            return 0;
    }
}

#endif // ISM3D_CLUSTERINGFACTORY_H
