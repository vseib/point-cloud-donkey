/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "clustering_kmeans.h"
#include "../features/features.h"
#include "../utils/utils.h"
#include "../utils/distance.h"

namespace ism3d
{
    ClusteringKMeans::ClusteringKMeans()
    {
        //addParameter(m_clusterCountFactor, "ClusterCountFactor", 0.1f);
        //addParameter(m_desiredClusters, "DesiredClusters", 0);
        //addParameter(m_branching, "Branching", 10);
        addParameter(m_iterations, "Iterations", 1000);
        addParameter(m_centersInit, "CentersInit", flann::FLANN_CENTERS_KMEANSPP);
        addParameter(m_cbIndex, "CbIndex", 0.5f);
    }

    ClusteringKMeans::~ClusteringKMeans()
    {
    }

    void ClusteringKMeans::cluster(pcl::PointCloud<ISMFeature>::ConstPtr features, int clusterCount)
    {
        if (clusterCount == 0)
            clusterCount = 1;

        m_desiredClusters = clusterCount;
        m_branching = m_desiredClusters > 1 ? m_desiredClusters : 2;

        LOG_INFO("clustering " << features->size() << " features into " <<
                 m_desiredClusters << " clusters");

        if (getDistance().getType() == DistanceEuclidean::getTypeStatic())
            cluster<DistanceEuclidean::DistanceType>(features);
        else {
            LOG_WARN("The k-means algorithm is only defined on euclidean distance. Using other distance metrices may lead to unexpected results.");
            if (getDistance().getType() == DistanceChiSquared::getTypeStatic())
                cluster<DistanceChiSquared::DistanceType>(features);
            else
                throw RuntimeException("invalid distance type: " + getDistance().getType());
        }
    }

    std::string ClusteringKMeans::getTypeStatic()
    {
        return "KMeans";
    }

    std::string ClusteringKMeans::getType() const
    {
        return ClusteringKMeans::getTypeStatic();
    }
}
