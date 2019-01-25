/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */


#include "clustering_none.h"
#include "../utils/ism_feature.h"

namespace ism3d
{
    ClusteringNone::ClusteringNone()
    {
    }

    ClusteringNone::~ClusteringNone()
    {
    }

    void ClusteringNone::process(pcl::PointCloud<ISMFeature>::ConstPtr features)
    {
        // use descriptor data as clusters
        m_centers.resize(features->size());
        m_indices.resize(features->size());
        for (int i = 0; i < m_centers.size(); i++) {
            const ISMFeature& feature = features->at(i);
            m_centers[i] = feature.descriptor;
            m_indices[i] = i;
        }
    }

    std::string ClusteringNone::getTypeStatic()
    {
        return "None";
    }

    std::string ClusteringNone::getType() const
    {
        return ClusteringNone::getTypeStatic();
    }
}
