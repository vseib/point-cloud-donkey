/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "clustering.h"
#include "clustering_factory.h"
#include "../utils/ism_feature.h"
#include "../utils/utils.h"
#include "../utils/distance.h"

namespace ism3d
{
    Clustering::Clustering()
        : m_distance(0)
    {
    }

    Clustering::~Clustering()
    {
    }

    float Clustering::compactness(const std::vector<std::vector<float> >& clusters,
                                  pcl::PointCloud<ISMFeature>::ConstPtr data) const
    {
        float compactness = 0.0f;
        for (unsigned int i_point = 0; i_point < (int)data->size(); i_point++)
        {
            //LOG_INFO("processing point " << i_point << " of " << (int)data->size());
            const ISMFeature& sample = data->at(i_point);

            int k_best = 0;
            float min_dist = std::numeric_limits<float>::max ();

            for (int i_cluster = 0; i_cluster < (int)clusters.size(); i_cluster++)
            {
                const std::vector<float>& cluster = clusters[i_cluster];
                float dist = getDistance()(sample.descriptor, cluster);
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

    void Clustering::clear()
    {
        m_distance = 0;
        m_indices.clear();
        m_centers.clear();
    }

    const std::vector<int>& Clustering::operator()(pcl::PointCloud<ISMFeature>::ConstPtr features, const Distance* distance)
    {
        LOG_ASSERT(distance);

        m_distance = distance;

        m_indices.clear();
        m_centers.clear();

        process(features);

        //LOG_ASSERT(m_indices.size() == descriptors->size());
        LOG_INFO("found " << m_centers.size() << " clusters");

//        float cmp = -1;
//        if(m_centers.size() < 100) // VS: skip compactness calculation: takes too long and is not needed
//        {
//            cmp = compactness(m_centers, features);
//            LOG_INFO("compactness: " << cmp);
//        }
//        else
//        {
//            LOG_INFO("compactness: -- skipped, too many clusters");
//        }

        return m_indices;
    }

    const Distance& Clustering::getDistance() const
    {
        LOG_ASSERT(m_distance);
        return *m_distance;
    }

    const std::vector<int>& Clustering::getClusterIndices() const
    {
        return m_indices;
    }

    const std::vector<std::vector<float> >& Clustering::getClusterCenters() const
    {
        return m_centers;
    }
}
