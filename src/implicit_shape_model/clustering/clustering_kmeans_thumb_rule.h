/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef CLUSTERING_KMEANS_THUMB_RULE_H
#define CLUSTERING_KMEANS_THUMB_RULE_H

#include "clustering_kmeans.h"

namespace ism3d
{
    class ClusteringKMeansThumbRule
            : public ClusteringKMeans
    {
    public:
        ClusteringKMeansThumbRule();
        ~ClusteringKMeansThumbRule();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        void process(pcl::PointCloud<ISMFeature>::ConstPtr);
    };
}

#endif // CLUSTERING_KMEANS_THUMB_RULE_H
