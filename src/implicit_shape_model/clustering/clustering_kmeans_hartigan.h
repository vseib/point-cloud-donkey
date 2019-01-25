/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_CLUSTERINGKMEANSHARTIGAN_H
#define ISM3D_CLUSTERINGKMEANSHARTIGAN_H

#include "clustering_kmeans.h"
#include "../utils/distance.h"

namespace ism3d
{
    class ClusteringKMeansHartigan
            : public ClusteringKMeans
    {
    public:
        ClusteringKMeansHartigan();
        ~ClusteringKMeansHartigan();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        void process(pcl::PointCloud<ISMFeature>::ConstPtr);

    private:
        float withinClusterSumOfSquares(const std::vector<std::vector<float> >&,
                                        pcl::PointCloud<ISMFeature>::ConstPtr);

        DistanceEuclidean m_dist;
        int m_maxK;
    };
}

#endif // ISM3D_CLUSTERINGKMEANSHARTIGAN_H
