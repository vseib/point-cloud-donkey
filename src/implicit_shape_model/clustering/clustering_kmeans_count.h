/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_CLUSTERINGKMEANSCOUNT_H
#define ISM3D_CLUSTERINGKMEANSCOUNT_H

#include "clustering_kmeans.h"

namespace ism3d
{
    class ClusteringKMeansCount
            : public ClusteringKMeans
    {
    public:
        ClusteringKMeansCount();
        ~ClusteringKMeansCount();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        void process(pcl::PointCloud<ISMFeature>::ConstPtr);

    private:
        int m_clusterCount;
    };
}

#endif // ISM3D_CLUSTERINGKMEANSCOUNT_H
