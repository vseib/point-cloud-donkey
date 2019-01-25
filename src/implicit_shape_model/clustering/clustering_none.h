/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_CLUSTERINGNONE_H
#define ISM3D_CLUSTERINGNONE_H

#include "clustering.h"

namespace ism3d
{
    /**
     * @brief The ClusteringNone class
     * Performs no clustering at all and generates an output cluster for each input feature.
     */
    class ClusteringNone
        : public Clustering
    {
    public:
        ClusteringNone();
        ~ClusteringNone();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        void process(pcl::PointCloud<ISMFeature>::ConstPtr);
    };
}

#endif // ISM3D_CLUSTERINGNONE_H
