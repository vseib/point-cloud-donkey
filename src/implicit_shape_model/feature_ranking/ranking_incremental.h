/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_RANKINGINCREMENTAL_H
#define ISM3D_RANKINGINCREMENTAL_H

#include "feature_ranking.h"
#include <pcl/kdtree/kdtree_flann.h>

namespace ism3d
{
    /**
     * @brief The RankingIncremental class
     * Assigns each feature a score based on class posterior increments
     */
    class RankingIncremental
            : public FeatureRanking
    {
    public:
        RankingIncremental();
        ~RankingIncremental();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        std::map<unsigned, std::vector<float> > iComputeScores(
                const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &features);
    };
}

#endif // ISM3D_RANKINGINCREMENTAL_H
