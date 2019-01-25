/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_RANKINGUNIFORM_H
#define ISM3D_RANKINGUNIFORM_H

#include "feature_ranking.h"

namespace ism3d
{
    /**
     * @brief The RankingUniform class
     * Assigns each feature a uniform score
     */
    class RankingUniform
            : public FeatureRanking
    {
    public:
        RankingUniform();
        ~RankingUniform();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        std::map<unsigned, std::vector<float> > iComputeScores(const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &features);

    };
}

#endif // ISM3D_RANKINGUNIFORM_H
