/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "ranking_uniform.h"

namespace ism3d
{
    RankingUniform::RankingUniform()
    {
    }

    RankingUniform::~RankingUniform()
    {
    }

    std::map<unsigned, std::vector<float> > RankingUniform::iComputeScores(
            const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &features)
    {
        LOG_INFO("starting uniform ranking");
        std::map<unsigned, std::vector<float> > scores;

        for(auto it : features)
        {
            unsigned class_id = it.first;
            int num_features = 0;
            for(auto it2 : it.second)
            {
                num_features += it2->size();
            }
            scores.insert({class_id, std::vector<float>(num_features, 1.0f)});
        }

        return scores;
    }

    std::string RankingUniform::getTypeStatic()
    {
        return "Uniform";
    }

    std::string RankingUniform::getType() const
    {
        return RankingUniform::getTypeStatic();
    }
}
