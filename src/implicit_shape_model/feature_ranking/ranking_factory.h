/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_RANKINGFACTORY_H
#define ISM3D_RANKINGFACTORY_H

#include "../utils/factory.h"
#include "ranking_uniform.h"
#include "ranking_naive_bayes.h"
#include "ranking_incremental.h"
#include "ranking_knn_activation.h"
#include "ranking_strangeness.h"

namespace ism3d
{
    template <>
    FeatureRanking* Factory<FeatureRanking>::createByType(const std::string& type)
    {
        if (type == RankingUniform::getTypeStatic())
            return new RankingUniform();
        else if (type == RankingNaiveBayes::getTypeStatic())
            return new RankingNaiveBayes();
        else if (type == RankingIncremental::getTypeStatic())
            return new RankingIncremental();
        else if (type == RankingKnnActivation::getTypeStatic())
            return new RankingKnnActivation();
        else if (type == RankingStrangeness::getTypeStatic())
            return new RankingStrangeness();
        else
            return 0;
    }
}

#endif // ISM3D_RANKINGFACTORY_H
