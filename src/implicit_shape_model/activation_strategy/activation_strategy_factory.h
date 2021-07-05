/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_ACTIVATIONSTRATEGYFACTORY_H
#define ISM3D_ACTIVATIONSTRATEGYFACTORY_H

#include "../utils/factory.h"
#include "activation_strategy_threshold.h"
#include "activation_strategy_knn.h"
#include "activation_strategy_knn_rule.h"
#include "activation_strategy_inn.h"

namespace ism3d
{
    template <>
    ActivationStrategy* Factory<ActivationStrategy>::createByType(const std::string& type)
    {
        if (type == ActivationStrategyKNN::getTypeStatic())
            return new ActivationStrategyKNN();
        else if (type == ActivationStrategyKnnRule::getTypeStatic())
            return new ActivationStrategyKnnRule();
        else if (type == ActivationStrategyINN::getTypeStatic())
            return new ActivationStrategyINN();
        else if (type == ActivationStrategyThreshold::getTypeStatic())
            return new ActivationStrategyThreshold();
        else
            return 0;
    }
}

#endif // ISM3D_ACTIVATIONSTRATEGYFACTORY_H
