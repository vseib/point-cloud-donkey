/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2021, Viktor Seib
 * All rights reserved.
 *
 */

#include "activation_strategy_knn_rule.h"

#include <flann/flann.h>

namespace ism3d
{
    ActivationStrategyKnnRule::ActivationStrategyKnnRule()
    {
        addParameter(m_k, "K", 3);
        m_is_detection = false;

        m_k = 3; // always use 3 according to the implemented rule
    }

    ActivationStrategyKnnRule::~ActivationStrategyKnnRule()
    {
    }

    std::string ActivationStrategyKnnRule::getTypeStatic()
    {
        return "KNNRule";
    }

    std::string ActivationStrategyKnnRule::getType() const
    {
        return ActivationStrategyKnnRule::getTypeStatic();
    }
}
