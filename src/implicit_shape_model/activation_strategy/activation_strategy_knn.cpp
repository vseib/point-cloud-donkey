/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "activation_strategy_knn.h"

#include <flann/flann.h>

namespace ism3d
{
    ActivationStrategyKNN::ActivationStrategyKNN()
    {
        addParameter(m_k, "K", 1);
    }

    ActivationStrategyKNN::~ActivationStrategyKNN()
    {
    }

    std::string ActivationStrategyKNN::getTypeStatic()
    {
        return "KNN";
    }

    std::string ActivationStrategyKNN::getType() const
    {
        return ActivationStrategyKNN::getTypeStatic();
    }
}
