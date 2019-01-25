/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "activation_strategy_inn.h"

#include <flann/flann.h>

namespace ism3d
{
    ActivationStrategyINN::ActivationStrategyINN()
    {
        addParameter(m_num_iterations, "Iterations", 5);
        addParameter(m_residual_weight, "ResidualWeight", 0.01f);
    }

    ActivationStrategyINN::~ActivationStrategyINN()
    {
    }

    std::string ActivationStrategyINN::getTypeStatic()
    {
        return "INN";
    }

    std::string ActivationStrategyINN::getType() const
    {
        return ActivationStrategyINN::getTypeStatic();
    }
}
