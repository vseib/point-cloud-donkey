/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "activation_strategy.h"
#include "activation_strategy_factory.h"

namespace ism3d
{
ActivationStrategy::ActivationStrategy()
    : m_distance(0)
{
}

ActivationStrategy::~ActivationStrategy()
{
}

std::vector<std::shared_ptr<Codeword> > ActivationStrategy::operate(const ISMFeature& feature,
                                                                      const std::vector<std::shared_ptr<Codeword> >& codewords,
                                                                      const Distance* distance)
{
    LOG_ASSERT(distance);
    m_distance = distance;
    std::vector<std::shared_ptr<Codeword> > activatedWords = activate(feature, codewords);
    return activatedWords;
}

const Distance& ActivationStrategy::distance() const
{
    LOG_ASSERT(m_distance);
    return *m_distance;
}
}
