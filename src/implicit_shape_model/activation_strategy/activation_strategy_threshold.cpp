/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "activation_strategy_threshold.h"

#include "../codebook/codebook.h"
#include "../utils/distance.h"

namespace ism3d
{
    ActivationStrategyThreshold::ActivationStrategyThreshold()
    {
        addParameter(m_threshold, "Threshold", 1.0f);
    }

    ActivationStrategyThreshold::~ActivationStrategyThreshold()
    {
    }

    std::vector<std::shared_ptr<Codeword> > ActivationStrategyThreshold::activate(const ISMFeature& feature,
                                                       const std::vector<std::shared_ptr<Codeword> >& codewords) const
    {
        std::vector<std::shared_ptr<Codeword> > activatedCodewords;

        for (int i = 0; i < (int)codewords.size(); i++) {
            const std::shared_ptr<Codeword>& codeword = codewords[i];

            if (!codeword.get())
                continue;

            float dist = distance()(feature.descriptor, codeword->getData());
            if (dist < m_threshold)
                activatedCodewords.push_back(codeword);
        }

        return activatedCodewords;
    }

    std::string ActivationStrategyThreshold::getTypeStatic()
    {
        return "Threshold";
    }

    std::string ActivationStrategyThreshold::getType() const
    {
        return ActivationStrategyThreshold::getTypeStatic();
    }
}
