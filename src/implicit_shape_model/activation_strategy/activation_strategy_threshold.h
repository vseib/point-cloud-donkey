/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_ACTIVATIONSTRATEGYTHRESHOLD_H
#define ISM3D_ACTIVATIONSTRATEGYTHRESHOLD_H

#include "activation_strategy.h"

namespace ism3d
{
    /**
     * @brief The ActivationStrategyThreshold class
     * Activates the codewords that have a matching distance to the feature below a threshold.
     */
    class ActivationStrategyThreshold
            : public ActivationStrategy
    {
    public:
        ActivationStrategyThreshold();
        ~ActivationStrategyThreshold();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:

        std::vector<std::shared_ptr<Codeword> > activate(const ISMFeature& feature,
                                                           const std::vector<std::shared_ptr<Codeword> >& codewords) const;
    private:
        float m_threshold;
    };
}

#endif // ISM3D_ACTIVATIONSTRATEGYTHRESHOLD_H
