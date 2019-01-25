/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_CODEWORDDISTRIBUTIONFACTORY_H
#define ISM3D_CODEWORDDISTRIBUTIONFACTORY_H

#include "../utils/factory.h"
#include "codeword_distribution.h"

namespace ism3d
{
    template <>
    CodewordDistribution* Factory<CodewordDistribution>::createByType(const std::string& type)
    {
        if (type.empty())
            return new CodewordDistribution();
        else
            return 0;
    }
}

#endif // ISM3D_CODEWORDDISTRIBUTIONFACTORY_H
