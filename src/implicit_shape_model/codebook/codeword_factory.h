/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_CODEWORDFACTORY_H
#define ISM3D_CODEWORDFACTORY_H

#include "../utils/factory.h"
#include "codeword.h"

namespace ism3d
{
    template <>
    Codeword* Factory<Codeword>::createByType(const std::string& type)
    {
        if (type.empty())
            return new Codeword();
        else
            return 0;
    }
}

#endif // ISM3D_CODEWORDFACTORY_H
