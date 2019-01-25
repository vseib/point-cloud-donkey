/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_CODEBOOKFACTORY_H
#define ISM3D_CODEBOOKFACTORY_H

#include "../utils/factory.h"
#include "codebook.h"

namespace ism3d
{
    template <>
    Codebook* Factory<Codebook>::createByType(const std::string& type)
    {
        if (type.empty())
            return new Codebook();
        else
            return 0;
    }
}

#endif // ISM3D_CODEBOOKFACTORY_H
