/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "ism_feature.h"

namespace ism3d
{
    ISMFeature::ISMFeature()
        : descriptor(0)
    {
        globalDescriptorRadius = -1;
        classId = -1;
        centerDist = 0;
    }
}
