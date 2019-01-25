/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_VOTINGFACTORY_H
#define ISM3D_VOTINGFACTORY_H

#include "../utils/factory.h"
#include "voting_mean_shift.h"
#include "voting_hough_3d.h"

namespace ism3d
{
    template <>
    Voting* Factory<Voting>::createByType(const std::string& type)
    {
        if (type == VotingMeanShift::getTypeStatic())
            return new VotingMeanShift();
        else if (type == VotingHough3D::getTypeStatic())
            return new VotingHough3D();
        else
            return 0;
    }
}

#endif // ISM3D_VOTINGFACTORY_H
