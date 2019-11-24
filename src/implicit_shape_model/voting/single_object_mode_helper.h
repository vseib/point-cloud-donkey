/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2019, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef SINGLE_OBJECT_MODE_HELPERS_H
#define SINGLE_OBJECT_MODE_HELPERS_H

#include "../utils/utils.h"
#include "voting.h"

namespace ism3d
{
    class SingleObjectModeHelper
    {
    public:
        static float getModelRadius(pcl::PointCloud<PointT>::ConstPtr &points,
                                    const PointT &query);

        static float getVotingSpaceSize(const std::vector<Voting::Vote>& votes,
                                        const PointT &query);

    private:
        SingleObjectModeHelper();
        ~SingleObjectModeHelper();
    };
}

#endif // SINGLE_OBJECT_MODE_HELPERS_H
