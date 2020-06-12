/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2020, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_VOTINGMAXIMUM_H
#define ISM3D_VOTINGMAXIMUM_H

#include <vector>
#include <Eigen/Core>

#include "../utils/utils.h"

namespace ism3d
{
    /**
     * @brief The VotingMaximum struct
     * A voting maximum represents a found object occurrence and is characterized
     * by a position and a bounding box. The weight defines the certainty of the
     * detection and the class id is the class type of the object.
     * Voting and maxima detection is performed for each class individually.
     */
    struct VotingMaximum
    {
        VotingMaximum() {
            position = Eigen::Vector3f(0, 0, 0);
            weight = 0;
            classId = -1;
            unsigned max_value = std::numeric_limits<unsigned>::max();
            instanceIds = {max_value};
            boundingBox.size = Eigen::Vector3f(0,0,0);
            boundingBox.position = position;
            boundingBox.rotQuat = boost::math::quaternion<float>(1, 0, 0, 0);
            globalHypothesis = {max_value, -1};
        }

        Eigen::Vector3f position;
        float weight;
        unsigned classId;
        std::vector<unsigned> instanceIds;
        Utils::BoundingBox boundingBox;
        std::vector<int> voteIndices;

        struct GlobalHypothesis
        {
            unsigned classId;
            float classWeight;
            unsigned instanceId;
            float instanceWeight;
        };

        // TODO VS temp for testing: as soon as it is final include it in maxima merging: look for all occurences of TEMP FIX THIS!
        GlobalHypothesis globalHypothesis; // pair of: class_id and score
    };
}

#endif // ISM3D_VOTINGMAXIMUM_H
