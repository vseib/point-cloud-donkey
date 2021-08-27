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
     * @brief The Vote struct
     * The internal vote representation
     */
    struct Vote
    {
        Eigen::Vector3f position;
        float weight;
        unsigned classId;
        unsigned instanceId;
        Eigen::Vector3f keypoint;       // associated keypoint position in the scene
        Eigen::Vector3f keypoint_training; // associated keypoint position from training
        Utils::BoundingBox boundingBox; // associated bounding box
        int codewordId;                 // codeword the vote belongs to

        bool operator==(const ism3d::Vote &other) const
        {
            return position.x() == other.position.x() &&
                   position.y() == other.position.y() &&
                   position.z() == other.position.z();
        }
    };

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
            instanceId = max_value;
            boundingBox.size = Eigen::Vector3f(0,0,0);
            boundingBox.position = position;
            boundingBox.rotQuat = boost::math::quaternion<float>(1, 0, 0, 0);
            globalHypothesis = {max_value, -1};
        }

        Eigen::Vector3f position;
        float weight;
        unsigned classId;
        unsigned instanceId;
        float instanceWeight;

        Utils::BoundingBox boundingBox;
        std::vector<Vote> votes;

        struct GlobalHypothesis
        {
            unsigned classId;
            float classWeight;
            unsigned instanceId;
            float instanceWeight;
        };

        bool operator<(const ism3d::VotingMaximum &other) const
        {
            return weight < other.weight;
        }

        GlobalHypothesis globalHypothesis;
    };
}

#endif // ISM3D_VOTINGMAXIMUM_H
