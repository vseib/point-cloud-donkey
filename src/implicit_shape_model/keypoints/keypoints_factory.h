/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_KEYPOINTSFACTORY_H
#define ISM3D_KEYPOINTSFACTORY_H

#include "../utils/factory.h"
#include "keypoints_harris3d.h"
#include "keypoints_iss3d.h"
#include "keypoints_voxel_grid.h"
#include "keypoints_sift3d.h"

namespace ism3d
{
    template <>
    Keypoints* Factory<Keypoints>::createByType(const std::string& type)
    {
        if (type == KeypointsHarris3D::getTypeStatic())
            return new KeypointsHarris3D();
        else if (type == KeypointsISS3D::getTypeStatic())
            return new KeypointsISS3D();
        else if (type == KeypointsVoxelGrid::getTypeStatic())
            return new KeypointsVoxelGrid();
        else if (type == KeypointsSIFT3D::getTypeStatic())
            return new KeypointsSIFT3D();
        else
            return 0;
    }
}

#endif // ISM3D_KEYPOINTSFACTORY_H
