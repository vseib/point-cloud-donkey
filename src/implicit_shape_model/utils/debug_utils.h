/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_DEBUG_UTILS_H
#define ISM3D_DEBUG_UTILS_H

#include <string>
#include <map>
#include <vector>

#include <pcl/point_cloud.h>

#include "ism_feature.h"
#include "utils.h"

namespace ism3d
{
    /**
     * @brief The DebugUtils class
     * A pure static helper class providing helpful functions for debugging.
     */
    class DebugUtils
    {
    public:

        // get points that represent the box corners with correct rotation
        // indermediate points can be added for each of the box' edges
        static pcl::PointCloud<ism3d::PointNormalT>::Ptr getBoxCorners(const Utils::BoundingBox &box, int num_intermediate_points);

        // this method is used in feature ranking to write features to a file on disk
        static void writeToFile(std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> &features, std::string filename);
        // this method is used in feature ranking to write feature indices and scores to dsik
        static void writeOutForDebug(const std::map<unsigned, std::vector<std::pair<int, float>>> &sorted_list,
                                     const std::string &type);

    private:
        DebugUtils();
        ~DebugUtils();
    };
}

#endif // ISM3D_DEBUG_UTILS_H
