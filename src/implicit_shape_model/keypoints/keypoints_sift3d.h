/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_KEYPOINTSSIFT3D_H
#define ISM3D_KEYPOINTSSIFT3D_H

#include "keypoints.h"

#define PCL_NO_PRECOMPILE
#include <pcl/keypoints/sift_keypoint.h>

namespace ism3d
{
    /**
     * @brief The KeypointsSIFT3D class
     * Computes keyppoints using a 3d adapted version of the SIFT keypoint detector.
     */
    class KeypointsSIFT3D
            : public Keypoints
    {
    public:
        KeypointsSIFT3D();
        ~KeypointsSIFT3D();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        pcl::PointCloud<PointT>::ConstPtr iComputeKeypoints(pcl::PointCloud<PointT>::ConstPtr,
                                                            pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                            pcl::PointCloud<PointT>::ConstPtr,
                                                            pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                            pcl::search::Search<PointT>::Ptr);

    private:
        float m_radius;
    };
}

#endif // ISM3D_KEYPOINTSSIFT3D_H
