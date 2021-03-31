/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_KEYPOINTSISS3D_H
#define ISM3D_KEYPOINTSISS3D_H

#include "keypoints.h"

namespace ism3d
{
    /**
     * @brief The KeypointsISS3D class
     * Computes keypoints using intrinsic shape signatures.
     */
    class KeypointsISS3D
            : public Keypoints
    {
    public:
        KeypointsISS3D();
        ~KeypointsISS3D();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        pcl::PointCloud<PointT>::ConstPtr iComputeKeypoints(pcl::PointCloud<PointT>::ConstPtr points,
                                                            pcl::PointCloud<PointT>::ConstPtr eigenValues,
                                                            pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                            pcl::PointCloud<PointT>::Ptr pointsWithoutNaNNormals,
                                                            pcl::PointCloud<PointT>::Ptr eigenValuesWithoutNan,
                                                            pcl::PointCloud<pcl::Normal>::Ptr normalsWithoutNaN,
                                                            pcl::search::Search<PointT>::Ptr search);

    private:
        double m_salientRadius;
        double m_nonMaxRadius;
        double m_gamma21;
        double m_gamma32;
        int m_minNeighbors;
    };
}

#endif // ISM3D_KEYPOINTSISS3D_H
