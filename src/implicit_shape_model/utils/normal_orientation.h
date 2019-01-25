/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_NORMALORIENTATION_H
#define ISM3D_NORMALORIENTATION_H

#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include "utils.h"

namespace ism3d
{
    /**
     * @brief The NormalOrientation class
     * Computes consistent normal orientations for a given point cloud with associated normals.
     */

    class NormalOrientation
    {
    public:
        NormalOrientation();

        /**
         * @brief Constructor
         * @param k the number of nearest neighbors to build the riemannian graph, < 1 to use the radius instead
         * @param radius the radius for nearest neighbors to build the riemannian graph
         */
        NormalOrientation(int k);
        NormalOrientation(int k, float radius);
        NormalOrientation(float radius);
        ~NormalOrientation();

        //  Abuse SHOT reference frame for consistent normals
        //  z-axis is of reference frame comes from the 3rd eigenvector and is oriented towards higher point density
        //  reverting the z-axis should give us normals pointing out of objects
        // DRAWBACK:
        //  slow, since local reference frames for ALL points are computed
        //  NOTE: all normals at concave surfaces point inside the model
        bool processSHOTLRF(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                        pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                        pcl::PointCloud<pcl::Normal>::Ptr& orientedNormals, pcl::search::Search<PointT>::Ptr searchTree);

#ifdef USE_VCGLIB
        // normal computation and orientation using the VCG library
        void computeUsingEMST(pcl::PointCloud<PointT>::ConstPtr model,
                              pcl::PointCloud<pcl::Normal>::Ptr normals);
#endif

    private:

        int m_k;
        float m_radius;
    };
}

#endif // ISM3D_NORMALORIENTATION_H
