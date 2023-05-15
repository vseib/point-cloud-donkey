/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2023, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_FEATURESCOSPAIR_H
#define ISM3D_FEATURESCOSPAIR_H

#include "features.h"

namespace ism3d
{
    /**
     * @brief The FeaturesCospair class
     * Computes features using Colored Histograms of Spatial Concentric Surflet-Pairs for 3D object recognition.
     *
     * Paper: Logoglu, K. Berker, Sinan Kalkan, and Alptekin Temizel. "CoSPAIR: Colored Histograms of Spatial
     * Concentric Surflet-Pairs for 3D Object Recognition." Robotics and Autonomous Systems 75 (2016): 558-570.
     */
    class FeaturesCospair
            : public Features
    {
    public:
        FeaturesCospair();
        ~FeaturesCospair();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        pcl::PointCloud<ISMFeature>::Ptr iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr,
                                                             pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                             pcl::PointCloud<PointT>::ConstPtr,
                                                             pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                             pcl::PointCloud<pcl::ReferenceFrame>::Ptr,
                                                             pcl::PointCloud<PointT>::Ptr,
                                                             pcl::search::Search<PointT>::Ptr);

    private:

        double m_radius;
    };
}

#endif // ISM3D_FEATURESCOSPAIR_H
