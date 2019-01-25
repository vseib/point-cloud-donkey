/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_FEATURESFPFH_H
#define ISM3D_FEATURESFPFH_H

#include "features.h"

namespace ism3d
{
    /**
     * @brief The FeaturesFPFH class
     * Computes features using the fast point feature histograms descriptor.
     */
    class FeaturesFPFH
            : public Features
    {
    public:
        FeaturesFPFH();
        ~FeaturesFPFH();

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

#endif // ISM3D_FEATURESFPFH_H
