/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_FEATURESRSD_H
#define ISM3D_FEATURESRSD_H

#include "features.h"

namespace ism3d
{
    /**
     * @brief The FeaturesRSD class
     * Computes features using the  Radius-Based Surface Descriptor.
     */
    class FeaturesRSD
            : public Features
    {
    public:
        FeaturesRSD();
        ~FeaturesRSD();

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
        bool m_use_hist;
    };
}

#endif // ISM3D_FEATURESRSD_H
