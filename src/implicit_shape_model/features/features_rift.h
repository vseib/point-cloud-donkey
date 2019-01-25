/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_FEATURESRIFT_H
#define ISM3D_FEATURESRIFT_H

#include "features.h"

namespace ism3d
{
    /**
     * @brief The FeaturesRIFT class
     * Computes features using the RIFT descriptor.
     */
    class FeaturesRIFT
            : public Features
    {
    public:
        FeaturesRIFT();
        ~FeaturesRIFT();

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

#endif // ISM3D_FEATURESRIFT_H
