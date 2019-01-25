/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_FEATURESDUMMY_H
#define ISM3D_FEATURESDUMMY_H

#include "features.h"

namespace ism3d
{
    /**
     * @brief The FeaturesDummy class
     * Used as a dummy feature to enable usage of old config files and data files that did not have a global feature
     */
    class FeaturesDummy
            : public Features
    {
    public:
        FeaturesDummy();
        ~FeaturesDummy();

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
    };
}

#endif // ISM3D_FEATURESDUMMY_H
