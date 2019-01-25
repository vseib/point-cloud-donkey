/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_FEATURESSHOTGLOBAL_H
#define ISM3D_FEATURESSHOTGLOBAL_H

#include "features.h"

namespace ism3d
{
    /**
     * @brief The FeaturesSHOTGlobal class
     * Computes a global feature using the signature of histograms of orientations descriptor.
     */
    class FeaturesSHOTGlobal
            : public Features
    {
    public:
        FeaturesSHOTGlobal();
        ~FeaturesSHOTGlobal();

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
    };
}

#endif // ISM3D_FEATURESSHOTGLOBAL_H
