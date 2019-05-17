/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2019, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_FEATURESGASD_H
#define ISM3D_FEATURESGASD_H

#include "features.h"
#define PCL_NO_PRECOMPILE
#include <pcl/features/gasd.h>

namespace ism3d
{
    /**
     * @brief The FeaturesGASD class
     * Computes features using the Globally Aligned Spatial Distribution descriptor.
     */
    class FeaturesGASD
            : public Features
    {
    public:
        FeaturesGASD();
        ~FeaturesGASD();

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

        template<typename T>
        void computeDescriptor(pcl::PointCloud<T> &descriptor, pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals)
        {
            // Create the GASD estimation class, and pass the input dataset to it
            pcl::GASDColorEstimation<PointT, T> gasd_est;
            gasd_est.setInputCloud(pointCloudWithoutNaNNormals);
            if(!m_use_color)
            {
                gasd_est.setColorHistsSize(0);
            }
            gasd_est.compute(descriptor);
        }

    private:

        bool m_use_color;
    };
}

#endif // ISM3D_FEATURESGASD_H
