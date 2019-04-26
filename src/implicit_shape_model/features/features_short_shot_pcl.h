/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_FEATURESSHORTSHOTPCL_H
#define ISM3D_FEATURESSHORTSHOTPCL_H

#include "features.h"

namespace ism3d
{
    /**
     * @brief The FeaturesSHORTSHOT class
     * Computes features using the shortened signature of histograms of orientations descriptor.
     *
     * For more details see:
     *
     * Viktor Seib, Dietrich Paulus
     * "A Low-Dimensional Feature Transform for Keypoint Matching and Classification of Point Clouds without Normal Computation"
     * 25th IEEE International Conference on Image Processing (ICIP), 2018
     * pages: 2949--2953
     *
     */

    class FeaturesSHORTSHOTPCL
            : public Features
    {
    public:
        FeaturesSHORTSHOTPCL();
        ~FeaturesSHORTSHOTPCL();

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

#endif // ISM3D_FEATURESSHORTSHOTPCL_H
