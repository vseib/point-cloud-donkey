/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2019, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_FEATURESSHORTSHOTGLOBAL_H
#define ISM3D_FEATURESSHORTSHOTGLOBAL_H

#include "features.h"

namespace ism3d
{
    /**
     * @brief The FeaturesSHORTSHOTGlobal class
     * Computes a global SHORT SHOT feature.
     * (reuses code from the Compact Geometric Features (CGF), see third_party/CGF/cgf.cpp)
     *
     * Original CGF repo: https://marckhoury.github.io/CGF/ and https://github.com/marckhoury/CGF
     */
    class FeaturesSHORTSHOTGlobal
            : public Features
    {
    public:
        FeaturesSHORTSHOTGlobal();
        ~FeaturesSHORTSHOTGlobal();

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

        std::vector<std::vector<double>> compute_descriptor(pcl::PointCloud<PointT>::ConstPtr cloud,
                                                pcl::PointCloud<PointT>::Ptr keypoints,
                                                pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames);

        void configureSphericalGrid();

        double m_radius;
        double m_min_radius;
        bool m_log_radius;
        int m_feature_dims;
        int m_r_bins;
        int m_e_bins;
        int m_a_bins;
        std::string m_bin_type;
    };
}

#endif // ISM3D_FEATURESSHORTSHOTGLOBAL_H
