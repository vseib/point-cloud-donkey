/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2019, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_FEATURESHORTCSHOT_H
#define ISM3D_FEATURESHORTCSHOT_H

#include "features.h"

namespace ism3d
{
    /**
     * @brief The FeaturesSHORTCSHOT class
     * Computes features using the generalized Short C-SHOT (reuses code from the Compact Geometric Features (CGF), see third_party/CGF/cgf.cpp)
     *
     * Original CGF repo: https://marckhoury.github.io/CGF/ and https://github.com/marckhoury/CGF
     */
    class FeaturesSHORTCSHOT
            : public Features
    {
    public:
        FeaturesSHORTCSHOT();
        ~FeaturesSHORTCSHOT();

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

        // distribute the increment linearly between bins
        // return value:
        //  first: portion for current bin
        //  second: +1 to assign the portion (1-first) to next bin
        //          -1 to assign the portion (1-first) to previous bin
        std::pair<float,int> linearDistribution(float raw_bin_id);

        std::pair<float,int> linearDistribution2(float raw_bin_id, float bin_size, float signal_range);

        // check if bin is in valid range and correct if necessary
        int correct_bin(int bin, int total_bins, bool is_cyclic);

        void configureSphericalGrid();

        void RGB2CIELAB (unsigned char R, unsigned char G, unsigned char B, float &L, float &A, float &B2);

        static float sRGB_LUT[256];
        static float sXYZ_LUT[4000];

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

#endif // ISM3D_FEATURESHORTCSHOT_H
