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
    // TODO VS update doc

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

        void compute_shape_descriptor(std::vector<double> &shape_descriptor,
                double r, double theta, double phi, double ln_rmin, double ln_rmax_rmin);

        void compute_color_descriptor(std::vector<double> &color_descriptor,
                double r, double theta, double phi, double ln_rmin, double ln_rmax_rmin, double color_distance);

        // distribute the increment linearly between bins
        // return value:
        //  first: portion for current bin
        //  second: +1 to assign the portion (1-first) to next bin
        //          -1 to assign the portion (1-first) to previous bin
        std::pair<float,int> linear_interpolation(float raw_bin_id);

//        std::pair<float,int> linearDistribution2(float raw_bin_id, float bin_size, float signal_range);

        // check if bin is in valid range and correct if necessary
        int correct_bin(int bin, int total_bins, bool is_cyclic);

        void configureSphericalGrid();
        void configureSphericalColorGrid();

        // NOTE: next three lines taken from PCL (Point Cloud Library): cshot.hpp
        void RGB2CIELAB (unsigned char R, unsigned char G, unsigned char B, float &L, float &A, float &B2);
        static float sRGB_LUT[256];
        static float sXYZ_LUT[4000];

        double m_radius;
        double m_min_radius;
        double m_min_radius_relative;
        bool m_log_radius;
        bool m_use_min_radius;
        int m_total_feature_dims;
        int m_shape_feature_dims;
        int m_color_feature_dims;
        int m_color_hist_size;
        int m_r_bins;
        int m_e_bins;
        int m_a_bins;
        int m_r_color_bins;
        int m_e_color_bins;
        int m_a_color_bins;

        std::string m_bin_type;
    };
}

#endif // ISM3D_FEATURESHORTCSHOT_H
