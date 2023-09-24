/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2019, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_short_cshot.h"

#define PCL_NO_PRECOMPILE
#include <pcl/common/angles.h>
#include <pcl/common/centroid.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/shot_lrf_omp.h>

namespace ism3d
{
    FeaturesSHORTCSHOT::FeaturesSHORTCSHOT()
    {
        addParameter(m_radius, "Radius", 0.1);
        addParameter(m_use_min_radius, "UseMinRadius", false);
        addParameter(m_min_radius_relative, "ShortShotMinRadius", 0.0);
        addParameter(m_shape_feature_dims, "ShortShotDims", 32);
        addParameter(m_color_feature_dims, "ShortColorShotDims", 32);
        addParameter(m_color_hist_size, "ShortColorShotHistSize", 15);
        addParameter(m_log_radius, "ShortShotLogRadius", false);
        addParameter(m_r_bins, "ShortShotRBins", 2);
        addParameter(m_e_bins, "ShortShotEBins", 2);
        addParameter(m_a_bins, "ShortShotABins", 8);
        addParameter(m_bin_type, "ShortShotBinType", std::string("auto"));

        // initialize look up tables here, to avoid race conditions as in PCL when initialized in method RGB2CIELAB();
        if (sRGB_LUT[0] < 0)
        {
          for (int i = 0; i < 256; i++)
          {
            float f = static_cast<float> (i) / 255.0f;
            if (f > 0.04045)
              sRGB_LUT[i] = powf ((f + 0.055f) / 1.055f, 2.4f);
            else
              sRGB_LUT[i] = f / 12.92f;
          }

          for (int i = 0; i < 4000; i++)
          {
            float f = static_cast<float> (i) / 4000.0f;
            if (f > 0.008856)
              sXYZ_LUT[i] = static_cast<float> (powf (f, 0.3333f));
            else
              sXYZ_LUT[i] = static_cast<float>((7.787 * f) + (16.0 / 116.0));
          }
        }
    }

    FeaturesSHORTCSHOT::~FeaturesSHORTCSHOT()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesSHORTCSHOT::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                         pcl::search::Search<PointT>::Ptr search)
    {
        configureSphericalGrid();
        configureSphericalColorGrid();
        m_total_feature_dims = m_shape_feature_dims + m_color_feature_dims * m_color_hist_size;

        // compute features
        std::vector<std::vector<double>> raw_features = compute_descriptor(
                    pointCloudWithoutNaNNormals, keypoints, referenceFrames);

        Eigen::Vector4d centroid;
        pcl::compute3DCentroid(*pointCloudWithoutNaNNormals, centroid);

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(raw_features.size());

        for (int i = 0; i < (int)raw_features.size(); i++)
        {
            ISMFeature& feature = features->at(i);
            std::vector<double>& raw_hist = raw_features.at(i);

            // store the descriptor
            feature.descriptor.resize(m_total_feature_dims);
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = static_cast<float>(raw_hist[j]);

            // store distance to centroid
            feature.centerDist = (keypoints->at(i).getVector3fMap() - Eigen::Vector3f(centroid.x(), centroid.y(), centroid.z())).norm();
        }

        return features;
    }

    // NOTE: the following method reuses code from CGF, see method "compute_intensities" in "third_party/cgf/cgf.cpp"
    std::vector<std::vector<double>> FeaturesSHORTCSHOT::compute_descriptor(
                                            pcl::PointCloud<PointT>::ConstPtr cloud,
                                            pcl::PointCloud<PointT>::Ptr keypoints,
                                            pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames)
    {
        std::vector<std::vector<double>> descriptors;
        descriptors.resize(keypoints->points.size());

        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
        tree->setInputCloud(cloud);

        if(m_use_min_radius)
        {
            m_min_radius_relative = m_radius * m_min_radius_relative;
        }
        else
        {
            if(m_log_radius) // if log radius is enable, min radius MUST be used
            {
                // if we are here this means no min radius is enabled, use default relative value
                m_min_radius_relative = m_radius * 0.1f;
            }
            else
            {
                m_min_radius_relative = 0.0;
            }
        }

        double ln_rmin = m_min_radius_relative == 0 ? 0 : log(m_min_radius_relative);
        double ln_rmax_rmin = m_min_radius_relative == 0 ? 0 : log(m_radius/m_min_radius_relative);

        #pragma omp parallel for num_threads(m_numThreads)
        for(int i = 0; i < keypoints->points.size(); i++)
        {
            std::vector<int> indices;
            std::vector<float> distances;

            // reference values for shape part of descriptor
            pcl::ReferenceFrame current_frame = (*referenceFrames)[i];
            Eigen::Vector4f current_frame_x (current_frame.x_axis[0], current_frame.x_axis[1], current_frame.x_axis[2], 0);
            Eigen::Vector4f current_frame_y (current_frame.y_axis[0], current_frame.y_axis[1], current_frame.y_axis[2], 0);
            Eigen::Vector4f current_frame_z (current_frame.z_axis[0], current_frame.z_axis[1], current_frame.z_axis[2], 0);

            // reference values for color part of descriptor
            // NOTE: this block taken from PCL (Point Cloud Library): cshot.hpp
            unsigned char redRef   = keypoints->points[i].r;
            unsigned char greenRef = keypoints->points[i].g;
            unsigned char blueRef  = keypoints->points[i].b;
            float LRef, aRef, bRef;
            RGB2CIELAB (redRef, greenRef, blueRef, LRef, aRef, bRef);
            LRef /= 100.0f;
            aRef /= 120.0f;
            bRef /= 120.0f;    //normalized LAB components (0<L<1, -1<a<1, -1<b<1)

            std::vector<double> shape_descriptor;
            shape_descriptor.resize(m_r_bins * m_e_bins * m_a_bins);
            std::fill(shape_descriptor.begin(), shape_descriptor.end(), 0);

            std::vector<double> color_descriptor;
            color_descriptor.resize(m_r_color_bins * m_e_color_bins * m_a_color_bins * m_color_hist_size);
            std::fill(color_descriptor.begin(), color_descriptor.end(), 0);

            tree->radiusSearch(keypoints->points[i], m_radius, indices, distances);

            for(int j = 0; j < indices.size(); j++)
            {
                if(distances[j] > 1E-15)
                {
                    Eigen::Vector4f v = cloud->points[indices[j]].getVector4fMap() - keypoints->points[i].getVector4fMap();
                    double x_l = (double)v.dot(current_frame_x);
                    double y_l = (double)v.dot(current_frame_y);
                    double z_l = (double)v.dot(current_frame_z);

                    double r = sqrt(x_l*x_l + y_l*y_l + z_l*z_l);
                    double theta = pcl::rad2deg(acos(z_l / r));
                    double phi = pcl::rad2deg(atan2(y_l, x_l));

                    compute_shape_descriptor(shape_descriptor, r, theta, phi, ln_rmin, ln_rmax_rmin);

                    // prepare for color histogram
                    // NOTE: this block taken from PCL (Point Cloud Library): cshot.hpp
                    // (with some modifications)
                    unsigned char red = cloud->points[indices[j]].r;
                    unsigned char green = cloud->points[indices[j]].g;
                    unsigned char blue = cloud->points[indices[j]].b;
                    float L, a, b;
                    RGB2CIELAB (red, green, blue, L, a, b);
                    L /= 100.0f;
                    a /= 120.0f;
                    b /= 120.0f;   //normalized LAB components (0<L<1, -1<a<1, -1<b<1)
                    double color_distance = (fabs (LRef - L) + ((fabs (aRef - a) + fabs (bRef - b)) / 2)) /3;
                    if (color_distance > 1.0)
                      color_distance = 1.0;
                    if (color_distance < 0.0)
                      color_distance = 0.0;

                    compute_color_descriptor(color_descriptor, r, theta, phi, ln_rmin, ln_rmax_rmin, color_distance);
                }
            }

            // fuse descriptors
            std::vector<double> fused_descriptor;
            fused_descriptor.insert(fused_descriptor.end(), shape_descriptor.begin(), shape_descriptor.end());
            fused_descriptor.insert(fused_descriptor.end(), color_descriptor.begin(), color_descriptor.end());

            // normalize descriptor with l2 norm
            double acc_norm = 0;
            for(int j = 0; j < fused_descriptor.size(); j++)
            {
                acc_norm += fused_descriptor[j] * fused_descriptor[j];
            }
            acc_norm = std::sqrt(acc_norm);
            for(int j = 0; j < fused_descriptor.size(); j++)
            {
                fused_descriptor[j] /= acc_norm;
            }
            descriptors[i] = fused_descriptor;
        }
        return descriptors;
    }

    void FeaturesSHORTCSHOT::compute_shape_descriptor(
            std::vector<double> &shape_descriptor,
            double r, double theta, double phi, double ln_rmin, double ln_rmax_rmin)
    {
        // compute bin for shape descriptor
        int bin_r;
        float raw_r;
        if(m_log_radius)
        {
            raw_r = (m_r_bins - 1) * (log(r) - ln_rmin) / ln_rmax_rmin + 1;
            bin_r = int(raw_r);
        }
        else
        {
            raw_r = m_r_bins * r / m_radius;
            bin_r = int(raw_r);
        }

        float raw_theta = m_e_bins * theta / 180;
        int bin_theta = int(raw_theta);
        float raw_phi = m_a_bins * (phi + 180) / 360;
        int bin_phi = int(raw_phi);

        // check primary bin range
        bin_r = bin_r >= 0 ? bin_r : 0;
        bin_r = bin_r < m_r_bins ? bin_r : m_r_bins - 1;
        bin_theta = bin_theta < m_e_bins ? bin_theta : m_e_bins - 1;
        bin_phi = bin_phi < m_a_bins ? bin_phi : m_a_bins - 1;

        // init secondary bins (for interpolation)
        int bin_r2 = bin_r;
        int bin_theta2 = bin_theta;
        int bin_phi2 = bin_phi;
        bool bin_r2_ok = false, bin_theta2_ok = false, bin_phi2_ok = false;

        // compute and check secondary bins
        auto result_r = linear_interpolation(raw_r);
        if(m_r_bins > 1)
        {
            bin_r2 = bin_r + result_r.second;
            bin_r2 = correct_bin(bin_r2, m_r_bins, false);
            if (bin_r2 != bin_r)
                bin_r2_ok = true;
        }
        auto result_theta = linear_interpolation(raw_theta);
        if(m_e_bins > 1)
        {
            bin_theta2 = bin_theta + result_theta.second;
            bin_theta2 = correct_bin(bin_theta2, m_e_bins, false);
            if (bin_theta2 != bin_theta)
                bin_theta2_ok = true;
        }
        auto result_phi = linear_interpolation(raw_phi);
        if(m_a_bins > 1)
        {
            bin_phi2 = bin_phi + result_phi.second;
            bin_phi2 = correct_bin(bin_phi2, m_a_bins, true);
            if (bin_phi2 != bin_phi)
                bin_phi2_ok = true;
        }

        // compute all possible bins and update values
        // first: bin, second: update value
        std::vector<int> bins;
        bins.push_back(bin_r + bin_theta * m_r_bins + bin_phi * m_r_bins * m_e_bins);
        if(bin_phi2_ok)
            bins.push_back(bin_r + bin_theta * m_r_bins + bin_phi2 * m_r_bins * m_e_bins);
        if(bin_theta2_ok)
            bins.push_back(bin_r + bin_theta2 * m_r_bins + bin_phi * m_r_bins * m_e_bins);
        if(bin_r2_ok)
            bins.push_back(bin_r2 + bin_theta * m_r_bins + bin_phi * m_r_bins * m_e_bins);

        // compute corresponding increments
        std::vector<float> increments;
        increments.push_back(result_r.first + result_theta.first + result_phi.first);
        if(bin_phi2_ok)
            increments.push_back(result_r.first + result_theta.first + (1-result_phi.first));
        if(bin_theta2_ok)
            increments.push_back(result_r.first + (1-result_theta.first) + result_phi.first);
        if(bin_r2_ok)
            increments.push_back((1-result_r.first) + result_theta.first + result_phi.first);

        // update bins
        for(int idx = 0; idx < bins.size(); idx++)
            shape_descriptor[bins[idx]] += (increments[idx]);
    }

    void FeaturesSHORTCSHOT::compute_color_descriptor(
            std::vector<double> &color_descriptor,
            double r, double theta, double phi, double ln_rmin,
            double ln_rmax_rmin, double color_distance)
    {
        // compute geometrical bin for color histogram
        int bin_r;
        float raw_r;
        if(m_log_radius)
        {
            raw_r = (m_r_color_bins - 1) * (log(r) - ln_rmin) / ln_rmax_rmin + 1;
            bin_r = int(raw_r);
        }
        else
        {
            raw_r = m_r_color_bins * r / m_radius;
            bin_r = int(raw_r);
        }

        float raw_theta = m_e_color_bins * theta / 180;
        int bin_theta = int(raw_theta);
        float raw_phi = m_a_color_bins * (phi + 180) / 360;
        int bin_phi = int(raw_phi);
        // bin inside the color histogram inside the geometrical bin
        float raw_c = color_distance * m_color_hist_size;
        int bin_c = int(raw_c);

        // check primary bin range
        bin_r = bin_r >= 0 ? bin_r : 0;
        bin_r = bin_r < m_r_color_bins ? bin_r : m_r_color_bins - 1;
        bin_theta = bin_theta < m_e_color_bins ? bin_theta : m_e_color_bins - 1;
        bin_phi = bin_phi < m_a_color_bins ? bin_phi : m_a_color_bins - 1;
        bin_c = bin_c < m_color_hist_size ? bin_c : m_color_hist_size - 1;

        // init secondary bins (for interpolation)
        int bin_r2 = bin_r;
        int bin_theta2 = bin_theta;
        int bin_phi2 = bin_phi;
        int bin_c2 = bin_c;
        bool bin_r2_ok = false, bin_theta2_ok = false, bin_phi2_ok = false, bin_c2_ok = false;

        // compute and check secondary bins
        auto result_r = linear_interpolation(raw_r);
        if(m_r_color_bins > 1)
        {
            bin_r2 = bin_r + result_r.second;
            bin_r2 = correct_bin(bin_r2, m_r_color_bins, false);
            if (bin_r2 != bin_r)
                bin_r2_ok = true;
        }
        auto result_theta = linear_interpolation(raw_theta);
        if(m_e_color_bins > 1)
        {
            bin_theta2 = bin_theta + result_theta.second;
            bin_theta2 = correct_bin(bin_theta2, m_e_color_bins, false);
            if (bin_theta2 != bin_theta)
                bin_theta2_ok = true;
        }
        auto result_phi = linear_interpolation(raw_phi);
        if(m_a_color_bins > 1)
        {
            bin_phi2 = bin_phi + result_phi.second;
            bin_phi2 = correct_bin(bin_phi2, m_a_color_bins, true);
            if (bin_phi2 != bin_phi)
                bin_phi2_ok = true;
        }
        auto result_c = linear_interpolation(raw_c);
        if(m_color_hist_size > 1)
        {
            bin_c2 = bin_c + result_c.second;
            bin_c2 = correct_bin(bin_c2, m_color_hist_size, false);
            if (bin_c2 != bin_c)
                bin_c2_ok = true;
        }

        // compute all possible bins
        std::vector<int> bins;
        bins.push_back(bin_c
                       + bin_r     * m_color_hist_size
                       + bin_theta * m_color_hist_size * m_r_color_bins
                       + bin_phi   * m_color_hist_size * m_r_color_bins * m_e_color_bins);
        if(bin_phi2_ok)
            bins.push_back(bin_c
                           + bin_r     * m_color_hist_size
                           + bin_theta * m_color_hist_size * m_r_color_bins
                           + bin_phi2  * m_color_hist_size * m_r_color_bins * m_e_color_bins);
        if(bin_theta2_ok)
            bins.push_back(bin_c
                           + bin_r      * m_color_hist_size
                           + bin_theta2 * m_color_hist_size * m_r_color_bins
                           + bin_phi    * m_color_hist_size * m_r_color_bins * m_e_color_bins);
        if(bin_r2_ok)
            bins.push_back(bin_c
                           + bin_r2    * m_color_hist_size
                           + bin_theta * m_color_hist_size * m_r_color_bins
                           + bin_phi   * m_color_hist_size * m_r_color_bins * m_e_color_bins);
        if(bin_c2_ok)
            bins.push_back(bin_c2
                           + bin_r     * m_color_hist_size
                           + bin_theta * m_color_hist_size * m_r_color_bins
                           + bin_phi   * m_color_hist_size * m_r_color_bins * m_e_color_bins);

        // compute corresponding increments
        std::vector<float> increments;
        increments.push_back(result_c.first + result_r.first + result_theta.first + result_phi.first);
        if(bin_phi2_ok)
            increments.push_back(result_c.first + result_r.first + result_theta.first + (1-result_phi.first));
        if(bin_theta2_ok)
            increments.push_back(result_c.first + result_r.first + (1-result_theta.first) + result_phi.first);
        if(bin_r2_ok)
            increments.push_back(result_c.first + (1-result_r.first) + result_theta.first + result_phi.first);
        if(bin_c2_ok)
            increments.push_back((1-result_c.first) + (1-result_r.first) + result_theta.first + result_phi.first);

        // update bins
        for(int idx = 0; idx < bins.size(); idx++)
            color_descriptor[bins[idx]] += (increments[idx]);
    }

    // return pair: first: increment value, second: neighboring bin (next or previous compared to the current one)
    std::pair<float,int> FeaturesSHORTCSHOT::linear_interpolation(float raw_bin_id)
    {
        float decimals = raw_bin_id - (int)raw_bin_id;
        if(decimals <= 0.5)
        {
            return {decimals+0.5, -1};
        }
        else
        {
            return {(1-decimals)+0.5,1};
        }
    }

//    std::pair<float,int> FeaturesSHORTCSHOT::linearDistribution2(float raw_bin_id,
//                                                                float bin_size,
//                                                                float signal_range)
//    {
//        float decimals = raw_bin_id - int(raw_bin_id);
//        bool lower_bin = true;
//        // decimals indicate whether the current bin position
//        // is closer to the lower or to the higher neighboring bin
//        if (decimals > 0.5)
//        {
//            decimals = 1-decimals;
//            lower_bin = false;
//        }

//        // bin_size: value range that is discretized per bin (e.g. 90 degrees)
//        // cur_range: portion of the bin_size closest to bin border
//        float cur_range = bin_size * decimals;

//        // check if a neighboring bin is affected to distribute bin updates
//        // signal_range: value range to discretized per bin (e.g. 30 degrees)
//        // if the decimal position inside the bin is further away from the bin border than
//        // half the signal range, no neighboring bins are affected
//        if(cur_range < 0.5 * signal_range)
//        {
//            // portion of the signal range that falls into neighboring bin
//            float neighbor_value_raw = 0.5 * signal_range - cur_range;
//            // normalized portion that falls into current bin
//            float update_value_normalized = (signal_range - neighbor_value_raw) / signal_range;
//            return {update_value_normalized, lower_bin ? -1 : 1};

//            // this simulates a triangle function (triangle area = 1, base = 1, height = 2)
////            float xx = neighbor_value_raw / signal_range; // portion of the base = 1 in neighboring bin
////            float yy = 1 - 2*xx*xx; // 1 - area of triangle in neighboring bin --> area in current bin
////            return {yy, lower_bin ? -1 : 1};

//        }
//        else
//        {   // only current bin is updated
//            return {1, 0};
//        }
//    }

    int FeaturesSHORTCSHOT::correct_bin(int bin, int total_bins, bool is_cyclic)
    {
        if(is_cyclic)
        {
            if(bin < 0)
                return total_bins-1;
            else if(bin >= total_bins)
                return 0;
            else
                return bin;
        }
        else
        {
            if(bin < 0)
                return 0;
            else if(bin >= total_bins)
                return total_bins-1;
            else
                return bin;
        }
    }

    void FeaturesSHORTCSHOT::configureSphericalGrid()
    {
        // automatically set bins to default configuration to match the given dimensionality
        if(m_bin_type == "auto")
        {
            if(m_shape_feature_dims == 8)
            {
                m_r_bins = 1;
                m_e_bins = 1;
                m_a_bins = 8;
            }
            else if(m_shape_feature_dims == 16)
            {
                m_r_bins = 2;
                m_e_bins = 2;
                m_a_bins = 4;
            }
            else if(m_shape_feature_dims == 24)
            {
                m_r_bins = 2;
                m_e_bins = 2;
                m_a_bins = 6;
            }
            else if(m_shape_feature_dims == 32)
            {
                m_r_bins = 2;
                m_e_bins = 2;
                m_a_bins = 8;
            }
            else if(m_shape_feature_dims == 64)
            {
                m_r_bins = 2;
                m_e_bins = 4;
                m_a_bins = 8;
            }
            else if(m_shape_feature_dims == 96)
            {
                m_r_bins = 3;
                m_e_bins = 4;
                m_a_bins = 8;
            }
            else if(m_shape_feature_dims == 128)
            {
                m_r_bins = 4;
                m_e_bins = 4;
                m_a_bins = 8;
            }
            else if(m_shape_feature_dims == 192)
            {
                m_r_bins = 6;
                m_e_bins = 4;
                m_a_bins = 8;
            }
            else if(m_shape_feature_dims == 256)
            {
                m_r_bins = 8;
                m_e_bins = 4;
                m_a_bins = 8;
            }
            else
            {
                LOG_ERROR("Unsupported Short CSHOT dimensions for automatic bin configuration: " << m_shape_feature_dims << "! Setting to 32 dimensions with default bins.");
                m_r_bins = 2;
                m_e_bins = 2;
                m_a_bins = 8;
                m_shape_feature_dims = 32;
            }
        }
        else if(m_bin_type == "manual")
        {
            // bins are read from config, do nothing, but update dimensionality
            m_shape_feature_dims = m_r_bins * m_e_bins * m_a_bins;
        }
        else
        {
            LOG_ERROR("Unsupported Short SHOT bins configuration type: " << m_bin_type << "! Setting to 32 dimensions with default bins.");
            m_r_bins = 2;
            m_e_bins = 2;
            m_a_bins = 8;
            m_shape_feature_dims = 32;
        }
    }

    void FeaturesSHORTCSHOT::configureSphericalColorGrid()
    {
        // automatically set bins to default configuration to match the given dimensionality
        // NOTE: in contrast to shape bins, here only automatically setting bins is supported
        if(m_color_feature_dims == 8)
        {
            m_r_color_bins = 1;
            m_e_color_bins = 1;
            m_a_color_bins = 8;
        }
        else if(m_color_feature_dims == 16)
        {
            m_r_color_bins = 2;
            m_e_color_bins = 2;
            m_a_color_bins = 4;
        }
        else if(m_color_feature_dims == 24)
        {
            m_r_color_bins = 2;
            m_e_color_bins = 2;
            m_a_color_bins = 6;
        }
        else if(m_color_feature_dims == 32)
        {
            m_r_color_bins = 2;
            m_e_color_bins = 2;
            m_a_color_bins = 8;
        }
        else if(m_color_feature_dims == 64)
        {
            m_r_color_bins = 2;
            m_e_color_bins = 4;
            m_a_color_bins = 8;
        }
        else if(m_color_feature_dims == 96)
        {
            m_r_color_bins = 3;
            m_e_color_bins = 4;
            m_a_color_bins = 8;
        }
        else if(m_color_feature_dims == 128)
        {
            m_r_color_bins = 4;
            m_e_color_bins = 4;
            m_a_color_bins = 8;
        }
        else
        {
            LOG_ERROR("Unsupported Short Color SHOT dimensions for bin configuration: " << m_color_feature_dims << "! Setting to 32 dimensions.");
            m_r_color_bins = 2;
            m_e_color_bins = 2;
            m_a_color_bins = 8;
            m_color_feature_dims = 32;
        }
    }

    // NOTE: next block taken from PCL (Point Cloud Library): cshot.hpp
    float FeaturesSHORTCSHOT::sRGB_LUT[256] = {- 1};
    float FeaturesSHORTCSHOT::sXYZ_LUT[4000] = {- 1};
    void FeaturesSHORTCSHOT::RGB2CIELAB (unsigned char R, unsigned char G, unsigned char B,
                                         float &L, float &A, float &B2)
    {

      float fr = sRGB_LUT[R];
      float fg = sRGB_LUT[G];
      float fb = sRGB_LUT[B];

      // Use white = D65
      const float x = fr * 0.412453f + fg * 0.357580f + fb * 0.180423f;
      const float y = fr * 0.212671f + fg * 0.715160f + fb * 0.072169f;
      const float z = fr * 0.019334f + fg * 0.119193f + fb * 0.950227f;

      float vx = x / 0.95047f;
      float vy = y;
      float vz = z / 1.08883f;

      vx = sXYZ_LUT[int(vx*4000)];
      vy = sXYZ_LUT[int(vy*4000)];
      vz = sXYZ_LUT[int(vz*4000)];

      L = 116.0f * vy - 16.0f;
      if (L > 100)
        L = 100.0f;

      A = 500.0f * (vx - vy);
      if (A > 120)
        A = 120.0f;
      else if (A <- 120)
        A = -120.0f;

      B2 = 200.0f * (vy - vz);
      if (B2 > 120)
        B2 = 120.0f;
      else if (B2<- 120)
        B2 = -120.0f;
    }

    std::string FeaturesSHORTCSHOT::getTypeStatic()
    {
        return "SHORT_CSHOT";
    }

    std::string FeaturesSHORTCSHOT::getType() const
    {
        return FeaturesSHORTCSHOT::getTypeStatic();
    }
}
