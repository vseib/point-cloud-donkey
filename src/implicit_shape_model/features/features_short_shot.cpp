/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2019, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_short_shot.h"

#define PCL_NO_PRECOMPILE
#include <pcl/common/angles.h>
#include <pcl/common/centroid.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/shot_lrf_omp.h>

namespace ism3d
{
    FeaturesSHORTSHOT::FeaturesSHORTSHOT()
    {
        addParameter(m_radius, "Radius", 0.1);
        addParameter(m_use_min_radius, "UseMinRadius", false);
        addParameter(m_min_radius_relative, "ShortShotMinRadius", 0.0);
        addParameter(m_feature_dims, "ShortShotDims", 32);
        addParameter(m_log_radius, "ShortShotLogRadius", false);
        addParameter(m_r_bins, "ShortShotRBins", 2);
        addParameter(m_e_bins, "ShortShotEBins", 2);
        addParameter(m_a_bins, "ShortShotABins", 8);
        addParameter(m_bin_type, "ShortShotBinType", std::string("auto"));
    }

    FeaturesSHORTSHOT::~FeaturesSHORTSHOT()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesSHORTSHOT::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                         pcl::search::Search<PointT>::Ptr search)
    {
        configureSphericalGrid();

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
            feature.descriptor.resize(m_feature_dims);
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = static_cast<float>(raw_hist[j]);

            // store distance to centroid
            feature.centerDist = (keypoints->at(i).getVector3fMap() - Eigen::Vector3f(centroid.x(), centroid.y(), centroid.z())).norm();
        }

        return features;
    }

    // NOTE: the following method reuses code from CGF, see method "compute_intensities" in "third_party/cgf/cgf.cpp"
    std::vector<std::vector<double>> FeaturesSHORTSHOT::compute_descriptor(
                                            pcl::PointCloud<PointT>::ConstPtr cloud,
                                            pcl::PointCloud<PointT>::Ptr keypoints,
                                            pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames)
    {
        std::vector<std::vector<double>> intensities;
        intensities.resize(keypoints->points.size());

        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
        tree->setInputCloud(cloud);

        if(m_use_min_radius)
            m_min_radius = m_radius * m_min_radius_relative;
        else
            m_min_radius = 0.0;

        double ln_rmin = m_min_radius == 0 ? 0 : log(m_min_radius);
        double ln_rmax_rmin = m_min_radius == 0 ? 0 : log(m_radius/m_min_radius);

        #pragma omp parallel for num_threads(m_numThreads)
        for(int i = 0; i < keypoints->points.size(); i++)
        {
            std::vector<int> indices;
            std::vector<float> distances;

            pcl::ReferenceFrame current_frame = (*referenceFrames)[i];
            Eigen::Vector4f current_frame_x (current_frame.x_axis[0], current_frame.x_axis[1], current_frame.x_axis[2], 0);
            Eigen::Vector4f current_frame_y (current_frame.y_axis[0], current_frame.y_axis[1], current_frame.y_axis[2], 0);
            Eigen::Vector4f current_frame_z (current_frame.z_axis[0], current_frame.z_axis[1], current_frame.z_axis[2], 0);

            std::vector<double> shape_descriptor;
            shape_descriptor.resize(m_r_bins * m_e_bins * m_a_bins);
            std::fill(shape_descriptor.begin(), shape_descriptor.end(), 0);

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
                    if(r < m_min_radius) continue;
                    double theta = pcl::rad2deg(acos(z_l / r));
                    double phi = pcl::rad2deg(atan2(y_l, x_l));

                    compute_shape_descriptor(shape_descriptor, r, theta, phi, ln_rmin, ln_rmax_rmin);
                }
            }
            // normalize descriptor with l2 norm
            double acc_norm = 0;
            for(int j = 0; j < shape_descriptor.size(); j++)
            {
                acc_norm += shape_descriptor[j] * shape_descriptor[j];
            }
            acc_norm = std::sqrt(acc_norm);
            for(int j = 0; j < shape_descriptor.size(); j++)
            {
                shape_descriptor[j] /= acc_norm;
            }
            intensities[i] = shape_descriptor;
        }
        return intensities;
    }


    void FeaturesSHORTSHOT::compute_shape_descriptor(
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

        // compute all possible bins
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


    std::pair<float,int> FeaturesSHORTSHOT::linear_interpolation(float raw_bin_id)
    {
        // decimals indicates where inside the bin the value lies (e.g. 0.5 -> center of bin)
        float decimals = raw_bin_id - (int)raw_bin_id;

        // determine update values
        if(decimals <= 0.5)
        {
            return {decimals+0.5, -1};
        }
        else
        {
            return {(1-decimals)+0.5,1};
        }
    }


    int FeaturesSHORTSHOT::correct_bin(int bin, int total_bins, bool is_cyclic)
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

    void FeaturesSHORTSHOT::configureSphericalGrid()
    {
        // automatically set bins to default configuration to match the given dimensionality
        if(m_bin_type == "auto")
        {
            if(m_feature_dims == 8)
            {
                m_r_bins = 1;
                m_e_bins = 1;
                m_a_bins = 8;
            }
            else if(m_feature_dims == 16)
            {
                m_r_bins = 2;
                m_e_bins = 2;
                m_a_bins = 4;
            }
            else if(m_feature_dims == 24)
            {
                m_r_bins = 2;
                m_e_bins = 2;
                m_a_bins = 6;
            }
            else if(m_feature_dims == 32)
            {
                m_r_bins = 2;
                m_e_bins = 2;
                m_a_bins = 8;
            }
            else if(m_feature_dims == 64)
            {
                m_r_bins = 2;
                m_e_bins = 4;
                m_a_bins = 8;
            }
            else if(m_feature_dims == 96)
            {
                m_r_bins = 3;
                m_e_bins = 4;
                m_a_bins = 8;
            }
            else if(m_feature_dims == 128)
            {
                m_r_bins = 4;
                m_e_bins = 4;
                m_a_bins = 8;
            }
            else if(m_feature_dims == 192)
            {
                m_r_bins = 6;
                m_e_bins = 4;
                m_a_bins = 8;
            }
            else if(m_feature_dims == 256)
            {
                m_r_bins = 8;
                m_e_bins = 4;
                m_a_bins = 8;
            }
            else
            {
                LOG_ERROR("Unsupported Short SHOT dimensions for automatic bin configuration: " << m_feature_dims << "! Setting to 32 dimensions with default bins.");
                m_r_bins = 2;
                m_e_bins = 2;
                m_a_bins = 8;
                m_feature_dims = 32;
            }
        }
        else if(m_bin_type == "manual")
        {
            // bins are read from config, do nothing, but update dimensionality
            m_feature_dims = m_r_bins * m_e_bins * m_a_bins;
        }
        else
        {
            LOG_ERROR("Unsupported Short SHOT bins configuration type: " << m_bin_type << "! Setting to 32 dimensions with default bins.");
            m_r_bins = 2;
            m_e_bins = 2;
            m_a_bins = 8;
            m_feature_dims = 32;
        }
    }

    std::string FeaturesSHORTSHOT::getTypeStatic()
    {
        return "SHORT_SHOT";
    }

    std::string FeaturesSHORTSHOT::getType() const
    {
        return FeaturesSHORTSHOT::getTypeStatic();
    }
}
