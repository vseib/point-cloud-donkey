/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2019, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_short_shot_global.h"

#define PCL_NO_PRECOMPILE
#include <pcl/features/shot_omp.h>
#include <pcl/common/angles.h>
#include <pcl/common/centroid.h>
#include <pcl/search/kdtree.h>

namespace ism3d
{
    FeaturesSHORTSHOTGlobal::FeaturesSHORTSHOTGlobal()
    {
        addParameter(m_min_radius, "ShortShotMinRadius", 0.1);
        addParameter(m_feature_dims, "ShortShotDims", 32);
        addParameter(m_log_radius, "ShortShotLogRadius", false);
        addParameter(m_r_bins, "ShortShotRBins", 1);
        addParameter(m_e_bins, "ShortShotEBins", 1);
        addParameter(m_a_bins, "ShortShotABins", 8);
        addParameter(m_bin_type, "ShortShotBinType", std::string("auto"));
    }

    FeaturesSHORTSHOTGlobal::~FeaturesSHORTSHOTGlobal()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesSHORTSHOTGlobal::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                       pcl::search::Search<PointT>::Ptr search)
    {
        configureSphericalGrid();
        m_radius = getCloudRadius(pointCloudWithoutNaNNormals);

        // for global SHORT SHOT use centroid instead of keypoints
        Eigen::Vector4f centroid4f;
        pcl::compute3DCentroid(*pointCloudWithoutNaNNormals, centroid4f);
        pcl::PointCloud<PointT>::Ptr centroid_cloud(new pcl::PointCloud<PointT>());
        PointT centroid_point;
        centroid_point.x = centroid4f[0];
        centroid_point.y = centroid4f[1];
        centroid_point.z = centroid4f[2];
        centroid_cloud->push_back(centroid_point);
        centroid_cloud->height = 1;
        centroid_cloud->width = 1;
        centroid_cloud->is_dense = false;
        keypoints = centroid_cloud;

        // for global SHORT SHOT: compute centroid reference frame
        pcl::PointCloud<pcl::ReferenceFrame>::Ptr centroid_frame(new pcl::PointCloud<pcl::ReferenceFrame>());
        pcl::SHOTLocalReferenceFrameEstimationOMP<PointT, pcl::ReferenceFrame> refEst;
        refEst.setRadiusSearch(m_radius);
        refEst.setInputCloud(centroid_cloud);
        refEst.setSearchSurface(pointCloudWithoutNaNNormals);
        refEst.setSearchMethod(search);
        refEst.compute(*centroid_frame);

        // compute features
        std::vector<std::vector<double>> raw_features = compute_descriptor(
                    pointCloudWithoutNaNNormals, keypoints, centroid_frame);

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
        }

        return features;
    }


    // NOTE: the following method reuses code from CGF, see method "compute_intensities" in "third_party/cgf/cgf.cpp"
    std::vector<std::vector<double>> FeaturesSHORTSHOTGlobal::compute_descriptor(
                                            pcl::PointCloud<PointT>::ConstPtr cloud,
                                            pcl::PointCloud<PointT>::Ptr keypoints,
                                            pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames)
    {
        std::vector<std::vector<double>> intensities;
        intensities.resize(keypoints->points.size());

        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
        tree->setInputCloud(cloud);

        double ln_rmin = log(m_min_radius);
        double ln_rmax_rmin = log(m_radius/m_min_radius);

        //#pragma omp parallel for num_threads(m_numThreads) NOTE: no need for OMP - we have only 1 keypoint
        for(int i = 0; i < keypoints->points.size(); i++)
        {
            std::vector<int> indices;
            std::vector<float> distances;
            std::vector<double> intensity;
            intensity.resize(m_r_bins * m_e_bins * m_a_bins);

            pcl::ReferenceFrame current_frame = (*referenceFrames)[i];
            Eigen::Vector4f current_frame_x (current_frame.x_axis[0], current_frame.x_axis[1], current_frame.x_axis[2], 0);
            Eigen::Vector4f current_frame_y (current_frame.y_axis[0], current_frame.y_axis[1], current_frame.y_axis[2], 0);
            Eigen::Vector4f current_frame_z (current_frame.z_axis[0], current_frame.z_axis[1], current_frame.z_axis[2], 0);

            std::fill(intensity.begin(), intensity.end(), 0);
            tree->radiusSearch(keypoints->points[i], m_radius, indices, distances);
            for(int j = 0; j < indices.size(); j++) {
                if(distances[j] > 1E-15) {
                    Eigen::Vector4f v = cloud->points[indices[j]].getVector4fMap() - keypoints->points[i].getVector4fMap();
                    double x_l = (double)v.dot(current_frame_x);
                    double y_l = (double)v.dot(current_frame_y);
                    double z_l = (double)v.dot(current_frame_z);

                    double r = sqrt(x_l*x_l + y_l*y_l + z_l*z_l);
                    double theta = pcl::rad2deg(acos(z_l / r));
                    double phi = pcl::rad2deg(atan2(y_l, x_l));

                    int bin_r;
                    if(m_log_radius)
                        bin_r = int((m_r_bins - 1) * (log(r) - ln_rmin) / ln_rmax_rmin + 1);
                    else
                        bin_r = int(m_r_bins * r / m_radius);

                    int bin_theta = int(m_e_bins * theta / 180);
                    int bin_phi = int(m_a_bins * (phi + 180) / 360);

                    bin_r = bin_r >= 0 ? bin_r : 0;
                    bin_r = bin_r < m_r_bins ? bin_r : m_r_bins - 1;
                    bin_theta = bin_theta < m_e_bins ? bin_theta : m_e_bins - 1;
                    bin_phi = bin_phi < m_a_bins ? bin_phi : m_a_bins - 1;
                    int idx = bin_r + bin_theta * m_r_bins + bin_phi * m_r_bins * m_e_bins;
                    intensity[idx] += 1;
                }
            }
            // normalize descriptor with l2 norm
            double acc_norm = 0;
            for(int j = 0; j < intensity.size(); j++)
            {
                acc_norm += intensity[j] * intensity[j];
            }
            acc_norm = std::sqrt(acc_norm);
            for(int j = 0; j < intensity.size(); j++)
            {
                intensity[j] /= acc_norm;
            }
            intensities[i] = intensity;
        }
        return intensities;
    }


    void FeaturesSHORTSHOTGlobal::configureSphericalGrid()
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
                m_r_bins = 4;
                m_e_bins = 2;
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

    std::string FeaturesSHORTSHOTGlobal::getTypeStatic()
    {
        return "SHORT_SHOT_GLOBAL";
    }

    std::string FeaturesSHORTSHOTGlobal::getType() const
    {
        return FeaturesSHORTSHOTGlobal::getTypeStatic();
    }
}
