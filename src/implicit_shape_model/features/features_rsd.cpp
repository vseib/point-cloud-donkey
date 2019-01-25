/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_rsd.h"

#define PCL_NO_PRECOMPILE
#include <pcl/features/rsd.h>


namespace ism3d
{
    FeaturesRSD::FeaturesRSD()
    {
        addParameter(m_radius, "Radius", 0.1);
        addParameter(m_use_hist, "UseFullRSDHistogram", true);
    }

    FeaturesRSD::~FeaturesRSD()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesRSD::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                         pcl::search::Search<PointT>::Ptr search)
    {
        // Object for storing the RSD descriptors for each point.
        pcl::PointCloud<pcl::PrincipalRadiiRSD>::Ptr descriptors(new pcl::PointCloud<pcl::PrincipalRadiiRSD>());

        // RSD estimation object.
        pcl::RSDEstimation<PointT, pcl::Normal, pcl::PrincipalRadiiRSD> rsd;
        rsd.setInputCloud(keypoints);
        rsd.setSearchSurface(pointCloudWithoutNaNNormals);
        rsd.setInputNormals(normalsWithoutNaN);
        rsd.setSearchMethod(search);

        // Search radius, to look for neighbors. Note: the value given here has to be
        // larger than the radius used to estimate the normals.
        rsd.setRadiusSearch(m_radius);
        // Plane radius. Any radius larger than this is considered infinite (a plane).
        rsd.setPlaneRadius(m_radius);
        // Do we want to save the full distance-angle histograms?
        if(m_use_hist)
            rsd.setSaveHistograms(true);
        else
            rsd.setSaveHistograms(false);

        rsd.compute(*descriptors);

        LOG_INFO("RSD obtained " << descriptors->size() << " descriptors!");

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());

        // store the descriptor
        if(m_use_hist)
        {
            boost::shared_ptr<std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf> > >
                    hist_descriptors = rsd.getHistograms();
            LOG_INFO("RSD obtained " << hist_descriptors->size() << " histograms!");

            features->resize(hist_descriptors->size());

            for (int i = 0; i < (int)hist_descriptors->size(); i++)
            {
                ISMFeature& feature = features->at(i);
                const Eigen::MatrixXf& rsd_hist = hist_descriptors->at(i);

                // store the descriptor
                feature.descriptor.resize(rsd_hist.size());
                for (int j = 0; j < feature.descriptor.size(); j++)
                    feature.descriptor[j] = rsd_hist(j);
            }
        }
        else
        {
            features->resize(descriptors->size());
            for (int i = 0; i < (int)descriptors->size(); i++)
            {
                ISMFeature& feature = features->at(i);
                const pcl::PrincipalRadiiRSD& rsdd = descriptors->at(i);
                feature.descriptor.resize(2);
                feature.descriptor[0] = rsdd.r_min;
                feature.descriptor[1] = rsdd.r_max;
            }
        }

        // if whole histograms are computed, it's better to normalize them before using
        if(rsd.getSaveHistograms())
            normalizeDescriptors(features);

        return features;
    }

    std::string FeaturesRSD::getTypeStatic()
    {
        return "RSD";
    }

    std::string FeaturesRSD::getType() const
    {
        return FeaturesRSD::getTypeStatic();
    }
}
