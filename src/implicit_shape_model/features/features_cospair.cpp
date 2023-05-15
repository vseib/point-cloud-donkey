/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2023, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_cospair.h"

#define PCL_NO_PRECOMPILE
#include <pcl/common/centroid.h>
#include "../third_party/cospair/cospair.h"

namespace ism3d
{
    FeaturesCospair::FeaturesCospair()
    {
        addParameter(m_radius, "Radius", 0.1);
    }

    FeaturesCospair::~FeaturesCospair()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesCospair::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                       pcl::search::Search<PointT>::Ptr search)
    {
        // default values from paper and git-repo https://github.com/berkerlogoglu/COSPAIR
        int num_levels = 7;
        int num_bins = 9;
        int rgb_type = 5; // NOTE: 1: RGB 2: RGB - L1 3: HSV 4: HSV-L1 5: CIELab (recommened and used in paper) 6: CIELab-L1
        int num_rgb_bins = 9;

        COSPAIR cospair;
        std::vector<std::vector<float>> cospair_features;
        Eigen::Vector4d centroid;

        if (pointCloud->isOrganized())
        {
            cospair_features = cospair.ComputeCOSPAIR(pointCloud, keypoints, normals,
                                                      m_radius, num_levels, num_bins, rgb_type, num_rgb_bins);
            pcl::compute3DCentroid(*pointCloud, centroid);
        }
        else {
            cospair_features = cospair.ComputeCOSPAIR(pointCloudWithoutNaNNormals, keypoints, normalsWithoutNaN,
                                                      m_radius, num_levels, num_bins, rgb_type, num_rgb_bins);
            pcl::compute3DCentroid(*pointCloudWithoutNaNNormals, centroid);
        }

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(cospair_features.size());

        for (int i = 0; i < (int)cospair_features.size(); i++)
        {
            ISMFeature& feature = features->at(i);
            const std::vector<float>& cospair = cospair_features.at(i);

            // store the descriptor
            feature.descriptor.resize(cospair.size());
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = cospair[j];

            // store distance to centroid
            feature.centerDist = (keypoints->at(i).getVector3fMap() - Eigen::Vector3f(centroid.x(), centroid.y(), centroid.z())).norm();
        }

        return features;
    }

    std::string FeaturesCospair::getTypeStatic()
    {
        return "CoSPAIR";
    }

    std::string FeaturesCospair::getType() const
    {
        return FeaturesCospair::getTypeStatic();
    }
}
