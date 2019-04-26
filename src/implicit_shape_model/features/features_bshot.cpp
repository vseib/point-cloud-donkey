/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */


/*
 * Ideas taken from
 *
 *      https://github.com/saimanoj18/iros_bshot
 *
 * Copyright by Sai Manoj Prakhya
 *
 */


#include "features_bshot.h"

#define PCL_NO_PRECOMPILE
#include <pcl/features/shot_omp.h>
#include <pcl/common/centroid.h>


namespace ism3d
{
    FeaturesBSHOT::FeaturesBSHOT()
    {
        addParameter(m_radius, "Radius", 0.1);
    }

    FeaturesBSHOT::~FeaturesBSHOT()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesBSHOT::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                       pcl::search::Search<PointT>::Ptr search)
    {

        pcl::SHOTEstimationOMP<PointT, pcl::Normal, pcl::SHOT352> shotEst;
        Eigen::Vector4d centroid;

        if (pointCloud->isOrganized()) {
            shotEst.setSearchSurface(pointCloud);
            shotEst.setInputNormals(normals);
            pcl::compute3DCentroid(*pointCloud, centroid);
        }
        else {
            shotEst.setSearchSurface(pointCloudWithoutNaNNormals);
            shotEst.setInputNormals(normalsWithoutNaN);
            pcl::compute3DCentroid(*pointCloudWithoutNaNNormals, centroid);
        }

        shotEst.setInputCloud(keypoints);
        shotEst.setInputReferenceFrames(referenceFrames);
        shotEst.setSearchMethod(search);

        // parameters
        shotEst.setRadiusSearch(m_radius);

        // compute features
        pcl::PointCloud<pcl::SHOT352>::Ptr shotFeatures(new pcl::PointCloud<pcl::SHOT352>());
        shotEst.compute(*shotFeatures);

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(shotFeatures->size());

        for (int i = 0; i < (int)shotFeatures->size(); i++)
        {
            ISMFeature& feature = features->at(i);
            const pcl::SHOT352& shot = shotFeatures->at(i);

            // store the descriptor
            std::vector<float> temp_vec(4);
            std::vector<int> temp_vec_bin(4);
            feature.descriptor.resize(352);
            for (int j = 0; j < shot.descriptorSize(); j+=4)
            {
                temp_vec[0] = shot.descriptor[j+0];
                temp_vec[1] = shot.descriptor[j+1];
                temp_vec[2] = shot.descriptor[j+2];
                temp_vec[3] = shot.descriptor[j+3];

                temp_vec_bin = getBinaryVector(temp_vec);

                feature.descriptor[j+0] = temp_vec_bin[0];
                feature.descriptor[j+1] = temp_vec_bin[1];
                feature.descriptor[j+2] = temp_vec_bin[2];
                feature.descriptor[j+3] = temp_vec_bin[3];
            }

            // store distance to centroid
            PointT keyp = keypoints->at(i);
            feature.centerDist = (Eigen::Vector3d(keyp.x, keyp.y, keyp.z)-Eigen::Vector3d(centroid.x(), centroid.y(), centroid.z())).norm();
        }

        return features;
    }

    std::vector<int> FeaturesBSHOT::getBinaryVector(const std::vector<float> &vec)
    {
        // TODO VS: use std::bitset
        std::vector<int> result(4,0); // initialization: case A

        float sum = vec[0]+vec[1]+vec[2]+vec[3];
        if(sum != 0)
        {
            // case B
            bool case_b = false;
            if(vec[0] > sum*0.9) result[0] = 1;
            if(vec[1] > sum*0.9) result[1] = 1;
            if(vec[2] > sum*0.9) result[2] = 1;
            if(vec[3] > sum*0.9) result[3] = 1;
            if(result[0]+result[1]+result[2]+result[3] == 1) case_b = true;

            // case C
            bool case_c = false;
            if(!case_b)
            {
                if(vec[0]+vec[1] > sum*0.9) result = {1,1,0,0};
                if(vec[0]+vec[2] > sum*0.9) result = {1,0,1,0};
                if(vec[0]+vec[3] > sum*0.9) result = {1,0,0,1};
                if(vec[1]+vec[2] > sum*0.9) result = {0,1,1,0};
                if(vec[1]+vec[3] > sum*0.9) result = {0,1,0,1};
                if(vec[2]+vec[3] > sum*0.9) result = {0,0,1,1};
                if(result[0]+result[1]+result[2]+result[3] == 2) case_c = true;
            }

            // case D
            bool case_d = false;
            if(!case_b && !case_c)
            {
                if(vec[0]+vec[1]+vec[2] > sum*0.9) result = {1,1,1,0};
                if(vec[0]+vec[1]+vec[3] > sum*0.9) result = {1,1,0,1};
                if(vec[0]+vec[2]+vec[3] > sum*0.9) result = {1,0,1,1};
                if(vec[1]+vec[2]+vec[3] > sum*0.9) result = {0,1,1,1};
                if(result[0]+result[1]+result[2]+result[3] == 3) case_d = true;
            }

            // case E
            if(!case_b && !case_c && !case_d)
            {
                result = {1,1,1,1};
            }
        }

        return result;
    }

    std::string FeaturesBSHOT::getTypeStatic()
    {
        return "BSHOT";
    }

    std::string FeaturesBSHOT::getType() const
    {
        return FeaturesBSHOT::getTypeStatic();
    }
}
