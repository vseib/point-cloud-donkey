/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features_esf_local.h"

#define PCL_NO_PRECOMPILE
#include <pcl/features/esf.h>


namespace ism3d
{
    FeaturesESFLocal::FeaturesESFLocal()
    {
        addParameter(m_radius, "Radius", 0.1f);
    }

    FeaturesESFLocal::~FeaturesESFLocal()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesESFLocal::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                       pcl::search::Search<PointT>::Ptr search)
    {
        // NOTE: local ESF: extract points in a radius and use them as input cloud

        // build flann dataset
        flann::Matrix<float> dataset(new float[pointCloudWithoutNaNNormals->size() * 3], pointCloudWithoutNaNNormals->size(), 3);
        for(int i = 0; i < pointCloudWithoutNaNNormals->size(); i++)
        {
            PointT p = pointCloudWithoutNaNNormals->at(i);
            dataset[i][0] = p.x;
            dataset[i][1] = p.y;
            dataset[i][2] = p.z;
        }
        flann::Index<flann::L2<float> > index(dataset, flann::KDTreeIndexParams(4));
        index.buildIndex();

        // prepare query points
        flann::Matrix<float> query(new float[keypoints->size() * 3], keypoints->size(), 3);
        for(int i = 0; i < keypoints->size(); i++)
        {
            // insert the query points
            PointT keyp = keypoints->at(i);
            query[i][0] = keyp.x;
            query[i][1] = keyp.y;
            query[i][2] = keyp.z;
        }

        // radius search
        std::vector<std::vector<int> > indices;
        std::vector<std::vector<float> > distances;
        index.radiusSearch(query, indices, distances, m_radius, flann::SearchParams(128));

        // ESF estimation object
        pcl::ESFEstimation<PointT, pcl::ESFSignature640> esf;
        // Object for storing the ESF descriptor
        pcl::PointCloud<pcl::ESFSignature640>::Ptr descriptor(new pcl::PointCloud<pcl::ESFSignature640>);
        pcl::PointCloud<pcl::ESFSignature640>::Ptr all_descriptors(new pcl::PointCloud<pcl::ESFSignature640>);

        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
        for(const std::vector<int> &idxs : indices)
        {
            cloud->clear();
            for(int idx : idxs)
                cloud->push_back(pointCloudWithoutNaNNormals->at(idx));

            esf.setInputCloud(cloud);
            esf.compute(*descriptor);
            all_descriptors->push_back(descriptor->at(0));
        }

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(all_descriptors->size());

        for (int i = 0; i < (int)all_descriptors->size(); i++)
        {
            ISMFeature& feature = features->at(i);
            const pcl::ESFSignature640& esf = all_descriptors->at(i);

            // store the descriptor
            feature.descriptor.resize(640);
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = esf.histogram[j];
        }

        return features;
    }

    std::string FeaturesESFLocal::getTypeStatic()
    {
        return "ESF_LOCAL";
    }

    std::string FeaturesESFLocal::getType() const
    {
        return FeaturesESFLocal::getTypeStatic();
    }
}
