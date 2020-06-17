/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "features.h"
#include "features_factory.h"
#include "../utils/utils.h"
#include "../utils/distance.h"

#define PCL_NO_PRECOMPILE
#include <pcl/features/board.h>

#ifdef WITH_PCL_GREATER_1_8
    #include <pcl/features/flare.h>
#endif

#include <pcl/features/shot_lrf_omp.h>
#include "../third_party/pcl_shot_na_lrf/shot_na_lrf.h"  // TODO: remove from project if it gets merged into pcl
#include "../third_party/pcl_shot_na_lrf/shot_na_lrf.hpp"

namespace ism3d
{
Features::Features()
    : m_numThreads(1)
{
    addParameter(m_referenceFrameRadius, "ReferenceFrameRadius", 0.2f);
    addParameter(m_referenceFrameType, "ReferenceFrameType", std::string("SHOT"));
}

Features::~Features()
{
}

pcl::PointCloud<ISMFeature>::ConstPtr Features::operator()(pcl::PointCloud<PointT>::ConstPtr points,
                                                           pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                           pcl::PointCloud<PointT>::ConstPtr pointsWithoutNaNNormals,
                                                           pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                           pcl::PointCloud<PointT>::ConstPtr keypoints,
                                                           pcl::search::Search<PointT>::Ptr search)
{
    LOG_ASSERT(points->size() == normals->size());

    // prepare for filtering
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr cleanReferenceFrames(new pcl::PointCloud<pcl::ReferenceFrame>());
    pcl::PointCloud<PointT>::Ptr cleanKeypoints(new pcl::PointCloud<PointT>());

    // this means we are computing a LOCAL feature, so we need reference frames and clean keypoints
    if(keypoints->size() != 0)
    {
        // compute reference frames before descriptors - there might be descriptors depending on it
        pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr referenceFrames;
        LOG_INFO("computing reference frames");

        referenceFrames = computeReferenceFrames(points, normals, pointsWithoutNaNNormals, normalsWithoutNaN, keypoints, search);
        LOG_ASSERT(referenceFrames.get() != 0);
        LOG_ASSERT(referenceFrames->size() == keypoints->size());

        // sort out invalid reference frames and associated keypoints
        unsigned missedFrames = 0;
        for (int i = 0; i < (int)referenceFrames->size(); i++) {
            const pcl::ReferenceFrame& frame = referenceFrames->at(i);
            if (std::isfinite (frame.x_axis[0]) &&
                    std::isfinite (frame.y_axis[0]) &&
                    std::isfinite (frame.z_axis[0])) {
                cleanReferenceFrames->push_back(frame);
                cleanKeypoints->push_back(keypoints->at(i));
            }
            else
                missedFrames++;
        }
        LOG_ASSERT(cleanReferenceFrames->size() == cleanKeypoints->size());

        if(missedFrames > 0)
            LOG_WARN("found " << missedFrames << " invalid reference frame(s), discarding associated keypoint(s)");
    }

    // compute descriptors
    LOG_INFO("computing descriptors at keypoint positions");
    pcl::PointCloud<ISMFeature>::Ptr features = iComputeDescriptors(points, normals,
                                                                    pointsWithoutNaNNormals, normalsWithoutNaN,
                                                                    cleanReferenceFrames, cleanKeypoints,
                                                                    search);
    // set additional information
    // first assertion for local features, second for global features
    // NOTE: CVFH and OURCVFH produce a different number of features
    if (getType() != "CVFH" && getType() != "OURCVFH")
        LOG_ASSERT(features->size() == cleanKeypoints->size() || features->size() == 1);

    LOG_INFO("obtained " << features->size() << " " << getType() << " descriptors");

    for (int i = 0; i < (int)features->size(); i++)
    {
        ISMFeature& feature = features->at(i);

        // skip if computing global featues that do not return their reference frame and keypoint
        if(cleanReferenceFrames->size() != 0)
        {
            // store keypoint position
            const PointT& keypoint = cleanKeypoints->at(i);
            feature.x = keypoint.x;
            feature.y = keypoint.y;
            feature.z = keypoint.z;

            // store reference frame
            feature.referenceFrame = cleanReferenceFrames->at(i);
        }
    }

    return features;
}

void Features::setNumThreads(int numThreads)
{
    m_numThreads = numThreads;
}

int Features::getNumThreads() const
{
    return m_numThreads;
}

float Features::getCloudRadius(pcl::PointCloud<PointT>::ConstPtr &cloud) const
{
    // compute the object centroid
    Eigen::Vector4f centroid4f;
    pcl::compute3DCentroid(*cloud, centroid4f);
    Eigen::Vector3f centroid(centroid4f[0], centroid4f[1], centroid4f[2]);

    float temp_radius = 0;

    // find farthest point
    for(int i = 0; i < cloud->size(); i++)
    {
        PointT pointPCL = cloud->at(i);
        Eigen::Vector3f pointEigen(pointPCL.x, pointPCL.y, pointPCL.z);
        Eigen::Vector3f result_point = pointEigen - centroid;

        if(result_point.norm() > temp_radius)
        {
            temp_radius = result_point.norm();
        }
    }

    return temp_radius;
}

pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr Features::computeReferenceFrames(pcl::PointCloud<PointT>::ConstPtr points,
                                                                                pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                                pcl::PointCloud<PointT>::ConstPtr pointsWithoutNaNNormals,
                                                                                pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                                pcl::PointCloud<PointT>::ConstPtr keypoints,
                                                                                pcl::search::Search<PointT>::Ptr search) const
{
    LOG_ASSERT(points->size() == normals->size());
    LOG_ASSERT(pointsWithoutNaNNormals->size() == normalsWithoutNaN->size());

    // compute reference frames at keypoint positions
    if (m_referenceFrameType == "BOARD")
        return computeBOARDReferenceFrames(points, normals, pointsWithoutNaNNormals, normalsWithoutNaN, keypoints, search);
#ifdef WITH_PCL_GREATER_1_8
    else if (m_referenceFrameType == "FLARE")
        return computeFLAREReferenceFrames(points, normals, pointsWithoutNaNNormals, normalsWithoutNaN, keypoints, search);
#endif
    else if (m_referenceFrameType == "SHOT")
        return computeSHOTReferenceFrames(pointsWithoutNaNNormals, keypoints, search);
    else if (m_referenceFrameType == "SHOTNA")
        return computeSHOTNAReferenceFrames(points, normals, pointsWithoutNaNNormals, normalsWithoutNaN, keypoints, search);
    else
        throw BadParamExceptionType<std::string>("invalid reference frame type", m_referenceFrameType);
}

pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr Features::computeBOARDReferenceFrames(pcl::PointCloud<PointT>::ConstPtr points,
                                                                                     pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                                     pcl::PointCloud<PointT>::ConstPtr pointsWithoutNaNNormals,
                                                                                     pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                                     pcl::PointCloud<PointT>::ConstPtr keypoints,
                                                                                     pcl::search::Search<PointT>::Ptr search) const
{
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames(new pcl::PointCloud<pcl::ReferenceFrame>());
    pcl::BOARDLocalReferenceFrameEstimation<PointT, pcl::Normal, pcl::ReferenceFrame> refEst;

    if (points->isOrganized()) {
        refEst.setSearchSurface(points);
        refEst.setInputNormals(normals);
    }
    else {
        refEst.setSearchSurface(pointsWithoutNaNNormals);
        refEst.setInputNormals(normalsWithoutNaN);
    }

    refEst.setRadiusSearch(m_referenceFrameRadius);
    refEst.setInputCloud(keypoints);
    refEst.setSearchMethod(search);
    refEst.compute(*referenceFrames);

    return referenceFrames;
}

#ifdef WITH_PCL_GREATER_1_8
pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr Features::computeFLAREReferenceFrames(pcl::PointCloud<PointT>::ConstPtr points,
                                                                                     pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                                     pcl::PointCloud<PointT>::ConstPtr pointsWithoutNaNNormals,
                                                                                     pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                                     pcl::PointCloud<PointT>::ConstPtr keypoints,
                                                                                     pcl::search::Search<PointT>::Ptr search) const
{
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames(new pcl::PointCloud<pcl::ReferenceFrame>());
    pcl::FLARELocalReferenceFrameEstimation<PointT, pcl::Normal, pcl::ReferenceFrame> refEst;

    if (points->isOrganized()) {
        refEst.setSearchSurface(points);
        refEst.setInputNormals(normals);
    }
    else {
        refEst.setSearchSurface(pointsWithoutNaNNormals);
        refEst.setInputNormals(normalsWithoutNaN);
    }

    refEst.setRadiusSearch(m_referenceFrameRadius);
    refEst.setTangentRadius(m_referenceFrameRadius/5.0f);
    refEst.setInputCloud(keypoints);
    refEst.setSearchMethod(search);
    refEst.compute(*referenceFrames);

    return referenceFrames;
}
#endif

pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr Features::computeSHOTReferenceFrames(pcl::PointCloud<PointT>::ConstPtr points,
                                                                                    pcl::PointCloud<PointT>::ConstPtr keypoints,
                                                                                    pcl::search::Search<PointT>::Ptr search) const
{
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames(new pcl::PointCloud<pcl::ReferenceFrame>());
    pcl::SHOTLocalReferenceFrameEstimationOMP<PointT, pcl::ReferenceFrame> refEst;

    refEst.setRadiusSearch(m_referenceFrameRadius);
    refEst.setInputCloud(keypoints);
    refEst.setSearchSurface(points);
    refEst.setSearchMethod(search);
    refEst.compute(*referenceFrames);

    return referenceFrames;
}

pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr Features::computeSHOTNAReferenceFrames(pcl::PointCloud<PointT>::ConstPtr points,
                                                                                      pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                                      pcl::PointCloud<PointT>::ConstPtr pointsWithoutNaNNormals,
                                                                                      pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                                      pcl::PointCloud<PointT>::ConstPtr keypoints,
                                                                                      pcl::search::Search<PointT>::Ptr search) const
{
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames(new pcl::PointCloud<pcl::ReferenceFrame>());
    pcl::SHOTNALocalReferenceFrameEstimation<PointT, pcl::Normal, pcl::ReferenceFrame> refEst;

    if (points->isOrganized()) {
        refEst.setSearchSurface(points);
        refEst.setInputNormals(normals);
    }
    else {
        refEst.setSearchSurface(pointsWithoutNaNNormals);
        refEst.setInputNormals(normalsWithoutNaN);
    }

    refEst.setRadiusSearch(m_referenceFrameRadius);
    refEst.setInputCloud(keypoints);
    refEst.setSearchMethod(search);
    refEst.compute(*referenceFrames);

    return referenceFrames;
}


void Features::normalizeDescriptors(pcl::PointCloud<ISMFeature>::Ptr &features) const
{
    for(ISMFeature &f : features->points)
    {
        float sum = 0;
        std::vector<float> &descr = f.descriptor;
        for(int i = 0; i < descr.size(); i++)
        {
            sum += i;
        }
        for(int i = 0; i < descr.size(); i++)
        {
            descr.at(i) /= sum;
        }
    }
}
}
