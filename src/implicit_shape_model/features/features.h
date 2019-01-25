/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_FEATURES_H
#define ISM3D_FEATURES_H

#include "../utils/ism_feature.h"
#include "../utils/json_object.h"
#include "../utils/utils.h"

#define PCL_NO_PRECOMPILE
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/search.h>

namespace ism3d
{
    /**
     * @brief The Features class
     * Works as a functor and computes features on keypoint positions in the input point cloud. A
     * feature is a point representation containing a position and descriptor information.
     */
    class Features
            : public JSONObject
    {
    public:
        virtual ~Features();

        /**
         * @brief Interface function to compute features on keypoint positions in the input point cloud.
         * @param points the input point cloud
         * @param normals normals for the input point cloud (normals->size() == points->size())
         * @param pointsWithoutNaNNormals the input point cloud without points with corresponding
         * NaN normals
         * @param normalsWithoutNaN normals without NaN values (normalsWithoutNaN->size() == pointsWithoutNaNNormals->size())
         * @param keypoints a point cloud containing keypoints
         * @return a point cloud containing feature representations with position and descriptor information
         */
        pcl::PointCloud<ISMFeature>::ConstPtr operator()(pcl::PointCloud<PointT>::ConstPtr points,
                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                         pcl::PointCloud<PointT>::ConstPtr pointsWithoutNaNNormals,
                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                         pcl::PointCloud<PointT>::ConstPtr keypoints,
                                                         pcl::search::Search<PointT>::Ptr search);

        /**
         * @brief Set the number of threads to use. The derived classes do not need to use it.
         * @param numThreads the number of threads to use
         */
        void setNumThreads(int numThread);

    protected:
        Features();

        int getNumThreads() const;
        virtual pcl::PointCloud<ISMFeature>::Ptr iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr,
                                                                     pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                                     pcl::PointCloud<PointT>::ConstPtr,
                                                                     pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                                     pcl::PointCloud<pcl::ReferenceFrame>::Ptr,
                                                                     pcl::PointCloud<PointT>::Ptr,
                                                                     pcl::search::Search<PointT>::Ptr) = 0;

        float getCloudRadius(pcl::PointCloud<PointT>::ConstPtr &cloud) const;

        void normalizeDescriptors(pcl::PointCloud<ISMFeature>::Ptr &features) const;

        int m_numThreads;


    private:
        pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr computeReferenceFrames(pcl::PointCloud<PointT>::ConstPtr,
                                                                              pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                                              pcl::PointCloud<PointT>::ConstPtr,
                                                                              pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                                              pcl::PointCloud<PointT>::ConstPtr,
                                                                              pcl::search::Search<PointT>::Ptr) const;
        pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr computeBOARDReferenceFrames(pcl::PointCloud<PointT>::ConstPtr,
                                                                                   pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                                                   pcl::PointCloud<PointT>::ConstPtr,
                                                                                   pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                                                   pcl::PointCloud<PointT>::ConstPtr,
                                                                                   pcl::search::Search<PointT>::Ptr) const;
#ifdef WITH_PCL_GREATER_1_8
        pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr computeFLAREReferenceFrames(pcl::PointCloud<PointT>::ConstPtr,
                                                                                   pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                                                   pcl::PointCloud<PointT>::ConstPtr,
                                                                                   pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                                                   pcl::PointCloud<PointT>::ConstPtr,
                                                                                   pcl::search::Search<PointT>::Ptr) const;
#endif
        pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr computeSHOTReferenceFrames(pcl::PointCloud<PointT>::ConstPtr,
                                                                                  pcl::PointCloud<PointT>::ConstPtr,
                                                                                  pcl::search::Search<PointT>::Ptr) const;
        pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr computeSHOTNAReferenceFrames(pcl::PointCloud<PointT>::ConstPtr,
                                                                                    pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                                                    pcl::PointCloud<PointT>::ConstPtr,
                                                                                    pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                                                    pcl::PointCloud<PointT>::ConstPtr,
                                                                                    pcl::search::Search<PointT>::Ptr) const;

        float m_referenceFrameRadius;
        std::string m_referenceFrameType;
    };
}

#endif // ISM3D_FEATURES_H
