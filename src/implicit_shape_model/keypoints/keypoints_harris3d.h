/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_KEYPOINTSHARRIS3D_H
#define ISM3D_KEYPOINTSHARRIS3D_H

#include "keypoints.h"

#define PCL_NO_PRECOMPILE
#include <pcl/keypoints/harris_3d.h>

namespace ism3d
{
    /**
     * @brief The KeypointsHarris3D class
     * Computes keyppoints using a 3d adapted version of the harris corner detector.
     * Several different response methods can be chosen.
     */
    class KeypointsHarris3D
            : public Keypoints
    {
    public:
        KeypointsHarris3D();
        ~KeypointsHarris3D();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        pcl::PointCloud<PointT>::ConstPtr iComputeKeypoints(pcl::PointCloud<PointT>::ConstPtr,
                                                            pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                            pcl::PointCloud<PointT>::ConstPtr,
                                                            pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                            pcl::search::Search<PointT>::Ptr);

    private:
        pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI, pcl::Normal>::ResponseMethod m_responseMethod;
        float m_radius;
        float m_threshold;
        bool m_nonMaxSupression;
        bool m_refine;
    };

    template <>
    struct JSONParameterTraits<pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI, pcl::Normal>::ResponseMethod>
    {
        static bool check(const Json::Value& object) {
            return object.isString();
        }

        static pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI, pcl::Normal>::ResponseMethod fromJson(const Json::Value& object) {
            std::string value = object.asString();
            if (value == "HARRIS")
                return pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI, pcl::Normal>::HARRIS;
            else if (value == "NOBLE")
                return pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI, pcl::Normal>::NOBLE;
            else if (value == "LOWE")
                return pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI, pcl::Normal>::LOWE;
            else if (value == "TOMASI")
                return pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI, pcl::Normal>::TOMASI;
            else if (value == "CURVATURE")
                return pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI, pcl::Normal>::CURVATURE;
            throw BadParamExceptionType<std::string>("invalid response method", value);
        }

        static Json::Value toJson(pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI, pcl::Normal>::ResponseMethod value) {
            if (value == pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI, pcl::Normal>::HARRIS)
                return Json::Value("HARRIS");
            else if (value == pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI, pcl::Normal>::NOBLE)
                return Json::Value("NOBLE");
            else if (value == pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI, pcl::Normal>::LOWE)
                return Json::Value("LOWE");
            else if (value == pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI, pcl::Normal>::TOMASI)
                return Json::Value("TOMASI");
            else if (value == pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI, pcl::Normal>::CURVATURE)
                return Json::Value("CURVATURE");
            throw BadParamExceptionType<int>("invalid response method", (int)value);
        }
    };
}

#endif // ISM3D_KEYPOINTSHARRIS3D_H
