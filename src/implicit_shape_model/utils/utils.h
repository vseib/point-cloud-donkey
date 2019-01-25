/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_UTILS_H
#define ISM3D_UTILS_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <jsoncpp/json/json.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/filter.h>
#include <pcl/common/common.h>
#include <boost/math/quaternion.hpp>
#include <log4cxx/logger.h>

namespace ism3d
{
    #define GET_MACRO(_1, _2, NAME, ...) NAME

    // logging
    #define LOG_WARN(message) LOG4CXX_WARN(log4cxx::Logger::getRootLogger(), message)
    #define LOG_INFO(message) LOG4CXX_INFO(log4cxx::Logger::getRootLogger(), message)
    #define LOG_ERROR(message) LOG4CXX_ERROR(log4cxx::Logger::getRootLogger(), message)
    #define LOG_DEBUG(message) LOG4CXX_DEBUG(log4cxx::Logger::getRootLogger(), message)
    #define LOG_FATAL(message) LOG4CXX_FATAL(log4cxx::Logger::getRootLogger(), message)

    #define LOG_ASSERT(...) GET_MACRO(__VA_ARGS__, LOG_ASSERT_MSG, LOG_ASSERT_COND)(__VA_ARGS__)
    #define LOG_ASSERT_MSG(condition, message) LOG4CXX_ASSERT(log4cxx::Logger::getRootLogger(), condition, message)
    #define LOG_ASSERT_COND(condition) LOG4CXX_ASSERT(log4cxx::Logger::getRootLogger(), condition, "assertion failed: " << #condition)

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointXYZRGBNormal PointNormalT;

    /**
     * @brief The Utils class
     * A pure static helper class providing helpful functions.
     */
    class Utils
    {
    public:
        struct BoundingBox
        {
            BoundingBox() {
                position = Eigen::Vector3f(0, 0, 0);
                rotQuat = boost::math::quaternion<float>(1, 0, 0, 0);
                size = Eigen::Vector3f(1, 1, 1);
            }

            Eigen::Vector3f position;
            boost::math::quaternion<float> rotQuat;
            Eigen::Vector3f size;
        };

        // json helper functions
        static Json::Value vector3fToJson(const Eigen::Vector3f&);
        static bool jsonToVector3f(Json::Value, Eigen::Vector3f&);
        static Json::Value quatToJson(const boost::math::quaternion<float>&);
        static bool jsonToQuat(Json::Value, boost::math::quaternion<float>&);

        // math
        static float ln(float x);
        static boost::math::quaternion<float> ln(const boost::math::quaternion<float>&);
        static boost::math::quaternion<float> exp(const boost::math::quaternion<float>&);
        static float deg2rad(float);
        static float rad2deg(float);

        // reference frames
        static void getRotQuaternion(const pcl::ReferenceFrame&, boost::math::quaternion<float>&);
        static Eigen::Vector3f rotateInto(const Eigen::Vector3f&, const pcl::ReferenceFrame&);
        static Eigen::Vector3f rotateBack(const Eigen::Vector3f&, const pcl::ReferenceFrame&);
        static std::vector<pcl::ReferenceFrame> generateFrames(const pcl::ReferenceFrame&);

        template<typename T>
        static BoundingBox computeAABB(const typename pcl::PointCloud<T>::ConstPtr &cloud);
        template<typename T>
        static BoundingBox computeMVBB(const typename pcl::PointCloud<T>::ConstPtr &cloud);

        // misc helper functions
        template<typename T>
        static bool containsValue(const std::vector<T> &vec, const T &val)
        {
            for(T elem : vec)
                if (elem == val) return true;

            return false;
        }
        static float computeHingeLoss(const std::vector<float> &class_distances, const unsigned class_id);

        // quaternions
        static void matrix2Quat(const float* rot, float* quat);
        static void matrix2Quat(const Eigen::Matrix3f& rot, boost::math::quaternion<float>& quat);
        static void quat2Matrix(const float* quat, float* rot);
        static void quat2Matrix(const boost::math::quaternion<float>& quat, Eigen::Matrix3f& rot);
        static void euler2Quat(boost::math::quaternion<float>& quat, float angleX, float angleY, float angleZ);
        static void quat2Euler(const boost::math::quaternion<float>& quat, float& angleX, float& angleY, float& angleZ);
        static void axis2Quat(boost::math::quaternion<float>& quat, const Eigen::Vector3f& axis, float angle);
        static void quat2Axis(const boost::math::quaternion<float>& quat, Eigen::Vector3f& axis, float& angle);
        static void quatRotate(const boost::math::quaternion<float>& quat, Eigen::Vector3f& point);
        static void quatRotateInv(const boost::math::quaternion<float>& quat, Eigen::Vector3f& point);
        static void quatGetRotationTo(boost::math::quaternion<float>& quat, const Eigen::Vector3f& src, const Eigen::Vector3f& dest);
        static bool quatWeightedAverage(const std::vector<boost::math::quaternion<float> >&,
                                        const std::vector<float>&,
                                        boost::math::quaternion<float>&);

    private:
        Utils();
        ~Utils();
    };
}

#endif // ISM3D_UTILS_H
