/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "utils.h"
#include <Eigen/Eigenvalues>

#include "../third_party/libgdiam-1.3/gdiam.hpp"

#include <boost/math/constants/constants.hpp>

namespace ism3d
{
    Json::Value Utils::vector3fToJson(const Eigen::Vector3f& vector)
    {
        Json::Value json(Json::arrayValue);

        json.append(vector[0]);
        json.append(vector[1]);
        json.append(vector[2]);

        return json;
    }

    bool Utils::jsonToVector3f(Json::Value json, Eigen::Vector3f& vector)
    {
        if (json.isNull() || !json.isArray() || json.size() != 3)
            return false;

        vector[0] = json[0].asFloat();
        vector[1] = json[1].asFloat();
        vector[2] = json[2].asFloat();

        return true;
    }

    Json::Value Utils::quatToJson(const boost::math::quaternion<float>& quat)
    {
        Json::Value json(Json::arrayValue);

        json.append(quat.R_component_1());
        json.append(quat.R_component_2());
        json.append(quat.R_component_3());
        json.append(quat.R_component_4());

        return json;
    }

    bool Utils::jsonToQuat(Json::Value json, boost::math::quaternion<float>& quat)
    {
        if (json.isNull() || !json.isArray() || json.size() != 4)
            return false;

        quat = boost::math::quaternion<float>(json[0].asFloat(),
                json[1].asFloat(), json[2].asFloat(), json[3].asFloat());

        return true;
    }

    float Utils::ln(float x)
    {
        return log(x) / log(10);
    }

    boost::math::quaternion<float> Utils::ln(const boost::math::quaternion<float>& quat)
    {
        float w = quat.R_component_1();
        float x = quat.R_component_2();
        float y = quat.R_component_3();
        float z = quat.R_component_4();

        if (fabs(w) < 1.0f) {
            float angle = acosf(w);
            float sinAngle = sinf(angle);
            if (sinAngle >= 0.00001) {
                float coeff = angle / sinAngle;
                x *= coeff;
                y *= coeff;
                z *= coeff;
            }
        }

        boost::math::quaternion<float> result(w, x, y, z);
        return result;

        /*float qNorm = boost::math::norm(quat);
        float vNorm = sqrtf(quat.R_component_2() * quat.R_component_2() +
                            quat.R_component_3() * quat.R_component_3() +
                            quat.R_component_4() * quat.R_component_4());

        float w = ln(qNorm);
        float x = (quat.R_component_2() / vNorm) * acosf(quat.R_component_1() / qNorm);
        float y = (quat.R_component_3() / vNorm) * acosf(quat.R_component_1() / qNorm);
        float z = (quat.R_component_4() / vNorm) * acosf(quat.R_component_1() / qNorm);*/
    }

    boost::math::quaternion<float> Utils::exp(const boost::math::quaternion<float>& quat)
    {
        float w = quat.R_component_1();
        float x = quat.R_component_2();
        float y = quat.R_component_3();
        float z = quat.R_component_4();

        float angle = sqrtf(x * x + y * y + z * z);
        float sinAngle = sinf(angle);

        w = cosf(angle);

        if (fabs(sinAngle) >= 0.00001f) {
            float coeff = sinAngle / angle;
            x *= coeff;
            y *= coeff;
            z *= coeff;
        }

        boost::math::quaternion<float> result(w, x, y, z);
        return result;
    }

    float Utils::deg2rad(float deg)
    {
        return deg * (M_PI / 180.0f);
    }

    float Utils::rad2deg(float rad)
    {
        return (rad * 180.0f) / M_PI;
    }

    void Utils::getRotQuaternion(const pcl::ReferenceFrame& refFrame, boost::math::quaternion<float>& quaternion)
    {
        // transform the point from the world into the keypoint coordinate system
        Eigen::Matrix3f keypointCoord = Eigen::Matrix3f::Identity();
        keypointCoord(0, 0) = refFrame.x_axis[0];
        keypointCoord(1, 0) = refFrame.x_axis[1];
        keypointCoord(2, 0) = refFrame.x_axis[2];
        keypointCoord(0, 1) = refFrame.y_axis[0];
        keypointCoord(1, 1) = refFrame.y_axis[1];
        keypointCoord(2, 1) = refFrame.y_axis[2];
        keypointCoord(0, 2) = refFrame.z_axis[0];
        keypointCoord(1, 2) = refFrame.z_axis[1];
        keypointCoord(2, 2) = refFrame.z_axis[2];

        // create rotation quaternion
        matrix2Quat(keypointCoord, quaternion);
    }

    Eigen::Vector3f Utils::rotateInto(const Eigen::Vector3f& point, const pcl::ReferenceFrame& refFrame)
    {
        // create rotation quaternion
        boost::math::quaternion<float> rotQuat;
        getRotQuaternion(refFrame, rotQuat);

        // rotate point into coordinate system
        Eigen::Vector3f result(point);
        quatRotate(rotQuat, result);

        return result;
    }

    Eigen::Vector3f Utils::rotateBack(const Eigen::Vector3f& point, const pcl::ReferenceFrame& refFrame)
    {
        // create rotation quaternion
        boost::math::quaternion<float> rotQuat;
        getRotQuaternion(refFrame, rotQuat);

        // rotate point from coordinate system back into world
        Eigen::Vector3f result(point);
        quatRotateInv(rotQuat, result);

        return result;
    }

    std::vector<pcl::ReferenceFrame> Utils::generateFrames(const pcl::ReferenceFrame& refFrame)
    {
        // This function 4 different coordinates system by rotating the reference frame with the
        // right hand rule (see http://www.pointclouds.org/assets/uploads/cglibs13_features.pdf).

        std::vector<pcl::ReferenceFrame> frames(4, refFrame);

        // frames[0] = refFrame

        pcl::ReferenceFrame& frame1 = frames[1];
        frame1.x_axis[0] *= -1;
        frame1.x_axis[1] *= -1;
        frame1.x_axis[2] *= -1;
        frame1.y_axis[0] *= -1;
        frame1.y_axis[1] *= -1;
        frame1.y_axis[2] *= -1;

        pcl::ReferenceFrame& frame2 = frames[2];
        frame2.x_axis[0] *= -1;
        frame2.x_axis[1] *= -1;
        frame2.x_axis[2] *= -1;
        frame2.z_axis[0] *= -1;
        frame2.z_axis[1] *= -1;
        frame2.z_axis[2] *= -1;

        pcl::ReferenceFrame& frame3 = frames[3];
        frame3.y_axis[0] *= -1;
        frame3.y_axis[1] *= -1;
        frame3.y_axis[2] *= -1;
        frame3.z_axis[0] *= -1;
        frame3.z_axis[1] *= -1;
        frame3.z_axis[2] *= -1;

        return frames;
    }

    template
    Utils::BoundingBox Utils::computeAABB<PointT>(const pcl::PointCloud<PointT>::ConstPtr &model);
    template
    Utils::BoundingBox Utils::computeAABB<PointNormalT>(const pcl::PointCloud<PointNormalT>::ConstPtr &model);

    template<typename T>
    Utils::BoundingBox Utils::computeAABB(const typename pcl::PointCloud<T>::ConstPtr &model)
    {
        T minP, maxP;
        pcl::getMinMax3D(*model, minP, maxP);

        Utils::BoundingBox box;
        box.rotQuat = boost::math::quaternion<float>(1, 0, 0, 0);
        box.size = Eigen::Vector3f(maxP.x - minP.x, maxP.y - minP.y, maxP.z - minP.z);
        box.position = Eigen::Vector3f(minP.x + (box.size[0] / 2),
                minP.y + (box.size[1] / 2), minP.z + (box.size[2] / 2));
        return box;
    }

    template
    Utils::BoundingBox Utils::computeMVBB<PointT>(const pcl::PointCloud<PointT>::ConstPtr&);

    template
    Utils::BoundingBox Utils::computeMVBB<PointNormalT>(const pcl::PointCloud<PointNormalT>::ConstPtr&);

    template<typename T>
    Utils::BoundingBox Utils::computeMVBB(const typename pcl::PointCloud<T>::ConstPtr &model)
    {
        // remove nan points to avoid infinite loops
        typename pcl::PointCloud<T>::Ptr cloud(new pcl::PointCloud<T>());
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*model,*cloud, indices);

        gdiam_real* points = (gdiam_point)malloc(sizeof(gdiam_point_t) * cloud->points.size());
        assert(points != NULL);

        for (int i = 0; i < (int)cloud->size(); i++) {
            const T& point = cloud->at(i);
            points[i * 3 + 0] = point.x;
            points[i * 3 + 1] = point.y;
            points[i * 3 + 2] = point.z;
        }

        // compute minimum volume bounding box
        gdiam_point* pnt_arr = gdiam_convert((gdiam_real*)points, cloud->points.size());
        gdiam_bbox bb = gdiam_approx_mvbb(pnt_arr, cloud->points.size(), 0.0);
        free(points);

        Eigen::Vector3d minP, maxP;
        bb.get_min(&minP[0], &minP[1], &minP[2]);
        bb.get_max(&maxP[0], &maxP[1], &maxP[2]);

        Eigen::Vector3f dirX = Eigen::Vector3f(bb.get_dir(0)[0], bb.get_dir(0)[1], bb.get_dir(0)[2]);
        Eigen::Vector3f dirY = Eigen::Vector3f(bb.get_dir(1)[0], bb.get_dir(1)[1], bb.get_dir(1)[2]);
        Eigen::Vector3f dirZ = Eigen::Vector3f(bb.get_dir(2)[0], bb.get_dir(2)[1], bb.get_dir(2)[2]);
        Eigen::Vector3f size = Eigen::Vector3f(maxP[0] - minP[0], maxP[1] - minP[1], maxP[2] - minP[2]);
        Eigen::Vector3f pos = Eigen::Vector3f(minP[0] + (size[0] / 2.0f),
                minP[1] + (size[1] / 2.0f), minP[2] + (size[2] / 2.0f));

        Eigen::Matrix3f rot = Eigen::Matrix3f::Identity();
        rot(0, 0) = dirX[0];
        rot(1, 0) = dirX[1];
        rot(2, 0) = dirX[2];
        rot(0, 1) = dirY[0];
        rot(1, 1) = dirY[1];
        rot(2, 1) = dirY[2];
        rot(0, 2) = dirZ[0];
        rot(1, 2) = dirZ[1];
        rot(2, 2) = dirZ[2];

        Utils::BoundingBox box;
        Utils::matrix2Quat(rot, box.rotQuat);
        box.size = size;
        box.position = pos;
        Utils::quatRotate(box.rotQuat, box.position);

        return box;
    }

    template
    float Utils::computeCloudRadius<PointT>(const pcl::PointCloud<PointT>::Ptr &cloud);

    template
    float Utils::computeCloudRadius<PointNormalT>(const pcl::PointCloud<PointNormalT>::Ptr &cloud);

    template<typename T>
    float Utils::computeCloudRadius(const typename pcl::PointCloud<T>::Ptr &cloud)
    {
        // compute the object centroid
        Eigen::Vector4f centroid4f;
        pcl::compute3DCentroid(*cloud, centroid4f);
        Eigen::Vector3f centroid(centroid4f[0], centroid4f[1], centroid4f[2]);

        // compute radius (maximum distance of a point to centroid)
        float radius = 0.0f;
        for(const T &point : cloud->points)
        {
            Eigen::Vector3f eigpoint = point.getArray3fMap();
            float temp_radius = (eigpoint-centroid).norm();
            if(temp_radius > radius)
            {
                radius = temp_radius;
            }
        }
        return radius;
    }


    // TODO VS check if this is ever used
    float Utils::computeHingeLoss(const std::vector<float> &class_distances, const unsigned class_id)
    {
        float sum = 0;
        float true_class_dist = fabs(class_distances.at(class_id));

        for(int i = 0; i < class_distances.size(); i++)
        {
            if(class_id == i) continue;

            float class_dist = fabs(class_distances.at(i));
            sum += std::max(0.0f, class_dist - true_class_dist + 1);
        }

        return sum;
    }


    void Utils::matrix2Quat(const float* rot, float* quat)
    {
        // convert rotation matrix to quaternion (adapted from OgreQuaternion.cpp)

        float matrix[3][3] = {
            {rot[0], rot[1], rot[2]},
            {rot[3], rot[4], rot[5]},
            {rot[6], rot[7], rot[8]}
        };

        float trace = matrix[0][0] + matrix[1][1] + matrix[2][2];
        float root;

        if (trace > 0.0f) {
            root = sqrtf(trace + 1.0f);
            quat[3] = 0.5f * root;
            root = 0.5f / root;

            quat[0] = (matrix[2][1] - matrix[1][2]) * root;
            quat[1] = (matrix[0][2] - matrix[2][0]) * root;
            quat[2] = (matrix[1][0] - matrix[0][1]) * root;
        }
        else {
            static size_t next[3] = {1, 2, 0};
            size_t i = 0;
            if (matrix[1][1] > matrix[0][0])
                i = 1;
            if (matrix[2][2] > matrix[i][i])
                i = 2;
            size_t j = next[i];
            size_t k = next[j];

            root = sqrtf(matrix[i][i] - matrix[j][j] - matrix[k][k] + 1.0);
            float* apkQuat[3] = {&quat[0], &quat[1], &quat[2]};
            *apkQuat[i] = 0.5f * root;
            root = 0.5f / root;
            quat[3] = (matrix[k][j] - matrix[j][k]) * root;
            *apkQuat[j] = (matrix[j][i] + matrix[i][j]) * root;
            *apkQuat[k] = (matrix[k][i] + matrix[i][k]) * root;
        }
    }

    void Utils::matrix2Quat(const Eigen::Matrix3f& rot, boost::math::quaternion<float>& quat)
    {
        float q[4] = { quat.R_component_2(),
            quat.R_component_3(),
            quat.R_component_4(),
            quat.R_component_1() };

        matrix2Quat((float*)&rot(0, 0), q);

        quat = boost::math::quaternion<float>(q[3], q[0], q[1], q[2]);
    }

    void Utils::quat2Matrix(const float* quat, float* rot)
    {
        // convert quaternion to rotation matrix (adapted from OgreQuaternion.cpp)

        float tx = quat[0] + quat[0];
        float ty = quat[1] + quat[1];
        float tz = quat[2] + quat[2];
        float twx = tx * quat[3];
        float twy = ty * quat[3];
        float twz = tz * quat[3];
        float txx = tx * quat[0];
        float txy = ty * quat[0];
        float txz = tz * quat[0];
        float tyy = ty * quat[1];
        float tyz = tz * quat[1];
        float tzz = tz * quat[2];

        float matrix[3][3];

        matrix[0][0] = 1.0 - (tyy + tzz);
        matrix[0][1] = txy - twz;
        matrix[0][2] = txz + twy;
        matrix[1][0] = txy + twz;
        matrix[1][1] = 1.0 - (txx + tzz);
        matrix[1][2] = tyz - twx;
        matrix[2][0] = txz - twy;
        matrix[2][1] = tyz + twx;
        matrix[2][2] = 1.0 - (txx + tyy);

        rot[0] = matrix[0][0];
        rot[1] = matrix[0][1];
        rot[2] = matrix[0][2];
        rot[3] = matrix[1][0];
        rot[4] = matrix[1][1];
        rot[5] = matrix[1][2];
        rot[6] = matrix[2][0];
        rot[7] = matrix[2][1];
        rot[8] = matrix[2][2];
    }

    void Utils::quat2Matrix(const boost::math::quaternion<float>& quat, Eigen::Matrix3f& rot)
    {
        float q[4] = { quat.R_component_2(),
            quat.R_component_3(),
            quat.R_component_4(),
            quat.R_component_1() };

        quat2Matrix(q, (float*)&rot(0, 0));
    }

    void Utils::euler2Quat(boost::math::quaternion<float>& quat, float angleX, float angleY, float angleZ)
    {
        float r = deg2rad(angleX / 2.0f);
        float p = deg2rad(angleY / 2.0f);
        float y = deg2rad(angleZ / 2.0f);

        float sinp = sinf(p);
        float siny = sinf(y);
        float sinr = sinf(r);
        float cosp = cosf(p);
        float cosy = cosf(y);
        float cosr = cosf(r);

        quat = boost::math::quaternion<float>(
            cosr * cosp * cosy + sinr * sinp * siny,
            sinr * cosp * cosy - cosr * sinp * siny,
            cosr * sinp * cosy + sinr * cosp * siny,
            cosr * cosp * siny - sinr * sinp * cosy);

        // normalize
        quat /= boost::math::norm(quat);
    }

    Eigen::Vector3f Utils::quat2EulerAsVector(const boost::math::quaternion<float>& quat)
    {
        float x, y, z;
        quat2Euler(quat, x, y, z);
        return Eigen::Vector3f(x,y,z);
    }

    void Utils::quat2Euler(const boost::math::quaternion<float>& quat, float& angleX, float& angleY, float& angleZ)
    {
        boost::math::quaternion<float> myQuat = quat;
        myQuat /= boost::math::norm(myQuat);

        Eigen::Vector3f euler(0, 0, 0);

        float qW = myQuat.R_component_1();
        float qX = myQuat.R_component_2();
        float qY = myQuat.R_component_3();
        float qZ = myQuat.R_component_4();

        float test = (qW * qY - qZ * qX);
        float unit = qX * qX + qY * qY + qZ * qZ + qW * qW;

        // handle singularities
        if (test > 0.4999999f * unit) {
            euler[0] = 2.0f * atan2(qX, qW);
            euler[1] = M_PI / 2.0f;
            euler[2] = 0;
        }
        else if (test < -0.4999999f * unit) {
            euler[0] = 2.0f * atan2(qX, qW);
            euler[1] = -M_PI / 2.0f;
            euler[2] = 0;
        }
        else {
            euler[0] = atan2(2.0f * (qW * qX + qY * qZ), 1.0f - 2.0f * (qX * qX + qY * qY));
            euler[1] = asin(2.0f * test);
            euler[2] = atan2(2.0f * (qW * qZ + qX * qY), 1.0f - 2.0f * (qY * qY + qZ * qZ));
        }

        angleX = rad2deg(euler[0]);
        angleY = rad2deg(euler[1]);
        angleZ = rad2deg(euler[2]);
    }

    void Utils::axis2Quat(boost::math::quaternion<float>& quat, const Eigen::Vector3f& axis, float angle)
    {
        Eigen::Vector3f myAxis = axis;
        myAxis.normalize();

        float halfAngle = deg2rad(angle / 2.0f);
        float sinAngle = sinf(halfAngle);

        quat = boost::math::quaternion<float>(cosf(halfAngle), myAxis[0] * sinAngle,
                myAxis[1] * sinAngle, myAxis[2] * sinAngle);
    }

    void Utils::quat2Axis(const boost::math::quaternion<float>& quat, Eigen::Vector3f& axis, float& angle)
    {
        boost::math::quaternion<float> myQuat = quat;
        float qw = myQuat.R_component_1();

        // normalize
        if (qw > 1.0f)
            myQuat /= boost::math::norm(myQuat);

        angle = rad2deg(2.0f * acos(qw));
        float s = sqrtf(1.0f - qw * qw);

        if (s < 0.0001f) {
            // avoid divbyzero, any arbitrary axis is valid
            axis[0] = 0;
            axis[1] = 1;
            axis[2] = 0;
        }
        else {
            axis[0] = myQuat.R_component_2() / s;
            axis[1] = myQuat.R_component_3() / s;
            axis[2] = myQuat.R_component_4() / s;
        }
    }

    void Utils::quatRotateInv(const boost::math::quaternion<float>& quat, Eigen::Vector3f& point)
    {
        boost::math::quaternion<float> pointTemp(0, point[0], point[1], point[2]);
        pointTemp = boost::math::conj(quat) * pointTemp * quat;
        point = Eigen::Vector3f(pointTemp.R_component_2(), pointTemp.R_component_3(),
            pointTemp.R_component_4());
    }

    void Utils::quatRotate(const boost::math::quaternion<float>& quat, Eigen::Vector3f& point)
    {
        boost::math::quaternion<float> pointTemp(0, point[0], point[1], point[2]);
        pointTemp = quat * pointTemp * boost::math::conj(quat);
        point = Eigen::Vector3f(pointTemp.R_component_2(), pointTemp.R_component_3(),
            pointTemp.R_component_4());
    }

    void Utils::quatGetRotationTo(boost::math::quaternion<float>& quat,
                                  const Eigen::Vector3f& src, const Eigen::Vector3f& dest)
    {
        // Based on Stan Melax's article in Game Programming Gems

        // Copy, since cannot modify local
        Eigen::Vector3f v0 = src;
        Eigen::Vector3f v1 = dest;
        v0.normalize();
        v1.normalize();

        float d = v0.dot(v1);
        // If dot == 1, vectors are the same
        if (d >= 1.0f) {
            // identity quaternion
            quat = boost::math::quaternion<float>(1, 0, 0, 0);
        }
        else if (d < (1e-6f - 1.0f)) {
            // Generate an axis
            Eigen::Vector3f axis = Eigen::Vector3f(1, 0, 0).cross(src);
            if (axis.norm() < (1e-06 * 1e-06)) // pick another if colinear
                axis = Eigen::Vector3f(0, 1, 0).cross(src);
            axis.normalize();
            axis2Quat(quat, axis, M_PI);
        }
        else {
            float s = sqrtf((1 + d) * 2);
            float invS = 1 / s;

            Eigen::Vector3f c = v0.cross(v1);

            float x = c[0] * invS;
            float y = c[1] * invS;
            float z = c[2] * invS;
            float w = s * 0.5f;

            quat = boost::math::quaternion<float>(w, x, y, z);
            quat /= boost::math::norm(quat);
        }
    }

    void Utils::quatWeightedAverage(const std::vector<boost::math::quaternion<float> >& quaternions,
                                    const std::vector<float>& weights,
                                    boost::math::quaternion<float>& result)
    {
        Eigen::Matrix4f scatterMatrix;
        scatterMatrix.setZero();

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                float value = 0;

                #pragma omp parallel for
                for (int k = 0; k < (int)quaternions.size(); k++)
                {
                    const boost::math::quaternion<float>& quat1 = quaternions[k];
                    Eigen::Vector4f quatVec1(quat1.R_component_1(), quat1.R_component_2(), quat1.R_component_3(), quat1.R_component_4());
                    float weight = weights[k];
                    float factor = weight * quatVec1[i] * quatVec1[j];
                    #pragma omp critical
                    {
                        value += factor;
                    }
                }
                scatterMatrix(i, j) = value;
            }
        }

        Eigen::EigenSolver<Eigen::Matrix4f> solver(scatterMatrix);
        Eigen::EigenSolver<Eigen::Matrix4f>::EigenvalueType eigenvalues = solver.eigenvalues();
        Eigen::EigenSolver<Eigen::Matrix4f>::EigenvectorsType eigenvectors = solver.eigenvectors();

        int maxEigenvalueIndex = 0;
        float maxEigenvalue = 0;
        for (int i = 0; i < eigenvalues.cols(); i++) {
            const std::complex<float> eigenvalue = eigenvalues[i];
            if (eigenvalue.real() > maxEigenvalue) {
                maxEigenvalue = eigenvalue.real();
                maxEigenvalueIndex = i;
            }
        }

        Eigen::Vector4cf maxEigenvector(eigenvectors(0, maxEigenvalueIndex),
                                       eigenvectors(1, maxEigenvalueIndex),
                                       eigenvectors(2, maxEigenvalueIndex),
                                       eigenvectors(3, maxEigenvalueIndex));

        result = boost::math::quaternion<float>(maxEigenvector[0].real(), maxEigenvector[1].real(),
                maxEigenvector[2].real(), maxEigenvector[3].real());
    }
}
