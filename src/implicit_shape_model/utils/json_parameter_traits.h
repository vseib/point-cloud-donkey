/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_JSONPARAMETERTRAITS_H
#define ISM3D_JSONPARAMETERTRAITS_H

#include <string>
#include <jsoncpp/json/json.h>
#include <Eigen/Core>
#include <flann/defines.h>
#include "exception.h"

namespace ism3d
{
    template <typename T>
    struct JSONParameterTraits
    {
    };

    template <>
    struct JSONParameterTraits<int>
    {
        static bool check(const Json::Value& object) {
            return object.isNumeric();
        }

        static int fromJson(const Json::Value& object) {
            return object.asInt();
        }

        static Json::Value toJson(int value) {
            return Json::Value(value);
        }
    };

    template <>
    struct JSONParameterTraits<float>
    {
        static bool check(const Json::Value& object) {
            return object.isNumeric();
        }

        static float fromJson(const Json::Value& object) {
            return object.asFloat();
        }

        static Json::Value toJson(float value) {
            return Json::Value(value);
        }
    };

    template <>
    struct JSONParameterTraits<double>
    {
        static bool check(const Json::Value& object) {
            return object.isNumeric();
        }

        static double fromJson(const Json::Value& object) {
            return object.asDouble();
        }

        static Json::Value toJson(double value) {
            return Json::Value(value);
        }
    };

    template <>
    struct JSONParameterTraits<bool>
    {
        static bool check(const Json::Value& object) {
            return object.isBool();
        }

        static bool fromJson(const Json::Value& object) {
            return object.asBool();
        }

        static Json::Value toJson(bool value) {
            return Json::Value(value);
        }
    };

    template <>
    struct JSONParameterTraits<std::string>
    {
        static bool check(const Json::Value& object) {
            return object.isString();
        }

        static std::string fromJson(const Json::Value& object) {
            return object.asString();
        }

        static Json::Value toJson(std::string value) {
            return Json::Value(value);
        }
    };

    template <>
    struct JSONParameterTraits<Eigen::Vector3d>
    {
        static bool check(const Json::Value& object) {
            return object.isArray() && object.size() == 3 &&
                    object[0].isNumeric() && object[1].isNumeric() &&
                    object[2].isNumeric();
        }

        static Eigen::Vector3d fromJson(const Json::Value& object) {
            return Eigen::Vector3d(object[0].asDouble(),
                    object[1].asDouble(), object[2].asDouble());
        }

        static Json::Value toJson(Eigen::Vector3d value) {
            Json::Value jsonVector(Json::arrayValue);
            jsonVector.append(value[0]);
            jsonVector.append(value[1]);
            jsonVector.append(value[2]);
            return jsonVector;
        }
    };

    template <>
    struct JSONParameterTraits<Eigen::Vector3f>
    {
        static bool check(const Json::Value& object) {
            return object.isArray() && object.size() == 3 &&
                    object[0].isNumeric() && object[1].isNumeric() &&
                    object[2].isNumeric();
        }

        static Eigen::Vector3f fromJson(const Json::Value& object) {
            return Eigen::Vector3f(object[0].asFloat(),
                    object[1].asFloat(), object[2].asFloat());
        }

        static Json::Value toJson(Eigen::Vector3f value) {
            Json::Value jsonVector(Json::arrayValue);
            jsonVector.append(value[0]);
            jsonVector.append(value[1]);
            jsonVector.append(value[2]);
            return jsonVector;
        }
    };
}

#endif // ISM3D_JSONPARAMETERTRAITS_H
