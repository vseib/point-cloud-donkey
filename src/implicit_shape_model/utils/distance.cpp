/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "distance.h"

namespace ism3d
{
    // Distance
    Distance::Distance()
    {
    }

    float Distance::operator()(const Eigen::VectorXf& vec1, const Eigen::VectorXf& vec2) const
    {
        std::vector<float> data1(vec1.data(), vec1.data() + vec1.size());
        std::vector<float> data2(vec2.data(), vec2.data() + vec2.size());
        return getDistance(data1, data2);
    }

    float Distance::operator()(const std::vector<float>& data1, const std::vector<float>& data2) const
    {
        return getDistance(data1, data2);
    }

    // DistanceEuclidean
    std::string DistanceEuclidean::getType() const
    {
        return DistanceEuclidean::getTypeStatic();
    }

    std::string DistanceEuclidean::getTypeStatic()
    {
        return "Euclidean";
    }

    float DistanceEuclidean::getDistance(const std::vector<float>& data1, const std::vector<float>& data2) const
    {
        return distance(data1.begin(), data2.begin(), data1.size());
    }

    // DistanceChiSquared
    std::string DistanceChiSquared::getType() const
    {
        return DistanceChiSquared::getTypeStatic();
    }

    std::string DistanceChiSquared::getTypeStatic()
    {
        return "ChiSquared";
    }

    float DistanceChiSquared::getDistance(const std::vector<float>& data1, const std::vector<float>& data2) const
    {
        return distance(data1.begin(), data2.begin(), data1.size());
    }

    // DistanceHellinger
    std::string DistanceHellinger::getType() const
    {
        return DistanceHellinger::getTypeStatic();
    }

    std::string DistanceHellinger::getTypeStatic()
    {
        return "Hellinger";
    }

    float DistanceHellinger::getDistance(const std::vector<float>& data1, const std::vector<float>& data2) const
    {
        return distance(data1.begin(), data2.begin(), data1.size());
    }

    // DistanceHistIntersection
    std::string DistanceHistIntersection::getType() const
    {
        return DistanceHistIntersection::getTypeStatic();
    }

    std::string DistanceHistIntersection::getTypeStatic()
    {
        return "HistIntersection";
    }

    float DistanceHistIntersection::getDistance(const std::vector<float>& data1, const std::vector<float>& data2) const
    {
        return distance(data1.begin(), data2.begin(), data1.size());
    }
}
