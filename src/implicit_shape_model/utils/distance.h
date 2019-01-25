/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_DISTANCE_H
#define ISM3D_DISTANCE_H

#include <vector>
#include <flann/flann.hpp>
#include <Eigen/Core>

namespace ism3d
{
    /**
     * @brief The Distance struct
     * The distance base class.
     */
    struct Distance
    {
        typedef float ElementType;

        virtual std::string getType() const = 0;
        float operator()(const Eigen::VectorXf&, const Eigen::VectorXf&) const;
        float operator()(const std::vector<float>&, const std::vector<float>&) const;

    protected:
        Distance();

        virtual float getDistance(const std::vector<float>&, const std::vector<float>&) const = 0;
    };

    /**
     * @brief The DistanceEuclidean struct
     * Euclidean distance class.
     */
    struct DistanceEuclidean
        : public Distance
    {
        typedef flann::L2<typename Distance::ElementType> DistanceType;

        std::string getType() const;
        static std::string getTypeStatic();

    protected:
        float getDistance(const std::vector<float>&, const std::vector<float>&) const;

    private:
        DistanceType distance;
    };


    /**
     * @brief The DistanceChiSquared struct
     * The chi-squared distance class.
     */
    struct DistanceChiSquared
            : public Distance
    {
        typedef flann::ChiSquareDistance<typename Distance::ElementType> DistanceType;

        std::string getType() const;
        static std::string getTypeStatic();

    protected:
        float getDistance(const std::vector<float>&, const std::vector<float>&) const;

    private:
        DistanceType distance;
    };

    /**
     * @brief The DistanceHellinger struct
     * The hellinger distance class.
     */
    struct DistanceHellinger
            : public Distance
    {
        typedef flann::HellingerDistance<typename Distance::ElementType> DistanceType;

        std::string getType() const;
        static std::string getTypeStatic();

    protected:
        float getDistance(const std::vector<float>&, const std::vector<float>&) const;

    private:
        DistanceType distance;
    };

    /**
     * @brief The DistanceHistIntersection struct
     * The histogram intersection distance class.
     */
    struct DistanceHistIntersection
            : public Distance
    {
        typedef flann::HistIntersectionDistance<typename Distance::ElementType> DistanceType;

        std::string getType() const;
        static std::string getTypeStatic();

    protected:
        float getDistance(const std::vector<float>&, const std::vector<float>&) const;

    private:
        DistanceType distance;
    };
}

#endif // ISM3D_DISTANCE_H
