/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2020, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_MAXIMAHANDLER_H
#define ISM3D_MAXIMAHANDLER_H

#include <vector>
#include <Eigen/Core>

namespace ism3d
{
    /**
     * @brief The MaximaHandler class
     * Allows to filter maxima according to some strategy.
     * Supported strategies are: suppress smaller maxima, average maxima, average maxima and shift to process neighbors
     */
    class MaximaHandler
    {

    public:
        MaximaHandler(std::string type);
        virtual ~MaximaHandler(){};

        void processMaxima(const std::vector<Eigen::Vector3f>& clusterCenters,
                           const float radius,
                           std::vector<Eigen::Vector3f>& clusters) const;

    private:

        // only the first maximum in the radius is retained
        void suppressNeighborMaxima(const std::vector<Eigen::Vector3f>&,
                                    const float radius,
                                    std::vector<Eigen::Vector3f>&) const;
        // the average of the maxima in the radius is retained
        void averageNeighborMaxima(const std::vector<Eigen::Vector3f>&,
                                   const float radius,
                                   std::vector<Eigen::Vector3f>&) const;
        // the average of the maxima and its [neighbor's neighbor's ...] neighbors in the radius is retained
        void averageShiftNeighborMaxima(const std::vector<Eigen::Vector3f>&,
                                        const float radius,
                                        std::vector<Eigen::Vector3f>&) const;

        std::string m_type;
    };
}

#endif // ISM3D_MAXIMAHANDLER_H
