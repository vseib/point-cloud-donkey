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

#include "voting_maximum.h"

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
        MaximaHandler();
        virtual ~MaximaHandler(){}

        static void setRadiusType(const std::string radius_type)
        {
            m_radius_type = radius_type;
        }

        static void setRadiusFactor(const float factor)
        {
            m_radius_factor = factor;
        }

        static void setRadius(const float radius)
        {
            m_radius = radius;
        }

        static void setBoundingBoxMaps(const std::map<unsigned, std::pair<float, float>> &dims_map,
                                       const std::map<unsigned, std::pair<float, float>> &vars_map)
        {
            m_dimensions_map = dims_map;
            m_variance_map = vars_map;
        }

        // helper methods
        static float getSearchDistForClass(const unsigned class_id);

        // -------------- these methods determine which maxima are kept during mean-shift ------------------
        static void processMaxima(const std::string &type,
                           const std::vector<Eigen::Vector3f>& clusterCenters,
                           const float radius,
                           std::vector<Eigen::Vector3f>& clusters);

        // -------------- these methods determine how maxima are treated after mean-shift ------------------
        static std::vector<VotingMaximum> filterMaxima(const std::string filter_type, const std::vector<VotingMaximum> &maxima);

    private:

        // -------------- these methods determine which maxima are kept during mean-shift ------------------

        // only the first maximum in the radius is retained
        static void suppressNeighborMaxima(const std::vector<Eigen::Vector3f>&,
                                    const float radius,
                                    std::vector<Eigen::Vector3f>&);
        // the average of the maxima in the radius is retained
        static void averageNeighborMaxima(const std::vector<Eigen::Vector3f>&,
                                   const float radius,
                                   std::vector<Eigen::Vector3f>&);
        // the average of the maxima and its [neighbor's neighbor's ...] neighbors in the radius is retained
        static void averageShiftNeighborMaxima(const std::vector<Eigen::Vector3f>&,
                                        const float radius,
                                        std::vector<Eigen::Vector3f>&);

        // -------------- these methods determine how maxima are treated after mean-shift ------------------
        static std::vector<VotingMaximum> mergeAndFilterMaxima(const std::vector<VotingMaximum> &maxima, bool merge);
        static VotingMaximum mergeMaxima(const std::vector<VotingMaximum> &max_list);

        static std::string m_radius_type;  // take value from config or used learned average bounding box dimensions
        static float m_radius_factor;  // factor for radius, in case radius type is NOT Config
        static float m_radius; // bandwidth or half of the bin size

        // maps class ids to average pairs of two longest bounding box dimensions
        // <first radius, second radius>
        static std::map<unsigned, std::pair<float, float>> m_dimensions_map;
        // maps class ids to average pairs of two longest bounding box dimension variances
        // <first variance, second variance>
        static std::map<unsigned, std::pair<float, float>> m_variance_map;
    };
}

#endif // ISM3D_MAXIMAHANDLER_H
