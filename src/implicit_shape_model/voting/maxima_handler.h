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
#include "voting.h"

namespace ism3d
{
    enum class SingleObjectMaxType
    {
        DEFAULT,    // default means no special treatment
        COMPLETE_VOTING_SPACE,
        BANDWIDTH,
        MODEL_RADIUS
    };

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

        static void setSingleObjectMaxType(const std::string max_type_param)
        {
            // set max type based on parameter value
            if(max_type_param == "None" || max_type_param == "Default")
                m_max_type = SingleObjectMaxType::DEFAULT;
            else if(max_type_param == "BandwidthVotes")
                m_max_type = SingleObjectMaxType::BANDWIDTH;
            else if(max_type_param == "VotingSpaceVotes")
                m_max_type = SingleObjectMaxType::COMPLETE_VOTING_SPACE;
            else if(max_type_param == "ModelRadiusVotes")
                m_max_type = SingleObjectMaxType::MODEL_RADIUS;
            else
            {
                LOG_WARN("Invalid single object maximum type: " << max_type_param << "! Using default instead.");
                m_max_type = SingleObjectMaxType::DEFAULT;
            }
        }

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

        static float getRadius()
        {
            return m_radius;
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
                                  const float radius,
                                  const std::vector<Eigen::Vector3f>& clusterCenters,
                                  std::vector<float> &densities,
                                  std::vector<Eigen::Vector3f>& clusters);

        // -------------- these methods determine how maxima are treated after mean-shift ------------------
        static std::vector<VotingMaximum> filterMaxima(const std::string filter_type, const std::vector<VotingMaximum> &maxima);

        static SingleObjectMaxType m_max_type;


        // -------------- these methods determine which maxima are kept during mean-shift ------------------

        // only the first maximum in the radius is retained
        static void suppressNeighborMaxima(const std::vector<Eigen::Vector3f>& cluster_centers,
                                           const std::vector<float> &densities,
                                           const float radius,
                                           std::vector<Eigen::Vector3f>& maxima);
        // the average of the maxima in the radius is retained
        static void averageNeighborMaxima(const std::vector<Eigen::Vector3f> &cluster_centers,
                                          const float radius,
                                          std::vector<Eigen::Vector3f> &maxima,
                                          std::vector<float> &densities);

        // the average of the maxima and its [neighbor's neighbor's ...] neighbors in the radius is retained
        static void averageShiftNeighborMaxima(const std::vector<Eigen::Vector3f>&,
                                               const float radius,
                                               std::vector<Eigen::Vector3f>&);

        static void suppressNeighborMaxima2(const std::vector<VotingMaximum> &input_maxima,
                                     const float radius,
                                     std::vector<VotingMaximum>& maxima);


    private:

        // -------------- these methods determine how maxima are treated after mean-shift ------------------
        static std::vector<VotingMaximum> mergeAndFilterMaxima(const std::vector<VotingMaximum> &maxima, bool merge);
        static VotingMaximum mergeMaxima(const std::vector<VotingMaximum> &max_list);
        static VotingMaximum::GlobalHypothesis mergeGlobalHypotheses(const std::vector<VotingMaximum::GlobalHypothesis> &global_hyps);

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
