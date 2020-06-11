/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2020, Viktor Seib
 * All rights reserved.
 *
 */

#include "maxima_handler.h"

std::string ism3d::MaximaHandler::m_radius_type = "";
float ism3d::MaximaHandler::m_radius_factor = 0.0f;
float ism3d::MaximaHandler::m_radius = 0.0f;
std::map<unsigned, std::pair<float, float>> ism3d::MaximaHandler::m_dimensions_map = {};
std::map<unsigned, std::pair<float, float>> ism3d::MaximaHandler::m_variance_map = {};

namespace ism3d
{
    MaximaHandler::MaximaHandler()
    {
    }

    void MaximaHandler::processMaxima(const std::string &type,
                                      const std::vector<Eigen::Vector3f>& clusterCenters,
                                      const float radius,
                                      std::vector<Eigen::Vector3f>& clusters)
    {
        // retrieve maximum points
        if(type == "Suppress")
        {
            suppressNeighborMaxima(clusterCenters, radius, clusters);
        }
        else if(type == "Average")
        {
            averageNeighborMaxima(clusterCenters, radius, clusters);
        }
        else if(type == "AverageShift")
        {
            averageShiftNeighborMaxima(clusterCenters, radius, clusters);
        }
    }

    void MaximaHandler::suppressNeighborMaxima(const std::vector<Eigen::Vector3f>& maxima,
                                               const float radius,
                                               std::vector<Eigen::Vector3f>& clusters)
    {
        std::vector<bool> duplicate(maxima.size());
        duplicate.assign(duplicate.size(), false);

        for (int i = 0; i < (int)maxima.size(); i++)
        {
            const Eigen::Vector3f& pointA = maxima[i];

            if (duplicate[i])
                continue;

            for (int j = i + 1; j < (int)maxima.size(); j++)
            {
                const Eigen::Vector3f& pointB = maxima[j];

                if (duplicate[j])
                    continue;

                float distance = (pointA - pointB).norm();

                if (distance < radius)
                    duplicate[j] = true;
            }
        }

        // add correct cluster centers
        for (int i = 0; i < (int)duplicate.size(); i++) {
            if (!duplicate[i])
                clusters.push_back(maxima[i]);
        }
    }

    void MaximaHandler::averageNeighborMaxima(const std::vector<Eigen::Vector3f>& maxima,
                                              const float radius,
                                              std::vector<Eigen::Vector3f>& clusters)
    {
        std::vector<std::vector<int>> duplicate_indices(maxima.size());
        for(int i = 0; i < duplicate_indices.size(); i++)
        {
            // add itself for simpler averaging later
            duplicate_indices.at(i).push_back(i);
        }

        std::vector<bool> duplicate(maxima.size());
        duplicate.assign(duplicate.size(), false);

        for (int k = 0; k < (int)maxima.size(); k++)
        {
            const Eigen::Vector3f& pointA = maxima[k];

            if (duplicate[k])
                continue;

            for (int j = k + 1; j < (int)maxima.size(); j++)
            {
                const Eigen::Vector3f& pointB = maxima[j];

                if (duplicate[j])
                    continue;

                float distance = (pointA - pointB).norm();

                if (distance < radius)
                {
                    duplicate[j] = true;
                    duplicate_indices.at(k).push_back(j);
                }
            }
        }

        // add correct cluster centers
        for (int i = 0; i < (int)duplicate_indices.size(); i++)
        {
            std::vector<int> index_list = duplicate_indices.at(i);
            if(index_list.size() == 1)
            {
                // maximum without neighbors can be added directly
                clusters.push_back(maxima[index_list.at(0)]);
            }
            else
            {
                // compute average of maximum and all neighbors
                Eigen::Vector3f average(0, 0, 0);
                for (int i = 0; i < index_list.size(); i++)
                {
                    // update shifted position
                    average += maxima[index_list.at(i)];
                }
                average /= index_list.size();
                clusters.push_back(average);
            }
        }
    }


    void MaximaHandler::averageShiftNeighborMaxima(const std::vector<Eigen::Vector3f>& maxima,
                                                   const float radius,
                                                   std::vector<Eigen::Vector3f>& clusters)
    {
        std::vector<std::vector<int>> duplicate_indices(maxima.size());
        for(int i = 0; i < duplicate_indices.size(); i++)
        {
            // add itself as seedpoint for maxima suppression
            duplicate_indices.at(i).push_back(i);
        }

        std::vector<bool> duplicate(maxima.size());
        duplicate.assign(duplicate.size(), false);

        // for each maximum ...
        for(int k = 0; k < (int)maxima.size(); k++)
        {
            if (duplicate[k])
                continue;

            // ... take the seedpoint
            for(int i : duplicate_indices.at(k))
            {
                const Eigen::Vector3f& pointA = maxima[i];

                // check all maxima within bandwidth
                for (int j = i + 1; j < (int)maxima.size(); j++)
                {
                    const Eigen::Vector3f& pointB = maxima[j];

                    float distance = (pointA - pointB).norm();

                    if (distance < radius)
                    {
                        // enables to check neighbors of neighbors etc.
                        duplicate[j] = true;
                        duplicate_indices.at(i).push_back(j);
                    }
                }
            }
        }

        // add correct cluster centers
        for (int i = 0; i < (int)duplicate_indices.size(); i++)
        {
            std::vector<int> index_list = duplicate_indices.at(i);
            if(index_list.size() == 1)
            {
                // maximum without neighbors can be added directly
                clusters.push_back(maxima[index_list.at(0)]);
            }
            else
            {
                // compute average of maximum and all neighbors
                Eigen::Vector3f shifted(0, 0, 0);
                for (int i = 0; i < index_list.size(); i++)
                {
                    // update shifted position
                    shifted += maxima[index_list.at(i)];
                }
                shifted /= index_list.size();
                clusters.push_back(shifted);
            }
        }
    }


    std::vector<VotingMaximum> MaximaHandler::filterMaxima(const std::string filter_type, const std::vector<VotingMaximum> &maxima)
    {
        if(filter_type == "Simple") // search in bandwith radius and keep only maximum with the highest weight, dont't merge
        {
            return mergeAndFilterMaxima(maxima, false);
        }
        else if(filter_type == "Merge")  // search in bandwith radius, merge maxima of same class and keep only maximum with the highest weight
        {
            return mergeAndFilterMaxima(maxima, true);
        }
        else
        {
            LOG_ERROR("Invalid maxima filter type specified: " << filter_type << "! No filtering is performed!");
            return maxima;
        }
    }

    std::vector<VotingMaximum> MaximaHandler::mergeAndFilterMaxima(const std::vector<VotingMaximum> &maxima, bool merge)
    {
        // find and merge maxima of different classes that are closer than bandwith or bin_size
        std::vector<VotingMaximum> close_maxima;
        std::vector<VotingMaximum> filtered_maxima;
        std::vector<bool> dirty_list(maxima.size(), false);

        for(unsigned i = 0; i < maxima.size(); i++)
        {
            if(dirty_list.at(i))
                continue;

            // set adaptive search distance depending on config and class id
            float search_dist = getSearchDistForClass(maxima.at(i).classId);

            // check distance to other maxima
            for(unsigned j = i+1; j < maxima.size(); j++)
            {
                if(dirty_list.at(j))
                    continue;

                float dist = (maxima.at(j).position - maxima.at(i).position).norm();
                float other_search_dist = getSearchDistForClass(maxima.at(j).classId);
                // only subsume maxima of classes with a smaller or equal search dist
                if(dist < search_dist && other_search_dist <= search_dist)
                {
                    close_maxima.push_back(maxima.at(j));
                    dirty_list.at(j) = true;
                }
            }

            // if some neighbors found, also add itself
            if(close_maxima.size() > 0)
            {
                close_maxima.push_back(maxima.at(i));
            }

            // merge close maxima of same classes before filtering
            if(merge && close_maxima.size() > 1) // > 1 because the maximum itself was added
            {
                std::vector<VotingMaximum> merged_maxima(maxima.size());
                std::map<unsigned, std::vector<VotingMaximum>> same_class_ids; // maps a class id to a list of close maxima with that id

                // create max list
                for(VotingMaximum m : close_maxima)
                {
                    unsigned class_id = m.classId;
                    if(same_class_ids.find(class_id) == same_class_ids.end())
                    {
                        same_class_ids.insert({class_id, {m}});
                    }
                    else
                    {
                        same_class_ids.at(class_id).push_back(m);
                    }
                }
                // merge maxima of same classes
                for(auto it : same_class_ids)
                {
                    VotingMaximum max = mergeMaxima(it.second);
                    merged_maxima.push_back(max);
                }
                close_maxima = merged_maxima;
            }

            // if a close maximum was found, leave only the one with the highest weight
            if(close_maxima.size() > 1) // > 1 because the maximum itself was added
            {
                VotingMaximum best_max;
                for(VotingMaximum m : close_maxima)
                {
                    if(m.weight > best_max.weight)
                    {
                        best_max = m;
                    }
                }
                filtered_maxima.push_back(best_max);
            }
            else
            {
                filtered_maxima.push_back(maxima.at(i));
            }
            close_maxima.clear();
        }
        return filtered_maxima;
    }


    VotingMaximum MaximaHandler::mergeMaxima(const std::vector<VotingMaximum> &max_list)
    {
        VotingMaximum result;
        for(VotingMaximum m : max_list)
        {
            // NOTE: position and bounding box must be handled before changing weight!
            result.position = result.position * result.weight + m.position * m.weight;
            result.position /= (result.weight + m.weight);
            result.boundingBox.position = result.position;
            result.boundingBox.size = result.boundingBox.size * result.weight + m.boundingBox.size * m.weight;
            result.boundingBox.size /= (result.weight + m.weight);
            boost::math::quaternion<float> rotQuat;
            Utils::quatWeightedAverage({result.boundingBox.rotQuat, m.boundingBox.rotQuat}, {result.weight, m.weight}, rotQuat);
            result.boundingBox.rotQuat = rotQuat;

            result.classId = m.classId;
            result.weight += m.weight;
            result.voteIndices.insert(result.voteIndices.end(), m.voteIndices.begin(), m.voteIndices.end());

            //TODO VS TEMP FIX THIS! -- should be some kind of average
            result.globalHypothesis = m.globalHypothesis;
            result.currentClassHypothesis = m.currentClassHypothesis;
        }
        return result;
    }

    float MaximaHandler::getSearchDistForClass(const unsigned class_id)
    {
        // NOTE: m_radius is assigned in derived classes and is related either to the bandwidth or bin size
        if(m_radius_type == "Config")
            return m_radius;
        if(m_radius_type == "FirstDim")
            return m_dimensions_map.at(class_id).first * m_radius_factor;
        if(m_radius_type == "SecondDim")
            return m_dimensions_map.at(class_id).second * m_radius_factor;

        LOG_ERROR("Invalid radius type: " << m_radius_type << "! Using config value instead.");
        return m_radius;
    }
}
