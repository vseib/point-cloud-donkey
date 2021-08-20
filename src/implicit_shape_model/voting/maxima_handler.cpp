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

ism3d::SingleObjectMaxType ism3d::MaximaHandler::m_max_type = ism3d::SingleObjectMaxType::DEFAULT;
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

    // TODO VS remove this method
    void MaximaHandler::processMaxima(const std::string &type,
                                      const float radius,
                                      const std::vector<Eigen::Vector3f> &cluster_centers,
                                      std::vector<float> &densities,
                                      std::vector<Eigen::Vector3f>& maxima)
    {
        // retrieve maximum points
        if(type == "Suppress")
        {
            suppressNeighborMaxima(cluster_centers, densities, radius, maxima);
        }
        else if(type == "Average")
        {
            std::vector<Eigen::Vector3f> temp_clusters;
            averageNeighborMaxima(cluster_centers, radius, temp_clusters, densities);
            suppressNeighborMaxima(temp_clusters, densities, radius, maxima);
        }
        else if(type == "AverageShift") // TODO VS: check this
        {
            LOG_WARN("--- Maxima processing type 'AverageShift' is no longer supported - results might not be accurate! ---");
            averageShiftNeighborMaxima(cluster_centers, radius, maxima);
        }
    }

    void MaximaHandler::suppressNeighborMaxima(const std::vector<Eigen::Vector3f>& cluster_centers,
                                               const std::vector<float> &densities,
                                               const float radius,
                                               std::vector<Eigen::Vector3f>& maxima)
    {
        maxima.clear();
        bool done = false;
        std::vector<float> worklist(densities.size());
        std::copy(densities.begin(), densities.end(), worklist.begin());

        while(!done)
        {
            // find max index
            auto max_iter = std::max_element(std::begin(worklist), std::end(worklist));
            float max_elem = -1;
            if(max_iter != std::end(worklist))
                    max_elem = *max_iter;
            if(max_elem != -1)
            {
                // store best maximum as results
                unsigned max_idx = max_iter - std::begin(worklist);
                const Eigen::Vector3f& center = cluster_centers[max_idx];
                maxima.push_back(std::move(center));
                worklist[max_idx] = -1;

                // eliminate non-max neighbors
                for (unsigned i = 0; i < cluster_centers.size(); i++)
                {
                    const Eigen::Vector3f& neighbor = cluster_centers[i];
                    float distance = (center - neighbor).norm();
                    if (distance < radius)
                    {
                        worklist[i] = -1;
                    }
                }
            }
            else
            {
                done = true;
            }
        }
    }

    void MaximaHandler::averageNeighborMaxima(const std::vector<Eigen::Vector3f>& cluster_centers,
                                              const float radius,
                                              std::vector<Eigen::Vector3f>& maxima,
                                              std::vector<float> &densities)
    {
        std::vector<std::vector<int>> duplicate_indices(cluster_centers.size());
        for(int i = 0; i < duplicate_indices.size(); i++)
        {
            // add itself for simpler averaging later
            duplicate_indices.at(i).push_back(i);
        }

        std::vector<bool> duplicate(cluster_centers.size());
        duplicate.assign(duplicate.size(), false);

        for (int k = 0; k < (int)cluster_centers.size(); k++)
        {
            const Eigen::Vector3f& pointA = cluster_centers[k];

            if (duplicate[k])
                continue;

            for (int j = k + 1; j < (int)cluster_centers.size(); j++)
            {
                const Eigen::Vector3f& pointB = cluster_centers[j];

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
                maxima.push_back(cluster_centers[index_list.at(0)]);
            }
            else
            {
                // compute average of maximum and all neighbors
                Eigen::Vector3f average(0, 0, 0);
                float sum_densities = 0;
                for (int j = 0; j < index_list.size(); j++)
                {
                    // update shifted position
                    average += cluster_centers[index_list.at(j)] * densities[index_list.at(j)];
                    sum_densities += densities[index_list.at(j)];
                }
                average /= sum_densities;
                maxima.push_back(average);
            }
        }
    }


    // TODO VS: remove this method
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
        else if(filter_type == "None")  // do nothing
        {
            return maxima;
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
        std::vector<VotingMaximum::GlobalHypothesis> global_hyps;

        std::map<unsigned, float> instance_weights;

        VotingMaximum result;
        for(const VotingMaximum &m : max_list)
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

            if(instance_weights.find(m.instanceId) != instance_weights.end())
            {
                instance_weights.at(m.instanceId) += m.instanceWeight;
            }
            else
            {
                instance_weights.insert({m.instanceId, m.instanceWeight});
            }

            // find max value
            unsigned max_id_weights;
            float best_weight = 0;
            for(auto it : instance_weights)
            {
                float weight = instance_weights[it.first];
                if(weight > best_weight)
                {
                    best_weight = weight;
                    max_id_weights = it.first;
                }
            }
            result.instanceId = max_id_weights;
            result.instanceWeight = instance_weights[max_id_weights];

            // accumulate global hypotheses and merge afterwards
            global_hyps.push_back(m.globalHypothesis);
        }

        result.globalHypothesis = mergeGlobalHypotheses(global_hyps);

        return result;
    }

    VotingMaximum::GlobalHypothesis MaximaHandler::mergeGlobalHypotheses(
            const std::vector<VotingMaximum::GlobalHypothesis> &global_hyps)
    {
        // TODO VS: alternatively try using the most frequent label!
        std::map<unsigned,float> global_weights;
        // accumulate weights for each id
        for(auto &gh : global_hyps)
        {
            if(global_weights.find(gh.classId) != global_weights.end())
            {
                global_weights.at(gh.classId) += gh.classWeight;
            }
            else
            {
                global_weights.insert({gh.classId, gh.classWeight});
            }
        }

        // find class with hightest weight
        unsigned max_id;
        float max_weight = std::numeric_limits<float>::min();
        for(auto &r : global_weights)
        {
            if(r.second > max_weight)
            {
                max_weight = r.second;
                max_id = r.first;
            }
        }

        // find instance of highest class with highest score
        global_weights.clear();
        for(auto &gh : global_hyps)
        {
            if(gh.classId == max_id)
            {
                if(global_weights.find(gh.instanceId) != global_weights.end())
                {
                    global_weights.at(gh.instanceId) += gh.instanceWeight;
                }
                else
                {
                    global_weights.insert({gh.instanceId, gh.instanceWeight});
                }
            }
        }

        VotingMaximum::GlobalHypothesis global_result;
        global_result.classId = max_id;
        global_result.classWeight = max_weight;

        // find instance with hightest score
        max_weight = std::numeric_limits<float>::min();
        for(auto &r : global_weights)
        {
            if(r.second > max_weight)
            {
                max_weight = r.second;
                max_id = r.first;
            }
        }

        global_result.instanceId = max_id;
        global_result.instanceWeight = max_weight;
        return global_result;
    }

    float MaximaHandler::getSearchDistForClass(const unsigned class_id)
    {
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
