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
}
