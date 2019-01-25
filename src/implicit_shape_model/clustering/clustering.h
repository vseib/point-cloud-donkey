/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_CLUSTERING_H
#define ISM3D_CLUSTERING_H

#include "../utils/json_object.h"

#include <vector>

#define PCL_NO_PRECOMPILE
#include <pcl/point_cloud.h>

namespace ism3d
{
    class ISMFeature;
    class Distance;

    /**
     * @brief The Clustering class
     * Works as a functor and clusters the input features by using a specified distance function for
     * comparison.
     */
    class Clustering
            : public JSONObject
    {
    public:
        virtual ~Clustering();

        /**
         * @brief Cluster the input features by using a specified distance function.
         * @param features the input features
         * @param distance the distance function
         * @return a list of indices which maps each input feature to a cluster id
         */
        const std::vector<int>& operator()(pcl::PointCloud<ISMFeature>::ConstPtr features,
                                           const Distance* distance);

        /**
         * @brief Get the list of cluster indices.
         * @return the list of cluster indices which maps each input feature index to a cluster index
         */
        const std::vector<int>& getClusterIndices() const;

        /**
         * @brief Get the cluster data centers for each cluster index.
         * @return the list of cluster centers
         */
        const std::vector<std::vector<float> >& getClusterCenters() const;

        /**
         * @brief Clear data.
         */
        virtual void clear();

    protected:
        Clustering();

        virtual void process(pcl::PointCloud<ISMFeature>::ConstPtr) = 0;
        const Distance& getDistance() const;

        std::vector<std::vector<float> > m_centers; // stores the cluster centers (means) in data space
        std::vector<int> m_indices; // maps each input feature index to a cluster index

    private:
        float compactness(const std::vector<std::vector<float> >&,
                          pcl::PointCloud<ISMFeature>::ConstPtr) const;

        const Distance* m_distance;
    };
}

#endif // ISM3D_CLUSTERING_H
