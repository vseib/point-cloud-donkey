/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_CLUSTERINGKMEANS_H
#define ISM3D_CLUSTERINGKMEANS_H

#include "clustering.h"

#include <flann/flann.h>
#include "../utils/ism_feature.h"
#include "../utils/utils.h"

namespace ism3d
{
    /**
     * @brief The ClusteringKMeans class
     * Performs a k-means clustering on the input data. The number of clusters is determined by a
     * cluster count factor, which is multiplied by the number of input features. It can also be
     * specified explicitly by setting the factor to 0 and using the "DesiredClusters" parameter.
     */
    class ClusteringKMeans
            : public Clustering
    {
    public:
        ~ClusteringKMeans();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        ClusteringKMeans();

        void cluster(pcl::PointCloud<ISMFeature>::ConstPtr, int);

    private:
        template <typename DistanceType>
        void cluster(pcl::PointCloud<ISMFeature>::ConstPtr);

        int m_desiredClusters;
        int m_branching;    // has influence on the obtained cluster count
        int m_iterations;
        flann_centers_init_t m_centersInit;
        float m_cbIndex;
    };

    template <typename DistanceType>
    void ClusteringKMeans::cluster(pcl::PointCloud<ISMFeature>::ConstPtr features)
    {
        if (features->size() == 0)
            return;

        typedef typename DistanceType::ElementType ElementType;
        typedef typename DistanceType::ResultType ResultType;

        // get descriptor size
        const ISMFeature& firstFeature = features->at(0);

        const int descriptorSize = firstFeature.descriptor.size();
        const int featureCount = features->size();

        if (m_desiredClusters > featureCount) {
            LOG_WARN("Desired clusters is higher than available feature count. Creating " <<
                     featureCount << " individual clusters.");
            m_desiredClusters = featureCount;
        }

        // fill input data
        ElementType* input = new ElementType[featureCount * descriptorSize];
        flann::Matrix<ElementType> inputMatrix(input, featureCount, descriptorSize);
        for (int i = 0; i < featureCount; i++) {
            const ISMFeature& feature = features->at(i);

            if ((int)feature.descriptor.size() != descriptorSize) {
                LOG_ERROR("invalid descriptor size");
                return;
            }

            for (int j = 0; j < descriptorSize; j++)
                inputMatrix[i][j] = feature.descriptor[j];
        }

        // create output data for cluster centers
        ElementType* centers = new ElementType[featureCount * descriptorSize];
        flann::Matrix<ResultType> centerMatrix(centers, m_desiredClusters, descriptorSize);
        memset(centers, featureCount * descriptorSize, sizeof(ElementType));

        // perform kmeans clustering to obtain cluster centers
        flann::KMeansIndexParams indexParams(m_branching, m_iterations, m_centersInit, m_cbIndex);

        int count = flann::hierarchicalClustering<DistanceType>(inputMatrix, centerMatrix, indexParams);

        if (count != m_desiredClusters)
            LOG_WARN("Requested " << m_desiredClusters << " but extracted " << count << " clusters instead");

        // create cluster centers
        m_centers.resize(count);
        for (int i = 0; i < count; i++) {
            ElementType* row = centerMatrix[i];
            std::vector<float>& center = m_centers[i];
            center.resize(descriptorSize);

            for (int j = 0; j < descriptorSize; j++)
                center[j] = row[j];
        }

        // beside cluster centers, cluster ids for each keypoint need to be extracted. Perform a KNN-Search with K = 1
        // for each keypoint to find it's nearest cluster center and assign the cluster id to this keypoint, i.e. the
        // keypoint is a member of this cluster.
        flann::Matrix<ResultType> resultCenterMatrix(centers, count, descriptorSize);
        flann::Index<DistanceType> tree(resultCenterMatrix, indexParams);
        tree.buildIndex();

        std::vector<std::vector<int> > indices(inputMatrix.rows);
        std::vector<std::vector<ElementType> > distances(inputMatrix.rows);
        tree.knnSearch(inputMatrix, indices, distances, 1, flann::SearchParams());

        delete[] input;
        delete[] centers;

        // create cluster indices
        m_indices.resize(indices.size());
        for (int i = 0; i < (int)indices.size(); i++)
            m_indices[i] = indices[i][0];
    }

    template <>
    struct JSONParameterTraits<flann::flann_centers_init_t>
    {
        static bool check(const Json::Value& object) {
            return object.isString();
        }

        static flann::flann_centers_init_t fromJson(const Json::Value& object) {
            std::string value = object.asString();
            if (value == "FLANN_CENTERS_GONZALES")
                return flann::FLANN_CENTERS_GONZALES;
            else if (value == "FLANN_CENTERS_KMEANSPP")
                return flann::FLANN_CENTERS_KMEANSPP;
            else if (value == "FLANN_CENTERS_RANDOM")
                return flann::FLANN_CENTERS_RANDOM;
            throw BadParamExceptionType<std::string>("invalid flann centers init", value);
        }

        static Json::Value toJson(flann::flann_centers_init_t value) {
            if (value == flann::FLANN_CENTERS_GONZALES)
                return Json::Value("FLANN_CENTERS_GONZALES");
            else if (value == flann::FLANN_CENTERS_KMEANSPP)
                return Json::Value("FLANN_CENTERS_KMEANSPP");
            else if (value == flann::FLANN_CENTERS_RANDOM)
                return Json::Value("FLANN_CENTERS_RANDOM");
            throw BadParamExceptionType<int>("invalid flann centers init", (int)value);
        }
    };
}

#endif // ISM3D_CLUSTERINGKMEANS_H
