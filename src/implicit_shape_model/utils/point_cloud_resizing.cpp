/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "point_cloud_resizing.h"
#include <pcl/search/organized.h>
#include "exception.h"

namespace ism3d
{
    PointCloudResizing::PointCloudResizing()
        : m_resizeFactor(2.0f)
    {
    }

    PointCloudResizing::~PointCloudResizing()
    {
    }

    pcl::PointCloud<PointT>::ConstPtr PointCloudResizing::operator()(pcl::PointCloud<PointT>::ConstPtr pointCloud) const
    {
        if (!pointCloud->isOrganized()) {
            LOG_ERROR("point cloud is not organized");
            return pointCloud;
        }

        LOG_ASSERT(m_resizeFactor >= 1);

        pcl::PointCloud<PointT>::Ptr downsampled(new pcl::PointCloud<PointT>());
        downsampled->width = (int)(pointCloud->width / m_resizeFactor);
        downsampled->height = (int)(pointCloud->height / m_resizeFactor);
        downsampled->is_dense = pointCloud->is_dense;
        downsampled->resize(downsampled->width * downsampled->height);

        pcl::search::OrganizedNeighbor<PointT> search;
        search.setInputCloud(pointCloud);

        Eigen::Matrix3f cameraMatrix;
        search.computeCameraMatrix(cameraMatrix);

        for (int i = 0; i < (int)downsampled->width; i++) {
            for (int j = 0; j < (int)downsampled->height; j++) {
                PointT& point = downsampled->at(i, j);

                // retrieve nearest neighbors with organized nearest neighbor, using a dynamic distance
                // depending on the spatial resolution at the point

                const float startWidth = i * m_resizeFactor;
                const float endWidth = (i + 1) * m_resizeFactor;
                const float startHeight= j * m_resizeFactor;
                const float endHeight = (j + 1) * m_resizeFactor;

                float middleWidth = startWidth + ((endWidth - startWidth) / 2.0f);
                float middleHeight = startHeight + ((endHeight - startHeight) / 2.0f);

                const PointT& queryPoint = pointCloud->at((int)middleWidth, (int)middleHeight);

                if (pcl::isFinite(queryPoint)) {
                    Eigen::Vector3f centerPoint(queryPoint.x, queryPoint.y, queryPoint.z);
                    float radius = getRadius(cameraMatrix, centerPoint,
                                             (int)startWidth, (int)endWidth,
                                             (int)startHeight, (int)endHeight);

                    std::vector<int> indices;
                    std::vector<float> distances;
                    search.radiusSearch(queryPoint, radius, indices, distances);

                    // TOOD: possibility to create a weighted average, e.g. by applying a gaussian to
                    // the normalized distance from the query point?

                    // compute the average over all nearest neighbors
                    point.x = point.y = point.z = 0;
                    float totalWeight = 0;
                    for (int k = 0; k < (int)indices.size(); k++) {
                        const PointT& curPoint = pointCloud->at(indices[k]);
                        point.x += curPoint.x;
                        point.y += curPoint.y;
                        point.z += curPoint.z;
                        totalWeight += 1;
                    }
                    point.x /= totalWeight;
                    point.y /= totalWeight;
                    point.z /= totalWeight;
                }
                else
                    point = queryPoint;
            }
        }

        return downsampled;
    }

    Eigen::Vector3f PointCloudResizing::getPoint(const Eigen::Matrix3f& cameraMatrix, int i, int j, float z) const
    {
        float x = ((i - cameraMatrix(0, 2)) * z) / cameraMatrix(0, 0);
        float y = ((j - cameraMatrix(1, 2)) * z) / cameraMatrix(1, 1);
        return Eigen::Vector3f(x, y, z);
    }

    float PointCloudResizing::getRadius(const Eigen::Matrix3f& cameraMatrix, const Eigen::Vector3f& centerPoint,
                                        int minW, int maxW, int minH, int maxH) const
    {
        // get all point corresponding to the window corners at a constant distance z
        std::vector<Eigen::Vector3f> points;
        points.push_back(getPoint(cameraMatrix, minW, minH, centerPoint[2]));
        points.push_back(getPoint(cameraMatrix, minW, maxH, centerPoint[2]));
        points.push_back(getPoint(cameraMatrix, maxW, minH, centerPoint[2]));
        points.push_back(getPoint(cameraMatrix, maxW, maxH, centerPoint[2]));

        // get maximum distance (radius) between point and center point
        float maxDist = 0;
        for (int i = 0; i < (int)points.size(); i++) {
            float dist = (points[i] - centerPoint).norm();
            if (dist > maxDist)
                maxDist = dist;
        }

        return maxDist;
    }

    void PointCloudResizing::setResizeFactor(float factor)
    {
        if (factor < 1)
            throw BadParamExceptionType<float>("factor has to be >= 1", factor);

        m_resizeFactor = factor;
    }
}
