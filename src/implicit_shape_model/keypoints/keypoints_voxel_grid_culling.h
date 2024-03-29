/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2021, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_KEYPOINTSVOXELGRIDCULLING_H
#define ISM3D_KEYPOINTSVOXELGRIDCULLING_H

#include "keypoints.h"
#include "../third_party/pcl_color_conversion/color_conversion.h"
#include <pcl/features/principal_curvatures.h>

namespace ism3d
{
    /**
     * @brief The KeypointsVoxelGridCulling class
     * Computes keypoints by uniformly sampling the input point cloud with a low resolution
     * and computing the centroid of points within a voxel.
     * Then each keypoint is evaluated based on a quality metric and "bad" keypoints are discarded.
     */
    class KeypointsVoxelGridCulling
            : public Keypoints
    {
    public:
        KeypointsVoxelGridCulling();
        ~KeypointsVoxelGridCulling();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        pcl::PointCloud<PointT>::ConstPtr iComputeKeypoints(pcl::PointCloud<PointT>::ConstPtr points,
                                                            pcl::PointCloud<PointT>::ConstPtr eigenValues,
                                                            pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                            pcl::PointCloud<PointT>::Ptr pointsWithoutNaNNormals,
                                                            pcl::PointCloud<PointT>::Ptr eigenValuesWithoutNan,
                                                            pcl::PointCloud<pcl::Normal>::Ptr normalsWithoutNaN,
                                                            pcl::search::Search<PointT>::Ptr search);

        std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> getScoresForKeypoints(
                const pcl::PointCloud<PointNormalT>::Ptr points_with_normals,
                const pcl::PointCloud<PointNormalT>::Ptr keypoints_with_normals,
                const pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures);

        std::tuple<float, float, float> computeThresholds(
                const std::vector<float> &geo_scores,
                const std::vector<float> &color_scores,
                const std::vector<float> &combined_scores_orig);

        float computeKPQ(const std::vector<int> &pointIdxs,
                         pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principle_curvatures) const;

        float computeColorScore(const std::vector<int> &pointIdxs,
                                pcl::PointCloud<PointNormalT>::Ptr points_with_normals,
                                const PointNormalT &ref,
                                const ColorConversion &cc) const;

        PointT refineKeypointPosition(const std::vector<float> &geo_scores,
                                      const std::vector<float> &color_scores,
                                      const std::vector<float> &combined_scores,
                                      const PointT &keypoint,
                                      const pcl::PointCloud<PointT>::Ptr &all_keypoints,
                                      const bool geo_passed,
                                      const bool color_passed);

    private:
        float m_leafSize;
        float m_max_similar_color_distance;
        float m_filter_threshold_geometry;
        float m_filter_cutoff_ratio;
        float m_filter_threshold_color;
        std::string m_filter_method_geometry;
        std::string m_filter_type_geometry;
        std::string m_filter_method_color;
        std::string m_filter_type_color;
        std::string m_combine_filters;
        bool m_disable_filter_in_training;
        bool m_refine_position;
    };
}

#endif // ISM3D_KEYPOINTSVOXELGRIDCULLING_H
