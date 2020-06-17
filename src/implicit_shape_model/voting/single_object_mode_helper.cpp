/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2019, Viktor Seib
 * All rights reserved.
 *
 */

#include "single_object_mode_helper.h"

namespace ism3d
{
    float SingleObjectHelper::getModelRadius(pcl::PointCloud<PointT>::ConstPtr &points,
                                                 const PointT &query)
    {
        // find distance of farthest point from centroid
        float model_radius = 0;
        for(int i = 0; i < points->size(); i++)
        {
            float dist = (points->at(i).getVector3fMap() - query.getVector3fMap()).norm();
            if(dist > model_radius) model_radius = dist;
        }
        return model_radius;
    }

    float SingleObjectHelper::getVotingSpaceSize(const std::vector<Voting::Vote>& votes,
                                                     const PointT &query)
    {
        float max_dist = 0;
        Eigen::Vector3f query_vec = query.getArray3fMap();
        for(int i = 0; i < votes.size(); i++)
        {
            Voting::Vote v = votes.at(i);
            float dist = (v.position - query_vec).squaredNorm();
            max_dist = max_dist > dist ? max_dist : dist;
        }
        return sqrt(max_dist);
    }
}
