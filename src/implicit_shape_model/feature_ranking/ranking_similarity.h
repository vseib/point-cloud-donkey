/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2021, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_RANKINGSIMILARITY_H
#define ISM3D_RANKINGSIMILARITY_H

#include "feature_ranking.h"
#include <pcl/kdtree/kdtree_flann.h>

namespace ism3d
{
    /**
     * @brief The RankingSimilarity class
     * Assigns each feature a score based on a KNN activation rule-based measure involving similarities of descriptors
     *
     */
    class RankingSimilarity
            : public FeatureRanking
    {
    public:
        RankingSimilarity();
        ~RankingSimilarity();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        std::map<unsigned, std::vector<float> > iComputeScores(
                const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &features);

    private:
        float getScore(float distance, bool as_penalty = false);
    };
}

#endif // ISM3D_RANKINGSIMILARITY_H
