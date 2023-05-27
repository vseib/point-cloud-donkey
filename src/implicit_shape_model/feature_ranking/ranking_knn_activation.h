/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_RANKINGKNNACTIVATION_H
#define ISM3D_RANKINGKNNACTIVATION_H

#include "feature_ranking.h"

// TODO VS rename in activation ranking (omit the knn in the name)
namespace ism3d
{
    /**
     * @brief The RankingKnnActivation class
     * Assigns each feature a score based on the activity (activating and beeing activated) in the k-NN
     */
    class RankingKnnActivation
            : public FeatureRanking
    {
    public:
        RankingKnnActivation();
        ~RankingKnnActivation();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        std::map<unsigned, std::vector<float> > iComputeScores(const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &features);

    private:

        int m_score_increment_type;
        bool m_use_feature_position;
    };
}

#endif // ISM3D_RANKINGKNNACTIVATION_H
