/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_RANKINGNAIVEBAYES_H
#define ISM3D_RANKINGNAIVEBAYES_H

#include "feature_ranking.h"
#include <pcl/kdtree/kdtree_flann.h>

namespace ism3d
{
    /**
     * @brief The RankingNaiveBayes class
     * Assigns each feature a score based on naive bayes
     */
    class RankingNaiveBayes
            : public FeatureRanking
    {
    public:
        RankingNaiveBayes();
        ~RankingNaiveBayes();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        std::map<unsigned, std::vector<float> > iComputeScores(const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &features);

    private:

    };
}

#endif // ISM3D_RANKINGNAIVEBAYES_H
