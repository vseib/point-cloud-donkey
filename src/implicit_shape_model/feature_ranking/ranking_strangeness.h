/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_RANKINGSTRANGENESS_H
#define ISM3D_RANKINGSTRANGENESS_H

#include "feature_ranking.h"
#include <pcl/kdtree/kdtree_flann.h>

namespace ism3d
{
    /**
     * @brief The RankingStrangeness class
     * Assigns each feature a score based on a strangeness measure
     *
     * NOTE: this is based on the paper
     * "Strangeness based feature selection for part based recognition"
     * by Li, Fayin and Kosecka, Jana and Wechsler, Harry
     * presented at Computer Vision and Pattern Recognition Workshop, 2006. CVPRW'06
     */
    class RankingStrangeness
            : public FeatureRanking
    {
    public:
        RankingStrangeness();
        ~RankingStrangeness();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        std::map<unsigned, std::vector<float> > iComputeScores(
                const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &features);
    };
}

#endif // ISM3D_RANKINGSTRANGENESS_H
