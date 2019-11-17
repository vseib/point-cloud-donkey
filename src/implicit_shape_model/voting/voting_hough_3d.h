/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_VOTINGHOUGH3D_H
#define ISM3D_VOTINGHOUGH3D_H

#include "voting.h"

namespace ism3d
{
    /**
     * @brief The VotingHough3D class
     * Detects maxima in the voting space by using a binned voting accumulator and finding
     * bins with the highest accumulator value.
     */
    class VotingHough3D
            : public Voting
    {
    public:
        VotingHough3D();
        ~VotingHough3D();

        static std::string getTypeStatic();
        std::string getType() const;

    protected:
        void iFindMaxima(pcl::PointCloud<PointT>::ConstPtr &,
                         const std::vector<Voting::Vote>&,
                         std::vector<Eigen::Vector3f>&,
                         std::vector<double>&,
                         std::vector<std::vector<int> >&,
                         std::vector<std::vector<float> >&,
                         unsigned classId, float &radius);
        void iPostInitConfig();
        void clear();

    private:
        pcl::recognition::HoughSpace3D* m_houghSpace;

        bool m_useInterpolation;
        Eigen::Vector3d m_minCoord;
        Eigen::Vector3d m_maxCoord;
        Eigen::Vector3d m_binSize;
        float m_relThreshold;
    };
}

#endif // ISM3D_VOTINGHOUGH3D_H
