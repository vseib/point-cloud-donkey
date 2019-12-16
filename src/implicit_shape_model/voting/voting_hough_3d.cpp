/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "voting_hough_3d.h"

namespace ism3d
{
    VotingHough3D::VotingHough3D()
        : m_houghSpace(0)
    {
        addParameter(m_useInterpolation, "UseInterpolation", true);
        addParameter(m_minCoord, "MinCoord", Eigen::Vector3d(-5, -5, -5));
        addParameter(m_maxCoord, "MaxCoord", Eigen::Vector3d(5, 5, 5));
        addParameter(m_binSize, "BinSize", Eigen::Vector3d(0.2, 0.2, 0.2));
        addParameter(m_relThreshold, "RelThreshold", 0.8f);

        iPostInitConfig();
    }

    VotingHough3D::~VotingHough3D()
    {
        delete m_houghSpace;
    }

    void VotingHough3D::iFindMaxima(pcl::PointCloud<PointT>::ConstPtr &points,
                                    const std::vector<Voting::Vote>& votes,
                                    std::vector<Eigen::Vector3f>& clusters,
                                    std::vector<double>& maxima,
                                    std::vector<std::vector<int> >& voteIndices,
                                    std::vector<std::vector<float> >& reweightedVotes,
                                    unsigned classId)
    {

        LOG_WARN("SingleObjectMode is not supported with Hough3D - switch to MeanShift to use it!");
        // TODO VS only warn in single object mode

        if(m_radiusType == "Config")
        {
            // leave bin size as it is from config
        }
        else if(m_radiusType == "FirstDim")
        {
            float temp = m_id_bb_dimensions_map.at(classId).first * m_radiusFactor;
            temp *= 2; // bins are conceptually a "diameter" instead of radius
            m_binSize = Eigen::Vector3d(temp, temp, temp);
        }
        else if(m_radiusType == "SecondDim")
        {
            float temp = m_id_bb_dimensions_map.at(classId).second * m_radiusFactor;
            temp *= 2; // bins are conceptually a "diameter" instead of radius
            m_binSize = Eigen::Vector3d(temp, temp, temp);
        }

        // forward bin size to voting class
        m_radius = m_binSize[0] * 0.5;

        // cast votes into own voting space
        iPostInitConfig();
        m_houghSpace->reset();

        for (int i = 0; i < (int)votes.size(); i++)
        {
            const Voting::Vote& vote = votes[i];
            if (m_useInterpolation) {
                m_houghSpace->voteInt(Eigen::Vector3d(vote.position[0], vote.position[1], vote.position[2]),
                        vote.weight, i);
            }
            else {
                m_houghSpace->vote(Eigen::Vector3d(vote.position[0], vote.position[1], vote.position[2]),
                        vote.weight, i);
            }
        }

        // find maxima
        m_houghSpace->findMaxima(-m_relThreshold, maxima, voteIndices);

        // iterate through all found maxima and create a weighted cluster center
        reweightedVotes.resize(voteIndices.size());
        for (int i = 0; i < (int)voteIndices.size(); i++)
        {
            const std::vector<int>& clusterVotes = voteIndices[i];
            std::vector<float>& voteWeights = reweightedVotes[i];
            voteWeights.resize(clusterVotes.size());

            Eigen::Vector3f clusterCenter(0, 0, 0);
            float weight = 0;
            for (int j = 0; j < (int)clusterVotes.size(); j++)
            {
                int ind = clusterVotes[j];
                const Vote& vote = votes[ind];
                voteWeights[j] = vote.weight;

                clusterCenter += (vote.position * vote.weight);
                weight += vote.weight;
            }

            clusterCenter /= weight;
            clusters.push_back(clusterCenter);
        }
    }

    void VotingHough3D::clear()
    {
        m_houghSpace->reset();
        Voting::clear();
    }

    std::string VotingHough3D::getTypeStatic()
    {
        return "Hough3D";
    }

    std::string VotingHough3D::getType() const
    {
        return VotingHough3D::getTypeStatic();
    }

    void VotingHough3D::iPostInitConfig()
    {
        delete m_houghSpace;
        m_houghSpace = new pcl::recognition::HoughSpace3D(m_minCoord, m_binSize, m_maxCoord);
    }
}
