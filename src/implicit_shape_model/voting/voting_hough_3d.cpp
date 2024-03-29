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
#include "maxima_handler.h"

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
                                    std::vector<Vote>& votes,
                                    std::vector<Eigen::Vector3f>& maximum_positions,
                                    std::vector<double>& maxima,
                                    std::vector<std::vector<unsigned>>& instanceIdsPerCluster,
                                    std::vector<std::vector<Vote>>& cluster_votes,
                                    unsigned classId)
    {

        if(m_single_object_mode)
            LOG_WARN("SingleObjectMode is not supported with Hough3D - switch to MeanShift to use it!");

        MaximaHandler::setRadius(m_binSize[0]/2);
        float temp = MaximaHandler::getSearchDistForClass(classId);
        m_binSize = Eigen::Vector3d(temp*2, temp*2, temp*2); // bins are conceptually a "diameter" instead of radius

        // cast votes into own voting space
        iPostInitConfig();
        m_houghSpace->reset();

        for (int i = 0; i < (int)votes.size(); i++)
        {
            const Vote& vote = votes[i];
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
        std::vector<std::vector<int>> voteIndices;
        m_houghSpace->findMaxima(-m_relThreshold, maxima, voteIndices);
        cluster_votes.resize(maxima.size());

        // iterate through all found maxima and create a weighted cluster center
        for (int i = 0; i < (int)voteIndices.size(); i++)
        {
            const std::vector<int>& clusterVoteIndices = voteIndices[i];

            std::vector<unsigned> clusterInstances;
            clusterInstances.resize(clusterVoteIndices.size());

            Eigen::Vector3f clusterCenter(0, 0, 0);
            float weight = 0;
            for (int j = 0; j < (int)clusterVoteIndices.size(); j++)
            {
                int ind = clusterVoteIndices[j];
                const Vote& vote = votes[ind];
                clusterInstances[j] = vote.instanceId;
                clusterCenter += (vote.position * vote.weight);
                weight += vote.weight;
                cluster_votes[i].push_back(std::move(vote));
            }

            clusterCenter /= weight;
            maximum_positions.push_back(clusterCenter);
            instanceIdsPerCluster.push_back(clusterInstances);
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
