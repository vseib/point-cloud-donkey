/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_VOTINGMEANSHIFT_H
#define ISM3D_VOTINGMEANSHIFT_H

#include "voting.h"
#include <map>

namespace ism3d
{
    /**
     * @brief The VotingMeanShift class
     * Detects maxima in the voting space by performing a mean shift mode estimation. Starting from several
     * seed points, the algorithm computes a kernel-weighted mean of the local neighborhood inside a
     * bandwidth radius and shifts the position onto the mean. The algorithm is repeated until the
     * seed points converge to a maximum. Non-maxima supression is then performed to filter out those
     * points that ended up on the same maximum.
     */
    class VotingMeanShift
            : public Voting
    {
    public:
        VotingMeanShift();
        ~VotingMeanShift();

        static std::string getTypeStatic();
        std::string getType() const;

        virtual void clear();

        const std::map<unsigned, std::vector<std::vector<Eigen::Vector3f> > >& getTrajectories() const;

    protected:
        void iFindMaxima(pcl::PointCloud<PointT>::ConstPtr &points,
                         std::vector<Voting::Vote>&,
                         std::vector<Eigen::Vector3f>&,
                         std::vector<double>&,
                         std::vector<std::vector<unsigned>> &,
                         std::vector<std::vector<int>>&,
                         std::vector<std::vector<float>>&,
                         unsigned);
        float iGetSeedsRange() const;

        std::vector<Voting::Vote> getVotesInRadius(
                const pcl::search::KdTree<PointT>::Ptr search,
                const PointT query,
                const float radius,
                std::vector<Voting::Vote> votes);

        void iDoMeanShift(const std::vector<Voting::Vote>&,
                          const std::vector<Voting::Vote>&,
                          std::vector<Eigen::Vector3f>&,
                          std::vector<std::vector<Voting::Vote>> &cluster_votes,
                          std::vector<std::vector<Eigen::Vector3f> >&,
                          pcl::search::KdTree<PointT>::Ptr& search);

        float estimateDensity(Eigen::Vector3f position,
                              std::vector<float>& new_cluster_votes_weights,
                              const std::vector<Voting::Vote>& cluster_votes);

//        float estimateDensity(Eigen::Vector3f,
//                              int,
//                              std::vector<float>&,
//                              const std::vector<Voting::Vote>&,
//                              pcl::search::KdTree<PointT>::Ptr& search);

    private:
        bool computeMeanShift(const std::vector<Voting::Vote>&,
                              const Eigen::Vector3f& center,
                              std::vector<Voting::Vote> &current_votes,
                              Eigen::Vector3f& newCenter,
                              pcl::search::KdTree<PointT>::Ptr& search) const;


        static bool mapCompareVector(const Eigen::Vector3i&, const Eigen::Vector3i&);
        std::vector<Vote> createSeeds(const std::vector<Voting::Vote>&, float);

        float kernel(float) const;
        float kernelDerivative(float) const;
        float kernelGaussian(float) const;
        float kernelDerivedGaussian(float) const;
        float kernelUniform(float) const;
        float kernelDerivedUniform(float) const;

        // NOTE: trajectories are saved for each class id and store a path of 3d positions for each
        // seed point
        std::map<unsigned, std::vector<std::vector<Eigen::Vector3f> > > m_trajectories;

        std::string m_kernel; // kernel type
        float m_bandwidth;  // radius
        float m_threshold;  // termination threshold
        int m_maxIter;      // maximum number of iterations until termination
        std::string m_maxima_suppression_type;
    };
}

#endif // ISM3D_VOTINGMEANSHIFT_H
