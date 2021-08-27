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
                         std::vector<Vote>&,
                         std::vector<Eigen::Vector3f>&,
                         std::vector<double>&,
                         std::vector<std::vector<unsigned>> &,
                         std::vector<std::vector<Vote>>& votes_per_cluster,
                         unsigned);
        float iGetSeedsRange() const;

        std::vector<Vote> getVotesInRadius(
                const pcl::search::KdTree<PointT>::Ptr search,
                const PointT query,
                const float radius,
                std::vector<Vote> &votes);

        void iDoMeanShift(const std::vector<Vote>&,
                          const std::vector<Vote>&,
                          std::vector<Eigen::Vector3f>&,
                          std::vector<std::vector<Eigen::Vector3f> >& trajectories,
                          pcl::search::KdTree<PointT>::Ptr& search);

        // TODO VS: unify both methods with additional bool to enable / disable reweighting
        float estimateDensity(Eigen::Vector3f position,
                              const std::vector<Vote>& votes,
                              std::vector<Vote> &cluster_votes,
                              pcl::search::KdTree<PointT>::Ptr& search);

        float estimateDensityAndReweightVotes(Eigen::Vector3f,
                              std::vector<Vote>& votes,
                              std::vector<Vote>& cluster_votes,
                              pcl::search::KdTree<PointT>::Ptr& search);

    private:
        bool computeMeanShift(const std::vector<Vote>&,
                              const Eigen::Vector3f& center,
                              Eigen::Vector3f& newCenter,
                              pcl::search::KdTree<PointT>::Ptr& search) const;


        static bool mapCompareVector(const Eigen::Vector3i&, const Eigen::Vector3i&);
        std::vector<Vote> createSeeds(const std::vector<Vote>&, float);

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
