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
                         const std::vector<Voting::Vote>&,
                         std::vector<Eigen::Vector3f>&,
                         std::vector<double>&,
                         std::vector<std::vector<int> >&,
                         std::vector<std::vector<float> >&,
                         unsigned, float &radius);
        float iGetSeedsRange() const;
        void iDoMeanShift(const std::vector<Voting::Vote>&,
                          const std::vector<Voting::Vote>&,
                          std::vector<Eigen::Vector3f>&,
                          std::vector<std::vector<Eigen::Vector3f> >&,
                          pcl::search::KdTree<PointT>::Ptr& search);
        float estimateDensity(Eigen::Vector3f,
                              int,
                              std::vector<float>&,
                              const std::vector<Voting::Vote>&,
                              pcl::search::KdTree<PointT>::Ptr& search);

        // only the first maximum in the radius is retained
        void suppressNeighborMaxima(const std::vector<Eigen::Vector3f>&,
                                  std::vector<Eigen::Vector3f>&) const;
        // the average of the maxima in the radius is retained
        void averageNeighborMaxima(const std::vector<Eigen::Vector3f>&,
                                  std::vector<Eigen::Vector3f>&) const;
        // the average of the maxima and its [neighbor's neighbor's ...] neighbors in the radius is retained
        void averageShiftNeighborMaxima(const std::vector<Eigen::Vector3f>&,
                                  std::vector<Eigen::Vector3f>&) const;

    private:
        bool computeMeanShift(const std::vector<Voting::Vote>&,
                              const Eigen::Vector3f& center,
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

        // each position index in this vector represents a vote index, at each position index is the cluster index, the vote belongs to
        std::vector<int> m_clusterIndices;

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
