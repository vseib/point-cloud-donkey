/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "voting_mean_shift.h"
#include <omp.h>
#include <pcl/filters/filter.h>

#include "single_object_mode_helper.h"
#include "maxima_handler.h"

namespace ism3d
{
VotingMeanShift::VotingMeanShift()
{
    addParameter(m_bandwidth, "Bandwidth", 0.2f);
    addParameter(m_threshold, "Threshold", 1e-3f);
    addParameter(m_maxIter, "MaxIter", 1000);
    addParameter(m_kernel, "Kernel", std::string("Gaussian"));
    addParameter(m_maxima_suppression_type, "MaximaSuppression", std::string("Average"));
}

VotingMeanShift::~VotingMeanShift()
{
}

float VotingMeanShift::iGetSeedsRange() const
{
    // a cube with this length fits perfectly inside the circle with radius m_bandwidth
    return (m_bandwidth * 2.0f) / sqrtf(2);
}

void VotingMeanShift::iFindMaxima(pcl::PointCloud<PointT>::ConstPtr &points,
                                  std::vector<Vote>& votes, // all votes in the voting space for this class ID
                                  std::vector<Eigen::Vector3f>& maximum_positions, // maxima positions
                                  std::vector<double>& maximum_weights, // weights for all maxima
                                  std::vector<std::vector<unsigned>>& instanceIdsPerCluster, // list of instance ids that belong to each maximum index
                                  std::vector<std::vector<Vote>>& votes_per_cluster, // holds a list of votes that belong to each maximum index
                                  unsigned classId)
{
    // determine bandwidth based on config
    MaximaHandler::setRadius(m_bandwidth);
    m_bandwidth = MaximaHandler::getSearchDistForClass(classId);

    // basic ideas adapted from
    // https://github.com/daviddoria/vtkMeanShiftClustering/blob/master/vtkMeanShiftClustering.cxx and
    // https://code.google.com/p/accord/source/browse/trunk/Sources/Accord.MachineLearning/Clustering/MeanShift.cs

    // build dataset
    pcl::PointCloud<PointT>::Ptr dataset(new pcl::PointCloud<PointT>());
    dataset->resize(votes.size());
    for (int i = 0; i < (int)votes.size(); i++)
    {
        const Vote& vote = votes[i];
        PointT votePoint;
        votePoint.x = vote.position[0];
        votePoint.y = vote.position[1];
        votePoint.z = vote.position[2];
        dataset->at(i) = votePoint;
    }
    dataset->height = 1;
    dataset->width = dataset->size();
    dataset->is_dense = false;

    // use a kd-tree for exact nearest neighbor search
    pcl::search::KdTree<PointT>::Ptr search(new pcl::search::KdTree<PointT>());
    search->setInputCloud(dataset);


    // default behavior:
    // 1) not single object mode, max type doesn't matter --> perform mean-shift to find maxima
    // 2) single object mode only with default max type   --> perform mean-shift to find maxima
    //    (this effectively disables single object mode for local features, but not for global ones)
    if(!m_single_object_mode || (m_single_object_mode && MaximaHandler::m_max_type == SingleObjectMaxType::DEFAULT))
    {
        // create seed points using binning strategy
        std::vector<Vote> seeds = createSeeds(votes, iGetSeedsRange());

        // perform mean shift
        std::vector<Eigen::Vector3f> cluster_centers;
        iDoMeanShift(seeds, votes, cluster_centers, m_trajectories[classId], search);

        // estimate densities for cluster centers and reweight votes by the kernel value
        std::vector<float> densities;
        for (unsigned i = 0; i < cluster_centers.size(); i++)
        {
            std::vector<Vote> cluster_votes;
            float d = estimateDensity(cluster_centers[i], votes, cluster_votes, search);
            densities.push_back(std::move(d));
            votes_per_cluster.push_back(std::move(cluster_votes));
        }

        if(m_maxima_suppression_type == "Average")
        {
            std::vector<Eigen::Vector3f> maxima;
            // weighted average with densities
            MaximaHandler::averageNeighborMaxima(cluster_centers, m_bandwidth, maxima, densities);
            // cluster centers changed, recalculated densities
            densities.clear();
            votes_per_cluster.clear();
            for (unsigned i = 0; i < maxima.size(); i++)
            {
                std::vector<Vote> cluster_votes;
                float d = estimateDensity(maxima[i], votes, cluster_votes, search);
                densities.push_back(std::move(d));
                votes_per_cluster.push_back(std::move(cluster_votes));
            }
            cluster_centers = std::move(maxima);
        }

        // NOTE: if "Average", subsequent "Suppress" must be executed to avoid keeping maxima closer than m_bandwidth
        // TODO VS: this "if" is not necessary, since there is only "average" and "suppress"
        if(m_maxima_suppression_type == "Average" || m_maxima_suppression_type == "Suppress")
        {
            MaximaHandler::suppressNeighborMaxima(cluster_centers, densities, m_bandwidth, maximum_positions);
        }
    }
    // in single object mode we assume that the whole voting space contains only one object
    // in such case we do not need mean-shift, but solely estimate the density with differnt
    // bandwidths depending on the single object mode max type
    else
    {
        // use object's centroid as query point for search
        Eigen::Vector4f centr;
        pcl::compute3DCentroid(*points, centr);
        PointT query;
        query.x = centr[0];
        query.y = centr[1];
        query.z = centr[2];

	// TODO VS: for each of these: check if method
	// VotingMeanShift::getVotesInRadius is useful

        if(MaximaHandler::m_max_type == SingleObjectMaxType::BANDWIDTH)
        {
            m_bandwidth = MaximaHandler::getSearchDistForClass(classId);
        }
        if(MaximaHandler::m_max_type == SingleObjectMaxType::MODEL_RADIUS)
        {
            m_bandwidth = SingleObjectHelper::getModelRadius(points, query);
        }
        if(MaximaHandler::m_max_type == SingleObjectMaxType::COMPLETE_VOTING_SPACE)
        {
            m_bandwidth = SingleObjectHelper::getVotingSpaceSize(votes, query);
        }

        // single object mode has only one cluster
        maximum_positions.clear();
        maximum_positions.push_back(query.getVector3fMap());
    }

    // estimate densities for cluster positions and reweight votes by the kernel value
    votes_per_cluster.clear();
    maximum_weights.clear();
    for (unsigned i = 0; i < maximum_positions.size(); i++)
    {
        // assigned clusters indices are changed
        std::vector<Vote> cluster_votes;
        float density = estimateDensityAndReweightVotes(maximum_positions[i], votes, cluster_votes, search);

        std::vector<unsigned> clusterInstances;
        for(auto elem : cluster_votes)
        {
            clusterInstances.push_back(elem.instanceId);
        }
        instanceIdsPerCluster.push_back(clusterInstances);

        votes_per_cluster.push_back(std::move(cluster_votes));
        maximum_weights.push_back(std::move(density));
    }
}

// TODO VS: check if this method is still needed
std::vector<Vote> VotingMeanShift::getVotesInRadius(const pcl::search::KdTree<PointT>::Ptr search,
        const PointT query,
        const float radius,
        std::vector<Vote> &votes)
{
    // find nearest points within search window
    std::vector<int> indices;
    std::vector<float> distances;

    std::vector<Vote> resulting_votes;

    search->radiusSearch(query, radius, indices, distances);
    for (int i = 0; i < (int)indices.size(); i++)
    {
        const Vote& vote = votes[indices[i]];
        resulting_votes.push_back(vote);
    }
    return resulting_votes;
}


void VotingMeanShift::iDoMeanShift(const std::vector<Vote>& seeds,
                                   const std::vector<Vote>& votes,
                                   std::vector<Eigen::Vector3f>& clusterCenters,
                                   std::vector<std::vector<Eigen::Vector3f> >& trajectories,
                                   pcl::search::KdTree<PointT>::Ptr& search)
{
    // iterate all the points
    // OMP causes memory allocation errors in vector clusterCenters
    //#pragma omp parallel for shared(seeds, trajectories, clusterCenters)
    for (int i = 0; i < (int)seeds.size(); i++)
    {
        const Vote& seed = seeds[i];

        Eigen::Vector3f currentCenter = seed.position;

        // find cluster center for current point
        int iter = 0;
        float diff = 0;
        bool skipVote = false;
        std::vector<Eigen::Vector3f> trajectory;
        do {
            Eigen::Vector3f shiftedCenter;
            if (!computeMeanShift(votes, currentCenter, shiftedCenter, search))
            {
                skipVote = true;
                break;
            }
            else
                trajectory.push_back(currentCenter);

            diff = (currentCenter - shiftedCenter).norm();

            currentCenter = shiftedCenter;

            iter++;
        } while (diff > m_threshold && iter <= m_maxIter);

        if (!skipVote) {
            clusterCenters.push_back(currentCenter);
            trajectory.push_back(currentCenter);
            trajectories.push_back(trajectory);
        }
    }
}


float VotingMeanShift::estimateDensity(Eigen::Vector3f position,
                                       const std::vector<Vote> &votes,
                                       std::vector<Vote> &cluster_votes,
                                       pcl::search::KdTree<PointT>::Ptr &search)
{
    // find nearest points within search window
    PointT query;
    query.x = position[0];
    query.y = position[1];
    query.z = position[2];
    std::vector<int> indices;
    std::vector<float> distances;

    search->radiusSearch(query, m_bandwidth, indices, distances);

    // shouldn't happen
    if (indices.size() == 0 || distances.size() == 0)
        return 0;

    float density = 0;
    for (unsigned i = 0; i < indices.size(); i++)
    {
        const Vote& vote = votes[indices[i]];

        // get euclidean distance between current center and nearest neighbor
        float distanceSqr = distances[i];

        // compute a normalized distance in {0, 1}
        float u = distanceSqr / (m_bandwidth * m_bandwidth);

        // compute weights
        float weight = kernel(u) * vote.weight;

        cluster_votes.push_back(std::move(vote));

        density += weight;
    }
    return density;
}



float VotingMeanShift::estimateDensityAndReweightVotes(Eigen::Vector3f position,
                                       std::vector<Vote>& votes,
                                       std::vector<Vote>& cluster_votes,
                                       pcl::search::KdTree<PointT>::Ptr& search)
{
    // find nearest points within search window
    PointT query;
    query.x = position[0];
    query.y = position[1];
    query.z = position[2];
    std::vector<int> indices;
    std::vector<float> distances;

    search->radiusSearch(query, m_bandwidth, indices, distances);

    // shouldn't happen
    if (indices.size() == 0 || distances.size() == 0)
        return 0;

    float density = 0;
    for (unsigned i = 0; i < indices.size(); i++)
    {
        Vote& vote = votes[indices[i]];

        // get euclidean distance between current center and nearest neighbor
        float distanceSqr = distances[i];

        // compute a normalized distance in {0, 1}
        float u = distanceSqr / (m_bandwidth * m_bandwidth);

        // compute weights
        float weight = kernel(u) * vote.weight;

        vote.weight = weight;
        cluster_votes.push_back(vote);

        density += weight;
    }
    return density;
}


bool VotingMeanShift::computeMeanShift(const std::vector<Vote>& votes,
                                       const Eigen::Vector3f& center,
                                       Eigen::Vector3f& newCenter,
                                       pcl::search::KdTree<PointT>::Ptr& search) const
{
    // find nearest points within search window
    PointT query;
    query.x = center[0];
    query.y = center[1];
    query.z = center[2];
    std::vector<int> indices;
    std::vector<float> distances;
    search->radiusSearch(query, m_bandwidth, indices, distances);

    // shouldn't happen
    if (indices.size() == 0 || distances.size() == 0)
        return false;

    Eigen::Vector3f shifted(0, 0, 0);
    double totalWeight = 0;
    for (int i = 0; i < (int)indices.size(); i++)
    {
        const Vote& vote = votes[indices[i]];

        // get euclidean distance between current center and nearest neighbor
        float distanceSqr = distances[i];

        // compute a normalized distance in {0, 1}
        float u = distanceSqr / (m_bandwidth * m_bandwidth);

        // compute weights
        float g = -kernelDerivative(u) * vote.weight;

        // update shifted position
        shifted += g * vote.position;
        totalWeight += g;
    }

    // normalize by sum of weights
    if (totalWeight != 0)
        shifted /= totalWeight;

    newCenter = shifted;

    return true;
}

float VotingMeanShift::kernel(float x) const
{
    if (m_kernel == "Gaussian")
        return kernelGaussian(x);
    else if (m_kernel == "Uniform")
        return kernelUniform(x);
    return 0;
}

float VotingMeanShift::kernelDerivative(float x) const
{
    if (m_kernel == "Gaussian")
        return kernelDerivedGaussian(x);
    else if (m_kernel == "Uniform")
        return kernelDerivedUniform(x);
    return 0;
}

float VotingMeanShift::kernelGaussian(float x) const
{
    float profile = exp(-0.5 * x);
    return profile;
}

float VotingMeanShift::kernelDerivedGaussian(float x) const
{
    float profile = exp(-0.5 * x);
    float derivative = -0.5f * profile;
    return derivative;
}

float VotingMeanShift::kernelUniform(float x) const
{
    return 1;   // not sure if correct
}

float VotingMeanShift::kernelDerivedUniform(float x) const
{
    return 1;
}

bool VotingMeanShift::mapCompareVector(const Eigen::Vector3i& vec1, const Eigen::Vector3i& vec2)
{
    if (vec1[2] < vec2[2])
        return true;
    if ((vec1[2] == vec2[2]) && (vec1[1] < vec2[1]))
        return true;
    if ((vec1[2] == vec2[2]) && (vec1[1] == vec2[1]) && (vec1[0] < vec2[0]))
        return true;

    return false;
}

std::vector<Vote> VotingMeanShift::createSeeds(const std::vector<Vote>& votes,
                                                       float binSize)
{
    if (binSize == 0)
        return votes;
    else
    {
        int minBin = 1;

        // allow custom hashing for Eigen::Vector3f
        bool(*compFunc)(const Eigen::Vector3i&, const Eigen::Vector3i&) = VotingMeanShift::mapCompareVector;
        std::map<Eigen::Vector3i, std::pair<int, float>, bool(*)(const Eigen::Vector3i&, const Eigen::Vector3i&)> map(compFunc);

        for (int i = 0; i < (int)votes.size(); i++)
        {
            const Eigen::Vector3f& pos = votes[i].position;
            Eigen::Vector3i key((int)std::floor((pos[0] / binSize) + 0.5),
                    (int)std::floor((pos[1] / binSize) + 0.5),
                    (int)std::floor((pos[2] / binSize) + 0.5));

            // increase counter and weight
            if (map.find(key) != map.end()) {
                map[key].first = map[key].first + 1;
                map[key].second = map[key].second + votes[i].weight;
            }
            else {
                map[key].first = 1;
                map[key].second = votes[i].weight;
            }
        }

        // create seeds for bins which contain more than one point
        // alternatively: create seeds for bins which have a weight above a threshold
        std::vector<Vote> seeds;

        for (std::map<Eigen::Vector3i, std::pair<int, float> >::const_iterator it = map.begin();
             it != map.end(); it++)
        {
            if (it->second.first >= minBin) {
                Vote newVote;
                newVote.position = Eigen::Vector3f(it->first[0] * binSize,
                        it->first[1] * binSize,
                        it->first[2] * binSize);
                newVote.weight = it->second.second;
                seeds.push_back(newVote);
            }
        }

        return seeds;
    }
}

void VotingMeanShift::clear()
{
    m_trajectories.clear();
    Voting::clear();
}

const std::map<unsigned, std::vector<std::vector<Eigen::Vector3f> > >& VotingMeanShift::getTrajectories() const
{
    return m_trajectories;
}

std::string VotingMeanShift::getTypeStatic()
{
    return "MeanShift";
}

std::string VotingMeanShift::getType() const
{
    return VotingMeanShift::getTypeStatic();
}
}
