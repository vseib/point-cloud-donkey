/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "voting.h"
#include "voting_factory.h"
#include "maxima_handler.h"
#include "../codebook/codeword_distribution.h"

#include <fstream>

#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/common/centroid.h>

namespace ism3d
{

Voting::Voting()
{
    addParameter(m_minThreshold, "MinThreshold", 0.0f);
    addParameter(m_minVotesThreshold, "MinVotesThreshold", 1);
    addParameter(m_bestK, "BestK", -1);
    addParameter(m_averageRotation, "AverageRotation", false);
    addParameter(m_radiusType, "BinOrBandwidthType", std::string("Config"));
    addParameter(m_radiusFactor, "BinOrBandwidthFactor", 1.0f);
    addParameter(m_max_filter_type, "MaxFilterType", std::string("None"));
    addParameter(m_max_type_param, "SingleObjectMaxType", std::string("Default"));
    addParameter(m_single_object_mode, "SingleObjectMode", false);

    addParameter(m_use_global_features, "UseGlobalFeatures", false);
    addParameter(m_global_feature_method, "GlobalFeaturesStrategy", std::string("KNN"));
    addParameter(m_k_global_features, "GlobalFeaturesK", 1);
    addParameter(m_merge_function, "GlobalFeatureInfluenceType", 3);
    addParameter(m_min_svm_score, "GlobalParamMinSvmScore", 0.70f);
    addParameter(m_rate_limit, "GlobalParamRateLimit", 0.60f);
    addParameter(m_weight_factor, "GlobalParamWeightFactor", 1.5f);

    // TODO VS: use these params for later when adding ransac filtering
    addParameter(m_vote_filtering_with_ransac, "RansacVoteFiltering", false);
    addParameter(m_refine_model, "RansacRefineModel", false);
    addParameter(m_inlier_threshold, "RansacInlierThreshold", 0.1f);
}

Voting::~Voting()
{
    m_votes.clear();
}

void Voting::vote(Eigen::Vector3f position, float weight, unsigned classId, unsigned instanceId,
                  const Eigen::Vector3f& keypoint, const Utils::BoundingBox& boundingBox, const std::shared_ptr<Codeword> &codeword)
{
    // add the vote
    Vote newVote;
    newVote.position = position; // position of object center the vote votes for
    newVote.weight = weight;
    newVote.classId = classId;
    newVote.instanceId = instanceId;
    newVote.keypoint = keypoint;
    newVote.boundingBox = boundingBox;
    newVote.codewordId = codeword->getId();
    // this is used for ransac in maxima filtering
    newVote.keypoint_training = codeword->getFeaturePosition();

#pragma omp critical
    {
        m_votes[classId].push_back(newVote);
    }
}

std::vector<VotingMaximum> Voting::findMaxima(pcl::PointCloud<PointT>::ConstPtr &points,
                                              pcl::PointCloud<pcl::Normal>::ConstPtr &normals)
{
    // forward values to helper function
    MaximaHandler::setSingleObjectMaxType(m_max_type_param);
    MaximaHandler::setRadiusType(m_radiusType);
    MaximaHandler::setRadiusFactor(m_radiusFactor);

    if (m_votes.size() == 0)
        return std::vector<VotingMaximum>();

    std::vector<VotingMaximum> maxima;

    // find votes for each class individually
    // iterate over map that assigns each class id with a list of votes
    for(auto[classId, votes] : m_votes)
    {
        std::vector<Eigen::Vector3f> clusters;  // positions of maxima for current class id
        // NOTE: maximaValues can be removed, since it is not used before filtering and needs to be recomputed after filtering!
        std::vector<double> maximaValues;       // weights of maxima
        std::vector<std::vector<unsigned>> instanceIds; // list of instance ids for each maximum
        std::vector<std::vector<Vote>> cluster_votes;      // list of votes for each maximum

        // process the algorithm to find maxima on the votes of the current class
        // TODO VS: add openmp inside find maxima!
        iFindMaxima(points, votes, clusters, maximaValues, instanceIds, cluster_votes, classId);

        LOG_ASSERT(clusters.size() == maximaValues.size());
        LOG_ASSERT(clusters.size() == cluster_votes.size());

        if(m_vote_filtering_with_ransac)
        {
            auto[clusters_filtered, cluster_votes_filtered] = filterVotesWithRansac(clusters, cluster_votes);
            clusters = clusters_filtered;
            cluster_votes = cluster_votes_filtered;
        }

        // iterate through all found maxima for current class ID
        #pragma omp parallel for
        for (int i = 0; i < (int)clusters.size(); i++)
        {
            if (cluster_votes[i].size() < m_minVotesThreshold || cluster_votes[i].size() == 0) // catch an accidental threshold of 0
                continue;

            const std::vector<Vote>& current_votes = cluster_votes[i];

            // TODO VS this should only be executed if secondary labels are used
            // determine instance id based on all ids and corresponding vote weights
            std::map<unsigned, float> instance_weights;
            for(unsigned idx = 0; idx < current_votes.size(); idx++)
            {
                float weight = current_votes[idx].weight;
                unsigned instance_id = instanceIds[i][idx];

                if(instance_weights.find(instance_id) != instance_weights.end())
                {
                    instance_weights.at(instance_id) += weight;
                }
                else
                {
                    instance_weights.insert({instance_id, weight});
                }
            }

            // find max value
            unsigned max_id_weights;
            float best_weight = 0;
            for(auto it : instance_weights)
            {
                float weight = instance_weights[it.first];
                if(weight > best_weight)
                {
                    best_weight = weight;
                    max_id_weights = it.first;
                }
            }

            VotingMaximum maximum;
            maximum.classId = classId;
            maximum.instanceId = max_id_weights;
            maximum.instanceWeight = instance_weights[max_id_weights];
            maximum.position = clusters[i];
            maximum.votes = cluster_votes[i];
            // init global result with available data
            maximum.globalHypothesis.classId = classId;
            maximum.globalHypothesis.instanceId = max_id_weights;
            maximum.globalHypothesis.instanceWeight = instance_weights[max_id_weights];

            std::vector<boost::math::quaternion<float>> quats;
            std::vector<float> weights;

            // compute weighted maximum values
            float maxWeight = 0;
            maximum.boundingBox.size = Eigen::Vector3f(0, 0, 0);
            for (unsigned j = 0; j < current_votes.size(); j++)
            {
                const Vote& vote = current_votes[j];
                float newWeight = vote.weight;

                boost::math::quaternion<float> rotQuat = vote.boundingBox.rotQuat;
                quats.push_back(rotQuat);
                weights.push_back(newWeight);

                maximum.boundingBox.size += newWeight * vote.boundingBox.size;
                maxWeight += newWeight;
            }
            maximum.weight = maxWeight;
            maximum.globalHypothesis.classWeight = maxWeight;


            // weights should sum up to one
            for (int j = 0; j < (int)weights.size(); j++)
                weights[j] /= maxWeight;

            maximum.boundingBox.position = maximum.position;
            maximum.boundingBox.size /= maxWeight;

            // compute interpolation between quaternions
            if (m_averageRotation)
            {
                boost::math::quaternion<float> rotQuat;
                Utils::quatWeightedAverage(quats, weights, rotQuat);
                maximum.boundingBox.rotQuat = rotQuat;
            }

            // in non-single object mode: extract points around each maxima region and compute global features
            if(m_use_global_features && !m_single_object_mode)
            {
                pcl::PointCloud<PointT>::Ptr segmented_points(new pcl::PointCloud<PointT>());
                pcl::PointCloud<pcl::Normal>::Ptr segmented_normals(new pcl::PointCloud<pcl::Normal>());

                m_global_classifier->segmentROI(points, normals, maximum, segmented_points, segmented_normals);
                m_global_classifier->classify(segmented_points, segmented_normals, maximum);
            }

            #pragma omp critical
            {
                maxima.push_back(maximum);
            }
        }
    }

    // in single object mode: compute global features on the whole cloud once
    if(m_use_global_features && m_single_object_mode)
    {
        VotingMaximum global_result;
        m_global_classifier->classify(points, normals, global_result);

        // add global result to all maxima if in single object mode
        for(VotingMaximum &max : maxima)
            max.globalHypothesis = global_result.globalHypothesis;

        // if no maxima found in single object mode, use global hypothesis and fill in values
        if(maxima.size() == 0)
        {
            global_result.classId = global_result.globalHypothesis.classId;
            global_result.weight = global_result.globalHypothesis.classWeight;
            global_result.instanceId = global_result.globalHypothesis.instanceId;
            Eigen::Vector4d centroid;
            pcl::compute3DCentroid(*points, centroid);
            global_result.position = Eigen::Vector3f(centroid.x(), centroid.y(), centroid.z());
            global_result.boundingBox = Utils::computeMVBB<PointT>(points);
            maxima.push_back(global_result);
        }
    }

    // filter maxima if not in single object mode
    std::vector<VotingMaximum> filtered_maxima = maxima; // init for the case that no filtering type is selected
    if(!m_single_object_mode)
    {
        filtered_maxima = MaximaHandler::filterMaxima(m_max_filter_type, maxima);
    }
    maxima = filtered_maxima;

    // sort maxima
    std::sort(maxima.begin(), maxima.end(), Voting::sortMaxima);

    // add global features to result classification
    if(m_use_global_features)
    {
        m_global_classifier->setMergeParams(m_min_svm_score, m_rate_limit, m_weight_factor);
        m_global_classifier->mergeGlobalAndLocalHypotheses(m_merge_function, maxima);
        // global features might have changed weights
        std::sort(maxima.begin(), maxima.end(), Voting::sortMaxima);
    }

    // turn weights to probabilities
    normalizeWeights(maxima);
//    softmaxWeights(maxima);

    // filter low weight maxima
    // NOTE: if m_minThreshold is positive: filter absolute values
    //       if m_minThreshold is negative: filter relative to highest maximum
    float weight_threshold = m_minThreshold;
    if(weight_threshold < 0)
    {
        float max_weight = maxima.size() > 0 ? maxima.front().weight : 0.0f;
        weight_threshold = -weight_threshold * max_weight;
    }

    filtered_maxima.clear();
    for (int i = 0; i < (int)maxima.size(); i++)
    {
        if(maxima[i].weight >= weight_threshold)
        {
            filtered_maxima.push_back(maxima[i]);
        }
    }
    maxima = filtered_maxima;

    // only keep the best k maxima, if specified
    if (m_bestK > 0 && maxima.size() >= m_bestK)
        maxima.erase(maxima.begin() + m_bestK, maxima.end());

    printMaxima(maxima);

    return maxima;
}


void Voting::printMaxima(const std::vector<VotingMaximum> &maxima)
{
    for (int i = 0; i < (int)maxima.size(); i++)
    {
        // TODO VS: format output nicely
        const VotingMaximum& max = maxima[i];
        if(m_use_global_features)
        {
            LOG_INFO("maximum " << i << ", class: " << max.classId <<
                     ", weight: " << max.weight <<
                     ", instance: " << max.instanceId << " (" << max.instanceWeight << ")" <<
                     ", glob: (" << max.globalHypothesis.classId << ", " << max.globalHypothesis.classWeight << ")" <<
                     ", num votes: " << max.votes.size());
        }
        else
        {
            LOG_INFO("maximum " << i << ", class: " << max.classId <<
                     ", weight: " << max.weight <<
                     ", instance: " << max.instanceId << " (" << max.instanceWeight << ")" <<
                     ", num votes: " << max.votes.size());
        }
    }
}


std::tuple<std::vector<Eigen::Vector3f>, std::vector<std::vector<Vote>>> Voting::filterVotesWithRansac(
        const std::vector<Eigen::Vector3f> &clusters,
        const std::vector<std::vector<Vote>> &cluster_votes) const
{
    pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> corr_rejector;
    corr_rejector.setMaximumIterations(10000);
    // TODO VS params!!!
    // for classification: around 0.05 to 0.1 ? // for detection: around 0.01 ?
    corr_rejector.setInlierThreshold(m_inlier_threshold);
    corr_rejector.setRefineModel(m_refine_model);

    std::vector<Eigen::Matrix4f> transformations;
    std::vector<pcl::Correspondences> model_instances;
    std::vector<Eigen::Vector3f> clusters_filtered;
    std::vector<std::vector<Vote>> cluster_votes_filtered;

    for(unsigned j = 0; j < cluster_votes.size(); j++)
    {
        if (cluster_votes[j].size() < m_minVotesThreshold || cluster_votes[j].size() == 0) // catch an accidental threshold of 0
            continue;

        const std::vector<Vote>& vote_list = cluster_votes[j];

        // get corresponding keypoints
        pcl::PointCloud<PointT>::Ptr scene_keypoints(new pcl::PointCloud<PointT>());
        pcl::PointCloud<PointT>::Ptr object_keypoints(new pcl::PointCloud<PointT>());
        pcl::Correspondences initial_corrs, filtered_corrs;
        for(unsigned i = 0; i < vote_list.size(); i++)
        {
            const Vote &v = vote_list.at(i);

            PointT scenekeyp;
            scenekeyp.x = v.keypoint.x();
            scenekeyp.y = v.keypoint.y();
            scenekeyp.z = v.keypoint.z();
            scene_keypoints->push_back(scenekeyp);

            PointT objectkeyp;
            objectkeyp.x = v.keypoint_training.x();
            objectkeyp.y = v.keypoint_training.y();
            objectkeyp.z = v.keypoint_training.z();
            object_keypoints->push_back(objectkeyp);

            initial_corrs.emplace_back(pcl::Correspondence(i, i, 1)); // TODO VS: check if distance is used!
        }

        // correspondence rejection with ransac
        corr_rejector.setInputSource(object_keypoints);
        corr_rejector.setInputTarget(scene_keypoints);
        corr_rejector.setSaveInliers(true);
        corr_rejector.getRemainingCorrespondences(initial_corrs, filtered_corrs);
        std::vector<int> inlier_indices;
        corr_rejector.getInliersIndices(inlier_indices);

        Eigen::Matrix4f best_transform = corr_rejector.getBestTransformation();
        // save transformations for recognition if RANSAC was run successfully
        if(!best_transform.isIdentity(0.0001))
        {
            // keep good clusters
            clusters_filtered.push_back(clusters[j]);

            // keep only inlier votes of cluster
            std::vector<Vote> inlier_votes;
            for(int inlier_idx : inlier_indices)
            {
                inlier_votes.push_back(vote_list[inlier_idx]);
            }
            cluster_votes_filtered.push_back(inlier_votes);

            // keep transformation and correspondences
            transformations.push_back(best_transform);
            model_instances.push_back(filtered_corrs);
        }
    }

    return std::make_tuple(clusters_filtered, cluster_votes_filtered);
}

// TODO VS: check if this is still needed
bool Voting::sortMaxima(const VotingMaximum& maxA, const VotingMaximum& maxB)
{
    return maxA.weight > maxB.weight;
}

void Voting::normalizeWeights(std::vector<VotingMaximum> &maxima)
{
    float sum = 0;
    float sum_instance = 0;
    float sum_global = 0;
    float sum_instance_global = 0;
    for(const VotingMaximum &max : maxima)
    {
        sum += max.weight;
        sum_instance += max.instanceWeight;
        sum_global += max.globalHypothesis.classWeight;
        sum_instance_global += max.globalHypothesis.instanceWeight;
    }

    for(VotingMaximum &max : maxima)
    {
        max.weight = sum != 0 ? max.weight/sum : 0;
        max.instanceWeight = sum_instance != 0 ? max.instanceWeight/sum_instance : 0;
        max.globalHypothesis.classWeight = sum_global != 0 ? max.globalHypothesis.classWeight/sum_global : 0;
        max.globalHypothesis.instanceWeight = sum_instance_global != 0 ? max.globalHypothesis.instanceWeight/sum_instance_global : 0;
    }
}

void Voting::softmaxWeights(std::vector<VotingMaximum> &maxima)
{
    float sum = 0;
    float sum_instance = 0;
    float sum_global = 0;
    float sum_instance_global = 0;
    for(const VotingMaximum &max : maxima)
    {
        sum += std::exp(max.weight);
        sum_instance += std::exp(max.instanceWeight);
        sum_global += std::exp(max.globalHypothesis.classWeight);
        sum_instance_global += std::exp(max.globalHypothesis.instanceWeight);
    }

    for(VotingMaximum &max : maxima)
    {
        max.weight = std::exp(max.weight) / sum;
        max.instanceWeight = std::exp(max.instanceWeight) / sum_instance;
        max.globalHypothesis.classWeight = std::exp(max.globalHypothesis.classWeight) / sum_global;
        max.globalHypothesis.instanceWeight = std::exp(max.globalHypothesis.instanceWeight) / sum_instance_global;
    }
}

const std::map<unsigned, std::vector<Vote> >& Voting::getVotes() const
{
    return m_votes;
}

void Voting::clear()
{
    m_votes.clear();
}

void Voting::determineAverageBoundingBoxDimensions(const std::map<unsigned, std::vector<Utils::BoundingBox> > &boundingBoxes)
{
    m_dimensions_map.clear();
    m_variance_map.clear();

    for(auto it : boundingBoxes)
    {
        unsigned classId = it.first;
        float max_accu = 0;
        float max_accuSqr = 0;
        float med_accu = 0;
        float med_accuSqr = 0;

        // check each bounding box of this class id
        for(auto box : it.second)
        {
            float max = box.size.maxCoeff();
            float min = box.size.minCoeff();
            // find the other value
            float med = box.size[0];
            for(int i = 1; i < 3; i++)
            {
                if(med == max || med == min)
                {
                    med = box.size[i];
                }
            }

            // use "radius" of bb dimensions, i.e. half of the sizes
            max_accu += max/2;
            med_accu += med/2;
            max_accuSqr += ((max/2)*(max/2));
            med_accuSqr += ((med/2)*(med/2));
        }

        // compute average
        max_accu /= it.second.size();
        med_accu /= it.second.size();
        max_accuSqr /= it.second.size();
        med_accuSqr /= it.second.size();

        // compute variance
        float max_var = max_accuSqr - (max_accu*max_accu);
        float med_var = med_accuSqr - (med_accu*med_accu);
        m_dimensions_map.insert({classId, {max_accu, med_accu}});
        m_variance_map.insert({classId, {max_var, med_var}});
    }
}


void Voting::forwardGlobalFeatures(std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &globalFeatures)
{
    m_global_features = globalFeatures;
}

void Voting::iSaveData(boost::archive::binary_oarchive &oa) const
{
    // fill in bounding box information
    unsigned bb_dims_size = m_dimensions_map.size();
    oa << bb_dims_size;
    for(auto it : m_dimensions_map)
    {
        unsigned classId = it.first;
        float firstDim = it.second.first;
        float secondDim = it.second.second;
        oa << classId;
        oa << firstDim;
        oa << secondDim;
    }

    unsigned bb_vars_size = m_variance_map.size();
    oa << bb_vars_size;
    for(auto it : m_variance_map)
    {
        unsigned classId = it.first;
        float firstVar = it.second.first;
        float secondVar = it.second.second;
        oa << classId;
        oa << firstVar;
        oa << secondVar;
    }

    // fill in global features
    unsigned glob_feat_size = m_global_features.size();
    oa << glob_feat_size;
    for(auto it : m_global_features)
    {
        unsigned classId = it.first; // descriptor type is same for all features, only store it once
        oa << classId;

        unsigned cloud_size = it.second.size();
        oa << cloud_size;
        for(auto feat_cloud : it.second) // iterate over each vector element (point cloud) of one class
        {
            unsigned feat_size = feat_cloud->points.size();
            oa << feat_size;
            for(auto feat : feat_cloud->points) // iterate over each descriptor (point in the cloud)
            {
                // save reference frame
                for(unsigned i = 0; i < 9; i++)
                {
                    oa << feat.referenceFrame.rf[i];
                }
                // save descriptor
                oa << feat.descriptor;
                oa << feat.globalDescriptorRadius;
                oa << feat.instanceId;
            }
        }
    }
}

bool Voting::iLoadData(boost::archive::binary_iarchive &ia)
{
    // read bounding box data
    m_dimensions_map.clear();
    m_variance_map.clear();

    unsigned bb_dims_size;
    ia >> bb_dims_size;
    for(int i = 0; i < bb_dims_size; i++)
    {
        unsigned classId;
        float firstDim;
        float secondDim;
        ia >> classId;
        ia >> firstDim;
        ia >> secondDim;
        m_dimensions_map.insert({classId, {firstDim, secondDim}});
    }

    unsigned bb_vars_size;
    ia >> bb_vars_size;
    for(int i = 0; i < bb_vars_size; i++)
    {
        unsigned classId;
        float firstVar;
        float secondVar;
        ia >> classId;
        ia >> firstVar;
        ia >> secondVar;
        m_variance_map.insert({classId, {firstVar, secondVar}});
    }

    MaximaHandler::setBoundingBoxMaps(m_dimensions_map, m_variance_map);

    // NOTE: even if global features are not used, they must be completely deserialized!
    // load all global features from training into a single cloud
    pcl::PointCloud<ISMFeature>::Ptr global_features_cloud(new pcl::PointCloud<ISMFeature>());
    std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> global_features_map;
    int descriptor_length;

    unsigned global_feat_size;
    ia >> global_feat_size;
    for(int i = 0; i < global_feat_size; i++)
    {
        unsigned classId;
        ia >> classId;

        std::vector<pcl::PointCloud<ISMFeature>::Ptr> cloud_vector;

        unsigned cloud_size;
        ia >> cloud_size;
        for(unsigned j = 0; j < cloud_size; j++)
        {
            pcl::PointCloud<ISMFeature>::Ptr temp_cloud(new pcl::PointCloud<ISMFeature>());

            unsigned feat_size;
            ia >> feat_size;
            for(int k = 0; k < feat_size; k++)
            {
                pcl::ReferenceFrame referenceFrame;
                for(int i_ref = 0; i_ref < 9; i_ref++)
                {
                    float ref;
                    ia >> ref;
                    referenceFrame.rf[i_ref] = ref;
                }

                std::vector<float> descriptor;
                ia >> descriptor;
                float radius;
                ia >> radius;
                unsigned instanceId;
                ia >> instanceId;

                ISMFeature ism_feature;
                ism_feature.referenceFrame = referenceFrame;
                ism_feature.descriptor = descriptor;
                ism_feature.globalDescriptorRadius =  radius;
                ism_feature.classId = classId;
                ism_feature.instanceId = instanceId;
                temp_cloud->push_back(ism_feature);
                global_features_cloud->push_back(ism_feature);
                descriptor_length = ism_feature.descriptor.size(); // are all the same just overwrite
            }
            temp_cloud->height = 1;
            temp_cloud->width = temp_cloud->size();
            temp_cloud->is_dense = false;
            cloud_vector.push_back(temp_cloud);
        }
        global_features_map.insert({classId, cloud_vector});
    }

    if(m_use_global_features)
    {
        // create flann index
        std::shared_ptr<FlannHelper> fh = std::make_shared<FlannHelper>(descriptor_length, global_features_cloud->size());
        fh->createDataset(global_features_cloud);
        // NOTE: index will be build when the first object is recognized - otherwise parameters are not initialized from config, but with default values
        //fh->buildIndex(m_distanceType, m_num_kd_trees);

        m_global_classifier = std::make_shared<GlobalClassifier>(
                    m_global_feature_descriptor,
                    m_global_feature_method,
                    m_k_global_features);
        m_global_classifier->setFlannHelper(fh);
        m_global_classifier->setLoadedFeatures(global_features_cloud);
        m_global_classifier->computeAverageRadii(global_features_map);
        m_global_classifier->loadSVMModels(m_svm_path);

        if(m_single_object_mode)
        {
            m_global_classifier->enableSingleObjectMode();
        }
    }
    return true;
}

}
