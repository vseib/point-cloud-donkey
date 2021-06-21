#include "pipeline_building_blocks.h"
#include "../../implicit_shape_model/utils/utils.h"

#include <pcl/registration/correspondence_rejection_sample_consensus.h>


pcl::CorrespondencesPtr findNnCorrespondences(
        const pcl::PointCloud<ISMFeature>::Ptr& scene_features,
        const float matching_threshold,
        const flann::Index<flann::L2<float>> &index)
{
    pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());

    // loop over all features extracted from the scene
    #pragma omp parallel for
    for(int fe = 0; fe < scene_features->size(); fe++)
    {
        // insert the query point
        ISMFeature feature = scene_features->at(fe);
        flann::Matrix<float> query(new float[feature.descriptor.size()], 1, feature.descriptor.size());
        for(int i = 0; i < feature.descriptor.size(); i++)
        {
            query[0][i] = feature.descriptor.at(i);
        }

        // prepare results
        std::vector<std::vector<int>> indices;
        std::vector<std::vector<float>> distances;
        index.knnSearch(query, indices, distances, 1, flann::SearchParams(128));

        delete[] query.ptr();

        if(distances[0][0] < matching_threshold)
        {
            // query index is scene, match is codebook ("object")
            pcl::Correspondence corr(fe, indices[0][0], distances[0][0]);
            #pragma omp critical
            {
                model_scene_corrs->push_back(corr);
            }
        }
    }

    return model_scene_corrs;
}


void remapIndicesToLocalCloud(
        const pcl::CorrespondencesPtr object_scene_corrs,
        const pcl::PointCloud<ISMFeature>::Ptr all_features,
        const std::vector<Eigen::Vector3f> all_center_vectors,
        pcl::PointCloud<PointT>::Ptr &object_keypoints,
        pcl::PointCloud<ISMFeature>::Ptr &object_features,
        std::vector<Eigen::Vector3f> &object_center_vectors,
        pcl::PointCloud<pcl::ReferenceFrame>::Ptr &object_lrf)
{
    // object keypoints are simply the matched keypoints from the codebook
    // however in order not to pass the whole codebook, we need to adjust the index mapping
    for(unsigned i = 0; i < object_scene_corrs->size(); i++)
    {
        // create new list of keypoints and reassign the object (i.e. match) index
        pcl::Correspondence &corr = object_scene_corrs->at(i);
        int &index = corr.index_match;
        const ISMFeature &feat = all_features->at(index);
        object_features->push_back(feat);
        object_center_vectors.push_back(all_center_vectors.at(index));

        PointT keypoint = PointT(feat.x, feat.y, feat.z);
        object_keypoints->push_back(keypoint);

        pcl::ReferenceFrame lrf = feat.referenceFrame;
        object_lrf->push_back(lrf);
        // remap index to new position in the created point clouds
        // no longer referring to m_features, but now to object_features
        index = i;
    }
}


std::vector<Eigen::Vector3f> prepareCenterVotes(
        const pcl::CorrespondencesPtr object_scene_corrs,
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        const std::vector<Eigen::Vector3f> object_center_vectors)
{
    std::vector<Eigen::Vector3f> votelist;
    for(const pcl::Correspondence &corr : *object_scene_corrs)
    {
        const ISMFeature &scene_feat = scene_features->at(corr.index_query);
        pcl::ReferenceFrame ref = scene_feat.referenceFrame;
        Eigen::Vector3f keyPos(scene_feat.x, scene_feat.y, scene_feat.z);

        Eigen::Vector3f vote = object_center_vectors.at(corr.index_match);
        Eigen::Vector3f center = keyPos + Utils::rotateBack(vote, ref);
        votelist.push_back(center);
    }
    return votelist;
}


void castVotesAndFindMaxima(
        const pcl::CorrespondencesPtr object_scene_corrs,
        const std::vector<Eigen::Vector3f> &votelist,
        const float relative_threshold,
        const bool use_distance_weight,
        std::vector<double> &maxima,
        std::vector<std::vector<int>> &vote_indices,
        std::shared_ptr<pcl::recognition::HoughSpace3D> &hough_space)
{
    hough_space->reset();
    for(int vid = 0; vid < votelist.size(); vid++)
    {
        Eigen::Vector3d vote(votelist[vid].x(), votelist[vid].y(), votelist[vid].z());
        // voting with interpolation
        // votes are in the same order as correspondences, vid is therefore the index in the correspondence list
        if(use_distance_weight)
        {
            float dist = object_scene_corrs->at(vid).distance;
            float weight = 1 - dist;
            weight = weight > 0 ? weight : 0;
            hough_space->voteInt(vote, weight, vid);
        }
        else
        {
            hough_space->voteInt(vote, 1.0, vid);
        }
    }
    hough_space->findMaxima(-relative_threshold, maxima, vote_indices);
}


void generateHypothesesWithAbsoluteOrientation(
        const pcl::CorrespondencesPtr object_scene_corrs,
        const std::vector<std::vector<int>> &vote_indices,
        const pcl::PointCloud<PointT>::Ptr scene_keypoints,
        const pcl::PointCloud<PointT>::Ptr object_keypoints,
        const float inlier_threshold,
        const bool refine_model,
        const bool use_hv,
        std::vector<Eigen::Matrix4f> &transformations,
        std::vector<pcl::Correspondences> &model_instances)
{
    pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> corr_rejector;
    corr_rejector.setMaximumIterations(10000);
    corr_rejector.setInlierThreshold(inlier_threshold);
    corr_rejector.setInputSource(scene_keypoints);
    corr_rejector.setInputTarget(object_keypoints);
    corr_rejector.setRefineModel(refine_model); // slightly worse results if true

    for(size_t j = 0; j < vote_indices.size (); ++j)
    {
        pcl::Correspondences temp_corrs, filtered_corrs;
        for (size_t i = 0; i < vote_indices[j].size(); ++i)
        {
            temp_corrs.push_back(object_scene_corrs->at(vote_indices[j][i]));
        }

        if(use_hv)
        {
            // skip maxima with insufficient votes for ransac
            if(temp_corrs.size() < 3)
            {
                continue;
            }
            // correspondence rejection with ransac
            corr_rejector.getRemainingCorrespondences(temp_corrs, filtered_corrs);
            Eigen::Matrix4f best_transform = corr_rejector.getBestTransformation();
            // save transformations for recognition
            if(!best_transform.isIdentity(0.01))
            {
                // keep transformation and correspondences if RANSAC was run successfully
                transformations.push_back(best_transform);
                model_instances.push_back(filtered_corrs);
            }
        }
        else
        {
            model_instances.push_back(temp_corrs);
        }
    }
}


void findClassAndPositionFromCluster(
        const pcl::Correspondences &filtered_corrs,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        const std::vector<Eigen::Vector3f> &object_center_vectors,
        const int num_classes,
        unsigned &resulting_class,
        int &resulting_num_votes,
        Eigen::Vector3f &resulting_position)
{
    // determine position based on filtered correspondences for each remaining class
    // (ideally, only one class remains after filtering)
    std::vector<Eigen::Vector3f> all_centers(num_classes, Eigen::Vector3f(0.0f,0.0f,0.0f));
    std::vector<int> num_votes(num_classes, 0);

    // compute
    for(unsigned fcorr_idx = 0; fcorr_idx < filtered_corrs.size(); fcorr_idx++)
    {
        unsigned match_idx = filtered_corrs.at(fcorr_idx).index_match;
        const ISMFeature &cur_feat = object_features->at(match_idx);
        unsigned class_id = cur_feat.classId;

        unsigned scene_idx = filtered_corrs.at(fcorr_idx).index_query;
        const ISMFeature &scene_feat = scene_features->at(scene_idx);
        pcl::ReferenceFrame ref = scene_feat.referenceFrame;
        Eigen::Vector3f keyPos(scene_feat.x, scene_feat.y, scene_feat.z);
        Eigen::Vector3f vote = object_center_vectors.at(match_idx);
        Eigen::Vector3f pos = keyPos + Utils::rotateBack(vote, ref);
        all_centers[class_id] += pos;
        num_votes[class_id] += 1;
    }
    for(unsigned temp_idx = 0; temp_idx < all_centers.size(); temp_idx++)
    {
        all_centers[temp_idx] /= num_votes[temp_idx];
    }
    // find class with max votes
    unsigned cur_class = 0;
    int cur_best_num = 0;
    for(unsigned class_idx = 0; class_idx < num_votes.size(); class_idx++)
    {
        if(num_votes[class_idx] > cur_best_num)
        {
            cur_best_num = num_votes[class_idx];
            cur_class = class_idx;
        }
    }

    // fill in results
    resulting_class = cur_class;
    resulting_num_votes = cur_best_num;
    resulting_position = all_centers[cur_class];
}


