#include "pipeline_building_blocks.h"
#include "../../implicit_shape_model/utils/utils.h"
#include "../../implicit_shape_model/utils/distance.h"

#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/icp.h>
#include <pcl/recognition/hv/hv_go.h>

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
            // for all following PCL algorithms to work correctly, we need to swap query and match
            // this is because all PCL algrithms handle the object as source (query) and the scene as target (match)
            // (e.g. for registration, ransac, ...), but so far we have the scene as query and the object as match
            // !!!
            // now: query/source index is codebook ("object"), match/target index is scene
            // !!!
            pcl::Correspondence corr(indices[0][0], fe, distances[0][0]);
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
        // create new list of keypoints and reassign the object (i.e. query) index
        pcl::Correspondence &corr = object_scene_corrs->at(i);
        int index = corr.index_query;
        const ISMFeature &feat = all_features->at(index);
        object_features->push_back(feat);
        object_center_vectors.push_back(all_center_vectors.at(index));

        PointT keypoint = PointT(feat.x, feat.y, feat.z);
        object_keypoints->push_back(keypoint);

        pcl::ReferenceFrame lrf = feat.referenceFrame;
        object_lrf->push_back(lrf);
        // remap index to new position in the created point clouds
        // no longer referring to m_features, but now to object_features
        corr.index_query = i;
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
        const ISMFeature &scene_feat = scene_features->at(corr.index_match);
        pcl::ReferenceFrame ref = scene_feat.referenceFrame;
        Eigen::Vector3f keyPos(scene_feat.x, scene_feat.y, scene_feat.z);

        Eigen::Vector3f vote = object_center_vectors.at(corr.index_query);
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
    hough_space->findMaxima(relative_threshold, maxima, vote_indices);
}


void clusterCorrespondences(
        const pcl::CorrespondencesPtr object_scene_corrs,
        const pcl::PointCloud<PointT>::Ptr scene_keypoints,
        const pcl::PointCloud<PointT>::Ptr object_keypoints,
        const pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_lrf,
        const pcl::PointCloud<pcl::ReferenceFrame>::Ptr object_lrf,
        const bool use_distance_weight,
        const float bin_size,
        const float corr_threshold,
        const float lrf_radius,
        const bool use_hough,
        const bool recognize,
        std::vector<pcl::Correspondences> &clustered_corrs,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &transformations)
{
    //  Using Hough3D - i.e. hypothesis generation of tombari
    if(use_hough)
    {
        //  Clustering
        pcl::Hough3DGrouping<PointT, PointT, pcl::ReferenceFrame, pcl::ReferenceFrame> clusterer;
        clusterer.setHoughBinSize(bin_size);
        clusterer.setHoughThreshold(corr_threshold); // NOTE: relative hough threshold!
        clusterer.setUseInterpolation(true);
        clusterer.setUseDistanceWeight(use_distance_weight);
        clusterer.setInputRf(object_lrf);
        clusterer.setSceneRf(scene_lrf);
        clusterer.setLocalRfSearchRadius(lrf_radius);
        clusterer.setInputCloud(object_keypoints);
        clusterer.setSceneCloud(scene_keypoints);
        clusterer.setModelSceneCorrespondences(object_scene_corrs);
        if(recognize)
            clusterer.recognize(transformations, clustered_corrs);
        else
            clusterer.cluster(clustered_corrs);
    }
    else // Using GeometricConsistency - i.e. hypothesis generation of chen (default in the aldoma pipeline)
    {
        pcl::GeometricConsistencyGrouping<PointT, PointT> gc_clusterer;
        gc_clusterer.setGCSize(bin_size);
        gc_clusterer.setGCThreshold(corr_threshold); // NOTE: cluster size, i.e. num of votes!
        gc_clusterer.setInputCloud(object_keypoints);
        gc_clusterer.setSceneCloud(scene_keypoints);
        gc_clusterer.setModelSceneCorrespondences(object_scene_corrs);
        gc_clusterer.cluster(clustered_corrs);
        if(recognize)
            gc_clusterer.recognize(transformations, clustered_corrs);
        else
            gc_clusterer.cluster(clustered_corrs);
    }
}


void generateClassificationHypotheses(
        const pcl::CorrespondencesPtr object_scene_corrs,
        const std::vector<std::vector<int>> &vote_indices,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        std::vector<std::pair<unsigned, float>> &results)
{
    std::vector<pcl::Correspondences> clustered_corrs;
    for (size_t j = 0; j < vote_indices.size (); ++j)
    {
        pcl::Correspondences max_corrs;
        for (size_t i = 0; i < vote_indices[j].size(); ++i)
        {
            max_corrs.push_back(object_scene_corrs->at(vote_indices[j][i]));
        }
        clustered_corrs.push_back(max_corrs);
    }

    generateClassificationHypotheses(clustered_corrs, object_features, results);
}


void generateClassificationHypotheses(
        const std::vector<pcl::Correspondences> &clustered_corrs,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        std::vector<std::pair<unsigned, float>> &results)
{
    // check all maxima since highest valued maximum might still be composed of different class votes
    // therefore we need to count votes per class per maximum
    results.clear();
    for (size_t j = 0; j < clustered_corrs.size (); ++j)
    {
        std::map<unsigned, int> class_occurences;
        pcl::Correspondences max_corrs = clustered_corrs[j];

        // count class occurences in filtered corrs
        for(unsigned fcorr_idx = 0; fcorr_idx < max_corrs.size(); fcorr_idx++)
        {
            unsigned obj_idx = max_corrs.at(fcorr_idx).index_query;
            const ISMFeature &cur_feat = object_features->at(obj_idx);
            unsigned class_id = cur_feat.classId;
            // add occurence of this class_id
            if(class_occurences.find(class_id) != class_occurences.end())
            {
                class_occurences.at(class_id) += 1;
            }
            else
            {
                class_occurences.insert({class_id, 1});
            }
        }

        // determine most frequent label
        unsigned cur_class = 0;
        int cur_best_num = 0;
        for(auto [class_id, num_votes] : class_occurences)
        {
            if(num_votes > cur_best_num)
            {
                cur_best_num = num_votes;
                cur_class = class_id;
            }
        }
        results.push_back({cur_class, cur_best_num});
    }
}


void generateHypothesesWithAbsoluteOrientation(
        const pcl::CorrespondencesPtr object_scene_corrs,
        const std::vector<std::vector<int>> &vote_indices,
        const pcl::PointCloud<PointT>::Ptr scene_keypoints,
        const pcl::PointCloud<PointT>::Ptr object_keypoints,
        const float inlier_threshold,
        const bool refine_model,
        const bool separate_voting_spaces,
        const bool use_hv,
        std::vector<Eigen::Matrix4f> &transformations,
        std::vector<pcl::Correspondences> &model_instances)
{
    pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> corr_rejector;
    corr_rejector.setMaximumIterations(10000);
    corr_rejector.setInlierThreshold(inlier_threshold);
    corr_rejector.setRefineModel(refine_model);
    if(!separate_voting_spaces) // just use common clouds
    {
        corr_rejector.setInputSource(object_keypoints);
        corr_rejector.setInputTarget(scene_keypoints);
    }

    // correspondences were grouped by hough voting
    for(size_t j = 0; j < vote_indices.size (); ++j) // iterate over each maximum
    {
        // skip maxima with view votes, mostly FP
        if(vote_indices[j].size() < 3)
        {
            continue;
        }

        pcl::Correspondences temp_corrs, filtered_corrs;
        std::vector<int> mapping_object;
        std::vector<int> mapping_scene;

        if(separate_voting_spaces) // create clouds for each maximum and remap indices
        {
            // will be filled with points of current maximum
            pcl::PointCloud<PointT>::Ptr max_scene_keypoints(new pcl::PointCloud<PointT>());
            pcl::PointCloud<PointT>::Ptr max_object_keypoints(new pcl::PointCloud<PointT>());

            for (size_t i = 0; i < vote_indices[j].size(); ++i) // iterate over each vote of maximum
            {
                pcl::Correspondence corr = object_scene_corrs->at(vote_indices[j][i]);
                temp_corrs.push_back(pcl::Correspondence(i, i, corr.distance)); // now correspondences refer to local cloud copy of maximum
                max_scene_keypoints->push_back(scene_keypoints->at(corr.index_match));
                max_object_keypoints->push_back(object_keypoints->at(corr.index_query));
                mapping_scene.push_back(corr.index_match);
                mapping_object.push_back(corr.index_query);
            }

            corr_rejector.setInputSource(max_object_keypoints);
            corr_rejector.setInputTarget(max_scene_keypoints);
        }
        else
        {
            for (size_t i = 0; i < vote_indices[j].size(); ++i) // iterate over each vote of maximum
            {
                pcl::Correspondence corr = object_scene_corrs->at(vote_indices[j][i]);
                temp_corrs.push_back(corr);
            }
        }

        if(use_hv)
        {
            // correspondence rejection with ransac
            corr_rejector.getRemainingCorrespondences(temp_corrs, filtered_corrs);
            Eigen::Matrix4f best_transform = corr_rejector.getBestTransformation();
            // save transformations for recognition
            if(!best_transform.isIdentity(0.0001))
            {
                // keep transformation and correspondences if RANSAC was run successfully
                transformations.push_back(best_transform);
                if(separate_voting_spaces) // remap indices back to common clouds
                {
                    // remap indices from local maximum cloud to input clouds
                    for(int i = 0; i < filtered_corrs.size(); i++)
                    {
                        pcl::Correspondence &corr = filtered_corrs.at(i);
                        corr.index_query = mapping_object[corr.index_query];
                        corr.index_match = mapping_scene[corr.index_match];
                    }
                }
                model_instances.push_back(filtered_corrs);
            }
        }
        else
        {
            if(separate_voting_spaces) // remap indices back to common clouds
            {
                // remap indices from local maximum cloud to input clouds
                for(int i = 0; i < temp_corrs.size(); i++)
                {
                    pcl::Correspondence &corr = temp_corrs.at(i);
                    corr.index_query = mapping_object[corr.index_query];
                    corr.index_match = mapping_scene[corr.index_match];
                }
            }
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
        unsigned object_idx = filtered_corrs.at(fcorr_idx).index_query;
        const ISMFeature &cur_feat = object_features->at(object_idx);
        unsigned class_id = cur_feat.classId;

        unsigned scene_idx = filtered_corrs.at(fcorr_idx).index_match;
        const ISMFeature &scene_feat = scene_features->at(scene_idx);
        pcl::ReferenceFrame ref = scene_feat.referenceFrame;
        Eigen::Vector3f keyPos(scene_feat.x, scene_feat.y, scene_feat.z);
        Eigen::Vector3f vote = object_center_vectors.at(object_idx);
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


void findClassAndPositionFromTransformedObjectKeypoints(
        const pcl::Correspondences &filtered_corrs,
        const Eigen::Matrix4f &transformation,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        const pcl::PointCloud<PointT>::Ptr object_keypoints,
        const std::vector<Eigen::Vector3f> &object_center_vectors,
        const int num_classes,
        unsigned &resulting_class,
        int &resulting_num_votes,
        Eigen::Vector3f &resulting_position)
{
    // extract instance specific data based on correspondences
    pcl::PointCloud<PointT>::Ptr instance_object_keypoints(new pcl::PointCloud<PointT>());
    pcl::PointCloud<ISMFeature>::Ptr instance_object_features(new pcl::PointCloud<ISMFeature>());
    std::vector<Eigen::Vector3f> instance_center_vectors;
    for(unsigned corr_idx; corr_idx < filtered_corrs.size(); corr_idx++)
    {
        pcl::Correspondence corr = filtered_corrs.at(corr_idx);
        instance_object_keypoints->push_back(object_keypoints->at(corr.index_query));
        instance_object_features->push_back(object_features->at(corr.index_query));
        instance_center_vectors.push_back(object_center_vectors.at(corr.index_query));
    }

    // determine position based on keypoints
    std::vector<Eigen::Vector3f> all_centers(num_classes, Eigen::Vector3f(0.0f,0.0f,0.0f));
    std::vector<int> num_votes(num_classes, 0);

    // transform original (i.e. object) keypoints with found cluster transformation
    pcl::PointCloud<PointT>::Ptr rotated_model(new pcl::PointCloud<PointT>());
    pcl::transformPointCloud(*instance_object_keypoints, *rotated_model, transformation);

    // perform center voting with transformed object keypoints, using the object's lrf
    for(unsigned idx = 0; idx < rotated_model->size(); idx++)
    {
        const ISMFeature &cur_feat = instance_object_features->at(idx);
        unsigned class_id = cur_feat.classId;
        pcl::ReferenceFrame ref = cur_feat.referenceFrame; // NOTE: using object's lrf

        PointT keyPoint = rotated_model->at(idx);
        Eigen::Vector3f keyPos = keyPoint.getArray3fMap();
        Eigen::Vector3f vote = instance_center_vectors.at(idx);
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


void findClassAndPointsFromCorrespondences(
        const pcl::Correspondences &corrs,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        unsigned &res_class,
        int &res_num_votes,
        pcl::PointCloud<PointT>::Ptr scene_points)
{
    // count correspondences per class
    std::map<unsigned, int> corrs_per_class;
    for(unsigned i = 0; i < corrs.size(); i++)
    {
        unsigned object_idx = corrs.at(i).index_query;
        const ISMFeature &object_feat = object_features->at(object_idx);
        unsigned class_id = object_feat.classId;

        if(corrs_per_class.find(class_id) != corrs_per_class.end())
        {
            corrs_per_class[class_id]++;
        }
        else
        {
            corrs_per_class.insert({class_id, 1});
        }
    }

    // find most frequent class
    res_num_votes = 0;
    for(auto [class_id, num_votes] : corrs_per_class)
    {
        if(num_votes > res_num_votes)
        {
            res_num_votes = num_votes;
            res_class = class_id;
        }
    }

    // extract points corresponding to most frequent class
    scene_points->clear();
    for(unsigned i = 0; i < corrs.size(); i++)
    {
        unsigned object_idx = corrs.at(i).index_query;
        unsigned scene_idx = corrs.at(i).index_match;
        if(res_class == object_features->at(object_idx).classId)
        {
            const ISMFeature &scene_feat = scene_features->at(scene_idx);
            scene_points->push_back(PointT(scene_feat.x, scene_feat.y, scene_feat.z));
        }
    }
}


void findPositionFromCluster(
        const pcl::Correspondences &filtered_corrs,
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        const std::vector<Eigen::Vector3f> &object_center_vectors,
        Eigen::Vector3f &resulting_position)
{
    // determine position based on filtered correspondences
    // (ideally, only one class remains after filtering)
    Eigen::Vector3f center = Eigen::Vector3f(0.0f,0.0f,0.0f);
    int num = 0;

    // compute
    for(unsigned fcorr_idx = 0; fcorr_idx < filtered_corrs.size(); fcorr_idx++)
    {
        unsigned object_idx = filtered_corrs.at(fcorr_idx).index_query;
        unsigned scene_idx = filtered_corrs.at(fcorr_idx).index_match;
        const ISMFeature &scene_feat = scene_features->at(scene_idx);
        pcl::ReferenceFrame ref = scene_feat.referenceFrame;
        Eigen::Vector3f keyPos(scene_feat.x, scene_feat.y, scene_feat.z);
        Eigen::Vector3f vote = object_center_vectors.at(object_idx);
        Eigen::Vector3f pos = keyPos + Utils::rotateBack(vote, ref);
        center += pos;
        num++;
    }
    // fill in results
    center /= float(num);
    resulting_position = center;
}


void findPositionFromTransformedObjectKeypoints(
        const pcl::Correspondences &filtered_corrs,
        const Eigen::Matrix4f &transformation,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        const pcl::PointCloud<PointT>::Ptr object_keypoints,
        const std::vector<Eigen::Vector3f> &object_center_vectors,
        Eigen::Vector3f &resulting_position)
{
    // extract instance specific data based on correspondences
    pcl::PointCloud<PointT>::Ptr instance_object_keypoints(new pcl::PointCloud<PointT>());
    pcl::PointCloud<ISMFeature>::Ptr instance_object_features(new pcl::PointCloud<ISMFeature>());
    std::vector<Eigen::Vector3f> instance_center_vectors;
    for(unsigned corr_idx; corr_idx < filtered_corrs.size(); corr_idx++)
    {
        pcl::Correspondence corr = filtered_corrs.at(corr_idx);
        instance_object_keypoints->push_back(object_keypoints->at(corr.index_query));
        instance_object_features->push_back(object_features->at(corr.index_query));
        instance_center_vectors.push_back(object_center_vectors.at(corr.index_query));
    }

    // determine position based on keypoints
    Eigen::Vector3f center = Eigen::Vector3f(0.0f,0.0f,0.0f);
    int num = 0;

    // transform original (i.e. object) keypoints with found cluster transformation
    pcl::PointCloud<PointT>::Ptr rotated_model(new pcl::PointCloud<PointT>());
    pcl::transformPointCloud(*instance_object_keypoints, *rotated_model, transformation);

    // perform center voting with transformed object keypoints, using the object's lrf
    for(unsigned idx = 0; idx < rotated_model->size(); idx++)
    {
        const ISMFeature &cur_feat = instance_object_features->at(idx);
        pcl::ReferenceFrame ref = cur_feat.referenceFrame; // NOTE: using object's lrf

        PointT keyPoint = rotated_model->at(idx);
        Eigen::Vector3f keyPos = keyPoint.getArray3fMap();
        Eigen::Vector3f vote = instance_center_vectors.at(idx);
        Eigen::Vector3f pos = keyPos + Utils::rotateBack(vote, ref);
        center += pos;
        num++;
    }

    // fill in results
    center /= float(num);
    resulting_position = center;
}


void generateCloudsFromTransformations(
        const std::vector<pcl::Correspondences> clustered_corrs,
        const std::vector<Eigen::Matrix4f> transformations,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        std::vector<pcl::PointCloud<PointT>::ConstPtr> &instances)
{
    for(size_t i = 0; i < transformations.size (); ++i)
    {
        pcl::PointCloud<PointT>::Ptr transformed_object(new pcl::PointCloud<PointT>());
        // NOTE: object_keypoints might contain multiple objects, so use only keypoints of a single cluster
        pcl::PointCloud<PointT>::Ptr cluster_keypoints(new pcl::PointCloud<PointT>());
        pcl::Correspondences this_cluster = clustered_corrs[i];
        for(const pcl::Correspondence &corr : this_cluster)
        {
            const ISMFeature &feat = object_features->at(corr.index_query);
            PointT keypoint = PointT(feat.x, feat.y, feat.z);
            cluster_keypoints->push_back(keypoint);
        }
        pcl::transformPointCloud(*cluster_keypoints, *transformed_object, transformations[i]);
        instances.push_back(transformed_object);
    }
}


void alignCloudsWithICP(
        const float icp_max_iterations,
        const float icp_correspondence_distance,
        const pcl::PointCloud<PointT>::Ptr scene_keypoints,
        const std::vector<pcl::PointCloud<PointT>::ConstPtr> &instances,
        std::vector<pcl::PointCloud<PointT>::ConstPtr> &registered_instances,
        std::vector<Eigen::Matrix4f> &final_transformations)
{
    for (size_t i = 0; i < instances.size (); ++i)
    {
        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setMaximumIterations(icp_max_iterations);
        icp.setMaxCorrespondenceDistance(icp_correspondence_distance);
        icp.setInputSource(instances[i]);
        icp.setInputTarget(scene_keypoints); // scene keypoints slightly better than scene cloud
        pcl::PointCloud<PointT>::Ptr registered (new pcl::PointCloud<PointT>);
        icp.align(*registered);
        registered_instances.push_back(registered);
        final_transformations.push_back(icp.getFinalTransformation());
    //            std::cout << "Instance " << i << " ";
    //            if (icp.hasConverged ())
    //            {
    //                std::cout << "Aligned!" << std::endl;
    //            }
    //            else
    //            {
    //                std::cout << "Not Aligned!" << std::endl;
    //            }
    }
}


// Hypothesis Verification
void runGlobalHV(
        const pcl::PointCloud<PointT>::Ptr scene_cloud,
        std::vector<pcl::PointCloud<PointT>::ConstPtr> &registered_instances,
        const float inlier_threshold,
        const float occlusion_threshold,
        const float regularizer,
        const float clutter_regularizer,
        const float radius_clutter,
        const bool detect_clutter,
        const float normal_radius,
        std::vector<bool> &hypotheses_mask)
{
    pcl::GlobalHypothesesVerification<PointT, PointT> GoHv;
    GoHv.setSceneCloud(scene_cloud);  // Scene Cloud
    GoHv.addModels(registered_instances, true);  //Models to verify
    GoHv.setInlierThreshold(inlier_threshold);
    GoHv.setOcclusionThreshold(occlusion_threshold);
    GoHv.setRegularizer(regularizer);
    GoHv.setClutterRegularizer(clutter_regularizer);
    GoHv.setRadiusClutter(radius_clutter);
    GoHv.setDetectClutter(detect_clutter);
    GoHv.setRadiusNormals(normal_radius);
    std::cout << "-------- global hv before verify " << std::endl;
    GoHv.verify();
    std::cout << "-------- global hv after verify " << std::endl;
    GoHv.getMask(hypotheses_mask);  // i-element TRUE if hvModels[i] verifies hypotheses
}


void performSelfAdaptedHoughVoting(
        const pcl::CorrespondencesPtr &object_scene_corrs,
        const pcl::PointCloud<PointT>::Ptr object_keypoints,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        const pcl::PointCloud<pcl::ReferenceFrame>::Ptr object_lrf,
        const pcl::PointCloud<PointT>::Ptr scene_keypoints,
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        const pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_lrf,
        float initial_matching_threshold,
        float rel_threshold,
        std::vector<double> &maxima,
        std::vector<std::vector<int>> &vote_indices,
        pcl::CorrespondencesPtr &model_scene_corrs_filtered,
        float &found_bin_size)
{
    std::cout << "Total number of correspondences: " << object_scene_corrs->size() << std::endl;

    std::vector<double> filtered_maxima;
    std::vector<std::vector<int>> filtered_vote_indices;

    // self-adapt measure for number of correspondences after filtering
    float t_corr = initial_matching_threshold;
    float ratio;
    do
    {
        // increase number of matches by increasing the threshold
        t_corr += 0.1;

        // apply feature distance threshold
        model_scene_corrs_filtered->clear();
        for(const pcl::Correspondence &corr : *object_scene_corrs)
        {
            if(corr.distance < t_corr)
                model_scene_corrs_filtered->push_back(corr);
        }

        std::cout << "Selecting " << model_scene_corrs_filtered->size() << " correspondences with threshold " << t_corr << std::endl;

        // prepare votes and hough space
        std::vector<std::pair<float,float>> votes; // first: rmse_E, second: rmse_T
        std::pair<float,float> rmse_E_min_max = {std::numeric_limits<float>::max(), std::numeric_limits<float>::min()};
        std::pair<float,float> rmse_T_min_max = {std::numeric_limits<float>::max(), std::numeric_limits<float>::min()};
        prepareSelfAdaptedVoting(model_scene_corrs_filtered, object_keypoints, object_features, object_lrf,
                                 scene_keypoints, scene_features, scene_lrf, votes, rmse_E_min_max, rmse_T_min_max);

        // self-adapt measure for number of bins
        int h_n = 5; // number of bins per dimension, taken from paper
        while(h_n >= 3)
        {
            std::cout << "Constructing Houghspace with " << h_n << " bins per dimension" << std::endl;

            // construct hough space - degrade the 3rd dimension to represent 2d hough
            // size of dims
            float b_l = (rmse_E_min_max.second - rmse_E_min_max.first) / h_n;
            float b_w = (rmse_T_min_max.second - rmse_T_min_max.first) / h_n;

            Eigen::Vector3d min_coord = Eigen::Vector3d(0, 0, 0);
            Eigen::Vector3d max_coord = Eigen::Vector3d(rmse_E_min_max.second, rmse_T_min_max.second, 1);
            Eigen::Vector3d bin_size = Eigen::Vector3d(b_l,b_w,1);
            found_bin_size = 0.5 * (b_l + b_w); // forward bin size to calling method
            // using 3d since there is no 2d hough in pcl
            std::shared_ptr<pcl::recognition::HoughSpace3D> hough_space;
            hough_space = std::make_shared<pcl::recognition::HoughSpace3D>(min_coord, bin_size, max_coord);

            // vote all votes into empty hough space
            hough_space->reset();
            maxima.clear();
            vote_indices.clear();
            for(int vid = 0; vid < votes.size(); vid++)
            {
                Eigen::Vector3d vote(votes[vid].first, votes[vid].second, 0);
                // voting with interpolation
                // votes are in the same order as correspondences, vid is therefore the index in the correspondence list
                hough_space->voteInt(vote, 1.0, vid);
            }
            // find maxima
            hough_space->findMaxima(rel_threshold, maxima, vote_indices);

            // only keep maxima with at least 3 votes
            filtered_maxima.clear();
            filtered_vote_indices.clear();
            for(int idx = 0; idx < maxima.size(); idx++)
            {
                if(vote_indices[idx].size() >= 3)
                {
                    filtered_maxima.push_back(maxima[idx]);
                    filtered_vote_indices.push_back(vote_indices[idx]);
                }
            }

            // if maxima are found, continue with next step
            if(filtered_maxima.size() > 0)
            {
                break;
            }
            h_n--; // reduce number of bins and search again
        }

        // if maxima are found, continue with next step
        if(filtered_maxima.size() > 0)
        {
            break;
        }

        float n = float(object_scene_corrs->size());
        float m = float(model_scene_corrs_filtered->size());
        ratio = m/n;
    }while(ratio < 0.5); // check if we have already selected at least 50% of all matches

    std::cout << "Found " << filtered_maxima.size() << " maxima" << std::endl;
    maxima = filtered_maxima;
    vote_indices = filtered_vote_indices;
}


void prepareSelfAdaptedVoting(
        const pcl::CorrespondencesPtr &object_scene_corrs_filtered,
        const pcl::PointCloud<PointT>::Ptr object_keypoints,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        const pcl::PointCloud<pcl::ReferenceFrame>::Ptr object_lrf,
        const pcl::PointCloud<PointT>::Ptr scene_keypoints,
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        const pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_lrf,
        std::vector<std::pair<float,float>> &votes,
        std::pair<float,float> &rmse_E_min_max,
        std::pair<float,float> &rmse_T_min_max)
{
    std::vector<Eigen::Vector3f> rotations;
    std::vector<Eigen::Vector3f> translations;
    std::vector<float> matching_weights;
    float max_weight = 0;
    for(const pcl::Correspondence &corr : *object_scene_corrs_filtered)
    {
        // get rotation matrix
        pcl::ReferenceFrame lrf_scene = scene_lrf->at(corr.index_match);
        pcl::ReferenceFrame lrf_object = object_lrf->at(corr.index_query);
        Eigen::Matrix3f lrf_s = lrf_scene.getMatrix3fMap();
        Eigen::Matrix3f lrf_o = lrf_object.getMatrix3fMap();
        Eigen::Matrix3f R = lrf_s.transpose() * lrf_o;
        // get translation vector
        Eigen::Vector3f keypoint_scene = scene_keypoints->at(corr.index_match).getVector3fMap();
        Eigen::Vector3f keypoint_object = object_keypoints->at(corr.index_query).getVector3fMap();
        Eigen::Vector3f T = keypoint_scene - R * keypoint_object;
        // convert rotation to euler angles
        float phi, theta, psi;
        if(R(3,3) == 0)
        {
            phi = 0;
        }
        else
        {
            phi = std::atan(R(3,2) / R(3,3));
        }
        if(R(3,3) == 0 && R(3,2) == 0)
        {
            theta = 0;
        }
        else
        {
            theta = std::atan(-R(3,1) / std::sqrt(R(3,2)*R(3,2) + R(3,3)*R(3,3)));
        }
        if(R(1,1) == 0)
        {
            psi = 0;
        }
        else
        {
            psi = atan(R(2,1)/R(1,1));
        }
        // store translations and rotations
        rotations.push_back(Eigen::Vector3f(phi, theta, psi));
        translations.push_back(T);

        // compute descriptor distance as weight
        ISMFeature feat_scene = scene_features->at(corr.index_match);
        ISMFeature feat_object = object_features->at(corr.index_query);
        DistanceEuclidean dist;
        float weight_raw = dist(feat_object.descriptor, feat_scene.descriptor);
        max_weight = weight_raw > max_weight ? weight_raw : max_weight;
        matching_weights.push_back(weight_raw);
    }

    // compute the center values and all weights from raw values that were saved
    Eigen::Vector3f E_c(0,0,0);
    Eigen::Vector3f T_c(0,0,0);
    float sum_weights = 0;
    for(unsigned i = 0; i < rotations.size(); i++)
    {
        float &raw_weight = matching_weights.at(i);
        raw_weight = 1 - (raw_weight / max_weight);
        sum_weights += raw_weight;
        E_c += raw_weight * rotations.at(i);
        T_c += raw_weight * translations.at(i);
    }
    // not in the paper, but shouldn't it be normalized by sum of weights?
    // NOTE: missing normalization does not affect results negatively
    //E_c /= sum_weights;
    //T_c /= sum_weights;

    // compute RMSEs as votes
    for(unsigned i = 0; i < rotations.size(); i++)
    {
        Eigen::Vector3f rot = rotations.at(i);
        float ex = rot.x()-E_c.x();
        float ey = rot.y()-E_c.y();
        float ez = rot.z()-E_c.z();
        float rmse_e_i = std::sqrt((ex*ex + ey*ey + ez*ez)/3);
        if(rmse_e_i < rmse_E_min_max.first)
            rmse_E_min_max.first = rmse_e_i;
        if(rmse_e_i > rmse_E_min_max.second)
            rmse_E_min_max.second = rmse_e_i;

        Eigen::Vector3f tr = translations.at(i);
        float tx = tr.x() - T_c.x();
        float ty = tr.y() - T_c.y();
        float tz = tr.z() - T_c.z();
        float rmse_t_i = std::sqrt((tx*tx + ty*ty + tz*tz)/3);
        if(rmse_t_i < rmse_T_min_max.first)
            rmse_T_min_max.first = rmse_t_i;
        if(rmse_t_i > rmse_T_min_max.second)
            rmse_T_min_max.second = rmse_t_i;

        votes.push_back({rmse_e_i, rmse_t_i});
    }
}


void getMetricsAndInlierPoints(
        const std::vector<pcl::PointCloud<PointT>::ConstPtr> registered_instances,
        const pcl::PointCloud<PointT>::Ptr scene_cloud,
        const float threshold,
        std::vector<pcl::PointCloud<PointT>::ConstPtr> &inlier_points_of_instances,
        std::vector<float> &fs_metrics,
        std::vector<float> &mr_metrics)
{
    for(const pcl::PointCloud<PointT>::ConstPtr &cloud : registered_instances)
    {
        // fitness score 1
        float fs1 = 0;
        float point_inlier_threshold = threshold; // paper works with meshes: threshold is 2 * mesh resolution
        pcl::PointCloud<PointT>::Ptr inlier_cloud(new pcl::PointCloud<PointT>());

        pcl::KdTreeFLANN<PointT> kdtree;
        kdtree.setInputCloud(scene_cloud);
        for(const PointT &p : cloud->points)
        {
            std::vector<int> indices(1);
            std::vector<float> distances(1);
            kdtree.nearestKSearch(p, 1, indices, distances);
            if(distances.size() > 0)
            {
                fs1 += distances[0];
                if(distances[0] < point_inlier_threshold)
                {
                    inlier_cloud->push_back(p);
                }
            }
        }
        // average point distance after registration
        fs1 /= cloud->size();
        // ratio of inliers of this instance
        float mr1 = float(inlier_cloud->size()) / float(cloud->size());

        // fill in results
        inlier_points_of_instances.emplace_back(std::move(inlier_cloud));
        fs_metrics.emplace_back(std::move(fs1));
        mr_metrics.emplace_back(std::move(mr1));
    }
}
