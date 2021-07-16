#include <pcl/io/pcd_io.h>
#include <pcl/correspondence.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/registration/icp.h>
#include <pcl/recognition/hv/hv_go.h>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include "../../implicit_shape_model/utils/utils.h"
#include "../../implicit_shape_model/utils/distance.h"

#include "self_adapt_hghv.h"
#include "../pipeline_building_blocks/pipeline_building_blocks.h"
#include "../pipeline_building_blocks/feature_processing.h" // provides namespace fp::

/**
 * Evaluation pipeline for the approach described in
 *
 * W. Zhou, C. Ma, A. Kuijper:
 *     Hough-space-based hypothesis generation and hypothesis verification for 3D
 *     object recognition and 6D pose estimation.
 *     2018, Computers & Graphics
 *
 */


// TODO VS: selber experimentieren mit
// #include <pcl/registration/transformation_estimation.h>
// oder
// #include <pcl/registration/transformation_estimation_svd.h>
// in Verbindung mit
// transformation_validation.h
// siehe auch:
// #include <pcl/sample_consensus/msac.h>
//
// siehe auch in OpenCV (es nutzt Ransac) und gibt eine Konfidenz: cv::estimateAffine3D
// (also eine Kombination aus pcl transformation estimation und transformation validation)


SelfAdaptHGHV::SelfAdaptHGHV(std::string dataset, float bin, float th) :
    m_features(new pcl::PointCloud<ISMFeature>()),
    m_flann_index(flann::KDTreeIndexParams(4))
{
    std::cout << "-------- loading parameters for " << dataset << " dataset --------" << std::endl;

    if(dataset == "aim" || dataset == "mcgill" || dataset == "mcg" || dataset == "psb" ||
            dataset == "sh12" || dataset == "mn10" || dataset == "mn40")
    {
        /// classification
        // use this for datasets: aim, mcg, psb, shrec-12, mn10, mn40
        m_corr_threshold = -0.1;

        fp::normal_radius = 0.05;
        fp::reference_frame_radius = 0.3;
        fp::feature_radius = 0.4;
        fp::keypoint_sampling_radius = 0.2;
        fp::normal_method = 1;
        fp::feature_type = "SHOT";
    }
    else if(dataset == "wash" || dataset == "bigbird" || dataset == "ycb")
    {
        /// classification
        m_corr_threshold = -0.5; // TODO VS check param

        fp::normal_radius = 0.005;
        fp::reference_frame_radius = 0.05;
        fp::feature_radius = 0.05;
        fp::keypoint_sampling_radius = 0.02;
        fp::normal_method = 0;
        fp::feature_type = "CSHOT";
    }
    else if(dataset == "dataset1" || dataset == "dataset5")
    {
        /// detection
        m_corr_threshold = -th;

        fp::normal_radius = 0.005;
        fp::reference_frame_radius = 0.05;
        fp::feature_radius = 0.05;
        fp::keypoint_sampling_radius = 0.02;
        fp::normal_method = 0;

        m_icp_max_iter = 100;
        m_icp_corr_distance = 0.05;

        if(dataset == "dataset1")
        {
            fp::normal_method = 1;
            fp::feature_type = "SHOT";
        }
        if(dataset == "dataset5")
        {
            fp::normal_method = 0;
            fp::feature_type = "CSHOT";
        }
    }
    else
    {
        std::cerr << "ERROR: dataset with name " << dataset << " not supported!" << std::endl;
    }
}


void SelfAdaptHGHV::train(const std::vector<std::string> &filenames,
                     const std::vector<unsigned> &class_labels,
                     const std::vector<unsigned> &instance_labels,
                     const std::string &output_file)
{
   if(filenames.size() != class_labels.size())
   {
       std::cerr << "ERROR: number of clouds does not match number of labels!" << std::endl;
       return;
   }

   // contains the whole list of features for each class id
   std::map<unsigned, pcl::PointCloud<ISMFeature>::Ptr> all_features;
   for(unsigned i = 0; i < class_labels.size(); i++)
   {
       unsigned int tr_class = class_labels.at(i);
       pcl::PointCloud<ISMFeature>::Ptr cloud(new pcl::PointCloud<ISMFeature>());
       all_features.insert({tr_class, cloud});
   }

   // contains the whole list of center vectors for each class id
   std::map<unsigned, std::vector<Eigen::Vector3f>> all_vectors;
   for(unsigned i = 0; i < class_labels.size(); i++)
   {
       unsigned int tr_class = class_labels.at(i);
       all_vectors.insert({tr_class, {}});
   }

   int num_features = 0;
   // process each input file
   for(int i = 0; i < filenames.size(); i++)
   {
       std::cout << "Processing file " << (i+1) << " of " << filenames.size() << std::endl;
       std::string file = filenames.at(i);
       unsigned int tr_class = class_labels.at(i);
       unsigned int tr_instance = instance_labels.at(i);

       // load cloud
       pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
       if(pcl::io::loadPCDFile(file, *cloud) == -1)
       {
           std::cerr << "ERROR: loading file " << file << std::endl;
       }

       // all these pointers are initialized within the called method
       pcl::PointCloud<PointT>::Ptr keypoints;
       pcl::PointCloud<ISMFeature>::Ptr features;
       pcl::PointCloud<pcl::Normal>::Ptr normals;
       pcl::PointCloud<pcl::ReferenceFrame>::Ptr reference_frames;
       processPointCloud(cloud, keypoints, features, normals, reference_frames);

       for(ISMFeature& ismf : features->points)
       {
           // assign labels
           ismf.classId = tr_class;
           ismf.instanceId = tr_instance;
           // get activation position, relative to object center
           Eigen::Vector3f keyPos(ismf.x, ismf.y, ismf.z);
           Eigen::Vector4f centroid4f;
           pcl::compute3DCentroid(*cloud, centroid4f);
           Eigen::Vector3f centroid(centroid4f[0], centroid4f[1], centroid4f[2]);
           Eigen::Vector3f vote = centroid - keyPos;
           vote = Utils::rotateInto(vote, ismf.referenceFrame);
           all_vectors.at(tr_class).push_back(vote);
       }

       // add computed features to map
       (*all_features.at(tr_class)) += (*features);
       num_features += features->size();
   }

   std::cout << "Extracted " << num_features << " features." << std::endl;

   std::string save_file(output_file);
   if(saveModelToFile(save_file, all_features, all_vectors))
   {
       std::cout << "Hough3D training finished!" << std::endl;
   }
   else
   {
       std::cerr << "ERROR saving Hough3D model!" << std::endl;
   }
}


std::vector<std::pair<unsigned, float>> SelfAdaptHGHV::classify(const std::string &filename)
{
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if(pcl::io::loadPCDFile<PointT>(filename, *cloud) == -1)
    {
        std::cerr << "ERROR: loading file " << filename << std::endl;
        return std::vector<std::pair<unsigned, float>>();
    }

    // all these pointers are initialized within the called method
    pcl::PointCloud<PointT>::Ptr keypoints;
    pcl::PointCloud<ISMFeature>::Ptr features;
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr reference_frames;
    processPointCloud(cloud, keypoints, features, normals, reference_frames);

    // get results
    std::vector<std::pair<unsigned, float>> results;
    classifyObject(features, keypoints, reference_frames, results);

    // here higher values are better
    std::sort(results.begin(), results.end(), [](const std::pair<unsigned, float> &a, const std::pair<unsigned, float> &b)
    {
        return a.second > b.second;
    });

    return results;
}


void SelfAdaptHGHV::classifyObject(
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        const pcl::PointCloud<PointT>::Ptr scene_keypoints,
        const pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_lrf,
        std::vector<std::pair<unsigned, float>> &results)
{
    // get model-scene correspondences
    // !!!
    // query/source index is codebook ("object"), match/target index is scene
    // !!!
    // do not apply a threshold here, it is done during the self-adapted voting
    float matching_threshold = std::numeric_limits<float>::max();
    pcl::CorrespondencesPtr object_scene_corrs = findNnCorrespondences(scene_features, matching_threshold, m_flann_index);

    std::cout << "Found " << object_scene_corrs->size() << " correspondences" << std::endl;

    bool use_first_ransac = false;
    // NOTE: first RANSAC in the method of zhou et al. is used to eliminate false correspondences
    // they are checking object instances separately, however, here I implement a generic pipeline
    // this means: correspondences belong to multiple classes and are therefore ambiguous regarding the transformation
    // TODO rethink feature distance based ransac

    // TODO VS
    // run first RANSAC here to eliminate false correspondences - how exactly?
    // NOTE:
    // * find transformation while eliminating fp with ransac
    // * only keep inliers
    // --> this requires to have separate codebooks for matching, otherwise transformations won't make sense
    // alternative with a common codebook: matching threshold


    // object keypoints are simply the matched keypoints from the codebook
    // however in order not to pass the whole codebook, we need to adjust the index mapping
    pcl::PointCloud<PointT>::Ptr object_keypoints(new pcl::PointCloud<PointT>());
    pcl::PointCloud<ISMFeature>::Ptr object_features(new pcl::PointCloud<ISMFeature>());
    std::vector<Eigen::Vector3f> object_center_vectors; // NOTE: unused here
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr object_lrf(new pcl::PointCloud<pcl::ReferenceFrame>());
    remapIndicesToLocalCloud(object_scene_corrs, m_features, m_center_vectors,
            object_keypoints, object_features, object_center_vectors, object_lrf);

    // cast votes (without center vectors) and retrieve maxima
    std::vector<double> maxima;
    std::vector<std::vector<int>> vote_indices;
    pcl::CorrespondencesPtr model_scene_corrs_filtered(new pcl::Correspondences());
    float initial_matching_threshold = 0.9; // will be incremented and therefore means "no threshold"
    float rel_threshold = m_corr_threshold;
    float found_bin_size; // NOTE: unused here
    performSelfAdaptedHoughVoting(object_scene_corrs, object_keypoints, object_features, object_lrf,
                                  scene_keypoints, scene_features, scene_lrf, initial_matching_threshold,
                                  rel_threshold, maxima, vote_indices, model_scene_corrs_filtered, found_bin_size);

    // check all maxima since highest valued maximum might still be composed of different class votes
    // therefore we need to count votes per class per maximum
    generateClassificationHypotheses(object_scene_corrs, vote_indices, object_features, results);
}


std::vector<VotingMaximum> SelfAdaptHGHV::detect(const std::string &filename)
{
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if(pcl::io::loadPCDFile<PointT>(filename, *cloud) == -1)
    {
      std::cerr << "ERROR: loading file " << filename << std::endl;
      return std::vector<VotingMaximum>();
    }

    // all these pointers are initialized within the called method
    pcl::PointCloud<PointT>::Ptr keypoints;
    pcl::PointCloud<ISMFeature>::Ptr features;
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr reference_frames;
    processPointCloud(cloud, keypoints, features, normals, reference_frames);

    std::vector<VotingMaximum> result_maxima;

    // get results
    std::vector<std::pair<unsigned, float>> results;
    std::vector<Eigen::Vector3f> positions;
    bool use_hv = true; // TODO VS get from calling method
    findObjects(cloud, features, keypoints, reference_frames, use_hv, results, positions);

    // here higher values are better
    std::sort(results.begin(), results.end(), [](const std::pair<unsigned, float> &a, const std::pair<unsigned, float> &b)
    {
        return a.second > b.second;
    });

    // convert result to maxima
    for(unsigned i = 0; i < results.size(); i++)
    {
        auto res = results.at(i);
        VotingMaximum max;
        max.classId = res.first;
        max.globalHypothesis.classId = res.first;
        max.weight = res.second;
        max.position = positions.at(i);
        result_maxima.emplace_back(std::move(max));
    }

    return result_maxima;
}


void SelfAdaptHGHV::findObjects(
        const pcl::PointCloud<PointT>::Ptr scene_cloud,
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        const pcl::PointCloud<PointT>::Ptr scene_keypoints,
        const pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_lrf,
        const bool use_hv,
        std::vector<std::pair<unsigned, float>> &results,
        std::vector<Eigen::Vector3f> &positions)
{
    // get model-scene correspondences
    // !!!
    // query/source index is codebook ("object"), match/target index is scene
    // !!!
    // do not apply a threshold here, it is done during the self-adapted voting
    float matching_threshold = std::numeric_limits<float>::max();
    pcl::CorrespondencesPtr object_scene_corrs = findNnCorrespondences(scene_features, matching_threshold, m_flann_index);

    std::cout << "Found " << object_scene_corrs->size() << " correspondences" << std::endl;

    // TODO VS
    // run first RANSAC here to eliminate false correspondences - how exactly?
    // NOTE:
    // * find transformation while eliminating fp with ransac
    // * only keep inliers
    // --> this requires to have separate codebooks for matching, otherwise transformations won't make sense
    // alternative with a common codebook: matching threshold

    // object keypoints are simply the matched keypoints from the codebook
    // however in order not to pass the whole codebook, we need to adjust the index mapping
    pcl::PointCloud<PointT>::Ptr object_keypoints(new pcl::PointCloud<PointT>());
    pcl::PointCloud<ISMFeature>::Ptr object_features(new pcl::PointCloud<ISMFeature>());
    std::vector<Eigen::Vector3f> object_center_vectors; // NOTE: unused here
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr object_lrf(new pcl::PointCloud<pcl::ReferenceFrame>());
    remapIndicesToLocalCloud(object_scene_corrs, m_features, m_center_vectors,
            object_keypoints, object_features, object_center_vectors, object_lrf);

    // cast votes (without center vectors) and retrieve maxima
    std::vector<double> maxima;
    std::vector<std::vector<int>> vote_indices;
    pcl::CorrespondencesPtr model_scene_corrs_filtered(new pcl::Correspondences());
    float initial_matching_threshold = 0.4;
    float rel_threshold = m_corr_threshold;
    float found_bin_size;
    performSelfAdaptedHoughVoting(object_scene_corrs, object_keypoints, object_features, object_lrf,
                                  scene_keypoints, scene_features, scene_lrf, initial_matching_threshold,
                                  rel_threshold, maxima, vote_indices, model_scene_corrs_filtered, found_bin_size);

    // generate 6DOF hypotheses with absolute orientation
    std::vector<pcl::Correspondences> clustered_corrs;
    std::vector<Eigen::Matrix4f> transformations;
    bool refine_model = false; // helps improve the results sometimes
    float inlier_threshold = found_bin_size;
    bool separate_voting_spaces = false;
    // second RANSAC: filter correspondence groups by position
    generateHypothesesWithAbsoluteOrientation(object_scene_corrs, vote_indices, scene_keypoints, object_keypoints,
                                              inlier_threshold, refine_model, separate_voting_spaces, use_hv,
                                              transformations, clustered_corrs);
    // TODO VS check for background knowledge:
    // https://www.ais.uni-bonn.de/papers/RAM_2015_Holz_PCL_Registration_Tutorial.pdf
    // https://www.programmersought.com/article/93804217152/

    // Stop if no instances
    if (transformations.size () <= 0)
    {
        std::cout << "No instances found!" << std::endl;
        return;
    }
    else
    {
        std::cout << "Resulting clusters: " << transformations.size() << std::endl;
    }

    // ------ this is where the hypothesis verification starts -------

    // Generate clouds for each instance found
    std::vector<pcl::PointCloud<PointT>::ConstPtr> instances;
    generateCloudsFromTransformations(clustered_corrs, transformations, object_features, instances);

    // ---------------------------- first verify ----------------------------
    // ICP
    std::vector<pcl::PointCloud<PointT>::ConstPtr> registered_instances;
    float icp_max_iterations = 100;
    float icp_correspondence_distance = 0.05;
    // TODO VS try passing the whole scene cloud instead of only scene keypoints
    alignCloudsWithICP(icp_max_iterations, icp_correspondence_distance,
                       scene_keypoints, instances, registered_instances);

    // find nearest neighbors for each point of a registered instance in the scene
    std::vector<pcl::PointCloud<PointT>::ConstPtr> inlier_points_of_instances;
    std::vector<float> fs_metrics;
    std::vector<float> mr_metrics;
    getMetricsAndInlierPoints(registered_instances, scene_cloud, fp::normal_radius,
                              inlier_points_of_instances, fs_metrics, mr_metrics);

    std::vector<pcl::PointCloud<PointT>::ConstPtr> first_pass_instances;
    std::vector<int> first_pass_indices; // mapping the passed hypotheses to total list of registered instances
    for(int i = 0; i < inlier_points_of_instances.size(); i++)
    {
        float fs1 = fs_metrics[i];
        float mr1 = mr_metrics[i];

        // thresholds from the paper
        float tm1 = 0.15;
        float tm2 = 0.3;
        float tf1 = 0.01 * mr1;
        float tf2 = 0.1 * mr1;

        std::cout << "--------- avg.dist (fs1): " << fs1 << "    ratio (mr1): " << mr1 << std::endl;

        // first verify
        if((mr1 < tm1 && fs1 <= tf1) or (tm1 <= mr1 && mr1 <= tm2 && fs1 <= tf2) or (mr1 >= tm2))
        {
            // keep hypothesis
            // std::cout << "Accepting instance in first verify" << std::endl;
            first_pass_instances.push_back(inlier_points_of_instances[i]);
            first_pass_indices.push_back(i);
        }
        else
        {
            // std::cout << "Rejecting instance in first verify" << std::endl;
        }
    }
    std::cout << "Registered instances after 1 verify: " << first_pass_instances.size() << std::endl;


    // ---------------------------- second verify ----------------------------
    // ICP
    std::vector<pcl::PointCloud<PointT>::ConstPtr> registered_instances2;
    // TODO VS try passing the whole scene cloud instead of only scene keypoints
    alignCloudsWithICP(icp_max_iterations, icp_correspondence_distance,
                       scene_keypoints, first_pass_instances, registered_instances2);

    // find nearest neighbors for each point of a registered instance in the scene
    std::vector<pcl::PointCloud<PointT>::ConstPtr> inlier_points_of_instances2;
    std::vector<float> fs_metrics2;
    std::vector<float> mr_metrics2;
    getMetricsAndInlierPoints(registered_instances2, scene_cloud, fp::normal_radius,
                              inlier_points_of_instances2, fs_metrics2, mr_metrics2);

    std::vector<pcl::PointCloud<PointT>::ConstPtr> second_pass_instances;
    std::vector<int> second_pass_indices; // mapping the passed hypotheses to total list of registered instances
    for(int i = 0; i < inlier_points_of_instances2.size(); i++)
    {
        float fs2 = fs_metrics2[i];
        float mr2 = mr_metrics2[i];

        // thresholds from the paper
        float tm3 = 0.8;
        float tf3 = 0.001 * mr2;

        std::cout << "--------- avg.dist (fs2): " << fs2 << "    ratio (mr2): " << mr2 << std::endl;

        // second verify
        if(mr2 >= tm3 && fs2 <= tf3)
        {
            // keep hypothesis
            // std::cout << "Accepting instance in second verify" << std::endl;
            second_pass_instances.push_back(inlier_points_of_instances2[i]); // TODO VS check what exactly to save
            second_pass_indices.push_back(i);
        }
        else
        {
            // std::cout << "Rejecting instance in second verify" << std::endl;
        }
    }
    std::cout << "Registered instances after 2 verify: " << first_pass_instances.size() << std::endl;

    // get resulting instances
    std::vector<pcl::PointCloud<PointT>::ConstPtr> verified_instances;
    std::vector<pcl::PointCloud<PointT>::ConstPtr> verified_instances_inliers1;
    std::vector<pcl::PointCloud<PointT>::ConstPtr> verified_instances_inliers2 = second_pass_instances;
    std::vector<Eigen::Matrix4f> verified_transformations;
    for(int i = 0; i < second_pass_instances.size(); i++)
    {
        int second_pass_idx = second_pass_indices[i];
        int first_pass_idx = first_pass_indices[second_pass_idx];
        verified_instances_inliers1.push_back(inlier_points_of_instances[first_pass_idx]);
        verified_instances.push_back(registered_instances[first_pass_idx]);
        verified_transformations.push_back(transformations[first_pass_idx]);
    }

    // TODO VS: find and fill in results

//            if(hypotheses_mask[i])
//            {
//                std::cout << "Instance " << i << " is GOOD!" << std::endl;
//                pcl::Correspondences filtered_corrs = clustered_corrs[i];
//                unsigned res_class;
//                int res_num_votes;
//                Eigen::Vector3f res_position;
//                findClassAndPositionFromCluster(filtered_corrs, object_features, scene_features,
//                                                res_class, res_num_votes, res_position);

//                // find aligned position
//                pcl::PointCloud<PointT>::ConstPtr reg_inst = registered_instances[i];
//                Eigen::Vector4f centroid4f;
//                pcl::compute3DCentroid(*reg_inst, centroid4f);

//                // store results
//                results.push_back({res_class, res_num_votes});
//                // TODO VS: which position? transformed centroid or position from cluster (the latter was better in first test)
//                //positions.emplace_back(Eigen::Vector3f(centroid4f.x(), centroid4f.y(), centroid4f.z()));
//                positions.push_back(res_position);
//            }
}



std::vector<pcl::Correspondences> SelfAdaptHGHV::getCorrespondeceClustersFromMaxima(
        const std::vector<double> &maxima,
        const std::vector<std::vector<int>> &voteIndices,
        const pcl::CorrespondencesPtr &model_scene_corrs_filtered) const
{
    std::vector<pcl::Correspondences> clustered_corrs;
    for (size_t j = 0; j < maxima.size (); ++j) // loop over all maxima
    {
        pcl::Correspondences max_corrs;
        for (size_t i = 0; i < voteIndices[j].size(); ++i)
        {
            max_corrs.push_back(model_scene_corrs_filtered->at(voteIndices[j][i]));
        }
        clustered_corrs.push_back(max_corrs);
    }
    return clustered_corrs;
}



void SelfAdaptHGHV::findClassAndPositionFromCluster(
        const pcl::Correspondences &filtered_corrs,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        unsigned &resulting_class,
        int &resulting_num_votes,
        Eigen::Vector3f &resulting_position) const
{
    // determine position based on filtered correspondences for each remaining class
    // (ideally, only one class remains after filtering)
    std::vector<Eigen::Vector3f> all_centers(m_number_of_classes);
    std::vector<int> num_votes(m_number_of_classes);
    // init
    for(unsigned temp_idx = 0; temp_idx < all_centers.size(); temp_idx++)
    {
        all_centers[temp_idx] = Eigen::Vector3f(0.0f,0.0f,0.0f);
        num_votes[temp_idx] = 0;
    }
    // compute
    for(unsigned fcorr_idx = 0; fcorr_idx < filtered_corrs.size(); fcorr_idx++)
    {
        // match_idx refers to the modified list, not the original codebook!!!
        unsigned match_idx = filtered_corrs.at(fcorr_idx).index_match;
        // const ISMFeature &cur_feat = m_features->at(match_idx);
        const ISMFeature &cur_feat = object_features->at(match_idx);
        unsigned class_id = cur_feat.classId;

        unsigned scene_idx = filtered_corrs.at(fcorr_idx).index_query;
        const ISMFeature &scene_feat = scene_features->at(scene_idx);
        pcl::ReferenceFrame ref = scene_feat.referenceFrame;
        Eigen::Vector3f keyPos(scene_feat.x, scene_feat.y, scene_feat.z);
        Eigen::Vector3f vote = m_center_vectors.at(match_idx);
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


bool SelfAdaptHGHV::loadModel(std::string &filename)
{
    if(!loadModelFromFile(filename)) return false;

    flann::Matrix<float> dataset = createFlannDataset();

    m_flann_index = flann::Index<flann::L2<float>>(dataset, flann::KDTreeIndexParams(4));
    m_flann_index.buildIndex();

    return true;
}


flann::Matrix<float> SelfAdaptHGHV::createFlannDataset() const
{
    // create a dataset with all features for matching / activation
    int descriptor_size = m_features->at(0).descriptor.size();
    flann::Matrix<float> dataset(new float[m_features->size() * descriptor_size],
            m_features->size(), descriptor_size);

    // build dataset
    for(int i = 0; i < m_features->size(); i++)
    {
        ISMFeature ism_feat = m_features->at(i);
        std::vector<float> descriptor = ism_feat.descriptor;
        for(int j = 0; j < (int)descriptor.size(); j++)
        {
            dataset[i][j] = descriptor.at(j);
        }
    }

    return dataset;
}


bool SelfAdaptHGHV::saveModelToFile(std::string &filename,
                              std::map<unsigned, pcl::PointCloud<ISMFeature>::Ptr> &all_features,
                              std::map<unsigned, std::vector<Eigen::Vector3f>> &all_vectors) const
{
    // create boost data object
    std::ofstream ofs(filename);
    boost::archive::binary_oarchive oa(ofs);

    // store label maps
    unsigned size = m_instance_to_class_map.size();
    oa << size;
    for(auto const &it : m_instance_to_class_map)
    {
        unsigned label_inst = it.first;
        unsigned label_class = it.second;
        oa << label_inst;
        oa << label_class;
    }

    size = m_class_labels.size();
    oa << size;
    for(auto it : m_class_labels)
    {
        std::string label = it.second;
        oa << label;
    }
    size = m_instance_labels.size();
    oa << size;
    for(auto it : m_instance_labels)
    {
        std::string label = it.second;
        oa << label;
    }

    // write extracted features
    int num_features = 0;
    for(auto elem : all_features)
        num_features += elem.second->size();

    int descriptor_dim = all_features[0]->at(0).descriptor.size();

    size = all_features.size();
    oa << size;
    oa << num_features;
    oa << descriptor_dim;

    //write classes
    for(auto elem : all_features)
    {
        for(unsigned int feat = 0; feat < elem.second->size(); feat++)
        {
            unsigned class_id = elem.first;
            oa << class_id;
        }
    }

    //write features
    for(auto elem : all_features)
    {
        for(unsigned int feat = 0; feat < elem.second->size(); feat++)
        {
            // write descriptor
            for(unsigned int i_dim = 0; i_dim < descriptor_dim; i_dim++)
            {
                float temp = elem.second->at(feat).descriptor.at(i_dim);
                oa << temp;
            }
            // write keypoint position
            float pos = elem.second->at(feat).x;
            oa << pos;
            pos = elem.second->at(feat).y;
            oa << pos;
            pos = elem.second->at(feat).z;
            oa << pos;
            // write reference frame
            for(unsigned lrf = 0; lrf < 9; lrf++)
            {
                pos = elem.second->at(feat).referenceFrame.rf[lrf];
                oa << pos;
            }
        }
    }

    // write all vectors
    size = all_vectors.size();
    oa << size;
    for(auto elem : all_vectors)
    {
        unsigned class_id = elem.first;
        oa << class_id;
        size = elem.second.size();
        oa << size;

        for(unsigned int vec = 0; vec < elem.second.size(); vec++)
        {
            float pos;
            pos = elem.second.at(vec).x();
            oa << pos;
            pos = elem.second.at(vec).y();
            oa << pos;
            pos = elem.second.at(vec).z();
            oa << pos;
        }
    }

    ofs.close();
    return true;
}

bool SelfAdaptHGHV::loadModelFromFile(std::string& filename)
{
    std::ifstream ifs(filename);
    if(ifs)
    {
        boost::archive::binary_iarchive ia(ifs);

        // load original labels
        unsigned size;
        ia >> size;
        m_instance_to_class_map.clear();
        for(unsigned i = 0; i < size; i++)
        {
            unsigned label_inst, label_class;
            ia >> label_inst;
            ia >> label_class;
            m_instance_to_class_map.insert({label_inst, label_class});
        }
        ia >> size;
        m_class_labels.clear();
        for(unsigned i = 0; i < size; i++)
        {
            std::string label;
            ia >> label;
            m_class_labels.insert({i, label});
        }
        ia >> size;
        m_instance_labels.clear();
        for(unsigned i = 0; i < size; i++)
        {
            std::string label;
            ia >> label;
            m_instance_labels.insert({i, label});
        }

        int number_of_features;
        int descriptor_dim;

        ia >> size;
        m_number_of_classes = size;
        ia >> number_of_features;
        ia >> descriptor_dim;

        // read classes
        m_class_lookup.clear();
        m_class_lookup.resize (number_of_features, 0);
        for (unsigned int feat_i = 0; feat_i < number_of_features; feat_i++)
        {
            ia >> m_class_lookup[feat_i];
        }

        // read features
        m_features->clear();
        for (unsigned int feat_i = 0; feat_i < number_of_features; feat_i++)
        {
            ISMFeature feature;
            feature.descriptor.resize(descriptor_dim);

            // read descriptor
            for (unsigned int dim_i = 0; dim_i < descriptor_dim; dim_i++)
            {
                ia >> feature.descriptor[dim_i];
            }

            // read keypoint position
            float pos;
            ia >> pos;
            feature.x = pos;
            ia >> pos;
            feature.y = pos;
            ia >> pos;
            feature.z = pos;
            // read reference frame
            for(unsigned lrf = 0; lrf < 9; lrf++)
            {
                ia >> pos;
                feature.referenceFrame.rf[lrf] = pos;
            }
            feature.classId = m_class_lookup.at(feat_i);
            m_features->push_back(feature);
        }

        // read all vectors
        m_center_vectors.clear();
        ia >> size;
        for(unsigned vec = 0; vec < size; vec++)
        {
            unsigned class_id;
            ia >> class_id;
            unsigned vec_size;
            ia >> vec_size;
            for(unsigned elem = 0; elem < vec_size; elem++)
            {
                float x,y,z;
                ia >> x;
                ia >> y;
                ia >> z;
                m_center_vectors.push_back(Eigen::Vector3f(x,y,z));
            }
        }

        ifs.close();
    }
    else
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    return true;
}
