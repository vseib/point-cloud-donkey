#include "global_hv.h"

#include <pcl/io/pcd_io.h>
#include <pcl/correspondence.h>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include "../../implicit_shape_model/utils/utils.h"

#include "../pipeline_building_blocks/pipeline_building_blocks.h"
#include "../pipeline_building_blocks/feature_processing.h" // provides namespace fp::

/**
 * Evaluation pipeline for the approach described in
 *
 * A. Aldoma, F. Tombari, L. Di Stefano, M. Vincze:
 *     A global hypothesis verification method for 3D object recognition.
 *     2012, ECCV
 *
 * There is also a tutorial for this approach in
 *
 * A. Aldoma, Z.C. Marton, F. Tombari, W. Wohlkinger, C. Potthast, B. Zeisl, R.B. Rusu, S. Gedikli, M. Vincze:
 *      Point Cloud Library: Three-Dimensional Object Recognition and 6 DOF Pose Estimation
 *      2012, IEEE Robotics and Automation Magazine
 *
 * Also see:
 *      https://pcl.readthedocs.io/en/latest/correspondence_grouping.html
 *      https://pcl.readthedocs.io/en/latest/global_hypothesis_verification.html
 */


GlobalHV::GlobalHV(std::string dataset, float bin, float th) :
    m_features(new pcl::PointCloud<ISMFeature>()),
    m_flann_index(flann::KDTreeIndexParams(4))
{
    std::cout << "-------- loading parameters for " << dataset << " dataset --------" << std::endl;

    if(dataset == "aim" || dataset == "mcgill" || dataset == "mcg" || dataset == "psb" ||
            dataset == "sh12" || dataset == "mn10" || dataset == "mn40")
    {
        /// classification
        // use this for datasets: aim, mcg, psb, shrec-12, mn10, mn40
        m_bin_size = 0.5;
        m_corr_threshold = -0.1;
        fp::normal_radius = 0.05;
        fp::reference_frame_radius = 0.3;
        fp::feature_radius = 0.4;
        fp::keypoint_sampling_radius = 0.2;
        fp::normal_method = 1;
        fp::feature_type = "SHOT";
    }
    else if(dataset == "washington" || dataset == "bigbird" || dataset == "ycb")
    {
        /// classification
        m_bin_size = 0.02; // TODO VS check param
        m_corr_threshold = -0.1; // TODO VS check param
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
        m_bin_size = bin;
        m_corr_threshold = th;
        fp::normal_radius = 0.005;
        fp::reference_frame_radius = 0.05;
        fp::feature_radius = 0.05;
        fp::keypoint_sampling_radius = 0.02;
        fp::normal_method = 0;

        m_icp_max_iter = 100;
        m_icp_corr_distance = 0.05;

        if(dataset == "dataset1")
            fp::feature_type = "SHOT";
        if(dataset == "dataset5")
            fp::feature_type = "CSHOT";
    }
    else
    {
        std::cerr << "ERROR: dataset with name " << dataset << " not supported!" << std::endl;
    }
}


void GlobalHV::train(const std::vector<std::string> &filenames,
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
       std::cout << "GlobalHV training finished!" << std::endl;
   }
   else
   {
       std::cerr << "ERROR saving GlobalHV model!" << std::endl;
   }
}


std::vector<std::pair<unsigned, float>> GlobalHV::classify(const std::string &filename,
                                                           bool use_hough)
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
    classifyObject(features, keypoints, reference_frames, use_hough, results);

    // here higher values are better
    std::sort(results.begin(), results.end(), [](const std::pair<unsigned, float> &a, const std::pair<unsigned, float> &b)
    {
        return a.second > b.second;
    });

    return results;
}


std::vector<VotingMaximum> GlobalHV::detect(const std::string &filename,
                                           bool use_hough, bool use_global_hv)
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
    bool use_tombari_variant = use_hough;
    bool use_aldoma_hv = use_global_hv;
    findObjects(cloud, features, keypoints, reference_frames,
                use_tombari_variant, use_aldoma_hv, results, positions);

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


bool GlobalHV::loadModel(std::string &filename)
{
    if(!loadModelFromFile(filename)) return false;

    flann::Matrix<float> dataset = createFlannDataset();

    m_flann_index = flann::Index<flann::L2<float>>(dataset, flann::KDTreeIndexParams(4));
    m_flann_index.buildIndex();

    return true;
}


flann::Matrix<float> GlobalHV::createFlannDataset() const
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


void GlobalHV::classifyObject(
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        const pcl::PointCloud<PointT>::Ptr scene_keypoints,
        const pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_lrf,
        const bool use_hough,
        std::vector<std::pair<unsigned, float>> &results)
{
    // get model-scene correspondences
    // query index is scene, match index is codebook ("object")
    // PCL implementation has a threshold of 0.25, however, without a threshold we get better results
    float matching_threshold = std::numeric_limits<float>::max();
    pcl::CorrespondencesPtr object_scene_corrs = findNnCorrespondences(scene_features, matching_threshold, m_flann_index);

    std::cout << "Found " << object_scene_corrs->size() << " correspondences" << std::endl;

    // object keypoints are simply the matched keypoints from the codebook
    // however in order not to pass the whole codebook, we need to adjust the index mapping
    pcl::PointCloud<PointT>::Ptr object_keypoints(new pcl::PointCloud<PointT>());
    pcl::PointCloud<ISMFeature>::Ptr object_features(new pcl::PointCloud<ISMFeature>());
    std::vector<Eigen::Vector3f> object_center_vectors;
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr object_lrf(new pcl::PointCloud<pcl::ReferenceFrame>());
    remapIndicesToLocalCloud(object_scene_corrs, m_features, m_center_vectors,
            object_keypoints, object_features, object_center_vectors, object_lrf);

    // Actual Clustering
    std::vector<pcl::Correspondences> clustered_corrs;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> rototranslations;
    bool use_distance_weight = false;
    bool recognize = false; // false for classification
    // NOTE: if hough is used, m_corr_threshold is the relative hough threshold,
    // otherwise it is the min. number of votes to form a maximum
    if(!use_hough)
    {
        m_corr_threshold = 3; // does not really matter for classification
    }
    clusterCorrespondences(object_scene_corrs, scene_keypoints, object_keypoints,
                           scene_lrf, object_lrf, use_distance_weight, m_bin_size,
                           m_corr_threshold, fp::reference_frame_radius, use_hough,
                           recognize, clustered_corrs, rototranslations);

    std::cout << "Found " << clustered_corrs.size();
    if(clustered_corrs.size() != 1)
        std::cout << " clusters" << std::endl;
    else
        std::cout << " cluster" << std::endl;

    generateClassificationHypotheses(clustered_corrs, object_features, results);
}

void GlobalHV::findObjects(
        const pcl::PointCloud<PointT>::Ptr scene_cloud,
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        const pcl::PointCloud<PointT>::Ptr scene_keypoints,
        const pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_lrf,
        const bool use_hough,
        const bool use_global_hv,
        std::vector<std::pair<unsigned, float>> &results,
        std::vector<Eigen::Vector3f> &positions)
{
    // get model-scene correspondences
    // query index is scene, match index is codebook ("object")
    // PCL implementation has a threshold of 0.25, however, without a threshold we get better results
    float matching_threshold = std::numeric_limits<float>::max();
    pcl::CorrespondencesPtr object_scene_corrs = findNnCorrespondences(scene_features, matching_threshold, m_flann_index);

    std::cout << "Found " << object_scene_corrs->size() << " correspondences" << std::endl;

    // object keypoints are simply the matched keypoints from the codebook
    // however in order not to pass the whole codebook, we need to adjust the index mapping
    pcl::PointCloud<PointT>::Ptr object_keypoints(new pcl::PointCloud<PointT>());
    pcl::PointCloud<ISMFeature>::Ptr object_features(new pcl::PointCloud<ISMFeature>());
    std::vector<Eigen::Vector3f> object_center_vectors;
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr object_lrf(new pcl::PointCloud<pcl::ReferenceFrame>());
    remapIndicesToLocalCloud(object_scene_corrs, m_features, m_center_vectors,
            object_keypoints, object_features, object_center_vectors, object_lrf);

    // Actual Clustering
    std::vector<pcl::Correspondences> clustered_corrs;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> rototranslations;
    bool use_distance_weight = false;
    bool recognize = true; // true for detection
    // NOTE: if hough is used, m_corr_threshold is the relative hough threshold,
    // otherwise it is the min. number of votes to form a maximum
    if(!use_hough)
    {
        m_corr_threshold = 3;
    }
    clusterCorrespondences(object_scene_corrs, scene_keypoints, object_keypoints,
                           scene_lrf, object_lrf, use_distance_weight, m_bin_size,
                           m_corr_threshold, fp::reference_frame_radius, use_hough,
                           recognize, clustered_corrs, rototranslations);

    // ------ this is where the aldoma contribution starts -------

    // Stop if no instances
    if (rototranslations.size () <= 0)
    {
        std::cout << "No instances found!" << std::endl;
        return;
    }
    else
    {
        std::cout << "Resulting clusters: " << rototranslations.size() << std::endl;
    }

    if(use_global_hv) // aldoma global hypotheses verification
    {
        // Generates clouds for each instance found
        std::vector<pcl::PointCloud<PointT>::ConstPtr> instances;
        generateCloudsFromTransformations(clustered_corrs, rototranslations, object_features, instances);

        // ICP
        std::vector<pcl::PointCloud<PointT>::ConstPtr> registered_instances;
        float icp_max_iterations = m_icp_max_iter;
        float icp_correspondence_distance = m_icp_corr_distance;
        alignCloudsWithICP(icp_max_iterations, icp_correspondence_distance,
                           scene_keypoints, instances, registered_instances);

        // Hypothesis Verification
        std::vector<bool> hypotheses_mask;  // Mask Vector to identify positive hypotheses
        float inlier_threshold = 0.01;
        float occlusion_threshold = 0.02;
        float regularizer = 3.0;
        float clutter_regularizer = 5.0;
        float radius_clutter = 0.25;
        bool detect_clutter = true;
        runGlobalHV(scene_cloud, registered_instances, inlier_threshold, occlusion_threshold,
                    regularizer, clutter_regularizer, radius_clutter, detect_clutter,
                    fp::normal_radius, hypotheses_mask);

        for (int i = 0; i < hypotheses_mask.size (); i++)
        {
            if(hypotheses_mask[i])
            {
                // use aligned cloud or cluster for position
                //std::cout << "Instance " << i << " is GOOD!" << std::endl;
                bool use_aligned_cloud = true;

                pcl::Correspondences filtered_corrs = clustered_corrs[i];
                unsigned res_class;
                int res_num_votes;
                Eigen::Vector3f res_position;
                findClassAndPositionFromCluster(filtered_corrs, object_features, scene_features,
                                                object_center_vectors, m_number_of_classes,
                                                res_class, res_num_votes, res_position);
                if(use_aligned_cloud)
                {
                    // find aligned position
                    pcl::PointCloud<PointT>::ConstPtr reg_inst = registered_instances[i];
                    Eigen::Vector4f centroid4f;
                    pcl::compute3DCentroid(*reg_inst, centroid4f);
                    results.push_back({res_class, res_num_votes});
                    positions.emplace_back(Eigen::Vector3f(centroid4f.x(), centroid4f.y(), centroid4f.z()));
                }
                else
                {
                    results.push_back({res_class, res_num_votes});
                    positions.push_back(res_position);
                }
            }
            else
            {
                //std::cout << "Instance " << i << " is bad, discarding!" << std::endl;
            }
        }
    }
    // NOTE: else branch is just for verification and should not be used normally
    // NOTE: if above use_hough == true, this else branch should correspond to the
    //       tombari pipeline (i.e. executable "eval_pipeline_tombari_detection")
    else
    {
        for (size_t j = 0; j < clustered_corrs.size (); ++j) // loop over all maxima/clusters
        {
            pcl::Correspondences filtered_corrs = clustered_corrs[j];
            unsigned res_class;
            int res_num_votes;
            Eigen::Vector3f res_position;
            findClassAndPositionFromCluster(filtered_corrs, object_features, scene_features,
                                            object_center_vectors, m_number_of_classes,
                                            res_class, res_num_votes, res_position);
            results.push_back({res_class, res_num_votes});
            positions.push_back(res_position);
        }
    }
}


bool GlobalHV::saveModelToFile(std::string &filename,
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

bool GlobalHV::loadModelFromFile(std::string& filename)
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
