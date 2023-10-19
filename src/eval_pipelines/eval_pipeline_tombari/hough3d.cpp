#include <pcl/io/pcd_io.h>
#include <pcl/correspondence.h>
#include <pcl/registration/transformation_estimation.h>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>

#include "hough3d.h"
#include "../../implicit_shape_model/utils/utils.h"
#include "../pipeline_building_blocks/pipeline_building_blocks.h"
#include "../pipeline_building_blocks/feature_processing.h" // provides namespace fp::


/**
 * Implementation of the approach described in
 *
 * F. Tombari, L. Di Stefano:
 *     Object recognition in 3D scenes with occlusions and clutter by Hough voting.
 *     2010, Fourth Pacific-Rim Symposium on Image and Video Technology
 *
 */


Hough3d::Hough3d(std::string dataset, float bin, float th, float count, float count2) :
    m_features(new pcl::PointCloud<ISMFeature>()), m_keypoints(new pcl::PointCloud<PointT>()),
    m_flann_index(flann::KDTreeIndexParams(4))
{
    std::cout << "-------- loading parameters for " << dataset << " dataset --------" << std::endl;

    if(dataset == "aim" || dataset == "mcgill" || dataset == "mcg" || dataset == "psb" ||
            dataset == "sh12" || dataset == "mn10" || dataset == "mn40")
    {
        /// classification
        m_th = -0.1;
        m_min_coord = Eigen::Vector3d(-2.0, -2.0, -2.0);
        m_max_coord = Eigen::Vector3d(2.0, 2.0, 2.0);
        m_bin_size = Eigen::Vector3d(0.5, 0.5, 0.5);

        // use this for datasets: aim, mcg, psb, shrec-12, mn10, mn40
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
        m_th = -0.1;
        m_min_coord = Eigen::Vector3d(-1.0, -1.0, -1.0);
        m_max_coord = Eigen::Vector3d(1.0, 1.0, 1.0);
        m_bin_size = Eigen::Vector3d(0.02, 0.02, 0.02);
        fp::normal_radius = 0.005;
        fp::reference_frame_radius = 0.04;
        fp::feature_radius = 0.06;
        fp::keypoint_sampling_radius = 0.02;
        fp::normal_method = 0;
        fp::feature_type = "CSHOT";
    }
    else if(dataset == "dataset1" || dataset == "dataset5")
    {
        m_count = count; // TODO VS temp
        m_count2 = count2;

        /// detection
        m_min_coord = Eigen::Vector3d(-1.0, -1.0, -1.0);
        m_max_coord = Eigen::Vector3d(1.0, 1.0, 1.0);
        fp::normal_radius = 0.005;
        fp::reference_frame_radius = 0.04;
        fp::feature_radius = 0.04;
        fp::keypoint_sampling_radius = 0.02;

        if(dataset == "dataset1")
        {
            fp::normal_method = 2;
            fp::feature_type = "SHOT";
            m_th = -th;// -0.01;
            m_bin_size = Eigen::Vector3d(bin, bin, bin); //Eigen::Vector3d(0.01, 0.01, 0.01);
        }
        if(dataset == "dataset5")
        {
            fp::normal_method = 0;
            fp::feature_type = "CSHOT";
            m_th = -th;//-0.80;
            m_bin_size = Eigen::Vector3d(bin, bin, bin); //Eigen::Vector3d(0.05, 0.05, 0.05);
        }
    }
    else
    {
        std::cerr << "ERROR: dataset with name " << dataset << " not supported!" << std::endl;
    }

    m_hough_space = std::make_shared<pcl::recognition::HoughSpace3D>(m_min_coord, m_bin_size, m_max_coord);
}


void Hough3d::train(const std::vector<std::string> &filenames,
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

    std::map<unsigned, std::vector<Utils::BoundingBox>> all_bounding_boxes;

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

        Utils::BoundingBox bounding_box = Utils::computeMVBB<PointT>(cloud);
        all_bounding_boxes[tr_class].push_back(bounding_box);

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
    if(saveModelToFile(save_file, all_features, all_vectors, all_bounding_boxes))
    {
        std::cout << "Hough3D training finished!" << std::endl;
    }
    else
    {
        std::cerr << "ERROR saving Hough3D model!" << std::endl;
    }
}


std::vector<std::pair<unsigned, float>> Hough3d::classify(const std::string &filename,
                                                          bool useSingleVotingSpace)
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
    if(useSingleVotingSpace)
    {
         classifyObjectsWithSingleVotingSpace(features, results);
    }
    else
    {
        classifyObjectsWithSeparateVotingSpaces(features, results);
    }

    // here higher values are better
    std::sort(results.begin(), results.end(), [](const std::pair<unsigned, float> &a, const std::pair<unsigned, float> &b)
    {
        return a.second > b.second;
    });

    return results;
}


std::vector<VotingMaximum> Hough3d::detect(const std::string &filename,
                                           bool useHypothesisVerification)
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
    bool use_hv = useHypothesisVerification;
    findObjects(features, keypoints, use_hv, results, positions);

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


bool Hough3d::loadModel(std::string &filename)
{
    if(!loadModelFromFile(filename)) return false;

    flann::Matrix<float> dataset = createFlannDataset();

    m_flann_index = flann::Index<flann::L2<float>>(dataset, flann::KDTreeIndexParams(4));
    m_flann_index.buildIndex();

    return true;
}


flann::Matrix<float> Hough3d::createFlannDataset()
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


void Hough3d::classifyObjectsWithSeparateVotingSpaces(
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        std::vector<std::pair<unsigned, float>> &results)
{
    // get model-scene correspondences
    // !!!
    // query/source index is codebook ("object"), match/target index is scene
    // !!!
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

    // prepare voting
    std::vector<Eigen::Vector3f> all_votes = std::move(prepareCenterVotes(object_scene_corrs, scene_features, object_center_vectors));

    // separate votelist to individual classes
    std::map<unsigned, std::vector<Eigen::Vector3f>> class_votes;
    for(unsigned i = 0; i < all_votes.size(); i++)
    {
        Eigen::Vector3f vote = all_votes.at(i);
        unsigned object_idx = object_scene_corrs->at(i).index_query;
        unsigned class_id = object_features->at(object_idx).classId;
        if(class_votes.find(class_id) != class_votes.end())
        {
            class_votes.at(class_id).push_back(vote);
        }
        else
        {
            class_votes.insert({class_id, {vote}});
        }
    }

    // cast votes of each class separately
    results.clear();
    for(auto [class_id, votelist] : class_votes)
    {
        // cast votes and retrieve maxima per class
        std::vector<double> maxima;
        std::vector<std::vector<int>> vote_indices;
        float relative_threshold = m_th; // minimal weight of a maximum in percent of highest maximum to be considered a hypothesis
        bool use_distance_weight = false;
        castVotesAndFindMaxima(object_scene_corrs, votelist, relative_threshold, use_distance_weight,
                               maxima, vote_indices, m_hough_space);

        // this creates the results for classification
        // sorts ascendingly - get last element as the one with highest value in voting space
        std::sort(maxima.begin(), maxima.end());
        if(maxima.size() > 0)
        {
            results.push_back({class_id, maxima.back()});
        }
    }
}

void Hough3d::classifyObjectsWithSingleVotingSpace(
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        std::vector<std::pair<unsigned, float>> &results)
{
    // get model-scene correspondences
    // !!!
    // query/source index is codebook ("object"), match/target index is scene
    // !!!
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

    // prepare voting
    std::vector<Eigen::Vector3f> votelist = std::move(prepareCenterVotes(object_scene_corrs, scene_features, object_center_vectors));

    // cast votes and retrieve maxima
    std::vector<double> maxima;
    std::vector<std::vector<int>> vote_indices;
    float relative_threshold = m_th; // minimal weight of a maximum in percent of highest maximum to be considered a hypothesis
    bool use_distance_weight = false;
    castVotesAndFindMaxima(object_scene_corrs, votelist, relative_threshold, use_distance_weight,
                           maxima, vote_indices, m_hough_space);

    std::cout << "Found " << maxima.size();
    if(maxima.size() != 1)
        std::cout << " maxima" << std::endl;
    else
        std::cout << " maximum" << std::endl;

    generateClassificationHypotheses(object_scene_corrs, vote_indices, object_features, results);
}


void Hough3d::findObjects(
        const pcl::PointCloud<ISMFeature>::Ptr& scene_features,
        const pcl::PointCloud<PointT>::Ptr scene_keypoints,
        const bool use_hv,
        std::vector<std::pair<unsigned, float>> &results,
        std::vector<Eigen::Vector3f> &positions)
{
    // PCL implementation has a threshold of 0.25, however, with 0.75 or without a threshold we get better results
    float matching_threshold = std::numeric_limits<float>::max();
    // get model-scene correspondences
    pcl::CorrespondencesPtr object_scene_corrs = std::move(findNnCorrespondences(scene_features, matching_threshold, m_flann_index));
    std::cout << "Found " << object_scene_corrs->size() << " correspondences" << std::endl;

    // prepare voting
    std::vector<Eigen::Vector3f> votelist = std::move(prepareCenterVotes(object_scene_corrs, scene_features, m_center_vectors));

    // cast votes and retrieve maxima
    std::vector<double> maxima;
    std::vector<std::vector<int>> vote_indices;
    float relative_threshold = m_th; // minimal weight of a maximum in percent of highest maximum to be considered a hypothesis
    bool use_distance_weight = int(m_count2) >= 2;
    castVotesAndFindMaxima(object_scene_corrs, votelist, relative_threshold, use_distance_weight,
                           maxima, vote_indices, m_hough_space);

    std::cout << "Found " << maxima.size();
    if(maxima.size() != 1)
        std::cout << " maxima" << std::endl;
    else
        std::cout << " maximum" << std::endl;

    // generate 6DOF hypotheses with absolute orientation
    std::vector<Eigen::Matrix4f> transformations;
    std::vector<pcl::Correspondences> model_instances;
    bool refine_model = int(m_count2)%2 == 0;
    float inlier_threshold = m_count;
    generateHypothesesWithAbsoluteOrientation(object_scene_corrs, vote_indices, scene_keypoints,
                                              m_keypoints, inlier_threshold, refine_model,
                                              use_hv, transformations, model_instances);

    std::cout << "Remaining hypotheses after RANSAC: " << model_instances.size() << std::endl;

    bool useClustersForPosition = true;

    // process remaining hypotheses
    results.clear();
    positions.clear();
    for(unsigned idx = 0; idx < model_instances.size(); idx++)
    {
        unsigned res_class;
        int res_num_votes;
        Eigen::Vector3f res_position;
        const pcl::Correspondences &filtered_corrs = model_instances[idx];
        const Eigen::Matrix4f &transformation = transformations[idx];

        if(useClustersForPosition)
        {
            findClassAndPositionFromCluster(filtered_corrs, m_features, scene_features, m_center_vectors,
                                            m_number_of_classes, res_class, res_num_votes, res_position);
        }
        else
        {
            // in general worse results // TODO VS eval at the end again, when best params are found
            findClassAndPositionFromTransformedObjectKeypoints(filtered_corrs, transformation, m_features,
                                            m_keypoints, m_center_vectors, m_number_of_classes, res_class,
                                            res_num_votes, res_position);
        }

        results.push_back({res_class, res_num_votes});
        positions.push_back(res_position);
    }
}


bool Hough3d::saveModelToFile(
        std::string &filename,
        std::map<unsigned, pcl::PointCloud<ISMFeature>::Ptr> &all_features,
        std::map<unsigned, std::vector<Eigen::Vector3f>> &all_vectors,
        std::map<unsigned, std::vector<Utils::BoundingBox>> &all_bounding_boxes) const
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

    std::map<unsigned, float> dims = computeAverageClassRadii(all_bounding_boxes);
    // fill in bounding box information
    unsigned bb_dims_size = dims.size();
    oa << bb_dims_size;
    for(auto it : dims)
    {
        unsigned classId = it.first;
        float radius = it.second;
        oa << classId;
        oa << radius;
    }

    ofs.close();
    return true;
}

bool Hough3d::loadModelFromFile(std::string& filename)
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
        m_keypoints->clear();
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
            PointT point;
            point.x = feature.x;
            point.y = feature.y;
            point.z = feature.z;
            m_keypoints->push_back(point);
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

        m_class_radii.clear();
        unsigned bb_dims_size;
        ia >> bb_dims_size;
        for(int i = 0; i < bb_dims_size; i++)
        {
            unsigned classId;
            float radius;
            ia >> classId;
            ia >> radius;
            m_class_radii.insert({classId, radius});
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
