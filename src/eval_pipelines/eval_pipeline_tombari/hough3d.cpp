#include "hough3d.h"

#include <pcl/io/pcd_io.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/shot_lrf_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/centroid.h>
#include <pcl/correspondence.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation.h>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include "../../implicit_shape_model/utils/utils.h"


/**
 * Implementation of the approach described in
 *
 * F. Tombari, L. Di Stefano:
 *     Object recognition in 3D scenes with occlusions and clutter by Hough voting.
 *     2010, Fourth Pacific-Rim Symposium on Image and Video Technology
 *
 */


// TODO VS: find good params for eval, in detection try additionally to sample 1000 key points per model in training and 3000 per
// scene in testing as described by tombari in the paper

Hough3d::Hough3d(std::string dataset, float bin, float th) :
    m_features(new pcl::PointCloud<ISMFeature>()),
    m_scene_keypoints(new pcl::PointCloud<PointT>()),
    m_flann_index(flann::KDTreeIndexParams(4))
{
    std::cout << "-------- loading parameters for " << dataset << " dataset --------" << std::endl;

    if(dataset == "aim" || dataset == "mcgill" || dataset == "mcg" || dataset == "psb" ||
            dataset == "sh12" || dataset == "mn10" || dataset == "mn40")
    {
        /// classification
        m_min_coord = Eigen::Vector3d(-2.0, -2.0, -2.0);
        m_max_coord = Eigen::Vector3d(2.0, 2.0, 2.0);
        m_bin_size = Eigen::Vector3d(0.5, 0.5, 0.5);

        // use this for datasets: aim, mcg, psb, shrec-12, mn10, mn40
        m_normal_radius = 0.05;
        m_reference_frame_radius = 0.3;
        m_feature_radius = 0.4;
        m_keypoint_sampling_radius = 0.2;
        m_normal_method = 1;
        m_feature_type = "SHOT";
    }
    else if(dataset == "wash" || dataset == "bigbird" || dataset == "ycb")
    {
        /// classification
        m_min_coord = Eigen::Vector3d(-1.0, -1.0, -1.0);
        m_max_coord = Eigen::Vector3d(1.0, 1.0, 1.0);
        m_bin_size = Eigen::Vector3d(0.02, 0.02, 0.02); // TODO VS find good params
        m_normal_radius = 0.005;
        m_reference_frame_radius = 0.05;
        m_feature_radius = 0.05;
        m_keypoint_sampling_radius = 0.02;
        m_normal_method = 0;
        m_feature_type = "CSHOT";
    }
    else if(dataset == "dataset1" || dataset == "dataset5")
    {
        /// detection
        m_th = th;
        m_min_coord = Eigen::Vector3d(-1.0, -1.0, -1.0);
        m_max_coord = Eigen::Vector3d(1.0, 1.0, 1.0);
        m_bin_size = Eigen::Vector3d(bin, bin, bin); // TODO VS find good params
        m_normal_radius = 0.005;
        m_reference_frame_radius = 0.05;
        m_feature_radius = 0.05;
        m_keypoint_sampling_radius = 0.02;
        m_normal_method = 0;

        if(dataset == "dataset1")
            m_feature_type = "SHOT";
        if(dataset == "dataset5")
            m_feature_type = "CSHOT";
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

        pcl::PointCloud<ISMFeature>::Ptr features_cleaned = processPointCloud(cloud);
        for(ISMFeature& ismf : features_cleaned->points)
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
        (*all_features.at(tr_class)) += (*features_cleaned);
        num_features += features_cleaned->size();
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


std::vector<std::pair<unsigned, float>> Hough3d::classify(const std::string &filename,
                                                          bool useSingleVotingSpace)
{
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if(pcl::io::loadPCDFile<PointT>(filename, *cloud) == -1)
    {
        std::cerr << "ERROR: loading file " << filename << std::endl;
        return std::vector<std::pair<unsigned, float>>();
    }

    // extract features
    pcl::PointCloud<ISMFeature>::Ptr features = processPointCloud(cloud);

    // get results
    std::vector<std::pair<unsigned, float>> results;
    if(useSingleVotingSpace)
    {
        results = classifyObjectsWithUnifiedVotingSpaces(features);
    }
    else
    {
        results = classifyObjectsWithSeparateVotingSpaces(features);
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

    // extract features
    pcl::PointCloud<ISMFeature>::Ptr features = processPointCloud(cloud);

    std::vector<VotingMaximum> result_maxima;

    // get results
    std::vector<std::pair<unsigned, float>> results;
    std::vector<Eigen::Vector3f> positions;
    bool use_hv = useHypothesisVerification;
    std::tie(results, positions) = findObjects(features, use_hv);

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


pcl::PointCloud<ISMFeature>::Ptr Hough3d::processPointCloud(pcl::PointCloud<PointT>::Ptr cloud)
{
    // create search tree
    pcl::search::Search<PointT>::Ptr searchTree;
    searchTree = pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>());

    // compute normals
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    computeNormals(cloud, normals, searchTree);

    // filter normals
    pcl::PointCloud<pcl::Normal>::Ptr normals_without_nan;
    pcl::PointCloud<PointT>::Ptr cloud_without_nan;
    filterNormals(normals, normals_without_nan, cloud, cloud_without_nan);

    // compute keypoints
    pcl::PointCloud<PointT>::Ptr keypoints;
    computeKeypoints(keypoints, cloud_without_nan);
    m_scene_keypoints = keypoints;

    // compute reference frames
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr reference_frames;
    computeReferenceFrames(reference_frames, keypoints, cloud_without_nan, searchTree);

    // sort out invalid reference frames and associated keypoints
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr cleanReferenceFrames(new pcl::PointCloud<pcl::ReferenceFrame>());
    pcl::PointCloud<PointT>::Ptr cleanKeypoints(new pcl::PointCloud<PointT>());
    unsigned missedFrames = 0;
    for (int i = 0; i < (int)reference_frames->size(); i++) {
        const pcl::ReferenceFrame& frame = reference_frames->at(i);
        if (std::isfinite (frame.x_axis[0]) &&
                std::isfinite (frame.y_axis[0]) &&
                std::isfinite (frame.z_axis[0])) {
            cleanReferenceFrames->push_back(frame);
            cleanKeypoints->push_back(keypoints->at(i));
        }
        else
            missedFrames++;
    }

    // compute descriptors
    pcl::PointCloud<ISMFeature>::Ptr features;
    computeDescriptors(cloud_without_nan, normals_without_nan, cleanKeypoints, searchTree, cleanReferenceFrames, features);


    // store keypoint positions and reference frames
    for (int i = 0; i < (int)features->size(); i++)
    {
        ISMFeature& feature = features->at(i);
        const PointT& keypoint = cleanKeypoints->at(i);
        feature.x = keypoint.x;
        feature.y = keypoint.y;
        feature.z = keypoint.z;
        feature.referenceFrame = cleanReferenceFrames->at(i);
    }

    // remove NAN features
    pcl::PointCloud<ISMFeature>::Ptr features_cleaned;
    removeNanDescriptors(features, features_cleaned);

    return features_cleaned;
}


void Hough3d::computeNormals(pcl::PointCloud<PointT>::Ptr cloud,
                           pcl::PointCloud<pcl::Normal>::Ptr& normals,
                           pcl::search::Search<PointT>::Ptr searchTree) const
{
    normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());

//    if(m_normal_method == 0 && cloud->isOrganized())
//    {
//        std::cout << " --- 1 --- " << std::endl;
//        pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> normalEst;
//        normalEst.setInputCloud(cloud);
//        normalEst.setNormalEstimationMethod(normalEst.AVERAGE_3D_GRADIENT);
//        normalEst.setMaxDepthChangeFactor(0.02f);
//        normalEst.setNormalSmoothingSize(10.0f);
//        normalEst.useSensorOriginAsViewPoint();
//        normalEst.compute(*normals);
//    }
//    else if(m_normal_method == 0 && !cloud->isOrganized())
    if(m_normal_method == 0)
    {
         // prepare PCL normal estimation object
        pcl::NormalEstimationOMP<PointT, pcl::Normal> normalEst;
        normalEst.setInputCloud(cloud);
        normalEst.setNumberOfThreads(0);
        normalEst.setSearchMethod(searchTree);
        normalEst.setRadiusSearch(m_normal_radius);
        normalEst.setViewPoint(0,0,0);
        normalEst.compute(*normals);
    }
    else
    {
        // prepare PCL normal estimation object
        pcl::NormalEstimationOMP<PointT, pcl::Normal> normalEst;
        normalEst.setInputCloud(cloud);
        normalEst.setNumberOfThreads(0);
        normalEst.setSearchMethod(searchTree);
        normalEst.setRadiusSearch(m_normal_radius);

        // move model to origin, then point normals away from origin
        pcl::PointCloud<PointT>::Ptr model_no_centroid(new pcl::PointCloud<PointT>());
        pcl::copyPointCloud(*cloud, *model_no_centroid);

        // compute the object centroid
        Eigen::Vector4f centroid4f;
        pcl::compute3DCentroid(*model_no_centroid, centroid4f);
        Eigen::Vector3f centroid(centroid4f[0], centroid4f[1], centroid4f[2]);
        // remove centroid for normal computation
        for(PointT& point : model_no_centroid->points)
        {
            point.x -= centroid.x();
            point.y -= centroid.y();
            point.z -= centroid.z();
        }
        normalEst.setInputCloud(model_no_centroid);
        normalEst.setViewPoint(0,0,0);
        normalEst.compute(*normals);
        // invert normals
        for(pcl::Normal& norm : normals->points)
        {
            norm.normal_x *= -1;
            norm.normal_y *= -1;
            norm.normal_z *= -1;
        }
    }
}

void Hough3d::filterNormals(pcl::PointCloud<pcl::Normal>::Ptr normals,
                          pcl::PointCloud<pcl::Normal>::Ptr &normals_without_nan,
                          pcl::PointCloud<PointT>::Ptr cloud,
                          pcl::PointCloud<PointT>::Ptr &cloud_without_nan) const
{
    std::vector<int> mapping;
    normals_without_nan = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());
    pcl::removeNaNNormalsFromPointCloud(*normals, *normals_without_nan, mapping);

    // create new point cloud without NaN normals
    cloud_without_nan = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
    for (int i = 0; i < (int)mapping.size(); i++)
    {
        cloud_without_nan->push_back(cloud->at(mapping[i]));
    }
}


void Hough3d::computeKeypoints(pcl::PointCloud<PointT>::Ptr &keypoints, pcl::PointCloud<PointT>::Ptr cloud) const
{
    pcl::VoxelGrid<PointT> voxelGrid;
    voxelGrid.setInputCloud(cloud);
    voxelGrid.setLeafSize(m_keypoint_sampling_radius, m_keypoint_sampling_radius, m_keypoint_sampling_radius);
    keypoints = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
    voxelGrid.filter(*keypoints);
}


void Hough3d::computeReferenceFrames(pcl::PointCloud<pcl::ReferenceFrame>::Ptr &reference_frames,
                                   pcl::PointCloud<PointT>::Ptr keypoints,
                                   pcl::PointCloud<PointT>::Ptr cloud,
                                   pcl::search::Search<PointT>::Ptr searchTree) const
{
    reference_frames = pcl::PointCloud<pcl::ReferenceFrame>::Ptr(new pcl::PointCloud<pcl::ReferenceFrame>());
    pcl::SHOTLocalReferenceFrameEstimationOMP<PointT, pcl::ReferenceFrame> refEst;
    refEst.setRadiusSearch(m_reference_frame_radius);
    refEst.setInputCloud(keypoints);
    refEst.setSearchSurface(cloud);
    refEst.setSearchMethod(searchTree);
    refEst.compute(*reference_frames);

    pcl::PointCloud<pcl::ReferenceFrame>::Ptr cleanReferenceFrames(new pcl::PointCloud<pcl::ReferenceFrame>());
    pcl::PointCloud<PointT>::Ptr cleanKeypoints(new pcl::PointCloud<PointT>());
    for(int i = 0; i < (int)reference_frames->size(); i++)
    {
        const pcl::ReferenceFrame& frame = reference_frames->at(i);
        if(std::isfinite(frame.x_axis[0]) && std::isfinite(frame.y_axis[0]) && std::isfinite(frame.z_axis[0]))
        {
            cleanReferenceFrames->push_back(frame);
            cleanKeypoints->push_back(keypoints->at(i));
        }
    }

    keypoints = cleanKeypoints;
    reference_frames = cleanReferenceFrames;
}


void Hough3d::computeDescriptors(pcl::PointCloud<PointT>::Ptr cloud,
                               pcl::PointCloud<pcl::Normal>::Ptr normals,
                               pcl::PointCloud<PointT>::Ptr keypoints,
                               pcl::search::Search<PointT>::Ptr searchTree,
                               pcl::PointCloud<pcl::ReferenceFrame>::Ptr reference_frames,
                               pcl::PointCloud<ISMFeature>::Ptr &features) const
{
    if(m_feature_type == "SHOT")
    {
        pcl::SHOTEstimationOMP<PointT, pcl::Normal, pcl::SHOT352> shotEst;
        shotEst.setSearchSurface(cloud);
        shotEst.setInputNormals(normals);
        shotEst.setInputCloud(keypoints);
        shotEst.setInputReferenceFrames(reference_frames);
        shotEst.setSearchMethod(searchTree);
        shotEst.setRadiusSearch(m_feature_radius);
        pcl::PointCloud<pcl::SHOT352>::Ptr shot_features(new pcl::PointCloud<pcl::SHOT352>());
        shotEst.compute(*shot_features);

        // create descriptor point cloud
        features = pcl::PointCloud<ISMFeature>::Ptr(new pcl::PointCloud<ISMFeature>());
        features->resize(shot_features->size());

        for (int i = 0; i < (int)shot_features->size(); i++)
        {
            ISMFeature& feature = features->at(i);
            const pcl::SHOT352& shot = shot_features->at(i);

            // store the descriptor
            feature.descriptor.resize(352);
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = shot.descriptor[j];
        }
    }
    else if(m_feature_type == "CSHOT")
    {
        pcl::SHOTColorEstimationOMP<PointT, pcl::Normal, pcl::SHOT1344> shotEst;

        // temporary workaround to fix race conditions in OMP version of CSHOT in PCL
        if (shotEst.sRGB_LUT[0] < 0)
        {
          for (int i = 0; i < 256; i++)
          {
            float f = static_cast<float> (i) / 255.0f;
            if (f > 0.04045)
              shotEst.sRGB_LUT[i] = powf ((f + 0.055f) / 1.055f, 2.4f);
            else
              shotEst.sRGB_LUT[i] = f / 12.92f;
          }

          for (int i = 0; i < 4000; i++)
          {
            float f = static_cast<float> (i) / 4000.0f;
            if (f > 0.008856)
              shotEst.sXYZ_LUT[i] = static_cast<float> (powf (f, 0.3333f));
            else
              shotEst.sXYZ_LUT[i] = static_cast<float>((7.787 * f) + (16.0 / 116.0));
          }
        }

        shotEst.setSearchSurface(cloud);
        shotEst.setInputNormals(normals);
        shotEst.setInputCloud(keypoints);
        shotEst.setInputReferenceFrames(reference_frames);
        shotEst.setSearchMethod(searchTree);
        shotEst.setRadiusSearch(m_feature_radius);
        pcl::PointCloud<pcl::SHOT1344>::Ptr shot_features(new pcl::PointCloud<pcl::SHOT1344>());
        shotEst.compute(*shot_features);

        // create descriptor point cloud
        features = pcl::PointCloud<ISMFeature>::Ptr(new pcl::PointCloud<ISMFeature>());
        features->resize(shot_features->size());

        for (int i = 0; i < (int)shot_features->size(); i++)
        {
            ISMFeature& feature = features->at(i);
            const pcl::SHOT1344& shot = shot_features->at(i);

            // store the descriptor
            feature.descriptor.resize(1344);
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = shot.descriptor[j];
        }
    }
}


void Hough3d::removeNanDescriptors(pcl::PointCloud<ISMFeature>::Ptr features,
                                 pcl::PointCloud<ISMFeature>::Ptr &features_cleaned) const
{
    features_cleaned = pcl::PointCloud<ISMFeature>::Ptr(new pcl::PointCloud<ISMFeature>());
    features_cleaned->header = features->header;
    features_cleaned->height = 1;
    features_cleaned->is_dense = false;
    bool nan_found = false;
    for(int a = 0; a < features->size(); a++)
    {
        ISMFeature fff = features->at(a);
        for(int b = 0; b < fff.descriptor.size(); b++)
        {
            if(std::isnan(fff.descriptor.at(b)))
            {
                nan_found = true;
                break;
            }
        }
        if(!nan_found)
        {
            features_cleaned->push_back(fff);
        }
        nan_found = false;
    }
    features_cleaned->width = features_cleaned->size();
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


std::vector<std::pair<unsigned, float>>
Hough3d::classifyObjectsWithSeparateVotingSpaces(
        const pcl::PointCloud<ISMFeature>::Ptr& scene_features) const
{
    int k_search = 1;

    // loop over all features extracted from the input model
    std::map<unsigned, std::vector<Eigen::Vector3f>> all_votes;
    for(int fe = 0; fe < scene_features->size(); fe++)
    {
        // insert the query point
        ISMFeature scene_feat = scene_features->at(fe);
        flann::Matrix<float> query(new float[scene_feat.descriptor.size()], 1, scene_feat.descriptor.size());
        for(int i = 0; i < scene_feat.descriptor.size(); i++)
        {
            query[0][i] = scene_feat.descriptor.at(i);
        }

        // prepare results
        std::vector<std::vector<int>> indices;
        std::vector<std::vector<float>> distances;
        m_flann_index.knnSearch(query, indices, distances, k_search, flann::SearchParams(128));

        delete[] query.ptr();

        // tombari uses some kind of distance threshold but doesn't report it
        // PCL implementation has a threshold of 0.25, however, without a threshold we get better results
        float threshold = std::numeric_limits<float>::max();
        if(distances[0][0] < threshold)
        {
            const ISMFeature &object_feat = m_features->at(indices[0][0]);
            unsigned class_id = object_feat.classId;

            pcl::ReferenceFrame ref = scene_feat.referenceFrame;
            Eigen::Vector3f keyPos(scene_feat.x, scene_feat.y, scene_feat.z);
            Eigen::Vector3f vote = m_center_vectors.at(indices[0][0]);
            Eigen::Vector3f center = keyPos + Utils::rotateBack(vote, ref);
            // collect vote
            if(all_votes.find(class_id) != all_votes.end())
            {
                // known class - append vote
                all_votes.at(class_id).push_back(center);
            }
            else
            {
                // class not seen yet - insert first vote
                all_votes.insert({class_id, {center}});
            }
        }
    }

    // cast votes of each class separately
    // (tombari speaks of only one single hough space, but this makes implementation easier without drawbacks)
    std::vector<std::pair<unsigned, float>> results;
    for(auto elem : all_votes)
    {
        unsigned class_id = elem.first;
        std::vector<Eigen::Vector3f> votelist = elem.second;
        // vote all votes into empty hough space
        m_hough_space->reset();
        for(int vid = 0; vid < votelist.size(); vid++)
        {
            Eigen::Vector3d vote(votelist[vid].x(), votelist[vid].y(), votelist[vid].z());
            // voting with interpolation: tombari does not use interpolation, but later checks neighboring bins
            // interpolated voting should be the same or even better
            m_hough_space->voteInt(vote, 1.0, vid); // NOTE: voteIndices not used here, therefore just using some (meaningless) id
        }
        // find maxima for this class id
        std::vector<double> maxima;
        float m_relThreshold = 0.1f;
        std::vector<std::vector<int>> voteIndices;
        m_hough_space->findMaxima(-m_relThreshold, maxima, voteIndices);
        // this creates the results for classification
        // sorts ascendingly - get last element as the one with highest value in voting space
        std::sort(maxima.begin(), maxima.end());
        if(maxima.size() > 0)
        {
            results.push_back({class_id, maxima.back()});
        }
    }

    return results;
}

std::vector<std::pair<unsigned, float>>
Hough3d::classifyObjectsWithUnifiedVotingSpaces(
        const pcl::PointCloud<ISMFeature>::Ptr& scene_features) const
{
    int k_search = 1;

    // loop over all features extracted from the input model
    std::map<unsigned, std::vector<Eigen::Vector3f>> all_votes;
    std::map<unsigned, std::vector<int>> all_match_idxs;
    pcl::Correspondences object_scene_corrs;

    for(int fe = 0; fe < scene_features->size(); fe++)
    {
        // insert the query point
        ISMFeature scene_feat = scene_features->at(fe);
        flann::Matrix<float> query(new float[scene_feat.descriptor.size()], 1, scene_feat.descriptor.size());
        for(int i = 0; i < scene_feat.descriptor.size(); i++)
        {
            query[0][i] = scene_feat.descriptor.at(i);
        }

        // prepare results
        std::vector<std::vector<int>> indices;
        std::vector<std::vector<float>> distances;
        m_flann_index.knnSearch(query, indices, distances, k_search, flann::SearchParams(128));

        delete[] query.ptr();

        // tombari uses some kind of distance threshold but doesn't report it
        // PCL implementation has a threshold of 0.25, however, without a threshold we get better results
        float threshold = std::numeric_limits<float>::max();
        if(distances[0][0] < threshold)
        {
            // create current correspondence
            pcl::Correspondence corr;
            corr.index_query = fe; // query is from the list of scene features
            corr.index_match = indices[0][0];
            corr.distance = distances[0][0];
            int corr_id = object_scene_corrs.size();
            object_scene_corrs.push_back(corr);

            const ISMFeature &object_feat = m_features->at(indices[0][0]);
            unsigned class_id = object_feat.classId;

            pcl::ReferenceFrame ref = scene_feat.referenceFrame;
            Eigen::Vector3f keyPos(scene_feat.x, scene_feat.y, scene_feat.z);

            Eigen::Vector3f vote = m_center_vectors.at(indices[0][0]);
            Eigen::Vector3f center = keyPos + Utils::rotateBack(vote, ref);
            // collect vote
            if(all_votes.find(class_id) != all_votes.end())
            {
                // known class - append vote
                all_votes.at(class_id).push_back(center);
                all_match_idxs.at(class_id).push_back(corr_id);
            }
            else
            {
                // class not seen yet - insert first vote
                all_votes.insert({class_id, {center}});
                all_match_idxs.insert({class_id, {corr_id}});
            }
        }
    }

    std::vector<std::pair<unsigned, float>> results;
    // use all votes at once
    std::vector<Eigen::Vector3f> votelist;
    for(auto elem : all_votes)
        votelist.insert(votelist.end(), elem.second.begin(), elem.second.end());
    std::vector<int> idlist;
    for(auto elem : all_match_idxs)
        idlist.insert(idlist.end(), elem.second.begin(), elem.second.end());

    // vote all votes into empty hough space
    m_hough_space->reset();
    for(int vid = 0; vid < votelist.size(); vid++)
    {
        Eigen::Vector3d vote(votelist[vid].x(), votelist[vid].y(), votelist[vid].z());
        int vote_id = idlist.at(vid);
        // voting with interpolation
        // tombari does not use interpolation, but later checks neighboring bins
        // interpolated voting should be the same or even better
        m_hough_space->voteInt(vote, 1.0, vote_id);
    }
    // find maxima
    std::vector<double> maxima;
    float m_relThreshold = 0.1f;
    std::vector<std::vector<int>> voteIndices;
    m_hough_space->findMaxima(-m_relThreshold, maxima, voteIndices);

    // check all maxima since highest valued maximum might still be composed of different class votes
    // therefore we need to count votes per class per maximum
    for (size_t j = 0; j < maxima.size (); ++j)
    {
        std::map<unsigned, int> class_occurences;
        pcl::Correspondences max_corrs;
        for (size_t i = 0; i < voteIndices[j].size(); ++i)
        {
            max_corrs.push_back(object_scene_corrs.at(voteIndices[j][i]));
        }

        // count class occurences in filtered corrs
        for(unsigned fcorr_idx = 0; fcorr_idx < max_corrs.size(); fcorr_idx++)
        {
            unsigned match_idx = max_corrs.at(fcorr_idx).index_match;
            const ISMFeature &cur_feat = m_features->at(match_idx);
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
        for(auto occ_elem : class_occurences)
        {
            if(occ_elem.second > cur_best_num)
            {
                cur_best_num = occ_elem.second;
                cur_class = occ_elem.first;
            }
        }
        results.push_back({cur_class, cur_best_num});
    }

    return results;
}


std::tuple<std::vector<std::pair<unsigned, float>>,std::vector<Eigen::Vector3f>>
Hough3d::findObjects(
        const pcl::PointCloud<ISMFeature>::Ptr& scene_features,
        const bool use_hv)
{
    // get model-scene correspondences
    // query index is scene, match index is codebook ("object")
    // PCL implementation has a threshold of 0.25, however, without a threshold we get better results
    float matching_threshold = std::numeric_limits<float>::max();
    pcl::CorrespondencesPtr object_scene_corrs = findNnCorrespondences(scene_features, matching_threshold);
    std::cout << "Found " << object_scene_corrs->size() << " correspondences" << std::endl;

    // object keypoints are simply the matched keypoints from the codebook
    pcl::PointCloud<PointT>::Ptr object_keypoints(new pcl::PointCloud<PointT>());
    pcl::PointCloud<ISMFeature>::Ptr object_features(new pcl::PointCloud<ISMFeature>());
    std::vector<Eigen::Vector3f> object_center_vectors;
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr object_lrf(new pcl::PointCloud<pcl::ReferenceFrame>());
    // however in order not to pass the whole codebook, we need to adjust the index mapping
    for(unsigned i = 0; i < object_scene_corrs->size(); i++)
    {
        // create new list of keypoints and reassign the object (i.e. match) index
        pcl::Correspondence &corr = object_scene_corrs->at(i);
        int &index = corr.index_match;
        const ISMFeature &feat = m_features->at(index);
        object_features->push_back(feat);
        object_center_vectors.push_back(m_center_vectors.at(index));

        PointT keypoint = PointT(feat.x, feat.y, feat.z);
        object_keypoints->push_back(keypoint);

        pcl::ReferenceFrame lrf = feat.referenceFrame;
        object_lrf->push_back(lrf);
        // remap index to new position in the created point clouds
        // no longer referring to m_features, but now to object_features
        index = i;
    }

    // prepare voting
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

    // cast votes
    std::vector<std::pair<unsigned, float>> results;
    m_hough_space->reset();
    for(int vid = 0; vid < votelist.size(); vid++)
    {
        Eigen::Vector3d vote(votelist[vid].x(), votelist[vid].y(), votelist[vid].z());
        // voting with interpolation
        // votes are in the same order as correspondences, vid is therefore the index in the correspondence list
        m_hough_space->voteInt(vote, 1.0, vid);
    }
    // find maxima for this class id
    float m_relThreshold = m_th; // TODO VS check this param
    std::vector<double> maxima;
    std::vector<std::vector<int>> voteIndices;
    m_hough_space->findMaxima(-m_relThreshold, maxima, voteIndices);

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

    // generate 6DOF hypotheses with absolute orientation
    std::vector<Eigen::Matrix4f> transformations;
    std::vector<Eigen::Vector3f> positions;
    std::vector<pcl::Correspondences> model_instances;
    pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> corr_rejector;
    corr_rejector.setMaximumIterations(10000);
    corr_rejector.setInlierThreshold(m_bin_size(0));
    corr_rejector.setInputSource(m_scene_keypoints);
    corr_rejector.setInputTarget(object_keypoints); // idea: use keypoints of features matched in the codebook
    //corr_rejector.setRefineModel(true); // slightly worse results

    for(size_t j = 0; j < maxima.size (); ++j)
    {
        pcl::Correspondences temp_corrs, filtered_corrs;
        for (size_t i = 0; i < voteIndices[j].size(); ++i)
        {
            temp_corrs.push_back(object_scene_corrs->at(voteIndices[j][i]));
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

    // process remaining hypotheses
    for(const pcl::Correspondences &filtered_corrs : model_instances)
    {
        unsigned res_class;
        int res_num_votes;
        Eigen::Vector3f res_position;
        findClassAndPositionFromCluster(filtered_corrs, object_features, scene_features,
                                        object_center_vectors, res_class, res_num_votes, res_position);
        results.push_back({res_class, res_num_votes});
        positions.push_back(res_position);
    }

    return std::make_tuple(results, positions);
}


void Hough3d::findClassAndPositionFromCluster(
        const pcl::Correspondences &filtered_corrs,
        const pcl::PointCloud<ISMFeature>::Ptr object_features,
        const pcl::PointCloud<ISMFeature>::Ptr scene_features,
        const std::vector<Eigen::Vector3f> &object_center_vectors,
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


pcl::CorrespondencesPtr Hough3d::findNnCorrespondences(
        const pcl::PointCloud<ISMFeature>::Ptr& scene_features,
        const float matching_threshold) const
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
        m_flann_index.knnSearch(query, indices, distances, 1, flann::SearchParams(128));

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


bool Hough3d::saveModelToFile(std::string &filename,
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
