/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2020, Viktor Seib, Norman Link
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 * * list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "implicit_shape_model.h"

#define PCL_NO_PRECOMPILE
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include "third_party/pcl_normal_3d_omp_with_eigenvalues/normal_3d_omp_with_eigenvalues.hpp"
#include <pcl/features/integral_image_normal.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/centroid.h>

#include <fstream>
#include <boost/chrono.hpp>

#include <iostream>
#include <omp.h>
#include <opencv2/ml/ml.hpp>

#include "keypoints/keypoints_voxel_grid.h"
#include "features/features_shot.h"
#include "features/features_grsd.h"
#include "feature_ranking/ranking_uniform.h"
#include "clustering/clustering_agglomerative.h"
#include "voting/voting_hough_3d.h"
#include "voting/voting_mean_shift.h"

#include "classifier/custom_SVM.h"
#include "utils/distance.h"
#include "utils/normal_orientation.h"
#include "utils/factory.h"
#include "utils/exception.h"

#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/basicconfigurator.h>


namespace ism3d
{

ImplicitShapeModel::ImplicitShapeModel() : m_distance(0)
{
    // init logging
    log4cxx::LayoutPtr layout(new log4cxx::PatternLayout("[\%d{HH:mm:ss}] \%p: \%m\%n"));
    log4cxx::ConsoleAppender* consoleAppender = new log4cxx::ConsoleAppender(layout);
    log4cxx::BasicConfigurator::configure(log4cxx::AppenderPtr(consoleAppender));
    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());

    // parameters for preprocessing
    addParameter(m_use_smoothing, "UseSmoothing", false);
    addParameter(m_polynomial_order, "SmoothingPolynomialOrder", 1);
    addParameter(m_smoothing_radius, "SmoothingRadius", 0.01f);
    addParameter(m_use_statistical_outlier_removal, "UseStatisticalOutlierRemoval", false);
    addParameter(m_som_mean_k, "OutlierRemovalMeanK", 20);
    addParameter(m_som_std_dev_mul, "OutlierRemovalStddevMul", 2.0f);
    addParameter(m_use_radius_outlier_removal, "UseRadiusOutlierRemoval", false);
    addParameter(m_ror_min_neighbors, "OutlierRemovalMinNeighbors", 10);
    addParameter(m_ror_radius, "OutlierRemovalRadius", 0.005f);
    addParameter(m_use_voxel_filtering, "UseVoxelFiltering", false);
    addParameter(m_voxel_leaf_size, "VoxelLeafSize", 0.0015f);

    // detection threshold for evaluation
    addParameter(m_distance_detection_thresh, "DistanceThresholdDetection", 0.05f);

    addParameter(m_distanceType, "DistanceType", std::string("Euclidean"));
    addParameter(m_normal_radius, "NormalRadius", 0.05f);
    addParameter(m_consistent_normals_k, "ConsistentNormalsK", 10);
    addParameter(m_consistent_normals_method, "ConsistentNormalsMethod", 2);
    addParameter(m_num_threads, "NumThreads", 0);
    addParameter(m_bb_type, "BoundingBoxType", std::string("MVBB"));
    addParameter(m_set_color_to_zero, "SetColorToZero", false);
    addParameter(m_enable_voting_analysis, "EnableVotingAnalysis", false);
    addParameter(m_voting_analysis_output_path, "VotingAnalysisOutputPath", std::string("/home/vseib/Desktop/"));
    addParameter(m_use_svm, "UseSvmTraining", false);
    addParameter(m_svm_auto_train, "SvmAutoTrain", false);
    addParameter(m_svm_1_vs_all_train, "SvmOneVsAllTraining", false);
    addParameter(m_svm_param_c, "SvmParamC", 7.41);
    addParameter(m_svm_param_gamma, "SvmParamGamma", 2.96);
    addParameter(m_svm_param_k_fold, "SvmParamKfold", 10);
    addParameter(m_single_object_mode, "SingleObjectMode", false);
    addParameter(m_num_kd_trees, "FLANNNumKDTrees", 4);
    addParameter(m_flann_exact_match, "FLANNExactMatch", false);
    addParameter(m_instance_labels_primary, "InstanceLabelsPrimary", true);

    init();
}

ImplicitShapeModel::~ImplicitShapeModel()
{
    log4cxx::Logger::getRootLogger()->removeAllAppenders();

    delete m_codebook;
    delete m_keypoints_detector;
    delete m_feature_descriptor;
    delete m_global_feature_descriptor;
    delete m_clustering;
    delete m_voting;
    delete m_distance;
    delete m_feature_ranking;
}

void ImplicitShapeModel::setLogging(bool l)
{
    if(l)
        log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());
    else
        log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getWarn());
}

void ImplicitShapeModel::init()
{
    iPostInitConfig();

    m_enable_signals = true;
    m_index_created = false;

    m_processing_times = {{"complete",0}, {"features",0}, {"keypoints",0}, {"normals",0}, {"flann",0}, {"voting",0}, {"maxima",0}};

    m_codebook = new Codebook();
    m_keypoints_detector = new KeypointsVoxelGrid();
    m_feature_descriptor = new FeaturesSHOT();
    m_global_feature_descriptor = new FeaturesGRSD();
    m_clustering = new ClusteringAgglomerative();
    m_voting = new VotingMeanShift();
    m_feature_ranking = new RankingUniform();
}

void ImplicitShapeModel::clear()
{
    m_training_objects_filenames.clear();
    m_training_objects_instance_ids.clear();
    m_training_objects_has_normals.clear();
    m_codebook->clear();
    m_clustering->clear();
    m_voting->clear();
}

bool ImplicitShapeModel::addTrainingModel(const std::string& filename, unsigned class_id, unsigned instance_id)
{
    LOG_INFO("adding training model with class id " << class_id << " and instance id " << instance_id);

    // add model
    m_training_objects_filenames[class_id].push_back(filename);
    m_training_objects_instance_ids[class_id].push_back(instance_id);
    m_training_objects_has_normals[class_id].push_back(true); // NOTE: optimistic assumption, needs to be checked later
    return true;
}

pcl::PointCloud<PointNormalT>::Ptr ImplicitShapeModel::loadPointCloud(const std::string& filename)
{
    pcl::PointCloud<PointNormalT>::Ptr pointCloud(new pcl::PointCloud<PointNormalT>());

    // filename needs to have at least an extension like .pcd or .ply
    if (filename.size() < 5) {
        LOG_ERROR("invalid filename: " << filename);
        return pcl::PointCloud<PointNormalT>::Ptr();
    }

    std::string extension = filename.substr(filename.size() - 4, 4);

    // load the point cloud
    if (extension == ".pcd") {
        if (pcl::io::loadPCDFile(filename, *pointCloud) < 0) {
            LOG_ERROR("could not load pcd file: " << filename);
            return pcl::PointCloud<PointNormalT>::Ptr();
        }
    }
    else if (extension == ".ply") {
        if (pcl::io::loadPLYFile(filename, *pointCloud) < 0) {
            LOG_ERROR("could not load ply file: " << filename);
            return pcl::PointCloud<PointNormalT>::Ptr();
        }
    }
    else {
        LOG_ERROR("Unknown extension: " << extension);
        return pcl::PointCloud<PointNormalT>::Ptr();
    }

    if (pointCloud->size() == 0) {
        LOG_ERROR("point cloud is empty");
        return pcl::PointCloud<PointNormalT>::Ptr();
    }

    return pointCloud;
}


void ImplicitShapeModel::train()
{
    // clear data
    m_codebook->clear();

    if (m_training_objects_filenames.size() == 0) {
        LOG_WARN("no training models found");
        return;
    }

    // measure the time
    boost::timer::cpu_timer timer_all;

    // contains the whole list of features for each class id and for each model
    std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> features; // TODO VS rethink the whole feature data type!!!
    std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> globalFeatures;
    std::map<unsigned, std::vector<Utils::BoundingBox>> boundingBoxes;

    // compute features for all models and all classes
    for (auto it = m_training_objects_filenames.begin(); it != m_training_objects_filenames.end(); it++)
    {
        unsigned class_id = it->first;

        LOG_ASSERT(m_training_objects_filenames.size() == m_training_objects_has_normals.size());
        LOG_ASSERT(m_training_objects_filenames.size() == m_training_objects_instance_ids.size());

        const std::vector<std::string>& cloud_filenames = it->second;
        const std::vector<unsigned>& cloud_instance_ids = m_training_objects_instance_ids[class_id];
        const std::vector<bool>& clouds_have_normals = m_training_objects_has_normals[class_id];

        LOG_ASSERT(cloud_filenames.size() == clouds_have_normals.size());
        LOG_ASSERT(cloud_filenames.size() == cloud_instance_ids.size());

        LOG_INFO("----------------------------------------------------------------");
        LOG_INFO("training class " << class_id << " with " << cloud_filenames.size() << " objects");

        for (int j = 0; j < (int)cloud_filenames.size(); j++)
        {
            LOG_INFO("training class " << class_id << ", current file " << j+1 << " of " << cloud_filenames.size() << ": " << cloud_filenames[j]);
            pcl::PointCloud<PointNormalT>::Ptr point_cloud = loadPointCloud(cloud_filenames[j]);
            point_cloud->is_dense = false; // to prevent errors in some PCL algorithms
            unsigned instance_id = cloud_instance_ids[j];



            // TODO VS: remove this completely
            if(m_set_color_to_zero)
            {
                LOG_INFO("Setting color to 0 in loaded point cloud");
                for(int i = 0; i < point_cloud->size(); i++)
                {
                    point_cloud->at(i).r = 0;
                    point_cloud->at(i).g = 0;
                    point_cloud->at(i).b = 0;
                }
            }
            bool normal_xyz_color_transform = false;
            if(normal_xyz_color_transform)
            {
                LOG_INFO("Setting color to normal's XYZ in loaded point cloud");
                for(int i = 0; i < point_cloud->size(); i++)
                {
                    const PointNormalT point = point_cloud->at(i);
                    point_cloud->at(i).r = point.normal_z * 255.0f;
                    point_cloud->at(i).g = point.normal_y * 255.0f;
                    point_cloud->at(i).b = point.normal_x * 255.0f;
                }
            }
            bool jet_color_transform = false;
            if(jet_color_transform)
            {
                LOG_INFO("Setting color to JET color scheme in loaded point cloud");
                for(int i = 0; i < point_cloud->size(); i++)
                {
                    // note: assuming x,y,z are in [-1|1]
                    point_cloud->at(i).r = 127.5f + (point_cloud->at(i).x * 127.5f);
                    point_cloud->at(i).g = 127.5f + (point_cloud->at(i).y * 127.5f);
                    point_cloud->at(i).b = 127.5f + (point_cloud->at(i).z * 127.5f);
                }
//                std::string name = "/home/vseib/Desktop/"+cloud_filenames[j];
//                LOG_INFO("now saving, name is: "+ name);
//                pcl::io::savePCDFileASCII(name, *point_cloud);
            }





            // compute bounding box
            Utils::BoundingBox bounding_box;
            if (m_bb_type == "MVBB")
                bounding_box = Utils::computeMVBB<PointNormalT>(point_cloud);
            else if (m_bb_type == "AABB")
                bounding_box = Utils::computeAABB<PointNormalT>(point_cloud);
            else
                throw BadParamExceptionType<std::string>("invalid bounding box type", m_bb_type);

            if(m_enable_signals)
            {
                m_signalBoundingBox(bounding_box);
            }

            // check first normal
            bool has_normals = clouds_have_normals[j];
            if (has_normals) {
                const PointNormalT& firstNormal = point_cloud->at(0);
                if (firstNormal.normal_x == 0 && firstNormal.normal_y == 0 && firstNormal.normal_z == 0 ||
                        std::isnan(firstNormal.normal_x) ||
                        std::isnan(firstNormal.normal_y) ||
                        std::isnan(firstNormal.normal_z) ||
                        std::isnan(firstNormal.curvature)
                        )
                    has_normals = false;
            }

            // compute features
            pcl::PointCloud<ISMFeature>::ConstPtr cloud_features;
            pcl::PointCloud<ISMFeature>::ConstPtr global_features;
            boost::timer::cpu_timer timer_dummy;
            std::tie(cloud_features, global_features, std::ignore, std::ignore) = computeFeatures(point_cloud, has_normals, timer_dummy, timer_dummy, true);

            // check for NAN features
            pcl::PointCloud<ISMFeature>::Ptr cloud_features_cleaned = removeNaNFeatures(cloud_features);
            pcl::PointCloud<ISMFeature>::Ptr global_features_cleaned = removeNaNFeatures(global_features);

            // insert labels into features
            for(ISMFeature& ismf : cloud_features_cleaned->points)
            {
                ismf.classId = class_id;
                ismf.instanceId = instance_id;
            }
            for(ISMFeature& ismf : global_features_cleaned->points)
            {
                ismf.classId = class_id;
                ismf.instanceId = instance_id;
            }

            if(m_enable_signals)
            {
                m_signalFeatures(cloud_features_cleaned);
            }

            // concatenate features
            features[class_id].push_back(cloud_features_cleaned);
            globalFeatures[class_id].push_back(global_features_cleaned);
            boundingBoxes[class_id].push_back(bounding_box);
        }
    }

    LOG_ASSERT(features.size() == boundingBoxes.size());

    // train SVM with global features
    if(m_use_svm)
    {
        LOG_INFO("Training SVM");
        trainSVM(globalFeatures);
    }

    // store average sizes as a hint for bandwidth during detection
    m_voting->determineAverageBoundingBoxDimensions(boundingBoxes); // TODO VS: remove this if inferior to standard params
    // forward global feature to voting class to store them
    m_voting->forwardGlobalFeatures(globalFeatures);

    LOG_INFO("computing feature ranking");
    // remove features with low scores
    std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> features_ranked;
    pcl::PointCloud<ISMFeature>::Ptr allFeatures_ranked(new pcl::PointCloud<ISMFeature>());

    std::tie(features_ranked, allFeatures_ranked) =
            (*m_feature_ranking)(features, m_num_kd_trees, m_flann_exact_match);

    // cluster descriptors and extract cluster centers
    LOG_INFO("clustering");
    (*m_clustering)(allFeatures_ranked, m_distance);
    const std::vector<std::vector<float>> clusterCenters = m_clustering->getClusterCenters();
    const std::vector<int> clusterIndices = m_clustering->getClusterIndices();

    // compute which cluster indices are assigned which feature indices
    std::vector<std::vector<int>> clusters(clusterCenters.size()); // each position: list of feature indices of a cluster
    for (int i = 0; i < (int)allFeatures_ranked->size(); i++)
    {
        int clusterIndex = clusterIndices[i]; // this index indicates which cluster the feature i belongs to
        clusters[clusterIndex].push_back(i);
    }
    // NOTE: if no clustering is used, clusterIndices are just ascending numbers (0, 1, 2, 3, ...)
    // in that case, clusters at each position have size == 1
    LOG_ASSERT(clusterIndices.size() == allFeatures_ranked->size());

    // create codewords and add them to the codebook - NOTE: if no clustering is used: a codeword
    // is just one feature and its center vector
    LOG_INFO("creating codewords");
    std::vector<std::shared_ptr<Codeword>> codewords;
    for (int i = 0; i < (int)clusterCenters.size(); i++)
    {
        // adding the class id to codewords as workaround: need to have class access during activation
        unsigned cur_class = allFeatures_ranked->at(i).classId;
        Eigen::Vector3f keypoint = allFeatures_ranked->at(i).getVector3fMap();

         // init with uniform weights // TODO VS: codeword weight is never anything else than 1.0f - remove??
        std::shared_ptr<Codeword> codeword(new Codeword(clusterCenters[i], clusters[i].size(), 1.0f, keypoint, cur_class));
        codewords.push_back(codeword);
    }

    LOG_INFO("activating codewords");
    m_flann_helper = std::make_shared<FlannHelper>(codewords.at(0)->getData().size(), codewords.size());
    m_flann_helper->createDataset(codewords);
    m_flann_helper->buildIndex(m_distance->getType(), m_num_kd_trees);

    // create index depending on distance type
    if(m_distance->getType() == "Euclidean")
    {
        m_codebook->activate(codewords, features_ranked, boundingBoxes, m_distance, *m_flann_helper->getIndexL2(), m_flann_exact_match);
    }
    else if(m_distance->getType() == "ChiSquared")
    {
        m_codebook->activate(codewords, features_ranked, boundingBoxes, m_distance, *m_flann_helper->getIndexChi(), m_flann_exact_match);
    }

    if(m_enable_signals)
    {
        m_signalCodebook(*m_codebook);
    }

    // cpu time (%t) sums up the time used by all threads, so use wall time (%w) instead to show
    // performance increase in multithreading
    LOG_INFO("training processing time: " << timer_all.format(4, "%w") << " seconds");
}


bool ImplicitShapeModel::add_normals(const std::string& filename, const std::string& folder)
{
    pcl::PointCloud<PointNormalT>::Ptr points_in = loadPointCloud(filename);

    if (points_in.get() == 0)
        return false;

    if (points_in->empty())
    {
        LOG_WARN("point cloud is empty");
    }

    // filter out nan points
    std::vector<int> dummy;
    pcl::PointCloud<PointNormalT>::Ptr points(new pcl::PointCloud<PointNormalT>());
    pcl::removeNaNFromPointCloud(*points_in, *points, dummy);
    points->is_dense = false;

    pcl::search::Search<PointT>::Ptr searchTree;
    if (points->isOrganized())
    {
        searchTree = pcl::search::OrganizedNeighbor<PointT>::Ptr(new pcl::search::OrganizedNeighbor<PointT>());
    }
    else
    {
        searchTree = pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>());
    }

    pcl::PointCloud<PointT>::Ptr pointCloud(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr eigenValues;
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    pcl::PointCloud<PointT>::Ptr pointsWithoutNaN;
    pcl::PointCloud<PointT>::Ptr eigenValuesWithoutNan;
    pcl::PointCloud<pcl::Normal>::Ptr normalsWithoutNaN;

    // extract position point cloud
    pcl::copyPointCloud(*points, *pointCloud);

    LOG_INFO("computing normals");
    computeNormals(pointCloud, eigenValues, normals, searchTree);

    LOG_ASSERT(normals->size() == pointCloud->size());

    // filter normals with NaN and corresponding points
    filterNormals(pointCloud, eigenValues, normals, pointsWithoutNaN, eigenValuesWithoutNan, normalsWithoutNaN);

    LOG_ASSERT(pointsWithoutNaN.get() != 0);
    LOG_ASSERT(normalsWithoutNaN.get() != 0);
    LOG_ASSERT(pointsWithoutNaN->size() == normalsWithoutNaN->size());

    pcl::PointCloud<PointNormalT>::Ptr points_out(new pcl::PointCloud<PointNormalT>());
    pcl::concatenateFields(*pointsWithoutNaN, *normalsWithoutNaN, *points_out);

    // save result
    std::string file = filename.substr(filename.find_last_of('/')+1);
    //pcl::io::savePCDFileBinary(folder+"/"+file, *points_out);
    pcl::io::savePCDFileBinaryCompressed(folder+"/"+file, *points_out);

    return true;
}

bool ImplicitShapeModel::detect(const std::string& filename, std::vector<VotingMaximum>& maxima, std::map<std::string, double> &times)
{
    pcl::PointCloud<PointNormalT>::Ptr points = loadPointCloud(filename);

    if(m_set_color_to_zero)
    {
        LOG_INFO("Setting color to 0 in loaded model");
        for(int i = 0; i < points->size(); i++)
        {
            points->at(i).r = 0;
            points->at(i).g = 0;
            points->at(i).b = 0;
        }
    }

    if (points.get() == 0)
        return false;

    std::tie(maxima, times) = detect(points, true); // NOTE: true is assumed because of point type, needs to be checked later
    return true;
}

std::tuple<std::vector<VotingMaximum>,std::map<std::string, double>>
ImplicitShapeModel::detect(pcl::PointCloud<PointT>::ConstPtr points)
{
    pcl::PointCloud<PointNormalT>::Ptr pointCloud(new pcl::PointCloud<PointNormalT>());
    pcl::copyPointCloud(*points, *pointCloud);
    return detect(pointCloud, false);
}

std::tuple<std::vector<VotingMaximum>,std::map<std::string, double>>
ImplicitShapeModel::detect(pcl::PointCloud<PointNormalT>::ConstPtr points_in, bool hasNormals)
{
        /* 1.) Detect Keypoints + Keypoint-Features
         * 2.) Activate Codebook with Keypoints
         * 3.) Each activated Codeword casts its Activation-Distribution into Houghspace
         * 4.) Search Houghspace for maxima
         * 5.) The model can be considered found according to some heuristics in Houghspace
         */

    if (points_in->empty())
    {
        LOG_WARN("point cloud is empty");
        return std::make_tuple(std::vector<VotingMaximum>(), m_processing_times);
    }
    if(m_single_object_mode)
    {
        // to avoid errors if using old config files
        throw RuntimeException("The parameter for \"single object mode\" must be set inside the \"Voting\" section of the config file. You are using the \"Parameters\" section.");
    }

    // measure the time
    boost::timer::cpu_timer timer;
    boost::timer::cpu_timer timer_features;

    // filter out nan points
    std::vector<int> unused;
    pcl::PointCloud<PointNormalT>::Ptr points(new pcl::PointCloud<PointNormalT>());
    pcl::removeNaNFromPointCloud(*points_in, *points, unused);
    points->is_dense = false;

    // check first normal
    if (hasNormals)
    {
        const PointNormalT& firstNormal = points->at(0);
        if (firstNormal.normal_x == 0 &&
                firstNormal.normal_y == 0 &&
                firstNormal.normal_z == 0 ||
                std::isnan(firstNormal.normal_x) ||
                std::isnan(firstNormal.curvature)
                )
            hasNormals = false;
    }

    // compute features
    pcl::PointCloud<ISMFeature>::ConstPtr features;
    pcl::PointCloud<PointT>::ConstPtr pointsWithoutNaN;
    pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN;
    boost::timer::cpu_timer timer_normals;
    timer_normals.stop();
    boost::timer::cpu_timer timer_keypoints;
    timer_keypoints.stop();
    std::tie(features, std::ignore, pointsWithoutNaN, normalsWithoutNaN) = computeFeatures(points, hasNormals, timer_normals, timer_keypoints, false);
    m_processing_times["normals"] += getElapsedTime(timer_normals, "milliseconds");
    m_processing_times["keypoints"] += getElapsedTime(timer_keypoints, "milliseconds");
    m_processing_times["features"] -= getElapsedTime(timer_normals, "milliseconds");
    m_processing_times["features"] -= getElapsedTime(timer_keypoints, "milliseconds");

    // check for NAN features
    pcl::PointCloud<ISMFeature>::Ptr features_cleaned = removeNaNFeatures(features);
    m_processing_times["features"] += getElapsedTime(timer_features, "milliseconds");

    if(m_enable_signals)
    {
        m_signalFeatures(features_cleaned);
    }

    boost::timer::cpu_timer timer_flann;
    if(!m_index_created)
    {
        LOG_INFO("creating flann index");
        std::vector<std::shared_ptr<Codeword>> codewords = m_codebook->getCodewords();
        m_flann_helper = std::make_shared<FlannHelper>(m_codebook->getDim(), m_codebook->getSize());
        m_flann_helper->createDataset(codewords);
        m_flann_helper->buildIndex(m_distance->getType(), m_num_kd_trees);
        m_index_created = true;
        m_voting->setDistanceType(m_distance->getType());
    }
    m_processing_times["flann"] += getElapsedTime(timer_flann, "milliseconds");

    // activate codebook with current keypoints and cast votes
    boost::timer::cpu_timer timer_voting;
    LOG_INFO("activating codewords and casting votes");
    m_voting->clear();

    // use index depending on distance type
    if(m_distance->getType() == "Euclidean")
    {
        m_codebook->castVotes(features_cleaned, m_distance, *m_voting, *m_flann_helper->getIndexL2(), m_flann_exact_match);
    }
    else if(m_distance->getType() == "ChiSquared")
    {
        m_codebook->castVotes(features_cleaned, m_distance, *m_voting, *m_flann_helper->getIndexChi(), m_flann_exact_match);
    }

    m_processing_times["voting"] += getElapsedTime(timer_voting, "milliseconds");

    // analyze voting spaces - only for debug
    std::map<unsigned, pcl::PointCloud<PointT>::Ptr > all_votings;
    if(m_enable_voting_analysis)
    {
        all_votings = analyzeVotingSpacesForDebug(m_voting->getVotes(), points);
    }

    LOG_INFO("finding maxima");
    boost::timer::cpu_timer timer_maxima;
    std::vector<VotingMaximum> positions = m_voting->findMaxima(pointsWithoutNaN, normalsWithoutNaN);
    LOG_INFO("detected " << positions.size() << " maxima");
    m_processing_times["maxima"] += getElapsedTime(timer_maxima, "milliseconds");

    // only debug
    if(m_enable_voting_analysis)
    {
        addMaximaForDebug(all_votings, positions);
    }

    if(m_enable_signals)
    {
        m_signalMaxima(positions);
    }

    // cpu time (%t) sums up the time used by all threads, so use wall time (%w) instead to show
    // performance increase in multithreading
    LOG_INFO("detection processing time: " << timer.format(4, "%w") << " seconds");

    // measure time
    m_processing_times["complete"] += getElapsedTime(timer, "milliseconds");

    return std::make_tuple(positions, m_processing_times);
}

// TODO VS move this method to utils
double ImplicitShapeModel::getElapsedTime(boost::timer::cpu_timer timer, std::string format)
{
    // measure time
    auto nano = boost::chrono::nanoseconds(timer.elapsed().wall);
    if(format == "seconds")
    {
        auto msec = boost::chrono::duration_cast<boost::chrono::seconds>(nano);
        return msec.count();
    }
    else //if (format == "milliseconds")
    {
        auto msec = boost::chrono::duration_cast<boost::chrono::milliseconds>(nano);
        return msec.count();
    }
}



std::tuple<pcl::PointCloud<ISMFeature>::ConstPtr, pcl::PointCloud<ISMFeature>::ConstPtr,
pcl::PointCloud<PointT>::ConstPtr, pcl::PointCloud<pcl::Normal>::ConstPtr >
ImplicitShapeModel::computeFeatures(pcl::PointCloud<PointNormalT>::ConstPtr points,
                                    bool hasNormals, boost::timer::cpu_timer& timer_normals, boost::timer::cpu_timer& timer_keypoints,
                                    bool is_training)
{
    if(m_use_statistical_outlier_removal)
    {
        // filter cloud to remove outliers
        LOG_INFO("performing statistical outlier removal");
        pcl::PointCloud<PointNormalT>::Ptr filtered(new pcl::PointCloud<PointNormalT>());
        filtered->is_dense = false;
        m_stat_outlier_rem.setInputCloud(points);
        m_stat_outlier_rem.filter(*filtered);
        points = filtered;
    }
    if(m_use_radius_outlier_removal)
    {
        // filter cloud to remove outliers
        LOG_INFO("performing radius outlier removal");
        pcl::PointCloud<PointNormalT>::Ptr filtered(new pcl::PointCloud<PointNormalT>());
        filtered->is_dense = false;
        m_radius_outlier_rem.setInputCloud(points);
        m_radius_outlier_rem.filter(*filtered);
        points = filtered;
    }
    if(m_use_smoothing)
    {
        // smooth cloud to remove noise in normal's orientation
        LOG_INFO("performing MLS smoothing");
        pcl::PointCloud<pcl::PointNormal>::Ptr output(new pcl::PointCloud<pcl::PointNormal>());
        // copy data for a compatible point type
        pcl::PointCloud<pcl::PointXYZ>::Ptr input(new pcl::PointCloud<pcl::PointXYZ>());
        for(int i = 0; i < points->size(); i++)
        {
            const PointNormalT &p = points->at(i);
            pcl::PointXYZ point = pcl::PointXYZ(p.x, p.y, p.z);
            input->push_back(point);
        }
        input->width = points->width;
        input->height = points->height;
        input->is_dense = points->is_dense;

        m_mls_smoothing.setInputCloud(input);
        m_mls_smoothing.process(*output);
        // copy resulting geometric data
        pcl::PointCloud<PointNormalT>::Ptr filtered(new pcl::PointCloud<PointNormalT>());
        unsigned common_size = points->size();
        // size should stay the same, but often it does not
        if(output->size() < points->size())
        {
            common_size = output->size();
        }
        for(int i = 0; i < common_size; i++)
        {
            const pcl::PointNormal &fp = output->at(i);
            PointNormalT point = points->at(i);
            point.x = fp.x;
            point.y = fp.y;
            point.z = fp.z;
            filtered->push_back(point);
        }
        filtered->width = points->width;
        filtered->height = points->height;
        filtered->is_dense = false;
        points = filtered;
    }
    if(m_use_voxel_filtering)
    {
        // filter cloud to get a uniform point distribution
        LOG_INFO("performing voxel filtering");
        pcl::PointCloud<PointNormalT>::Ptr filtered(new pcl::PointCloud<PointNormalT>());
        filtered->is_dense = false;
        m_voxel_filtering.setInputCloud(points);
        m_voxel_filtering.filter(*filtered);
        points = filtered;
    }

    pcl::search::Search<PointT>::Ptr searchTree;
    if (points->isOrganized())
    {
        searchTree = pcl::search::OrganizedNeighbor<PointT>::Ptr(new pcl::search::OrganizedNeighbor<PointT>());
    }
    else
    {
        searchTree = pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>());
    }

    pcl::PointCloud<PointT>::Ptr pointCloud(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr eigenValues(new pcl::PointCloud<PointT>());
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    pcl::PointCloud<PointT>::Ptr pointsWithoutNaN;
    pcl::PointCloud<PointT>::Ptr eigenValuesWithoutNan;
    pcl::PointCloud<pcl::Normal>::Ptr normalsWithoutNaN;

    // extract position point cloud
    pcl::copyPointCloud(*points, *pointCloud);

    if(m_enable_signals)
    {
        m_signalPointCloud(pointCloud);
    }

    // skip normals on certain descriptors
    std::string descr_type = m_feature_descriptor->getType();

    // TODO VS: add descriptor type that can be checked for "needNormals" and "isBinary"
    if (!hasNormals && (descr_type != "SHORT_SHOT" && descr_type != "SHORT_CSHOT" &&
                        descr_type != "SHORT_SHOT_PCL"))
    {
        // compute normals on the cloud
        timer_normals.start();
        LOG_INFO("computing normals");
        computeNormals(pointCloud, eigenValues, normals, searchTree);
        timer_normals.stop();
    }
    else
    {
        // extract normals point cloud
        LOG_INFO("extracting normals");
        normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());
        pcl::copyPointCloud(*points, *normals);
    }

    LOG_ASSERT(normals->size() == pointCloud->size());

    // filter normals with NaN and corresponding points
    filterNormals(pointCloud, eigenValues, normals, pointsWithoutNaN, eigenValuesWithoutNan, normalsWithoutNaN);

    LOG_ASSERT(pointsWithoutNaN.get() != 0);
    LOG_ASSERT(normalsWithoutNaN.get() != 0);
    LOG_ASSERT(eigenValuesWithoutNan.get() != 0);
    LOG_ASSERT(pointsWithoutNaN->size() == normalsWithoutNaN->size());

    if(m_enable_signals)
    {
        m_signalNormals(pointsWithoutNaN, normalsWithoutNaN);
    }

    // detect interesting keypoints
    LOG_INFO("computing keypoints");
    timer_keypoints.start();
    m_keypoints_detector->setNumThreads(m_num_threads);
    if(is_training)
    {
        m_keypoints_detector->setIsTraining();
    }
    pcl::PointCloud<PointT>::ConstPtr keypoints = (*m_keypoints_detector)(pointCloud, eigenValues, normals,
                                                                          pointsWithoutNaN, eigenValuesWithoutNan,
                                                                          normalsWithoutNaN, searchTree);
    timer_keypoints.stop();

    // compute descriptors for keypoints
    LOG_INFO("computing features");
    m_feature_descriptor->setNumThreads(m_num_threads);
    pcl::PointCloud<ISMFeature>::ConstPtr features = (*m_feature_descriptor)(pointCloud, normals,
                                                                            pointsWithoutNaN, normalsWithoutNaN,
                                                                            keypoints,
                                                                            searchTree);

    // reference frames can be invalid, in which case associated keypoints are discarded
    LOG_ASSERT(features->size() <= keypoints->size());

    // in training: always compute global features
    if(is_training)
    {
        // compute global descriptors for objects
        LOG_INFO("computing global features");
        pcl::PointCloud<PointT>::ConstPtr dummy_keypoints(new pcl::PointCloud<PointT>());
        m_global_feature_descriptor->setNumThreads(m_num_threads);
        pcl::PointCloud<ISMFeature>::ConstPtr global_features = (*m_global_feature_descriptor)(pointCloud, normals,
                                                                                             pointsWithoutNaN, normalsWithoutNaN,
                                                                                             dummy_keypoints,
                                                                                             searchTree);
        return std::make_tuple(features, global_features, pointsWithoutNaN, normalsWithoutNaN);
    }
    else // for recognition/testing global features need to be computed later
    {
        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr global_features(new pcl::PointCloud<ISMFeature>());
        return std::make_tuple(features, global_features, pointsWithoutNaN, normalsWithoutNaN);
    }
}

const Codebook* ImplicitShapeModel::getCodebook() const
{
    return m_codebook;
}

const Voting* ImplicitShapeModel::getVoting() const
{
    return m_voting;
}


void ImplicitShapeModel::computeNormals(pcl::PointCloud<PointT>::ConstPtr points,
                                        pcl::PointCloud<PointT>::Ptr &eigen_values,
                                        pcl::PointCloud<pcl::Normal>::Ptr &normals,
                                        pcl::search::Search<PointT>::Ptr &searchTree) const
{
    LOG_ASSERT(normals.get() == 0);
    normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());

    if (points->isOrganized())
    {
        // TODO VS: check if detection datasets are organized in training and testing !!! if yes: also test with other normals

        pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> normalEst;
        normalEst.setInputCloud(points);
        // TODO VS: test this
//        normalEst.setNormalEstimationMethod(normalEst.COVARIANCE_MATRIX);
        normalEst.setNormalEstimationMethod(normalEst.AVERAGE_3D_GRADIENT);
        normalEst.setMaxDepthChangeFactor(0.02f);
        normalEst.setNormalSmoothingSize(10.0f);
        // TODO VS: test this
//        normalEst.setBorderPolicy(normalEst.BORDER_POLICY_MIRROR);

        // flip normals toward scene viewpoint
        LOG_INFO("orienting normals toward viewpoint");
        normalEst.useSensorOriginAsViewPoint();
        normalEst.compute(*normals);
        // TODO VS check if normals are flipped to viewpoint
    }
    else
    {
        LOG_INFO("computing consistent normal orientation (using method " << m_consistent_normals_method <<")");

        // prepare PCL normal estimation object
        pcl::NormalEstimationOMPWithEigVals<PointT, pcl::Normal> normalEst;
        normalEst.setInputCloud(points);
        normalEst.setSearchMethod(searchTree);
        normalEst.setRadiusSearch(m_normal_radius);
        normalEst.setNumberOfThreads(m_num_threads);

        NormalOrientation orient(m_consistent_normals_k, m_normal_radius);

        if(m_consistent_normals_method == 0)
        {
            // orient consistently towards the view point
            normalEst.setViewPoint(0,0,0);
            normalEst.compute(*normals);
        }
        else if(m_consistent_normals_method == 1)
        {
            // move model to origin, then point normals away from origin
            pcl::PointCloud<PointT>::Ptr model_no_centroid(new pcl::PointCloud<PointT>());
            pcl::copyPointCloud(*points, *model_no_centroid);

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
        else if(m_consistent_normals_method == 2)
        {
            normalEst.compute(*normals); // this is only needed for curvature at each point
            orient.processSHOTLRF(points, normals, normals, searchTree);
        }
#ifdef USE_VCGLIB
        else if(m_consistentNormalsMethod == 3)
        {
            orient.computeUsingEMST(model, normals);
        }
#endif
        else
        {
            LOG_WARN("Invalid consistent normals method: " << m_consistent_normals_method << "! Skipping consistent normals.");
        }

        // get eigenvalues if proper feature type is selected
        if(normalEst.feature_name_ == "NormalEstimationOMPWithEigVals")
        {
            eigen_values = normalEst.eigen_values_;
        }
    }
}


void ImplicitShapeModel::filterNormals(pcl::PointCloud<PointT>::ConstPtr points,
                                       pcl::PointCloud<PointT>::ConstPtr eigenValues,
                                       pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                       pcl::PointCloud<PointT>::Ptr &pointsWithoutNaN,
                                       pcl::PointCloud<PointT>::Ptr &eigenValuesWithoutNan,
                                       pcl::PointCloud<pcl::Normal>::Ptr &normalsWithoutNaN)
{
    LOG_ASSERT(pointsWithoutNaN.get() == 0);
    LOG_ASSERT(eigenValuesWithoutNan.get() == 0);
    LOG_ASSERT(normalsWithoutNaN.get() == 0);

    // filter NaN normals
    std::vector<int> mapping;
    normalsWithoutNaN = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());
    pcl::removeNaNNormalsFromPointCloud(*normals, *normalsWithoutNaN, mapping);

    // create new point cloud without NaN normals
    pointsWithoutNaN = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
    // ... and without corresponding eigenvalues
    eigenValuesWithoutNan = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
    for (int i = 0; i < (int)mapping.size(); i++)
    {
        pointsWithoutNaN->push_back(points->at(mapping[i]));
        if(eigenValues->size() == normals->size())
        {
            eigenValuesWithoutNan->push_back(eigenValues->at(mapping[i]));
        }
    }
}

Json::Value ImplicitShapeModel::iChildConfigsToJson() const
{
    Json::Value children(Json::objectValue);

    children["Codebook"] = m_codebook->configToJson();
    children["Keypoints"] = m_keypoints_detector->configToJson();
    children["Features"] = m_feature_descriptor->configToJson();
    children["GlobalFeatures"] = m_global_feature_descriptor->configToJson();
    children["Clustering"] = m_clustering->configToJson();
    children["Voting"] = m_voting->configToJson();
    children["FeatureWeighting"] = m_feature_ranking->configToJson(); // TODO VS: rename config entry to FeatureRanking in future

    return children;
}

bool ImplicitShapeModel::iChildConfigsFromJson(const Json::Value& object)
{
    const Json::Value *codebook = &(object["Codebook"]);
    const Json::Value *keypoints = &(object["Keypoints"]);
    const Json::Value *features = &(object["Features"]);
    const Json::Value *global_features = &(object["GlobalFeatures"]);
    const Json::Value *clustering = &(object["Clustering"]);
    const Json::Value *voting = &(object["Voting"]);
    const Json::Value *feature_ranking = &(object["FeatureWeighting"]);

    if (codebook->isNull() || !codebook->isObject() ||
            keypoints->isNull() || !keypoints->isObject() ||
            features->isNull() || !features->isObject() ||
            clustering->isNull() || !clustering->isObject() ||
            voting->isNull() || !voting->isObject() ||
            feature_ranking->isNull() || !feature_ranking->isObject()) {
        LOG_ERROR("could not find necessary json entries");
        return false;
    }

    // handle global features separately to be able to use old configs and trained files
    bool use_dummy = false;
    Json::Value dummy(Json::objectValue);
    dummy["Type"] = Json::Value("Dummy");

    if (global_features->isNull() || !global_features->isObject())
    {
        use_dummy = true;
        LOG_WARN("No global feature descriptor available, using Dummy global feature!");
    }

    // clear
    delete m_codebook;
    delete m_keypoints_detector;
    delete m_feature_descriptor;
    delete m_global_feature_descriptor;
    delete m_clustering;
    delete m_voting;
    delete m_feature_ranking;

    // create new child objects
    m_codebook = Factory<Codebook>::create(*codebook);
    m_keypoints_detector = Factory<Keypoints>::create(*keypoints);
    m_feature_descriptor = Factory<Features>::create(*features);
    m_global_feature_descriptor = use_dummy ? Factory<Features>::create(dummy) : Factory<Features>::create(*global_features);
    m_clustering = Factory<Clustering>::create(*clustering);
    m_voting = Factory<Voting>::create(*voting);
    m_feature_ranking = Factory<FeatureRanking>::create(*feature_ranking);

    if (!m_codebook || !m_keypoints_detector || !m_feature_descriptor || !m_global_feature_descriptor ||
            !m_clustering || !m_voting || !m_feature_ranking)
        throw RuntimeException("Could not create object(s). The configuration file might be corrupted.");

    // set global descriptor for the detection step
    m_voting->setGlobalFeatureDescriptor(m_global_feature_descriptor);

    return true;
}

void ImplicitShapeModel::iSaveData(boost::archive::binary_oarchive &oa) const
{
    unsigned size = m_instance_to_class_map.size();
    oa << size;
    for(auto const &it : m_instance_to_class_map)
    {
        unsigned label_inst = it.first;
        unsigned label_class = it.second;
        oa << label_inst;
        oa << label_class;
    }

    m_codebook->saveData(oa);
    m_keypoints_detector->saveData(oa);
    m_feature_descriptor->saveData(oa);
    m_global_feature_descriptor->saveData(oa);
    m_clustering->saveData(oa);
    m_voting->saveData(oa);
    m_feature_ranking->saveData(oa);

    // store label maps
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
}

bool ImplicitShapeModel::iLoadData(boost::archive::binary_iarchive &ia)
{
    // objects have to be initialized already
    if (!m_codebook || !m_keypoints_detector || !m_feature_descriptor || !m_global_feature_descriptor ||
            !m_clustering || !m_voting || !m_feature_ranking) {
        LOG_ERROR("object is not initialized");
        return false;
    }

    // this is necessary since objects are created before config is read in json_object.cpp
    // forward svm path as it might have been manually changed in config
    m_voting->setSVMPath(m_svm_path);

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

    // init data for objects
    if (!m_codebook->loadData(ia) ||
            !m_keypoints_detector->loadData(ia) ||
            !m_feature_descriptor->loadData(ia) ||
            !m_global_feature_descriptor->loadData(ia) ||
            !m_clustering->loadData(ia) ||
            !m_voting->loadData(ia) ||
            !m_feature_ranking->loadData(ia))
    {
        LOG_ERROR("could not load child objects");
        return false;
    }

    // load original labels
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
    return true;
}


void ImplicitShapeModel::iPostInitConfig()
{
    // init distance
    delete m_distance;
    if (m_distanceType == DistanceEuclidean::getTypeStatic())
        m_distance = new DistanceEuclidean;
    else if (m_distanceType == DistanceChiSquared::getTypeStatic())
        m_distance = new DistanceChiSquared;
    else
        throw RuntimeException("invalid distance type: " + m_distanceType);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    m_mls_smoothing.setSearchMethod(tree);
    m_mls_smoothing.setComputeNormals(false);
    m_mls_smoothing.setPolynomialOrder(m_polynomial_order);
    m_mls_smoothing.setSearchRadius(m_smoothing_radius);
    m_stat_outlier_rem.setMeanK(m_som_mean_k);
    m_stat_outlier_rem.setStddevMulThresh(m_som_std_dev_mul);
    m_radius_outlier_rem.setRadiusSearch(m_ror_radius);
    m_radius_outlier_rem.setMinNeighborsInRadius(m_ror_min_neighbors);
    m_voxel_filtering.setLeafSize(m_voxel_leaf_size, m_voxel_leaf_size, m_voxel_leaf_size);

    // m_numThreads == 0 is the default, so don't change anything
    if (m_num_threads > 0)
        omp_set_num_threads(m_num_threads);

    LOG_INFO("OpenMP is using " << omp_get_max_threads() << " threads");

    if(m_flann_exact_match && m_num_kd_trees > 1)
    {
        LOG_WARN("It does not make sense to use more than 1 kd-tree with exact nearest neighbor!");
        m_num_kd_trees = 1;
    }
}


pcl::PointCloud<ISMFeature>::Ptr ImplicitShapeModel::removeNaNFeatures(pcl::PointCloud<ISMFeature>::ConstPtr modelFeatures)
{
    // check for NAN features
    pcl::PointCloud<ISMFeature>::Ptr modelFeatures_cleaned (new pcl::PointCloud<ISMFeature>());
    modelFeatures_cleaned->header = modelFeatures->header;
    modelFeatures_cleaned->height = 1;
    modelFeatures_cleaned->is_dense = false;
    bool nan_found = false;
    int nan_features = 0;
    for(int a = 0; a < modelFeatures->size(); a++)
    {
        ISMFeature fff = modelFeatures->at(a);
        for(int b = 0; b < fff.descriptor.size(); b++)
        {
            if(std::isnan(fff.descriptor.at(b)))
            {
                nan_features++;
                nan_found = true;
                break;
            }
        }
        if(!nan_found)
        {
            modelFeatures_cleaned->push_back(fff);
        }
        nan_found = false;
    }
    modelFeatures_cleaned->width = modelFeatures_cleaned->size();
    if(nan_features > 0)
        LOG_WARN("Found " << nan_features << " NaN features!");

    return modelFeatures_cleaned;
}

void ImplicitShapeModel::trainSVM(std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &globalFeatures)
{
    // insert training data and labels
    std::vector< std::vector<float> > training_data;
    std::vector<int> labels;
    for(auto it : globalFeatures)
    {
        for(auto feat_cloud : it.second)
        {
            for(ISMFeature feat : feat_cloud->points)
            {
                labels.push_back(it.first);
                training_data.push_back(feat.descriptor);
            }
        }
    }

    LOG_INFO("Training SVM with " << training_data.size() << " features.");

    // train the SVM
    CustomSVM svm(m_output_file_name);
    svm.setData(training_data, labels);

    // find good params or use default
    if(m_svm_auto_train)
    {
        if(m_svm_1_vs_all_train) // manual 1 vs all training
        {
            svm.trainAutomatically(m_svm_param_gamma, m_svm_param_c, m_svm_param_k_fold, true);
        }
        else // standard OpenCV pairwise 1 vs 1
        {
            svm.trainAutomatically(m_svm_param_gamma, m_svm_param_c, m_svm_param_k_fold, false);
        }
    }
    else
    {
        if(m_svm_1_vs_all_train) // manual 1 vs all training
        {
            svm.trainSimple(m_svm_param_gamma, m_svm_param_c, true);
        }
        else // standard OpenCV pairwise 1 vs 1
        {
            svm.trainSimple(m_svm_param_gamma, m_svm_param_c, false);
        }
    }
}


// TODO VS: move to debug_utils.cpp
void ImplicitShapeModel::writeFeaturesToDisk(std::string file_name,
                                             const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> &features,
                                             const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> &globalFeatures,
                                             const std::map<unsigned, std::vector<Utils::BoundingBox>> &boundingBoxes)
{
    std::ofstream ofs(file_name, std::ios::binary | std::ios::trunc);
    boost::archive::binary_oarchive oa(ofs);

    // save local features
    int mapsize = features.size();
    oa << mapsize;
    for(auto it : features)
    {
        int class_id = it.first;
        oa << class_id;

        int num_pcs = it.second.size();
        oa << num_pcs; // num of point clouds in std::vector

        for(auto pc : it.second)
        {
            int pc_size = pc->size();
            oa << pc_size;
            for(auto f : pc->points)
            {
//                f.save(oa);
                // save reference frame
                for(unsigned i = 0; i < 9; i++)
                {
                    oa << f.referenceFrame.rf[i];
                }
                int dsize = f.descriptor.size();
                oa << dsize;
                for(float ff : f.descriptor)
                {
                    oa << ff;
                }

                oa << f.centerDist;
                oa << f.globalDescriptorRadius;
            }
        }
    }

    // save global features
    mapsize = globalFeatures.size();
    oa << mapsize;
    for(auto it : globalFeatures)
    {
        int class_id = it.first;
        oa << class_id;

        int num_pcs = it.second.size();
        oa << num_pcs; // num of point clouds in std::vector

        for(auto pc : it.second)
        {
            int pc_size = pc->size();
            oa << pc_size;
            for(auto f : pc->points)
            {
//                f.save(oa);
                // save reference frame
                for(unsigned i = 0; i < 9; i++)
                {
                    oa << f.referenceFrame.rf[i];
                }
                int dsize = f.descriptor.size();
                oa << dsize;
                for(float ff : f.descriptor)
                {
                    oa << ff;
                }
                oa << f.centerDist;
                oa << f.globalDescriptorRadius;
            }
        }
    }

    // save bounding boxes
    mapsize = boundingBoxes.size();
    oa << mapsize;
    for(auto it : boundingBoxes)
    {
        int class_id = it.first;
        oa << class_id;

        int num_bb = it.second.size();
        oa << num_bb;
        for(const auto &bb : it.second)
        {
            Eigen::Vector3f vec = bb.position;
            oa << vec.x(); oa << vec.y(); oa << vec.z();
            vec = bb.size;
            oa << vec.x(); oa << vec.y(); oa << vec.z();

            float quat1 = bb.rotQuat.R_component_1();
            float quat2 = bb.rotQuat.R_component_2();
            float quat3 = bb.rotQuat.R_component_3();
            float quat4 = bb.rotQuat.R_component_4();
            oa << quat1;
            oa << quat2;
            oa << quat3;
            oa << quat4;
        }
    }


    ofs.close();
}


void ImplicitShapeModel::readFeaturesFromDisk(std::string file_name,
                          std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &features,
                          std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &globalFeatures,
                          std::map<unsigned, std::vector<Utils::BoundingBox> > &boundingBoxes)
{
    // read boost data object
    std::ifstream ifs(file_name, std::ios::binary);
    if(ifs)
    {
        boost::archive::binary_iarchive ia(ifs);

        // load local features
        features.clear();
        int map_size;
        ia >> map_size;
        for(int i = 0; i < map_size; i++)
        {
            int class_id;
            ia >> class_id;

            std::vector<pcl::PointCloud<ISMFeature>::Ptr> map_elem;

            int num_clouds;
            ia >> num_clouds;
            for(int j = 0; j < num_clouds; j++)
            {
                pcl::PointCloud<ISMFeature>::Ptr cloud(new pcl::PointCloud<ISMFeature>());

                int cloud_size;
                ia >> cloud_size;
                for(int k = 0; k < cloud_size; k++)
                {
                    pcl::ReferenceFrame referenceFrame;
                    for(int i_ref = 0; i_ref < 9; i_ref++)
                    {
                        float ref;
                        ia >> ref;
                        referenceFrame.rf[i_ref] = ref;
                    }

                    int dsize;
                    ia >> dsize;
                    std::vector<float> descriptor;
                    for(int l = 0; l < dsize; l++)
                    {
                        float fff;
                        ia >> fff;
                        descriptor.push_back(fff);
                    }

                    float centerDist;
                    ia >> centerDist;
                    float radius;
                    ia >> radius;

                    ISMFeature ismf;
//                    ismf.load(ia);
                    ismf.referenceFrame = referenceFrame;
                    ismf.descriptor = descriptor;
                    ismf.centerDist = centerDist;
                    ismf.globalDescriptorRadius = radius;
                    ismf.classId = class_id;
                    cloud->push_back(ismf);
                }

                cloud->height = 1;
                cloud->width = cloud->size();
                cloud->is_dense = false;
                map_elem.push_back(cloud);
            }
            features.insert({class_id, map_elem});
        }

        // load global features
        ia >> map_size;
        for(int i = 0; i < map_size; i++)
        {
            int class_id;
            ia >> class_id;

            std::vector<pcl::PointCloud<ISMFeature>::Ptr> map_elem;

            int num_clouds;
            ia >> num_clouds;
            for(int j = 0; j < num_clouds; j++)
            {
                pcl::PointCloud<ISMFeature>::Ptr cloud(new pcl::PointCloud<ISMFeature>());

                int cloud_size;
                ia >> cloud_size;
                for(int k = 0; k < cloud_size; k++)
                {
                    pcl::ReferenceFrame referenceFrame;
                    for(int i_ref = 0; i_ref < 9; i_ref++)
                    {
                        float ref;
                        ia >> ref;
                        referenceFrame.rf[i_ref] = ref;
                    }

                    int dsize;
                    ia >> dsize;
                    std::vector<float> descriptor;
                    for(int l = 0; l < dsize; l++)
                    {
                        float fff;
                        ia >> fff;
                        descriptor.push_back(fff);
                    }
                    float centerDist;
                    ia >> centerDist;
                    float radius;
                    ia >> radius;

                    ISMFeature ismf;
//                    ismf.load(ia);
                    ismf.referenceFrame = referenceFrame;
                    ismf.descriptor = descriptor;
                    ismf.centerDist = centerDist;
                    ismf.globalDescriptorRadius = radius;
                    ismf.classId = class_id;
                    cloud->push_back(ismf);
                }

                cloud->height = 1;
                cloud->width = cloud->size();
                cloud->is_dense = false;
                map_elem.push_back(cloud);
            }
            globalFeatures.insert({class_id, map_elem});
        }

        // load bounding boxes
        ia >> map_size;
        for(int i = 0; i < map_size; i++)
        {
            int class_id;
            ia >> class_id;
            int num_elems;
            ia >> num_elems;

            std::vector<Utils::BoundingBox> map_elem;

            for(int j = 0; j < num_elems; j++)
            {
                Utils::BoundingBox bb;
                ia >> bb.position.x();
                ia >> bb.position.y();
                ia >> bb.position.z();
                ia >> bb.size.x();
                ia >> bb.size.y();
                ia >> bb.size.z();
                float quat1, quat2, quat3, quat4;
                ia >> quat1;
                ia >> quat2;
                ia >> quat3;
                ia >> quat4;
                bb.rotQuat = boost::math::quaternion<float>(quat1, quat2, quat3, quat4);

                map_elem.push_back(bb);
            }

            boundingBoxes[class_id] = map_elem;
        }

        ifs.close();
    }
    else
    {
        LOG_ERROR("Error opening file: " << file_name);
        exit(1);
    }
}


// DEBUG

std::map<unsigned, pcl::PointCloud<PointT>::Ptr > ImplicitShapeModel::analyzeVotingSpacesForDebug
(const std::map<unsigned, std::vector<Vote> > &all_votes, pcl::PointCloud<PointNormalT>::Ptr points)
{
    std::map<unsigned, pcl::PointCloud<PointT>::Ptr > all_votings;
    for (auto it : all_votes)
    {
        unsigned classId = it.first;
        const std::vector<Vote>& votes = it.second; // all votes for this class

        pcl::PointCloud<PointT>::Ptr dataset(new pcl::PointCloud<PointT>());

        // include original points
        int skip_counter = 0;
        for(auto p : points->points)
        {
            // use only each 20th point
            if(skip_counter++ % 20 != 0) continue;

            PointT origp;
            origp.x = p.x;
            origp.y = p.y;
            origp.z = p.z;
            if(p.r == 0 && p.g == 0 && p.b == 0)
            {
                origp.r = 150;
                origp.g = 150;
                origp.b = 150;
            }
            else
            {
                origp.r = p.r;
                origp.g = p.g;
                origp.b = p.b;
            }
            // NOTE: uncomment to include points of object
            // dataset->push_back(origp);
        }

        // build dataset of votes
        for (int i = 0; i < (int)votes.size(); i++)
        {
            const Vote& vote = votes[i];
            PointT votePoint;
            votePoint.x = vote.position[0];
            votePoint.y = vote.position[1];
            votePoint.z = vote.position[2];
            votePoint.r = 0;
            votePoint.g = 255;
            votePoint.b = 255;
            // NOTE: uncomment to include the actual votes
            // dataset->push_back(votePoint);
        }
        dataset->height = 1;
        dataset->width = dataset->size();
        dataset->is_dense = false;
        all_votings.insert({classId, dataset});
    }

    return all_votings;
}


void ImplicitShapeModel::addMaximaForDebug(std::map<unsigned, pcl::PointCloud<PointT>::Ptr > &all_votings,
                                           std::vector<VotingMaximum> &positions)
{
    for(VotingMaximum max : positions)
    {
        int classId = max.classId;
        Eigen::Vector3f pos = max.position;
        PointT votePoint;
        votePoint.x = pos[0];
        votePoint.y = pos[1];
        votePoint.z = pos[2];
        votePoint.r = 255;
        votePoint.g = 0;
        votePoint.b = 0;
        // NOTE: uncomment to include maxima locations
        //all_votings.at(classId)->push_back(votePoint);
    }

    // check if path terminates in /
    unsigned pos = m_voting_analysis_output_path.find_last_of('/');
    if(pos != m_voting_analysis_output_path.size()-1)
    {
        m_voting_analysis_output_path = m_voting_analysis_output_path.append("/");
        std::string command = "mkdir "+m_voting_analysis_output_path+" -p";
        int unused = std::system(command.c_str());
    }

    m_counter++;
    std::ofstream ofs;
    ofs.open(m_voting_analysis_output_path+"votes.txt", std::ofstream::out | std::ofstream::app);
    for(auto obj : all_votings)
    {
        pcl::io::savePCDFileBinary(m_voting_analysis_output_path+"file_idx_"+std::to_string(m_counter)+"_class_"+
                                   std::to_string(obj.first)+".pcd", *obj.second);
        ofs << "file: " << m_counter << ", class: " << obj.first << ", votes:" << obj.second->size() << std::endl;
    }
    ofs.close();
}

}
