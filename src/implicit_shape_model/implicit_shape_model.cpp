/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2018, Viktor Seib, Norman Link
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
#include <pcl/features/integral_image_normal.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/centroid.h>

#include <fstream>
#include <chrono>

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
#include "utils/point_cloud_resizing.h"
#include "utils/normal_orientation.h"
#include "utils/factory.h"
#include "utils/exception.h"

#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/basicconfigurator.h>

// TODO VS temp for debug
int ism3d::ImplicitShapeModel::m_counter = 0;

namespace ism3d
{

ImplicitShapeModel::ImplicitShapeModel() : m_distance(0)
{
    // init logging
    log4cxx::LayoutPtr layout(new log4cxx::PatternLayout("[\%d{HH:mm:ss}] \%p: \%m\%n"));
    log4cxx::ConsoleAppender* consoleAppender = new log4cxx::ConsoleAppender(layout);
    log4cxx::BasicConfigurator::configure(log4cxx::AppenderPtr(consoleAppender));
    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());

    addParameter(m_distanceType, "DistanceType", std::string("Euclidean"));
    addParameter(m_useVoxelFiltering, "UseVoxelFiltering", false);
    addParameter(m_voxelLeafSize, "VoxelLeafSize", 0.01f);
    addParameter(m_normalRadius, "NormalRadius", 0.05f);
    addParameter(m_consistentNormalsK, "ConsistentNormalsK", 10);
    addParameter(m_consistentNormalsMethod, "ConsistentNormalsMethod", 2);
    addParameter(m_numThreads, "NumThreads", 0);
    addParameter(m_bbType, "BoundingBoxType", std::string("MVBB"));
    addParameter(m_setColorToZero, "SetColorToZero", false);
    addParameter(m_enableVotingAnalysis, "EnableVotingAnalysis", false);
    addParameter(m_votingAnalysisOutputPath, "VotingAnalysisOutputPath", std::string("/home/vseib/Desktop/"));
    addParameter(m_use_svm, "UseSvmTraining", false);
    addParameter(m_svm_auto_train, "SvmAutoTrain", false);
    addParameter(m_svm_1_vs_all_train, "SvmOneVsAllTraining", false);
    addParameter(m_svm_param_c, "SvmParamC", 7.41);
    addParameter(m_svm_param_gamma, "SvmParamGamma", 2.96);
    addParameter(m_svm_param_k_fold, "SvmParamKfold", 10);
    addParameter(m_single_object_mode, "SingleObjectMode", false);
    addParameter(m_num_kd_trees, "FLANNNumKDTrees", 4);
    addParameter(m_flann_exact_match, "FLANNExactMatch", false);

    init();
}

ImplicitShapeModel::~ImplicitShapeModel()
{
    log4cxx::Logger::getRootLogger()->removeAllAppenders();

    delete m_codebook;
    delete m_keypointsDetector;
    delete m_featureDescriptor;
    delete m_globalFeatureDescriptor;
    delete m_clustering;
    delete m_voting;
    delete m_distance;
    delete m_featureRanking;
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
    m_keypointsDetector = new KeypointsVoxelGrid();
    m_featureDescriptor = new FeaturesSHOT();
    m_globalFeatureDescriptor = new FeaturesGRSD();
    m_clustering = new ClusteringAgglomerative();
    m_voting = new VotingMeanShift();
    m_featureRanking = new RankingUniform();
}

void ImplicitShapeModel::clear()
{
    m_trainingModelsFilenames.clear();
    m_trainingModelHasNormals.clear();
    m_codebook->clear();
    m_clustering->clear();
    m_voting->clear();
}

bool ImplicitShapeModel::addTrainingModel(const std::string& filename, unsigned classId)
{
    LOG_INFO("adding training model with class id " << classId);

    // add model
    m_trainingModelsFilenames[classId].push_back(filename);
    m_trainingModelHasNormals[classId].push_back(true); // NOTE: optimistic assumption, needs to be checked later
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

    if (m_trainingModelsFilenames.size() == 0) {
        LOG_WARN("no training models found");
        return;
    }

    // measure the time
    boost::timer::cpu_timer timer;
    boost::timer::cpu_timer timer_all;

    // contains the whole list of features for each class id and for each model
    std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > features; // TODO VS rethink the whole feature data type!!!
    std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > globalFeatures;
    std::map<unsigned, std::vector<Utils::BoundingBox> > boundingBoxes;

    // compute features for all models and all classes
    for (auto it = m_trainingModelsFilenames.begin(); it != m_trainingModelsFilenames.end(); it++)
    {
        unsigned classId = it->first;

        LOG_ASSERT(m_trainingModelsFilenames.size() == m_trainingModelHasNormals.size());

        const std::vector<std::string>& model_filenames = it->second;
        const std::vector<bool>& modelsHaveNormals = m_trainingModelHasNormals[classId];

        LOG_ASSERT(model_filenames.size() == modelsHaveNormals.size());

        LOG_INFO("----------------------------------------------------------------");
        LOG_INFO("training class " << classId << " with " << model_filenames.size() << " objects");

        for (int j = 0; j < (int)model_filenames.size(); j++)
        {
            pcl::PointCloud<PointNormalT>::Ptr model = loadPointCloud(model_filenames[j]);

            if(m_setColorToZero)
            {
                LOG_INFO("Setting color to 0 in loaded model");
                for(int i = 0; i < model->size(); i++)
                {
                    model->at(i).r = 0;
                    model->at(i).g = 0;
                    model->at(i).b = 0;
                }
            }

            // compute bounding box
            Utils::BoundingBox boundingBox;
            if (m_bbType == "MVBB")
                boundingBox = Utils::computeMVBB<PointNormalT>(model);
            else if (m_bbType == "AABB")
                boundingBox = Utils::computeAABB<PointNormalT>(model);
            else
                throw BadParamExceptionType<std::string>("invalid bounding box type", m_bbType);

            if(m_enable_signals)
            {
                timer.stop();
                m_signalBoundingBox(boundingBox);
                timer.resume();
            }

            // check first normal
            bool hasNormals = modelsHaveNormals[j];
            if (hasNormals) {
                const PointNormalT& firstNormal = model->at(0);
                if (firstNormal.normal_x == 0 && firstNormal.normal_y == 0 && firstNormal.normal_z == 0 ||
                        pcl_isnan(firstNormal.normal_x) ||
                        pcl_isnan(firstNormal.normal_y) ||
                        pcl_isnan(firstNormal.normal_z) ||
                        pcl_isnan(firstNormal.curvature)
                        )
                    hasNormals = false;
            }

            // compute features
            pcl::PointCloud<ISMFeature>::ConstPtr model_features;
            pcl::PointCloud<ISMFeature>::ConstPtr global_features;
            std::tie(model_features, global_features, std::ignore, std::ignore) = computeFeatures(model, hasNormals, timer, timer, true);

            // check for NAN features
            pcl::PointCloud<ISMFeature>::Ptr modelFeatures_cleaned = removeNaNFeatures(model_features);
            pcl::PointCloud<ISMFeature>::Ptr globalFeatures_cleaned = removeNaNFeatures(global_features);

            if(m_enable_signals)
            {
                timer.stop();
                m_signalFeatures(modelFeatures_cleaned);
                timer.resume();
            }

            // concatenate features
            features[classId].push_back(modelFeatures_cleaned);
            globalFeatures[classId].push_back(globalFeatures_cleaned);
            boundingBoxes[classId].push_back(boundingBox);
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
    m_voting->determineAverageBoundingBoxDimensions(boundingBoxes);
    // forward global feature to voting class to store them
    m_voting->forwardGlobalFeatures(globalFeatures);

    LOG_INFO("computing feature ranking"); // TODO VS: avoid copying features for ranking
    // remove features with low scores
    std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > features_ranked;
    pcl::PointCloud<ISMFeature>::Ptr allFeatures_ranked(new pcl::PointCloud<ISMFeature>());
    std::vector<unsigned> allFeatureClasses_ranked;

    std::tie(features_ranked, allFeatures_ranked, allFeatureClasses_ranked) =
            (*m_featureRanking)(features, m_num_kd_trees, m_flann_exact_match);

    // cluster descriptors and extract cluster centers
    LOG_INFO("clustering");
    (*m_clustering)(allFeatures_ranked, m_distance);
    const std::vector<std::vector<float> > clusterCenters = m_clustering->getClusterCenters();
    const std::vector<int> clusterIndices = m_clustering->getClusterIndices();

    // compute which cluster indices are assigned which feature indices
    std::vector<std::vector<int> > clusters(clusterCenters.size()); // each position: list of feature indices of a cluster
    for (int i = 0; i < (int)allFeatures_ranked->size(); i++)
    {
        int clusterIndex = clusterIndices[i]; // this index indicates which cluster the feature i belongs to
        clusters[clusterIndex].push_back(i);
    }
    // NOTE: if no clustering is used, clusterIndices are just ascending numbers (0, 1, 2, 3, ...)
    // in that case, clusters at each position have size == 1
    LOG_ASSERT(clusterIndices.size() == allFeatures_ranked->size());


    // create codewords and add them to the codebook - NOTE: if no clustering is used: a codeword is just one feature and its center vector
    LOG_INFO("creating codewords");
    std::vector<std::shared_ptr<Codeword> > codewords;
    for (int i = 0; i < (int)clusterCenters.size(); i++)
    {
        std::shared_ptr<Codeword> codeword(new Codeword(clusterCenters[i], clusters[i].size(), 1.0f)); // init with uniform weights

        for (int j = 0; j < (int)clusters[i].size(); j++)
        {
            codeword->addFeature(allFeatures_ranked->at(clusters[i][j]).getVector3fMap(),
                                 allFeatureClasses_ranked[clusters[i][j]]);
        }
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
    else if(m_distance->getType() == "Hellinger")
    {
        m_codebook->activate(codewords, features_ranked, boundingBoxes, m_distance, *m_flann_helper->getIndexHel(), m_flann_exact_match);
    }
    else if(m_distance->getType() == "HistIntersection")
    {
        m_codebook->activate(codewords, features_ranked, boundingBoxes, m_distance, *m_flann_helper->getIndexHist(), m_flann_exact_match);
    }

    if(m_enable_signals)
    {
        timer.stop();
        m_signalCodebook(*m_codebook);
        timer.resume();
    }

    // cpu time (%t) sums up the time used by all threads, so use wall time (%w) instead to show
    // performance increase in multithreading
    LOG_INFO("training processing time: " << timer.format(4, "%w") << " seconds");
    LOG_INFO("total processing time: " << timer_all.format(4, "%w") << " seconds");
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
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    pcl::PointCloud<PointT>::Ptr pointsWithoutNaN;
    pcl::PointCloud<pcl::Normal>::Ptr normalsWithoutNaN;

    // extract position point cloud
    pcl::copyPointCloud(*points, *pointCloud);

    LOG_INFO("computing normals");
    computeNormals(pointCloud, normals, searchTree);

    LOG_ASSERT(normals->size() == pointCloud->size());

    // filter normals with NaN and corresponding points
    filterNormals(pointCloud, normals, pointsWithoutNaN, normalsWithoutNaN);

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

    if(m_setColorToZero)
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

    // measure the time
    boost::timer::cpu_timer timer;
    boost::timer::cpu_timer timer_features;

    // filter out nan points
    std::vector<int> dummy;
    pcl::PointCloud<PointNormalT>::Ptr points(new pcl::PointCloud<PointNormalT>());
    pcl::removeNaNFromPointCloud(*points_in, *points, dummy);
    points->is_dense = false;

    // check first normal
    if (hasNormals)
    {
        const PointNormalT& firstNormal = points->at(0);
        if (firstNormal.normal_x == 0 &&
                firstNormal.normal_y == 0 &&
                firstNormal.normal_z == 0 ||
                pcl_isnan(firstNormal.normal_x) ||
                pcl_isnan(firstNormal.curvature)
                )
            hasNormals = false;
    }

    // compute features
    pcl::PointCloud<ISMFeature>::ConstPtr features;
    pcl::PointCloud<ISMFeature>::ConstPtr globalFeatures;
    pcl::PointCloud<PointT>::ConstPtr pointsWithoutNaN;
    pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN;
    bool compute_global = m_single_object_mode;
    boost::timer::cpu_timer timer_normals;
    timer_normals.stop();
    boost::timer::cpu_timer timer_keypoints;
    timer_keypoints.stop();
    std::tie(features, globalFeatures, pointsWithoutNaN, normalsWithoutNaN) = computeFeatures(points, hasNormals, timer_normals, timer_keypoints,
                                                                                              compute_global);
    m_processing_times["normals"] += getElapsedTime(timer_normals, "milliseconds");
    m_processing_times["keypoints"] += getElapsedTime(timer_keypoints, "milliseconds");
    m_processing_times["features"] -= getElapsedTime(timer_normals, "milliseconds");
    m_processing_times["features"] -= getElapsedTime(timer_keypoints, "milliseconds");

    // check for NAN features
    pcl::PointCloud<ISMFeature>::Ptr features_cleaned = removeNaNFeatures(features);
    pcl::PointCloud<ISMFeature>::Ptr globalFeatures_cleaned = removeNaNFeatures(globalFeatures);
    m_processing_times["features"] += getElapsedTime(timer_features, "milliseconds");

    if(m_enable_signals)
    {
        timer.stop();
        m_signalFeatures(features_cleaned);
        timer.resume();
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
    LOG_INFO("activating codewords and casting votes");
    m_voting->clear();

    boost::timer::cpu_timer timer_voting;
    // use index depending on distance type
    if(m_distance->getType() == "Euclidean")
    {
        m_codebook->castVotes(features_cleaned, m_distance, *m_voting, *m_flann_helper->getIndexL2(), m_flann_exact_match);
    }
    else if(m_distance->getType() == "ChiSquared")
    {
        m_codebook->castVotes(features_cleaned, m_distance, *m_voting, *m_flann_helper->getIndexChi(), m_flann_exact_match);
    }
    // TODO VS: remove hellinger and histintersection everywhere
    else if(m_distance->getType() == "Hellinger")
    {
        m_codebook->castVotes(features_cleaned, m_distance, *m_voting, *m_flann_helper->getIndexHel(), m_flann_exact_match);
    }
    else if(m_distance->getType() == "HistIntersection")
    {
        m_codebook->castVotes(features_cleaned, m_distance, *m_voting, *m_flann_helper->getIndexHist(), m_flann_exact_match);
    }
    m_processing_times["voting"] += getElapsedTime(timer_voting, "milliseconds");

    // analyze voting spaces - only for debug
    std::map<unsigned, pcl::PointCloud<PointT>::Ptr > all_votings;
    if(m_enableVotingAnalysis)
    {
        all_votings = analyzeVotingSpacesForDebug(m_voting->getVotes(), points);
    }

    // forward global feature to voting class in single object mode
    if(m_single_object_mode) m_voting->setGlobalFeatures(globalFeatures_cleaned);

    LOG_INFO("finding maxima");
    boost::timer::cpu_timer timer_maxima;
    std::vector<VotingMaximum> positions = m_voting->findMaxima(pointsWithoutNaN, normalsWithoutNaN);
    m_processing_times["maxima"] += getElapsedTime(timer_maxima, "milliseconds");
    LOG_INFO("detected " << positions.size() << " maxima");

    // only debug
    if(m_enableVotingAnalysis)
    {
        addMaximaForDebug(all_votings, positions);
    }

    if(m_enable_signals)
    {
        timer.stop();
        m_signalMaxima(positions);
        timer.resume();
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
                                    bool compute_global)
{
    if (m_useVoxelFiltering) {
        // filter cloud to get a uniform point distribution
        LOG_INFO("performing voxel filtering");
        pcl::PointCloud<PointNormalT>::Ptr filtered(new pcl::PointCloud<PointNormalT>());
        m_voxelFiltering.setInputCloud(points);
        m_voxelFiltering.filter(*filtered);
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
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    pcl::PointCloud<PointT>::Ptr pointsWithoutNaN;
    pcl::PointCloud<pcl::Normal>::Ptr normalsWithoutNaN;

    // extract position point cloud
    pcl::copyPointCloud(*points, *pointCloud);

    if(m_enable_signals)
    {
        m_signalPointCloud(pointCloud);
    }

    // skip normals on certain descriptors
    std::string descr_type = m_featureDescriptor->getType();

    // TODO VS: add descriptor type that can be checked for "needNormals" and "isBinary"
    if (!hasNormals && (descr_type != "SHORT_SHOT" && descr_type != "SHORT_CSHOT" &&
                        descr_type != "SHORT_SHOT_PCL"))
    {
        // compute normals on the model
        timer_normals.start();
        LOG_INFO("computing normals");
        computeNormals(pointCloud, normals, searchTree);
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
    filterNormals(pointCloud, normals, pointsWithoutNaN, normalsWithoutNaN);

    LOG_ASSERT(pointsWithoutNaN.get() != 0);
    LOG_ASSERT(normalsWithoutNaN.get() != 0);
    LOG_ASSERT(pointsWithoutNaN->size() == normalsWithoutNaN->size());

    if(m_enable_signals)
    {
        m_signalNormals(pointsWithoutNaN, normalsWithoutNaN);
    }

    // detect interesting keypoints
    LOG_INFO("computing keypoints");
    timer_keypoints.start();
    m_keypointsDetector->setNumThreads(m_numThreads);
    pcl::PointCloud<PointT>::ConstPtr keypoints = (*m_keypointsDetector)(pointCloud, normals,
                                                                         pointsWithoutNaN, normalsWithoutNaN,
                                                                         searchTree);
    timer_keypoints.stop();

    // compute descriptors for keypoints
    LOG_INFO("computing features");
    m_featureDescriptor->setNumThreads(m_numThreads);
    pcl::PointCloud<ISMFeature>::ConstPtr features = (*m_featureDescriptor)(pointCloud, normals,
                                                                            pointsWithoutNaN, normalsWithoutNaN,
                                                                            keypoints,
                                                                            searchTree);

    // reference frames can be invalid, in which case associated keypoints are discarded
    LOG_ASSERT(features->size() <= keypoints->size());

    // in training compute global features
    if(compute_global)
    {
        // compute global descriptors for objects
        LOG_INFO("computing global features");
        pcl::PointCloud<PointT>::ConstPtr dummy_keypoints(new pcl::PointCloud<PointT>());
        m_globalFeatureDescriptor->setNumThreads(m_numThreads);
        pcl::PointCloud<ISMFeature>::ConstPtr global_features = (*m_globalFeatureDescriptor)(pointCloud, normals,
                                                                                             pointsWithoutNaN, normalsWithoutNaN,
                                                                                             dummy_keypoints,
                                                                                             searchTree);
        return std::make_tuple(features, global_features, pointsWithoutNaN, normalsWithoutNaN);
    }
    else // for recognition global features need to be computed later
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


void ImplicitShapeModel::computeNormals(pcl::PointCloud<PointT>::ConstPtr model,
                                        pcl::PointCloud<pcl::Normal>::Ptr& normals,
                                        pcl::search::Search<PointT>::Ptr searchTree) const
{
    LOG_ASSERT(normals.get() == 0);
    normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());

    if (model->isOrganized())
    {
        pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> normalEst;
        normalEst.setInputCloud(model);
        normalEst.setNormalEstimationMethod(normalEst.AVERAGE_3D_GRADIENT);
        normalEst.setMaxDepthChangeFactor(0.02f);
        normalEst.setNormalSmoothingSize(10.0f);

        // flip normals toward scene viewpoint
        LOG_INFO("orienting normals toward viewpoint");
        normalEst.useSensorOriginAsViewPoint();
        normalEst.compute(*normals);
        // TODO VS check if normals are flipped to viewpoint
    }
    else
    {
        LOG_INFO("computing consistent normal orientation (using method " << m_consistentNormalsMethod <<")");

        // prepare PCL normal estimation object
        pcl::NormalEstimationOMP<PointT, pcl::Normal> normalEst;
        normalEst.setInputCloud(model);
        normalEst.setSearchMethod(searchTree);
        normalEst.setRadiusSearch(m_normalRadius);
        normalEst.setNumberOfThreads(m_numThreads);

        NormalOrientation orient(m_consistentNormalsK, m_normalRadius);

        if(m_consistentNormalsMethod == 0)
        {
            // no consitent orientation - just compute
            normalEst.compute(*normals);
        }
        else if(m_consistentNormalsMethod == 1)
        {
            // move model to origin, then point normals away from origin
            pcl::PointCloud<PointT>::Ptr model_no_centroid(new pcl::PointCloud<PointT>());
            pcl::copyPointCloud(*model, *model_no_centroid);

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
        else if(m_consistentNormalsMethod == 2)
        {
            normalEst.compute(*normals); // this is only needed for curvature at each point
            orient.processSHOTLRF(model, normals, normals, searchTree);
        }
#ifdef USE_VCGLIB
        else if(m_consistentNormalsMethod == 3)
        {
            orient.computeUsingEMST(model, normals);
        }
#endif
        else
        {
            LOG_WARN("Invalid consistent normals method: " << m_consistentNormalsMethod << "! Skipping consistent normals.");
        }
    }
}


void ImplicitShapeModel::filterNormals(pcl::PointCloud<PointT>::ConstPtr model,
                                       pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                       pcl::PointCloud<PointT>::Ptr& modelWithoutNaN,
                                       pcl::PointCloud<pcl::Normal>::Ptr& normalsWithoutNaN)
{
    LOG_ASSERT(modelWithoutNaN.get() == 0);
    LOG_ASSERT(normalsWithoutNaN.get() == 0);

    // filter NaN normals
    std::vector<int> mapping;
    normalsWithoutNaN = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());
    pcl::removeNaNNormalsFromPointCloud(*normals, *normalsWithoutNaN, mapping);

    // create new point cloud without NaN normals
    modelWithoutNaN = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
    for (int i = 0; i < (int)mapping.size(); i++)
        modelWithoutNaN->push_back(model->at(mapping[i]));
}

Json::Value ImplicitShapeModel::iChildConfigsToJson() const
{
    Json::Value children(Json::objectValue);

    children["Codebook"] = m_codebook->configToJson();
    children["Keypoints"] = m_keypointsDetector->configToJson();
    children["Features"] = m_featureDescriptor->configToJson();
    children["GlobalFeatures"] = m_globalFeatureDescriptor->configToJson();
    children["Clustering"] = m_clustering->configToJson();
    children["Voting"] = m_voting->configToJson();
    children["FeatureWeighting"] = m_featureRanking->configToJson(); // TODO VS: rename config entry to FeatureRanking in future

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
    delete m_keypointsDetector;
    delete m_featureDescriptor;
    delete m_globalFeatureDescriptor;
    delete m_clustering;
    delete m_voting;
    delete m_featureRanking;

    // create new child objects
    m_codebook = Factory<Codebook>::create(*codebook);
    m_keypointsDetector = Factory<Keypoints>::create(*keypoints);
    m_featureDescriptor = Factory<Features>::create(*features);
    m_globalFeatureDescriptor = use_dummy ? Factory<Features>::create(dummy) : Factory<Features>::create(*global_features);
    m_clustering = Factory<Clustering>::create(*clustering);
    m_voting = Factory<Voting>::create(*voting);
    m_featureRanking = Factory<FeatureRanking>::create(*feature_ranking);

    if (!m_codebook || !m_keypointsDetector || !m_featureDescriptor || !m_globalFeatureDescriptor ||
            !m_clustering || !m_voting || !m_featureRanking)
        throw RuntimeException("Could not create object(s). The configuration file might be corrupted.");

    // set global descriptor for the detection step
    m_voting->setGlobalFeatureDescriptor(m_globalFeatureDescriptor);

    return true;
}

void ImplicitShapeModel::iSaveData(boost::archive::binary_oarchive &oa) const
{
    m_codebook->saveData(oa);
    m_keypointsDetector->saveData(oa);
    m_featureDescriptor->saveData(oa);
    m_globalFeatureDescriptor->saveData(oa);
    m_clustering->saveData(oa);
    m_voting->saveData(oa);
    m_featureRanking->saveData(oa);
}

bool ImplicitShapeModel::iLoadData(boost::archive::binary_iarchive &ia)
{
    // objects have to be initialized already
    if (!m_codebook || !m_keypointsDetector || !m_featureDescriptor || !m_globalFeatureDescriptor ||
            !m_clustering || !m_voting || !m_featureRanking) {
        LOG_ERROR("object is not initialized");
        return false;
    }

    // TODO VS: this is necessary since objects are created before config is read in json_object.cpp
    m_voting->setSVMPath(m_svm_path);

    // init data for objects
    if (!m_codebook->loadData(ia) ||
            !m_keypointsDetector->loadData(ia) ||
            !m_featureDescriptor->loadData(ia) ||
            !m_globalFeatureDescriptor->loadData(ia) ||
            !m_clustering->loadData(ia) ||
            !m_voting->loadData(ia) ||
            !m_featureRanking->loadData(ia))
    {
        LOG_ERROR("could not load child objects");
        return false;
    }

    return true;
}

Json::Value ImplicitShapeModel::iDataToJson() const
{
    Json::Value data(Json::objectValue);
    data["Codebook"] = m_codebook->dataToJson();
    data["Keypoints"] = m_keypointsDetector->dataToJson();
    data["Features"] = m_featureDescriptor->dataToJson();
    data["GlobalFeatures"] = m_globalFeatureDescriptor->dataToJson();
    data["Clustering"] = m_clustering->dataToJson();
    data["Voting"] = m_voting->dataToJson();
    data["FeatureWeighting"] = m_featureRanking->dataToJson();
    return data;
}

bool ImplicitShapeModel::iDataFromJson(const Json::Value& object)
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
            voting->isNull() || !voting->isObject()) {
        LOG_ERROR("could not find necessary json entries");
        return false;
    }

    // for backward compatibility the following objects give an error, but do not return false
    if(feature_ranking->isNull() || !feature_ranking->isObject())
    {
        LOG_ERROR("could not find \"FeatureWeighting\" object in json file");
    }

    // handle global features separately to be able to use old configs and trained files
    bool use_dummy = false;
    Json::Value dummy(Json::objectValue);
    dummy["Type"] = Json::Value("Dummy");

    if (global_features->isNull() || !global_features->isObject())
    {
        use_dummy = true;
        LOG_WARN("No global features available in data file, using Dummy global feature!");
    }


    // objects have to be initialized already
    if (!m_codebook || !m_keypointsDetector || !m_featureDescriptor || !m_globalFeatureDescriptor ||
            !m_clustering || !m_voting || !m_featureRanking) {
        LOG_ERROR("object is not initialized");
        return false;
    }

    // descriptor needs to have the same type
    Json::Value featureType = (*features)["Type"];
    Json::Value globalFeatureType = use_dummy ? dummy["Type"] : (*global_features)["Type"];

    if (featureType.isNull() || !featureType.isString() ||
            globalFeatureType.isNull() || !globalFeatureType.isString() ) {
        LOG_ERROR("invalid feature type");
        return false;
    }

    std::string typeStr = featureType.asString();
    if (typeStr != m_featureDescriptor->getType())
        throw RuntimeException("Cannot change local descriptor type after learning.");

    std::string typeStrGlobal = globalFeatureType.asString();
    if (typeStrGlobal != m_globalFeatureDescriptor->getType())
        throw RuntimeException("Cannot change global descriptor type after learning.");

    // init data for objects
    if (!m_codebook->dataFromJson(*codebook) ||
            !m_keypointsDetector->dataFromJson(*keypoints) ||
            !m_featureDescriptor->dataFromJson(*features) ||
            !m_globalFeatureDescriptor->dataFromJson(*global_features) ||
            !m_clustering->dataFromJson(*clustering) ||
            !m_voting->dataFromJson(*voting) ||
            !m_featureRanking->dataFromJson(*feature_ranking)) {
        LOG_ERROR("could not create child objects");
        return false;
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
    else if (m_distanceType == DistanceHellinger::getTypeStatic())
        m_distance = new DistanceHellinger;
    else if (m_distanceType == DistanceHistIntersection::getTypeStatic())
        m_distance = new DistanceHistIntersection;
    else
        throw RuntimeException("invalid distance type: " + m_distanceType);

    m_voxelFiltering.setLeafSize(m_voxelLeafSize, m_voxelLeafSize, m_voxelLeafSize);

    // m_numThreads == 0 is the default, so don't change anything
    if (m_numThreads > 0)
        omp_set_num_threads(m_numThreads);

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


// DEBUG

std::map<unsigned, pcl::PointCloud<PointT>::Ptr > ImplicitShapeModel::analyzeVotingSpacesForDebug
(const std::map<unsigned, std::vector<Voting::Vote> > &all_votes, pcl::PointCloud<PointNormalT>::Ptr points)
{
    std::map<unsigned, pcl::PointCloud<PointT>::Ptr > all_votings;
    for (auto it : all_votes)
    {
        unsigned classId = it.first;
        const std::vector<Voting::Vote>& votes = it.second; // all votes for this class

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
            const Voting::Vote& vote = votes[i];
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
    unsigned pos = m_votingAnalysisOutputPath.find_last_of('/');
    if(pos != m_votingAnalysisOutputPath.size()-1)
    {
        m_votingAnalysisOutputPath = m_votingAnalysisOutputPath.append("/");
        std::string command = "mkdir "+m_votingAnalysisOutputPath+" -p";
        int unused = std::system(command.c_str());
    }

    m_counter++;
    std::ofstream ofs;
    ofs.open(m_votingAnalysisOutputPath+"votes.txt", std::ofstream::out | std::ofstream::app);
    for(auto obj : all_votings)
    {
        pcl::io::savePCDFileBinary(m_votingAnalysisOutputPath+"file_idx_"+std::to_string(m_counter)+"_class_"+
                                   std::to_string(obj.first)+".pcd", *obj.second);
        ofs << "file: " << m_counter << ", class: " << obj.first << ", votes:" << obj.second->size() << std::endl;
    }
    ofs.close();
}

}
