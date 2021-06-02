#include "pcl_orcg.h"

#include <pcl/io/pcd_io.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/shot_lrf_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/correspondence.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/registration/icp.h>
#include <pcl/recognition/hv/hv_go.h>

// NOTE: inspired by the tutorials
// http://pointclouds.org/documentation/tutorials/correspondence_grouping.php#correspondence-grouping
// http://pointclouds.org/documentation/tutorials/global_hypothesis_verification.php
// and the paper:
// Aldoma et al. 2012, Point Cloud Library - Three-Dimensional Object Recognition and 6 DoF Pose Estimation, IEEE ROBOTICS & AUTOMATION MAGAZINE

Orcg::Orcg() : m_scene(new pcl::PointCloud<PointT>())
{
    m_normal_radius = 0.05;
    m_reference_frame_radius = 0.3;
    m_feature_radius = 0.4;
    m_keypoint_sampling_radius = 0.2;
    m_k_search = 11;

    m_cg_size = 0.5;
    m_cg_thresh = 0.25;
    m_use_hough = false;
    m_icp_max_iter = 100;
    m_icp_corr_distance = 0.1;
}


bool Orcg::prepareScene(const std::string &filename_scene)
{
    // load scene
    if(pcl::io::loadPCDFile<PointT>(filename_scene, *m_scene) == -1)
    {
        std::cerr << "ERROR: loading file " << filename_scene << std::endl;
        return false;
    }

    // extract features
    m_scene_features = processPointCloud(m_scene, m_scene_keypoints);
    return true;
}


std::vector<std::pair<unsigned, float>> Orcg::findObjectInScene(const std::string &filename_model) const
{
    pcl::PointCloud<PointT>::Ptr model(new pcl::PointCloud<PointT>());
    if(pcl::io::loadPCDFile<PointT>(filename_model, *model) == -1)
    {
        std::cerr << "ERROR: loading file " << filename_model << std::endl;
        return std::vector<std::pair<unsigned, float>>();
    }

    // extract features
    pcl::PointCloud<PointT>::Ptr model_keypoints;
    pcl::PointCloud<ISMFeature>::Ptr model_features = processPointCloud(model, model_keypoints);

    // get model-scene correspondences
    pcl::CorrespondencesPtr model_scene_corrs = findNnCorrespondences(model_features);
    std::cout << "Found " << model_scene_corrs->size() << " correspondences" << std::endl;

    // Actual Clustering
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
    std::vector<pcl::Correspondences> clustered_corrs;

    //  Using Hough3D
    if (m_use_hough)
    {
        //  Clustering
        pcl::Hough3DGrouping<PointT, PointT, pcl::ReferenceFrame, pcl::ReferenceFrame> clusterer;
        clusterer.setHoughBinSize(m_cg_size);
        clusterer.setHoughThreshold(m_cg_thresh);
        clusterer.setUseInterpolation(true);
        clusterer.setUseDistanceWeight(false);
        clusterer.setLocalRfSearchRadius(m_reference_frame_radius);

        clusterer.setInputCloud(model_keypoints);
        clusterer.setSceneCloud(m_scene_keypoints);
        clusterer.setModelSceneCorrespondences(model_scene_corrs);

        //clusterer.cluster(clustered_corrs);
        clusterer.recognize(rototranslations, clustered_corrs);
    }
    else // Using GeometricConsistency
    {
        pcl::GeometricConsistencyGrouping<PointT, PointT> gc_clusterer;
        gc_clusterer.setGCSize(m_cg_size);
        gc_clusterer.setGCThreshold(m_cg_thresh);

        gc_clusterer.setInputCloud(model_keypoints);
        gc_clusterer.setSceneCloud(m_scene_keypoints);
        gc_clusterer.setModelSceneCorrespondences(model_scene_corrs);

        //gc_clusterer.cluster(clustered_corrs);
        gc_clusterer.recognize(rototranslations, clustered_corrs);
    }

    // Stop if no instances
    if (rototranslations.size () <= 0)
    {
        std::cout << "No instances found!" << std::endl;
        return std::vector<std::pair<unsigned, float>>();
    }
    else
    {
        std::cout << "Recognized Instances: " << rototranslations.size () << std::endl;
    }

    // Generates clouds for each instances found
    std::vector<pcl::PointCloud<PointT>::ConstPtr> instances;
    for(size_t i = 0; i < rototranslations.size (); ++i)
    {
        pcl::PointCloud<PointT>::Ptr rotated_model(new pcl::PointCloud<PointT>());
        pcl::transformPointCloud(*model, *rotated_model, rototranslations[i]);
        instances.push_back(rotated_model);
    }

    // ICP
    std::vector<pcl::PointCloud<PointT>::ConstPtr> registered_instances;
    if (true)
    {
        std::cout << "--- ICP ---------" << std::endl;

        for (size_t i = 0; i < rototranslations.size (); ++i)
        {
            pcl::IterativeClosestPoint<PointT, PointT> icp;
            icp.setMaximumIterations(m_icp_max_iter);
            icp.setMaxCorrespondenceDistance(m_icp_corr_distance);
            icp.setInputTarget(m_scene);
            icp.setInputSource(instances[i]);
            pcl::PointCloud<PointT>::Ptr registered (new pcl::PointCloud<PointT>);
            icp.align(*registered);
            registered_instances.push_back(registered);
            std::cout << "Instance " << i << " ";
            if (icp.hasConverged ())
            {
                std::cout << "Aligned!" << std::endl;
            }
            else
            {
                std::cout << "Not Aligned!" << std::endl;
            }
        }
    }

    // Hypothesis Verification
    std::cout << "--- Hypotheses Verification ---" << std::endl;

    // TODO VS: tune thresholds
    std::vector<bool> hypotheses_mask;  // Mask Vector to identify positive hypotheses
    pcl::GlobalHypothesesVerification<PointT, PointT> GoHv;
    GoHv.setSceneCloud(m_scene);  // Scene Cloud
    GoHv.addModels(registered_instances, true);  //Models to verify
    GoHv.setInlierThreshold(0.01);
    GoHv.setOcclusionThreshold(0.02);
    GoHv.setRegularizer(3.0);
    GoHv.setClutterRegularizer(5.0);
    GoHv.setRadiusClutter(0.25);
    GoHv.setDetectClutter(true);
    GoHv.setRadiusNormals(m_normal_radius);
    GoHv.verify();
    GoHv.getMask (hypotheses_mask);  // i-element TRUE if hvModels[i] verifies hypotheses

    for (int i = 0; i < hypotheses_mask.size (); i++)
    {
        if(hypotheses_mask[i])
        {
            std::cout << "Instance " << i << " is GOOD! <---" << std::endl;
        }
        else
        {
            std::cout << "Instance " << i << " is bad!" << std::endl;
        }
    }

    // TODO VS: create some kind of results vector based on hypothesis verification
    std::vector<std::pair<unsigned, float>> results;
    return results;
}


pcl::PointCloud<ISMFeature>::Ptr Orcg::processPointCloud(pcl::PointCloud<PointT>::Ptr &cloud,
                                                         pcl::PointCloud<PointT>::Ptr &keyp) const
{
    // create search tree
    pcl::search::Search<PointT>::Ptr searchTree;
    if (cloud->isOrganized())
    {
        searchTree = pcl::search::OrganizedNeighbor<PointT>::Ptr(new pcl::search::OrganizedNeighbor<PointT>());
    }
    else
    {
        searchTree = pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>());
    }

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
    keyp = keypoints;

    // compute reference frames
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr reference_frames;
    computeReferenceFrames(reference_frames, keypoints, cloud_without_nan, searchTree);

    // compute descriptors
    pcl::PointCloud<ISMFeature>::Ptr features;
    computeDescriptors(cloud_without_nan, normals_without_nan, keypoints, searchTree, reference_frames, features);

    // remove NAN features
    pcl::PointCloud<ISMFeature>::Ptr features_cleaned;
    removeNanDescriptors(features, features_cleaned);

    return features_cleaned;
}


void Orcg::computeNormals(pcl::PointCloud<PointT>::Ptr &cloud,
                           pcl::PointCloud<pcl::Normal>::Ptr& normals,
                           pcl::search::Search<PointT>::Ptr &searchTree) const
{
    normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());

    if (cloud->isOrganized())
    {
        pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> normalEst;
        normalEst.setInputCloud(cloud);
        normalEst.setNormalEstimationMethod(normalEst.AVERAGE_3D_GRADIENT);
        normalEst.setMaxDepthChangeFactor(0.02f);
        normalEst.setNormalSmoothingSize(10.0f);
        normalEst.useSensorOriginAsViewPoint();
        normalEst.compute(*normals);
        // TODO VS check if normals are flipped to viewpoint
    }
    else
    {
        // prepare PCL normal estimation object
        pcl::NormalEstimationOMP<PointT, pcl::Normal> normalEst;
        normalEst.setInputCloud(cloud);
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

void Orcg::filterNormals(pcl::PointCloud<pcl::Normal>::Ptr &normals,
                          pcl::PointCloud<pcl::Normal>::Ptr &normals_without_nan,
                          pcl::PointCloud<PointT>::Ptr &cloud,
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


void Orcg::computeKeypoints(pcl::PointCloud<PointT>::Ptr &keypoints, pcl::PointCloud<PointT>::Ptr &cloud) const
{
    pcl::VoxelGrid<PointT> voxelGrid;
    voxelGrid.setInputCloud(cloud);
    voxelGrid.setLeafSize(m_keypoint_sampling_radius, m_keypoint_sampling_radius, m_keypoint_sampling_radius);
    keypoints = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
    voxelGrid.filter(*keypoints);
}


void Orcg::computeReferenceFrames(pcl::PointCloud<pcl::ReferenceFrame>::Ptr &reference_frames,
                                   pcl::PointCloud<PointT>::Ptr &keypoints,
                                   pcl::PointCloud<PointT>::Ptr &cloud,
                                   pcl::search::Search<PointT>::Ptr &searchTree) const
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


void Orcg::computeDescriptors(pcl::PointCloud<PointT>::Ptr &cloud,
                               pcl::PointCloud<pcl::Normal>::Ptr &normals,
                               pcl::PointCloud<PointT>::Ptr &keypoints,
                               pcl::search::Search<PointT>::Ptr &searchTree,
                               pcl::PointCloud<pcl::ReferenceFrame>::Ptr &reference_frames,
                               pcl::PointCloud<ISMFeature>::Ptr &features) const
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


void Orcg::removeNanDescriptors(pcl::PointCloud<ISMFeature>::Ptr &features,
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


flann::Matrix<float> Orcg::createFlannDataset(const pcl::PointCloud<ISMFeature>::Ptr& features) const
{
    // create a dataset with all features for matching / activation
    int descriptor_size = features->at(0).descriptor.size();
    flann::Matrix<float> dataset(new float[features->size() * descriptor_size],
            features->size(), descriptor_size);

    // build dataset
    for(int i = 0; i < features->size(); i++)
    {
        ISMFeature ism_feat = features->at(i);
        std::vector<float> descriptor = ism_feat.descriptor;
        for(int j = 0; j < (int)descriptor.size(); j++)
        {
            dataset[i][j] = descriptor.at(j);
        }
    }

    return dataset;
}


pcl::CorrespondencesPtr Orcg::findNnCorrespondences(const pcl::PointCloud<ISMFeature>::Ptr& features) const
{
    // create flann index for search
    flann::Matrix<float> dataset = createFlannDataset(features);
    flann::Index<flann::L2<float>> flann_index = flann::Index<flann::L2<float>>(dataset, flann::KDTreeIndexParams(4));
    flann_index.buildIndex();

    pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());

    // loop over all features extracted from the input model
    #pragma omp parallel for
    for(int fe = 0; fe < m_scene_features->size(); fe++)
    {
        // insert the query point
        ISMFeature feature = m_scene_features->at(fe);
        flann::Matrix<float> query(new float[feature.descriptor.size()], 1, feature.descriptor.size());
        for(int i = 0; i < feature.descriptor.size(); i++)
        {
            query[0][i] = feature.descriptor.at(i);
        }

        // prepare results
        std::vector<std::vector<int> > indices;
        std::vector<std::vector<float> > distances;
        int found_neigbors = flann_index.knnSearch(query, indices, distances, 1, flann::SearchParams(128));

        delete[] query.ptr();

        if(found_neigbors == 1 && distances[0][0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
        {
            pcl::Correspondence corr(fe, indices[0][0], distances[0][0]);
            #pragma omp critical
            {
                model_scene_corrs->push_back(corr);
            }
        }
    }

    return model_scene_corrs;
}
