#include "feature_processing.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/shot_lrf_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>

void processPointCloud(
        const pcl::PointCloud<PointT>::Ptr cloud,
        pcl::PointCloud<PointT>::Ptr &keypoints_cleaned,
        pcl::PointCloud<ISMFeature>::Ptr &features_cleaned,
        pcl::PointCloud<pcl::Normal>::Ptr &normals_cleaned,
        pcl::PointCloud<pcl::ReferenceFrame>::Ptr &reference_frames_cleaned)
{
    // create search tree
    pcl::search::Search<PointT>::Ptr searchTree;
    searchTree = pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>());

    // compute normals
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    computeNormals(cloud, searchTree, normals);

    // filter normals
    pcl::PointCloud<PointT>::Ptr cloud_without_nan;
    filterNormals(normals, cloud, normals_cleaned, cloud_without_nan);

    // compute keypoints
    pcl::PointCloud<PointT>::Ptr keypoints;
    computeKeypoints(cloud_without_nan, normals_cleaned, keypoints);

    // compute reference frames
    computeReferenceFrames(cloud_without_nan, keypoints, searchTree, keypoints_cleaned, reference_frames_cleaned);

    // compute descriptors
    pcl::PointCloud<ISMFeature>::Ptr features;
    computeDescriptors(cloud_without_nan, normals_cleaned, keypoints_cleaned,
                       searchTree, reference_frames_cleaned, features);

    // store keypoint positions and reference frames
    for (unsigned i = 0; i < features->size(); i++)
    {
        ISMFeature& feature = features->at(i);
        const PointT& keypoint = keypoints_cleaned->at(i);
        feature.x = keypoint.x;
        feature.y = keypoint.y;
        feature.z = keypoint.z;
        feature.referenceFrame = reference_frames_cleaned->at(i);
    }

    // remove NAN features
    removeNanDescriptors(features, features_cleaned);
}


void computeNormals(
        const pcl::PointCloud<PointT>::Ptr cloud,
        const pcl::search::Search<PointT>::Ptr searchTree,
        pcl::PointCloud<pcl::Normal>::Ptr& normals)
{
    normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());

//    if(fp::m_normal_method == 0 && cloud->isOrganized())
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
//    else if(fp::m_normal_method == 0 && !cloud->isOrganized())
    if(fp::normal_method == 0)
    {
         // prepare PCL normal estimation object
        pcl::NormalEstimationOMP<PointT, pcl::Normal> normalEst;
        normalEst.setInputCloud(cloud);
        normalEst.setNumberOfThreads(0);
        normalEst.setSearchMethod(searchTree);
        normalEst.setRadiusSearch(fp::normal_radius);
        normalEst.setViewPoint(0,0,0);
        normalEst.compute(*normals);
    }
    else // TODO VS: das ist nur für classification gut!
    {
        // prepare PCL normal estimation object
        pcl::NormalEstimationOMP<PointT, pcl::Normal> normalEst;
        normalEst.setInputCloud(cloud);
        normalEst.setNumberOfThreads(0);
        normalEst.setSearchMethod(searchTree);
        normalEst.setRadiusSearch(fp::normal_radius);

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


void filterNormals(
        const pcl::PointCloud<pcl::Normal>::Ptr normals,
        const pcl::PointCloud<PointT>::Ptr cloud,
        pcl::PointCloud<pcl::Normal>::Ptr &normals_without_nan,
        pcl::PointCloud<PointT>::Ptr &cloud_without_nan)
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


void computeKeypoints(
        const pcl::PointCloud<PointT>::Ptr cloud,
        const pcl::PointCloud<pcl::Normal>::Ptr normals,
        pcl::PointCloud<PointT>::Ptr &keypoints)
{
    pcl::VoxelGrid<PointT> voxelGrid;
    voxelGrid.setInputCloud(cloud);
    voxelGrid.setLeafSize(fp::keypoint_sampling_radius, fp::keypoint_sampling_radius, fp::keypoint_sampling_radius);
    keypoints = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
    voxelGrid.filter(*keypoints);


//    // select only best keypoints -- temp code
//    float cutoff_ratio = 0.5;
//    keypoints = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
//    // create cloud containing filtered cloud points and filtered normals
//    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr points_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
//    pcl::concatenateFields(*cloud, *normals, *points_with_normals);

//    // compute voxel grid keypoints on cloud with normals
//    pcl::VoxelGrid<pcl::PointXYZRGBNormal> voxel_grid;
//    voxel_grid.setInputCloud(points_with_normals);
//    voxel_grid.setLeafSize(fp::keypoint_sampling_radius, fp::keypoint_sampling_radius, fp::keypoint_sampling_radius);
//    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr keypoints_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
//    voxel_grid.filter(*keypoints_with_normals);

//    // copy only point information without normals
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_without_normals(new pcl::PointCloud<pcl::PointXYZRGB>()); // these keypoints will be filtered here
//    pcl::copyPointCloud(*keypoints_with_normals, *keypoints_without_normals);

//    // estimate principle curvatures
//    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PrincipalCurvatures> curv_est;
//    curv_est.setInputCloud(keypoints_without_normals);
//    curv_est.setSearchSurface(cloud);
//    curv_est.setInputNormals(normals);
//    //curv_est.setSearchMethod(search);
//    curv_est.setRadiusSearch(fp::keypoint_sampling_radius);
//    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures (new pcl::PointCloud<pcl::PrincipalCurvatures>());
//    curv_est.compute(*principal_curvatures);

//    std::vector<float> geo_scores;
//    for(unsigned idx = 0; idx < keypoints_with_normals->size(); idx++)
//    {
//        // PCL curvature
//        pcl::PointXYZRGBNormal &reference_point = keypoints_with_normals->at(idx);

//        // gaussian curvature
//        const pcl::PrincipalCurvatures &pc_point = principal_curvatures->at(idx);
//        geo_scores.push_back(pc_point.pc1 * pc_point.pc2);
//        // overwrite curvature with current method's value
//        reference_point.curvature = pc_point.pc1 * pc_point.pc2;
//    }

//    // sort to determine cutoff threshold
//    std::sort(geo_scores.begin(), geo_scores.end());

//    unsigned cutoff_index = unsigned(cutoff_ratio * geo_scores.size());
//    float filter_threshold_geometry = geo_scores.at(cutoff_index);

//    for(unsigned idx = 0; idx < keypoints_with_normals->size(); idx++)
//    {
//        bool geo_passed = true;
//        pcl::PointXYZRGBNormal point = keypoints_with_normals->at(idx);
//        // NOTE: curvature corresponds to chosen geometry type value
//        if(point.curvature < filter_threshold_geometry)
//        {
//            geo_passed = false;
//        }

//        if(geo_passed)
//            keypoints->push_back(keypoints_without_normals->at(idx));
//    }

}


void computeReferenceFrames(
        const pcl::PointCloud<PointT>::Ptr cloud,
        const pcl::PointCloud<PointT>::Ptr keypoints,
        const pcl::search::Search<PointT>::Ptr searchTree,
        pcl::PointCloud<PointT>::Ptr &keypoints_clean,
        pcl::PointCloud<pcl::ReferenceFrame>::Ptr &reference_frames)
{
    reference_frames = pcl::PointCloud<pcl::ReferenceFrame>::Ptr(new pcl::PointCloud<pcl::ReferenceFrame>());
    pcl::SHOTLocalReferenceFrameEstimationOMP<PointT, pcl::ReferenceFrame> refEst;
    refEst.setRadiusSearch(fp::reference_frame_radius);
    refEst.setInputCloud(keypoints);
    refEst.setSearchSurface(cloud);
    refEst.setSearchMethod(searchTree);
    refEst.compute(*reference_frames);

    pcl::PointCloud<pcl::ReferenceFrame>::Ptr cleanReferenceFrames(new pcl::PointCloud<pcl::ReferenceFrame>());
    keypoints_clean = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
    for(int i = 0; i < (int)reference_frames->size(); i++)
    {
        const pcl::ReferenceFrame& frame = reference_frames->at(i);
        if(std::isfinite(frame.x_axis[0]) && std::isfinite(frame.y_axis[0]) && std::isfinite(frame.z_axis[0]))
        {
            cleanReferenceFrames->push_back(frame);
            keypoints_clean->push_back(keypoints->at(i));
        }
    }
}


void computeDescriptors(
        const pcl::PointCloud<PointT>::Ptr cloud,
        const pcl::PointCloud<pcl::Normal>::Ptr normals,
        const pcl::PointCloud<PointT>::Ptr keypoints,
        const pcl::search::Search<PointT>::Ptr searchTree,
        const pcl::PointCloud<pcl::ReferenceFrame>::Ptr reference_frames,
        pcl::PointCloud<ISMFeature>::Ptr &features)
{
    if(fp::feature_type == "SHOT")
    {
        pcl::SHOTEstimationOMP<PointT, pcl::Normal, pcl::SHOT352> shotEst;
        shotEst.setSearchSurface(cloud);
        shotEst.setInputNormals(normals);
        shotEst.setInputCloud(keypoints);
        shotEst.setInputReferenceFrames(reference_frames);
        shotEst.setSearchMethod(searchTree);
        shotEst.setRadiusSearch(fp::feature_radius);
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
    else if(fp::feature_type == "CSHOT")
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
        shotEst.setRadiusSearch(fp::feature_radius);
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


void removeNanDescriptors(
        const pcl::PointCloud<ISMFeature>::Ptr features,
        pcl::PointCloud<ISMFeature>::Ptr &features_cleaned)
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
