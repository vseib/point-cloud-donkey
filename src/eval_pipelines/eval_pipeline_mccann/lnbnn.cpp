#include "lnbnn.h"

#include <pcl/io/pcd_io.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/shot_lrf_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/filters/voxel_grid.h>


Lnbnn::Lnbnn() : m_features(new pcl::PointCloud<ISMFeature>()), m_flann_index(flann::KDTreeIndexParams(4))
{
    m_normal_radius = 0.05;
    m_reference_frame_radius = 0.3;
    m_feature_radius = 0.4;
    m_keypoint_sampling_radius = 0.2;
    m_k_search = 11;
    m_rgbd_camera_data = true;
}


void Lnbnn::train(const std::vector<std::string> &filenames, const std::vector<std::string> &labels, const std::string &output_file) const
{
    if(filenames.size() != labels.size())
    {
        std::cerr << "ERROR: number of clouds does not match number of labels!" << std::endl;
        return;
    }

    // contains the whole list of features for each class id
    std::map<unsigned, pcl::PointCloud<ISMFeature>::Ptr> all_features;
    for(unsigned i = 0; i < labels.size(); i++)
    {
        unsigned int tr_class = atoi(labels.at(i).c_str());
        pcl::PointCloud<ISMFeature>::Ptr cloud(new pcl::PointCloud<ISMFeature>());
        all_features.insert({tr_class, cloud});
    }

    int num_features = 0;

    // process each input file
    for(int i = 0; i < filenames.size(); i++)
    {
        std::cout << "Processing file " << (i+1) << " of " << filenames.size() << std::endl;
        std::string file = filenames.at(i);
        unsigned int tr_class = atoi(labels.at(i).c_str());

        // load cloud
        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
        if(pcl::io::loadPCDFile(file, *cloud) == -1)
        {
            std::cerr << "ERROR: loading file " << file << std::endl;
        }

        pcl::PointCloud<ISMFeature>::Ptr features_cleaned = processPointCloud(cloud);

        // add computed features to map
        (*all_features.at(tr_class)) += (*features_cleaned);
        num_features += features_cleaned->size();
    }

    std::cout << "Extracted " << num_features << " features." << std::endl;

    std::string save_file(output_file);
    if(saveModelToFile(save_file, all_features))
    {
        std::cout << "LNBNN training finished!" << std::endl;
    }
    else
    {
        std::cerr << "ERROR saving LNBNN model!" << std::endl;
    }
}


std::vector<std::pair<unsigned, float>> Lnbnn::classify(const std::string &filename) const
{
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if(pcl::io::loadPCDFile<PointT>(filename, *cloud) == -1)
    {
        std::cerr << "ERROR: loading file " << filename << std::endl;
        return std::vector<std::pair<unsigned, float>>();
    }

    // extract features
    pcl::PointCloud<ISMFeature>::Ptr features = processPointCloud(cloud);

    // get class distances
    std::vector<std::pair<unsigned, float>> results = accumulateClassDistances(features);

    // here smaller values are better
    std::sort(results.begin(), results.end(), [](const std::pair<unsigned, float> &a, const std::pair<unsigned, float> &b)
    {
        return a.second < b.second;
    });

    return results;
}


bool Lnbnn::loadModel(std::string &filename)
{
    if(!loadModelFromFile(filename)) return false;

    flann::Matrix<float> dataset = createFlannDataset();

    m_flann_index = flann::Index<flann::L2<float>>(dataset, flann::KDTreeIndexParams(4));
    m_flann_index.buildIndex();

    return true;
}


pcl::PointCloud<ISMFeature>::Ptr Lnbnn::processPointCloud(pcl::PointCloud<PointT>::Ptr &cloud) const
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


void Lnbnn::computeNormals(pcl::PointCloud<PointT>::Ptr &cloud,
                           pcl::PointCloud<pcl::Normal>::Ptr& normals,
                           pcl::search::Search<PointT>::Ptr &searchTree) const
{
    normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());

    // always use if-clause if data from kinect is used - no matter if organized or not
    if (cloud->isOrganized() || m_rgbd_camera_data)
    {
        pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> normalEst;
        normalEst.setInputCloud(cloud);
        normalEst.setNormalEstimationMethod(normalEst.AVERAGE_3D_GRADIENT);
        normalEst.setMaxDepthChangeFactor(0.02f);
        normalEst.setNormalSmoothingSize(10.0f);
        normalEst.useSensorOriginAsViewPoint();
        normalEst.compute(*normals);
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

void Lnbnn::filterNormals(pcl::PointCloud<pcl::Normal>::Ptr &normals,
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


void Lnbnn::computeKeypoints(pcl::PointCloud<PointT>::Ptr &keypoints, pcl::PointCloud<PointT>::Ptr &cloud) const
{
    pcl::VoxelGrid<PointT> voxelGrid;
    voxelGrid.setInputCloud(cloud);
    voxelGrid.setLeafSize(m_keypoint_sampling_radius, m_keypoint_sampling_radius, m_keypoint_sampling_radius);
    keypoints = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
    voxelGrid.filter(*keypoints);
}


void Lnbnn::computeReferenceFrames(pcl::PointCloud<pcl::ReferenceFrame>::Ptr &reference_frames,
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


void Lnbnn::computeDescriptors(pcl::PointCloud<PointT>::Ptr &cloud,
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


void Lnbnn::removeNanDescriptors(pcl::PointCloud<ISMFeature>::Ptr &features,
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


flann::Matrix<float> Lnbnn::createFlannDataset()
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


std::vector<std::pair<unsigned, float>> Lnbnn::accumulateClassDistances(const pcl::PointCloud<ISMFeature>::Ptr& features) const
{
    std::vector<std::pair<unsigned, float>> class_distances;
    for(int i = 0; i < m_number_of_classes; i++)
    {
        class_distances.push_back({i, 0.0f});
    }

    int k_search = m_k_search;

    // loop over all features extracted from the input model
    #pragma omp parallel for
    for(int fe = 0; fe < features->size(); fe++)
    {
        // insert the query point
        ISMFeature feature = features->at(fe);
        flann::Matrix<float> query(new float[feature.descriptor.size()], 1, feature.descriptor.size());
        for(int i = 0; i < feature.descriptor.size(); i++)
        {
            query[0][i] = feature.descriptor.at(i);
        }

        // prepare results
        std::vector<std::vector<int> > indices;
        std::vector<std::vector<float> > distances;
        m_flann_index.knnSearch(query, indices, distances, k_search, flann::SearchParams(128));

        delete[] query.ptr();

        // background distance
        float dist_b = 0;
        if(distances.size() > 0 && distances.at(0).size() > 1)
        {
            dist_b = distances.at(0).back(); // get last element
        }

        std::vector<unsigned> used_classes;
        if(distances.size() > 0 && distances.at(0).size() > 0)
        {
            for(int i = 0; i < distances[0].size()-1; i++)
            {
                unsigned class_idx = m_class_lookup[indices[0][i]];
                if(!containsValue(used_classes, class_idx))
                {
                    #pragma omp critical
                    {
                        class_distances.at(class_idx).second += distances[0][i] - dist_b;
                    }
                    used_classes.push_back(class_idx);
                }
            }
        }
    }

    return class_distances;
}


bool Lnbnn::saveModelToFile(std::string &filename, std::map<unsigned, pcl::PointCloud<ISMFeature>::Ptr> &all_features) const
{
    std::ofstream output_file (filename.c_str (), std::ios::trunc);
    if (!output_file)
    {
      output_file.close ();
      return false;
    }

    int num_features = 0;
    for(auto elem : all_features)
        num_features += elem.second->size();

    int descriptor_dim = all_features[0]->at(0).descriptor.size();

    output_file << all_features.size() << " ";
    output_file << num_features << " ";
    output_file << descriptor_dim << " ";

    //write classes
    for(auto elem : all_features)
    {
        for(unsigned int feat = 0; feat < elem.second->size(); feat++)
        {
            output_file << elem.first << " ";
        }
    }

    //write features
    for(auto elem : all_features)
    {
        for(unsigned int feat = 0; feat < elem.second->size(); feat++)
        {
            for(unsigned int i_dim = 0; i_dim < descriptor_dim; i_dim++)
            {
                output_file << elem.second->at(feat).descriptor.at(i_dim) << " ";
            }
        }
    }

    output_file.close ();

    return true;
}

bool Lnbnn::loadModelFromFile(std::string& filename)
{
    std::ifstream input_file (filename.c_str ());
    if (!input_file)
    {
        input_file.close ();
        return (false);
    }

    int number_of_features;
    int descriptor_dim;

    char line[256];

    input_file.getline (line, 256, ' ');
    m_number_of_classes = static_cast<unsigned int> (strtol (line, 0, 10));
    input_file.getline (line, 256, ' '); number_of_features = atoi (line);
    input_file.getline (line, 256, ' '); descriptor_dim = atoi (line);

    // read classes
    m_class_lookup.clear();
    m_class_lookup.resize (number_of_features, 0);
    for (unsigned int feat_i = 0; feat_i < number_of_features; feat_i++)
    {
        input_file >> m_class_lookup[feat_i];
    }

    // read features
    m_features->clear();
    for (unsigned int feat_i = 0; feat_i < number_of_features; feat_i++)
    {
        ISMFeature feature;
        feature.descriptor.resize(descriptor_dim);

        for (unsigned int dim_i = 0; dim_i < descriptor_dim; dim_i++)
        {
            input_file >> feature.descriptor[dim_i];
        }
        feature.classId = m_class_lookup.at(feat_i);
        m_features->push_back(feature);
    }

  input_file.close ();
  return (true);
}
