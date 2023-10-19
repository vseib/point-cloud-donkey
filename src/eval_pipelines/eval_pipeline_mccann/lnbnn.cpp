#include "lnbnn.h"

#include <pcl/io/pcd_io.h>
#include <pcl/correspondence.h>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>

#include "../pipeline_building_blocks/pipeline_building_blocks.h"
#include "../pipeline_building_blocks/feature_processing.h" // provides namespace fp::

/**
 * Implementation of the approach described in
 *
 * S. McCann, D.G. Lowe:
 *     Local naive bayes nearest neighbor for image classification.
 *     2012, Conference on Computer Vision and Pattern Recognition (CVPR)
 *
 */


Lnbnn::Lnbnn() : m_features(new pcl::PointCloud<ISMFeature>()), m_flann_index(flann::KDTreeIndexParams(4))
{
    // use this for datasets: aim, mcg, psb, shrec-12, mn10, mn40
//    fp::normal_radius = 0.05;
//    fp::reference_frame_radius = 0.3;
//    fp::feature_radius = 0.4;
//    fp::keypoint_sampling_radius = 0.25;
//    fp::normal_method = 1;
//    fp::feature_type = "SHOT";

    // use this for datasets: washington, bigbird, ycb
    fp::normal_radius = 0.005;
    fp::reference_frame_radius = 0.04;
    fp::feature_radius = 0.06;
    fp::keypoint_sampling_radius = 0.02;
    fp::normal_method = 0;
    fp::feature_type = "CSHOT";
}


void Lnbnn::train(const std::vector<std::string> &filenames,
                  const std::vector<unsigned> &class_labels,
                  const std::vector<unsigned> &instance_labels,
                  const std::string &output_file) const
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
            ismf.classId = tr_class;
            ismf.instanceId = tr_instance;
        }

        // add computed features to map
        (*all_features.at(tr_class)) += (*features);
        num_features += features->size();
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

    // all these pointers are initialized within the called method
    pcl::PointCloud<PointT>::Ptr keypoints;
    pcl::PointCloud<ISMFeature>::Ptr features;
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr reference_frames;
    processPointCloud(cloud, keypoints, features, normals, reference_frames);

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


std::vector<std::pair<unsigned, float>> Lnbnn::accumulateClassDistances(
        const pcl::PointCloud<ISMFeature>::Ptr& features) const
{
    std::vector<std::pair<unsigned, float>> class_distances;
    for(int i = 0; i < m_number_of_classes; i++)
    {
        class_distances.push_back({i, 0.0f});
    }

    int k_search = 11;

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


bool Lnbnn::saveModelToFile(std::string &filename,
                            std::map<unsigned, pcl::PointCloud<ISMFeature>::Ptr> &all_features) const
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
            for(unsigned int i_dim = 0; i_dim < descriptor_dim; i_dim++)
            {
                float temp = elem.second->at(feat).descriptor.at(i_dim);
                oa << temp;
            }
        }
    }

    ofs.close();
    return true;
}

bool Lnbnn::loadModelFromFile(std::string& filename)
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

            for (unsigned int dim_i = 0; dim_i < descriptor_dim; dim_i++)
            {
                ia >> feature.descriptor[dim_i];
            }
            feature.classId = m_class_lookup.at(feat_i);
            m_features->push_back(feature);
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
