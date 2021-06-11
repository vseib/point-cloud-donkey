#include "pcl_ism.h"

#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>

//#include <pcl/recognition/impl/implicit_shape_model.hpp>
#include "implicit_shape_model.hpp"

#include <pcl/features/fpfh.h>
#include <pcl/features/impl/fpfh.hpp>
#include <pcl/features/shot.h>

/**
 * PCL implementation of the approach described in
 *
 * Jan Knopp, Mukta Prasad, Geert Willems, Radu Timofte, and Luc Van Gool:
 *     Hough Transforms and 3D SURF for robust three dimensional classication.
 *     2010, European Conference on Computer Vision (ECCV)
 *
 */

/**
 * also see:
 *      https://pcl.readthedocs.io/en/latest/implicit_shape_model.html#implicit-shape-model
 */

PclIsm::PclIsm()
{
    // use this for datasets: aim, mcg, psb, shrec-12, mn10, mn40
    m_normal_radius = 0.05;
    m_feature_radius = 0.4;
    m_keypoint_sampling_radius = 0.2;

    // use this for datasets: washington, bigbird, ycb
//    m_normal_radius = 0.005;
//    m_feature_radius = 0.05;
//    m_keypoint_sampling_radius = 0.02;

    m_model = boost::shared_ptr<pcl::features::ISMModel>(new pcl::features::ISMModel);
}


void PclIsm::train(const std::vector<std::string> &filenames,
                   const std::vector<unsigned> &labels,
                   const std::string &output_file) const
{
    if(filenames.size() != labels.size())
    {
        std::cerr << "ERROR: number of clouds does not match number of labels!" << std::endl;
        return;
    }

    pcl::NormalEstimation<PointT, pcl::Normal> normal_estimator;
    normal_estimator.setRadiusSearch(m_normal_radius);

    std::vector<pcl::PointCloud<PointT>::Ptr> training_clouds;
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> training_normals;
    std::vector<unsigned int> training_classes;

    for(int i = 0; i < filenames.size(); i++)
    {
        std::string file = filenames.at(i);
        unsigned int tr_class = labels.at(i);

        pcl::PointCloud<PointT>::Ptr tr_cloud(new pcl::PointCloud<PointT>());
        if(pcl::io::loadPCDFile(file, *tr_cloud) == -1)
        {
            std::cerr << "ERROR: loading file " << file << std::endl;
        }

        pcl::PointCloud<pcl::Normal>::Ptr tr_normals(new pcl::PointCloud<pcl::Normal>);
        normal_estimator.setInputCloud(tr_cloud);
        normal_estimator.compute(*tr_normals);

        training_clouds.push_back(tr_cloud);
        training_normals.push_back(tr_normals);
        training_classes.push_back(tr_class);
    }
    std::cout << "Loaded " << training_clouds.size() << " clouds for training. " << std::endl;

    pcl::FPFHEstimation<PointT, pcl::Normal, pcl::Histogram<153> >::Ptr fpfh
      (new pcl::FPFHEstimation<PointT, pcl::Normal, pcl::Histogram<153>>);
    fpfh->setRadiusSearch(m_feature_radius);
    pcl::Feature<PointT, pcl::Histogram<153> >::Ptr feature_estimator(fpfh);

    pcl::ism::ImplicitShapeModelEstimation<153, PointT, pcl::Normal> ism;
    ism.setFeatureEstimator(feature_estimator);
    ism.setTrainingClouds(training_clouds);
    ism.setTrainingNormals(training_normals);
    ism.setTrainingClasses(training_classes);
    ism.setClusterRate(1.0f);
    ism.setSamplingSize(m_keypoint_sampling_radius);

    pcl::ism::ImplicitShapeModelEstimation<153, PointT, pcl::Normal>::ISMModelPtr model =
            boost::shared_ptr<pcl::features::ISMModel>(new pcl::features::ISMModel);
    ism.trainISM(model);

    std::cout << "Model has " << model->number_of_clusters_ << " clusters." << std::endl;
    std::cout << "Model has " << model->number_of_visual_words_ << " visual words." << std::endl;
    std::cout << "Feature estimation completed, saving model to " << output_file << std::endl;
    std::string save_file(output_file);
    if(model->saveModelToFile(save_file))
    {
        std::cout << "ISM training finished!" << std::endl;
    }
    else
    {
        std::cerr << "ERROR saving ISM model!" << std::endl;
    }
}


std::vector<std::pair<unsigned, float>> PclIsm::classify(const std::string &filename) const
{
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if(pcl::io::loadPCDFile<PointT>(filename, *cloud) == -1)
      return std::vector<std::pair<unsigned, float>>();

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<PointT, pcl::Normal> normal_estimator;
    normal_estimator.setRadiusSearch(m_normal_radius);
    normal_estimator.setInputCloud(cloud);
    normal_estimator.compute(*normals);

    pcl::FPFHEstimation<PointT, pcl::Normal, pcl::Histogram<153> >::Ptr fpfh
      (new pcl::FPFHEstimation<PointT, pcl::Normal, pcl::Histogram<153> >);
    fpfh->setRadiusSearch(m_feature_radius);
    pcl::Feature<PointT, pcl::Histogram<153> >::Ptr feature_estimator(fpfh);

    pcl::ism::ImplicitShapeModelEstimation<153, PointT, pcl::Normal> ism;
    ism.setFeatureEstimator(feature_estimator);
    ism.setSamplingSize(m_keypoint_sampling_radius);
//    ism.setNVotState(false);

    std::vector<std::pair<unsigned, float>> results;
    for(const unsigned &label : m_class_labels)
    {
        unsigned class_label = label;
        boost::shared_ptr<pcl::features::ISMVoteList<PointT>> vote_list = ism.findObjects(m_model, cloud, normals, class_label);

        // ignore class if there are 0 votes
        if(vote_list->getNumberOfVotes() != 0)
        {
            double radius = m_model->sigmas_[class_label] * 10.0;
            double sigma = m_model->sigmas_[class_label] * 8.0;
            std::vector<pcl::ISMPeak, Eigen::aligned_allocator<pcl::ISMPeak>> strongest_peaks;
            vote_list->findStrongestPeaks(strongest_peaks, class_label, radius, sigma);

            for(const pcl::ISMPeak &peak : strongest_peaks)
            {
                results.push_back({(unsigned)peak.class_id, (float)peak.density});
            }
        }
    }

    std::sort(results.begin(), results.end(), [](const std::pair<unsigned, float> &a, const std::pair<unsigned, float> &b)
    {
        return a.second > b.second;
    });

    return results;
}


bool PclIsm::loadModel(std::string &filename)
{
    return (m_model->loadModelFromfile(filename));
}


void PclIsm::setClassLabels(const std::vector<unsigned> &class_labels)
{
    m_class_labels = class_labels;
}
