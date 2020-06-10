/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2020, Viktor Seib
 * All rights reserved.
 *
 */

#include "global_classifier.h"

#define PCL_NO_PRECOMPILE
#include <pcl/filters/extract_indices.h>

namespace ism3d
{
    std::string exec(const char *cmd); // defined at the end of file

    GlobalClassifier::GlobalClassifier(Features* global_descriptor,
                                       std::string method,
                                       int k_global)
    {
        m_feature_algorithm = global_descriptor;
        m_global_feature_method = method; // TODO VS replace by enum
        m_k_global_features = k_global;

        m_single_object_mode = false;
        m_index_created = false;
        m_svm_error = false;
    }

    GlobalClassifier::~GlobalClassifier()
    {
        // delete files that were unpacked for recognition
        if(m_svm_files.size() > 1)
        {
            for(std::string s : m_svm_files)
            {
                std::ignore = std::system(("rm " + s).c_str());
            }
        }
    }

    void GlobalClassifier::computeAverageRadii(
            // maps class ids to a vector of global features,
            // number of objects per class = number of global features per class
            std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> &global_features)
    {
        // compute average radii of each class from training
        for(auto it = global_features.begin(); it != global_features.end(); it++)
        {
            float avg_radius = 0;
            int num_points = 0;
            unsigned classID = it->first;

            std::vector<pcl::PointCloud<ISMFeature>::Ptr> cloud_vector = it->second;
            for(auto cloud : cloud_vector)
            {
                for(ISMFeature ism_feature : cloud->points)
                {
                    avg_radius += ism_feature.globalDescriptorRadius;
                    num_points += 1;
                }
            }
            m_average_radii.insert({classID, avg_radius / num_points});
        }
    }


    void GlobalClassifier::loadSVMModels(std::string &svm_path)
    {
        // load SVM for global features
        if(svm_path != "")
        {
            // get path and check for errors
            boost::filesystem::path path(svm_path);
            boost::filesystem::path p_comp = boost::filesystem::complete(path);

            if(boost::filesystem::exists(p_comp) && boost::filesystem::is_regular_file(p_comp))
            {
                m_svm_files.clear();
                // check if multiple svm files are available (i.e. 1 vs all svm)
                if(svm_path.find("tar") != std::string::npos)
                {
                    // show the content of the tar file
                    std::string result = exec(("tar -tf " + p_comp.string()).c_str());
                    // split the string and add to list
                    std::stringstream sstr;
                    sstr.str(result);
                    std::string item;
                    while (std::getline(sstr, item, '\n'))
                    {
                        boost::filesystem::path paths(item);
                        boost::filesystem::path ppp = boost::filesystem::complete(paths);
                        m_svm_files.push_back(ppp.string());
                    }
                    // unzip tar file
                    int ret = std::system(("tar -xzf " + p_comp.string()).c_str());
                    sleep(2);
                }
                else
                {
                    // only one file: standard OpenCV SVM (i.e. pairwise 1 vs 1 svm)
                    m_svm_files.push_back(p_comp.string());
                }
            }
            else
            {
                LOG_ERROR("SVM file not valid or missing!");
                m_svm_error = true;
            }
        }
        else
        {
            LOG_ERROR("SVM path is NULL!");
            m_svm_error = true;
        }
    }

    void GlobalClassifier::segmentROI(
            const pcl::PointCloud<PointT>::ConstPtr &points,
            const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
            const ism3d::VotingMaximum &maximum,
            pcl::PointCloud<PointT>::Ptr &segmented_points,
            pcl::PointCloud<pcl::Normal>::Ptr &segmented_normals)
    {
        // used to extract a portion of the input cloud to estimage a global feature
        pcl::KdTreeFLANN<PointT> input_points_kdtree;
        input_points_kdtree.setInputCloud(points);

        // first segment region cloud from input with typical radius for this class id
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        PointT query;
        float radius = m_average_radii.at(maximum.classId);
        query.x = maximum.position.x();
        query.y = maximum.position.y();
        query.z = maximum.position.z();

        if(input_points_kdtree.radiusSearch(query, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
        {
            // segment points
            pcl::ExtractIndices<PointT> extract;
            extract.setInputCloud(points);
            pcl::PointIndices::Ptr indices (new pcl::PointIndices());
            indices->indices = pointIdxRadiusSearch;
            extract.setIndices(indices);
            extract.filter(*segmented_points);
            // segment normals
            pcl::ExtractIndices<pcl::Normal> extract_normals;
            extract_normals.setInputCloud(normals);
            extract_normals.setIndices(indices); // use same indices
            extract_normals.filter(*segmented_normals);
        }
        else
        {
            LOG_WARN("Error during nearest neighbor search.");
        }
    }

    void GlobalClassifier::classify(
            const pcl::PointCloud<PointT>::ConstPtr &points,
            const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
            VotingMaximum &maximum)
    {
        // compute global features
        pcl::PointCloud<ISMFeature>::ConstPtr global_features = computeGlobalFeatures(points, normals);

        // if no SVM data available defaul to KNN
        if(m_svm_error) m_global_feature_method = "KNN";

        // process current global features according to some strategy
        if(!m_index_created)
        {
            LOG_INFO("creating flann index for global features");
            m_flann_helper->buildIndex(m_distance_type, 1);
            m_index_created = true;
        }

        if(m_global_feature_method == "KNN")
        {
            LOG_INFO("starting global classification with knn");
            std::map<unsigned, GlobalResultAccu> max_global_voting; // maps class id to struct with number of occurences and score
            int num_all_entries = 0;

            // find nearest neighbors to current global features in learned data
            // NOTE: some global features produce more than 1 descriptor per object, hence the loop
            for(ISMFeature query_feature : global_features->points)
            {
                LOG_INFO("   query_feature.descriptor.size() " << query_feature.descriptor.size());
                // insert the query point
                flann::Matrix<float> query(new float[query_feature.descriptor.size()], 1, query_feature.descriptor.size());
                for(int i = 0; i < query_feature.descriptor.size(); i++)
                {
                    query[0][i] = query_feature.descriptor.at(i);
                }

                // search
                std::vector<std::vector<int>> indices;
                std::vector<std::vector<float>> distances;
                if(m_flann_helper->getDistType() == "Euclidean")
                {
                    m_flann_helper->getIndexL2()->knnSearch(query, indices, distances, m_k_global_features, flann::SearchParams(-1));
                }
                else if(m_flann_helper->getDistType() == "ChiSquared")
                {
                    m_flann_helper->getIndexChi()->knnSearch(query, indices, distances, m_k_global_features, flann::SearchParams(-1));
                }

                delete[] query.ptr();

                // classic KNN approach
                num_all_entries += indices[0].size(); // NOTE: is not necessaraly k, because only (k-x) might have been found
                // loop over results

                for(int i = 0; i < indices[0].size(); i++)
                {
                    // insert result
                    ISMFeature temp = m_global_features->at(indices[0].at(i));
                    float dist_squared = distances[0].at(i);
                    const float sigma = 0.1;
                    const float denom = 2 * sigma * sigma;
                    float score = std::exp(-dist_squared/denom);
                    insertGlobalResult(max_global_voting, temp.classId, score);
                }
            }

            std::pair<unsigned, float> global_result = {maximum.classId, 0}; // pair of class id and score
            // determine score based on all votes
            unsigned max_occurences = 0;
            if(m_single_object_mode)
            {
                // find class with most occurences
                for(auto it : max_global_voting)
                {
                    if(it.second.num_occurences > max_occurences)
                    {
                        max_occurences = it.second.num_occurences;
                        global_result.first = it.first;
                    }
                }
                // compute score for best class (NOTE: determining best class based on score did not work well)
                GlobalResultAccu gra = max_global_voting.at(global_result.first);
                global_result.second = gra.score_sum / gra.num_occurences;
            }
            else // determine score based on current class
            {
                if(max_global_voting.find(maximum.classId) != max_global_voting.end())
                {
                    GlobalResultAccu gra = max_global_voting.at(maximum.classId);
                    global_result.second = gra.num_occurences > 0 ? gra.score_sum / gra.num_occurences : 0;
                }
            }

            // assign global result
            maximum.globalHypothesis = global_result;
            maximum.currentClassHypothesis = global_result;
        }
        else if(m_global_feature_method == "SVM")
        {
            LOG_INFO("starting global classification with svm");
            CustomSVM::SVMResponse svm_response;
            std::vector<CustomSVM::SVMResponse> all_responses; // in case one object has multiple global features
            // NOTE: some global features produce more than 1 descriptor per object, hence the loop
            for(ISMFeature query_feature : global_features->points)
            {
                // convert to SVM data format
                std::vector<float> data_raw = query_feature.descriptor;
                float data[data_raw.size()];
                for(unsigned i = 0; i < data_raw.size(); i++)
                {
                    data[i] = data_raw.at(i);
                }
                cv::Mat svm_input_data(1, data_raw.size(), CV_32FC1, data);

                CustomSVM::SVMResponse temp_response = m_svm.predictUnifyScore(svm_input_data, m_svm_files);
                all_responses.push_back(temp_response);
            }

            // check if several responses are available
            if(all_responses.size() > 1)
            {
                std::map<unsigned, GlobalResultAccu> global_result_per_class;
                for(const CustomSVM::SVMResponse &resp : all_responses)
                {
                    insertGlobalResult(global_result_per_class, (unsigned) resp.label, resp.score);
                }

                int best_class = 0;
                int best_occurences = 0;
                for(const auto it : global_result_per_class)
                {
                    // NOTE: what if there are 2 equal classes?
                    if(it.second.num_occurences > best_occurences) // find highest number of occurences
                    {
                        best_occurences = it.second.num_occurences;
                        best_class = it.first;
                    }
                }

                // find best class in list of responses with "best" (highest) score
                float best_score = std::numeric_limits<float>::min();
                CustomSVM::SVMResponse best_response = all_responses.at(0); // init with first value
                for(int i = 0; i < all_responses.size(); i++)
                {
                    if(all_responses.at(i).label == best_class)
                    {
                        if(all_responses.at(i).score > best_score)
                        {
                            best_score = all_responses.at(i).score;
                            best_response = all_responses.at(i);
                        }
                    }
                }
                svm_response = best_response;
            }
            else if(all_responses.size() == 1)
            {
                svm_response = all_responses.at(0);
            }

            // assign global result
            float cur_score = m_single_object_mode ? 0 : svm_response.all_scores.at(maximum.classId);
            maximum.globalHypothesis = {svm_response.label, svm_response.score};
            maximum.currentClassHypothesis = {maximum.classId, cur_score};
        }
    }

    pcl::PointCloud<ISMFeature>::ConstPtr GlobalClassifier::computeGlobalFeatures(const pcl::PointCloud<PointT>::ConstPtr points,
                                                 const pcl::PointCloud<pcl::Normal>::ConstPtr normals)
    {
        pcl::PointCloud<PointT>::ConstPtr dummy_keypoints(new pcl::PointCloud<PointT>());
        pcl::search::Search<PointT>::Ptr search = pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>());
        return (*m_feature_algorithm)(points, normals, points, normals, dummy_keypoints, search);
    }

    void GlobalClassifier::insertGlobalResult(std::map<unsigned, ism3d::GlobalResultAccu> &max_global_voting,
                                    unsigned found_class,
                                    float score) const
    {
        if(max_global_voting.find(found_class) != max_global_voting.end())
        {
            // found
            GlobalResultAccu &prev = max_global_voting.at(found_class);
            prev.num_occurences++;
            prev.score_sum += score;
        }
        else
        {
            // not found
            max_global_voting.insert({found_class, {1,score} });
        }
    }

    // NOTE: from http://stackoverflow.com/questions/478898/how-to-execute-a-command-and-get-output-of-command-within-c-using-posix
    std::string exec(const char* cmd)
    {
        std::array<char, 128> buffer;
        std::string result;
        std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
        if (!pipe) throw std::runtime_error("popen() failed!");
        while (!feof(pipe.get())) {
            if (fgets(buffer.data(), 128, pipe.get()) != NULL)
                result += buffer.data();
        }
        return result;
    }
}
