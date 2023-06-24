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
        m_global_feature_method = method;
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

    // TODO VS get rid of this method
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


    void GlobalClassifier::loadSVMModels(std::string &input_config_path, std::string &svm_path)
    {
        // load SVM for global features
        if(svm_path != "")
        {
            // in case ism is not loaded from current working directory, construct the real path
            std::size_t pos = input_config_path.find_last_of('/');
            // replace config file name by svm file name
            if(pos != std::string::npos)
            {
                svm_path = input_config_path.substr(0, pos+1) + svm_path;
            }

            // get path and check for errors
            boost::filesystem::path path(svm_path);
            boost::filesystem::path p_comp = boost::filesystem::complete(path);

            if(boost::filesystem::exists(p_comp) && boost::filesystem::is_regular_file(p_comp))
            {
                m_svm_files.clear();
                // check if multiple svm files are available (i.e. 1 vs all svm)
                if(svm_path.find("tar") != std::string::npos)
                {
                    LOG_INFO("Found 1 vs all SVM (binary)");
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
                    LOG_INFO("Found pairwise 1 vs 1 SVM (multiclass)");
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
            const int min_points,
            VotingMaximum &maximum)
    {
        // compute global features
        // require a minimum number of points for feature computation
        if(points->size() > min_points || min_points < 0)
        {
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
                classifyWithKNN(global_features, maximum);
            }
            else if(m_global_feature_method == "SVM")
            {
                LOG_INFO("starting global classification with svm");
                classifyWithSVM(global_features, maximum);
                // SVM does not support instance labels, so we have to get it from KNN classifier
                VotingMaximum instance_maximum;
                instance_maximum.classId = maximum.classId;
                // TODO VS: use only if
                // if( ... secondary labels ...) --> else: init instance id and weight with class id and weight
                // 1. secondary labels are used
                // 2. feature type of training and test is the same (usually this is the case, but it's possible
                //    to use an SVM with a different descriptor type than was used for training the global KNN classifier)
                if(global_features->at(0).descriptor.size() == m_global_features->at(0).descriptor.size())
                {
                    classifyWithKNN(global_features, instance_maximum);
                    // fill in instance result into actual maximum
                    maximum.globalHypothesis.instanceId = instance_maximum.globalHypothesis.instanceId;
                    maximum.globalHypothesis.instanceWeight = instance_maximum.globalHypothesis.instanceWeight;
                }
                else
                {
                    LOG_ERROR("ERROR: Loaded descriptors and computed descriptors do not match in dimensionality!");
                    LOG_ERROR("       Loaded features have dimension " << m_global_features->at(0).descriptor.size() <<
                              " while computed features have " << global_features->at(0).descriptor.size() << "!");
                    LOG_ERROR("Check your config (global feature type vs. SVM feature type)!");
                    exit(1);
                }
            }
        }
        else
        {
            LOG_WARN("----------- DEBUG TODO VS ---------- skipping global feature, with num points = " << points->size());
            // assign global hypothesis with zero weight
            VotingMaximum::GlobalHypothesis glob;
            glob.classId = maximum.classId;
            glob.classWeight = 0.0f;
            glob.instanceId = maximum.instanceId;
            glob.instanceWeight = 0.0f;
            maximum.globalHypothesis = glob;
        }
    }

    void GlobalClassifier::classifyWithKNN(pcl::PointCloud<ISMFeature>::ConstPtr global_features,
                                           VotingMaximum &maximum)
    {
        std::map<unsigned, GlobalResultAccu> max_global_voting; // maps class id to struct with number of occurences and score

        // find nearest neighbors to current global features in learned data
        // NOTE: some global features produce more than 1 descriptor per object, hence the loop
        for(ISMFeature query_feature : global_features->points)
        {
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
                // TODO VS: occasional segfault on knn search: check if flann helper is used correctly for global features (because it never happens with local features)
                m_flann_helper->getIndexChi()->knnSearch(query, indices, distances, m_k_global_features, flann::SearchParams(-1));
            }
            delete[] query.ptr();

            // classic KNN approach
            // loop over results
            for(int i = 0; i < indices[0].size(); i++)
            {
                // insert result
                ISMFeature temp = m_global_features->at(indices[0].at(i));
                float dist_squared = distances[0].at(i);
                // TODO VS which score is better?
                float score = std::exp(-sqrt(dist_squared));
//                float score = 1.0 / (1.0 + std::exp(-sqrt(dist_squared)));
                insertGlobalResult(max_global_voting, temp.classId, temp.instanceId, score);
            }
        }

        // determine score based on all votes
        VotingMaximum::GlobalHypothesis global_result = {maximum.classId, 0}; // pair of class id and score
        unsigned max_occurences = 0;

        if(m_single_object_mode)
        {
            // find class with most occurences
            for(auto it : max_global_voting)
            {
                GlobalResultAccu gra = it.second;
                if(gra.num_occurences > max_occurences)
                {
                    max_occurences = gra.num_occurences;
                    global_result.classId = it.first;
                }
            }
            // TODO VS - test again: determining best class based on score because score computation was changed!
            // use best class and compute score (NOTE: determining best class based on score did not work well)
            GlobalResultAccu gra = max_global_voting.at(global_result.classId);
            global_result.classWeight = gra.score_sum / gra.num_occurences;

            // find instance with most occurences
            max_occurences = 0;
            for(auto it : gra.instance_ids)
            {
                std::pair<int, float> elem = it.second;
                if(elem.first > max_occurences)
                {
                    max_occurences = elem.first;
                    global_result.instanceId = it.first;
                }
            }
            std::pair<int, float> elem = gra.instance_ids.at(global_result.instanceId);
            global_result.instanceWeight = elem.second / elem.first;
        }
        else
        {
            // determine score based on current class
            if(max_global_voting.find(maximum.classId) != max_global_voting.end())
            {
                GlobalResultAccu gra = max_global_voting.at(maximum.classId);
                global_result.classWeight = gra.num_occurences > 0 ? gra.score_sum / gra.num_occurences : 0;

                // find instance with most occurences
                max_occurences = 0;
                for(auto it : gra.instance_ids)
                {
                    std::pair<int, float> elem = it.second;
                    if(elem.first > max_occurences)
                    {
                        max_occurences = elem.first;
                        global_result.instanceId = it.first;
                    }
                }
                std::pair<int, float> elem = gra.instance_ids.at(global_result.instanceId);
                global_result.instanceWeight = elem.second / elem.first;
            }
        }

        // assign global result
        maximum.globalHypothesis = global_result;
    }

    void GlobalClassifier::classifyWithSVM(pcl::PointCloud<ISMFeature>::ConstPtr global_features,
                                           VotingMaximum &maximum)
    {
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
                // for now passing class label as instance label, will be replaced later
                insertGlobalResult(global_result_per_class, (unsigned) resp.label, resp.score, (unsigned) resp.label);
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
        if(m_single_object_mode)
        {
            // just get the best overall result
            maximum.globalHypothesis.classId = (unsigned)svm_response.label;
            maximum.globalHypothesis.classWeight = svm_response.score;
        }
        else
        {
            // get the result for current class id
            maximum.globalHypothesis.classId = maximum.classId;
            maximum.globalHypothesis.classWeight = svm_response.all_scores.at(maximum.classId);
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
                                            unsigned instance_id,
                                            float score) const
    {
        if(max_global_voting.find(found_class) != max_global_voting.end())
        {
            // found
            GlobalResultAccu &prev = max_global_voting.at(found_class);
            prev.num_occurences++;
            prev.score_sum += score;
            prev.insertInstanceLabel(instance_id, score);
        }
        else
        {
            // not found
            GlobalResultAccu gra = GlobalResultAccu(1, score, instance_id);
            max_global_voting.insert({found_class, gra});
        }
    }

    void GlobalClassifier::mergeGlobalAndLocalHypotheses(const int merge_function,
                                                         std::vector<VotingMaximum> &maxima)
    {
        if(merge_function >= 1 && merge_function <= 3 && !m_single_object_mode)
        {
            LOG_WARN("Merge functions 1, 2 and 3 are defined properly only in single object mode, which you did not enable!");
        }

        if(merge_function == 1)
        {
            // type 1: blind belief in good scores
            if(maxima.at(0).globalHypothesis.classWeight > m_min_svm_score)
            {
                maxima.at(0).classId = maxima.at(0).globalHypothesis.classId;
                maxima.at(0).instanceId = maxima.at(0).globalHypothesis.instanceId;
            }
        }
        else if(merge_function == 2) // this method's name in the phd thesis: fm2
        {
            // type 2: belief in good scores if global class is among the top classes
            if(maxima.at(0).globalHypothesis.classWeight > m_min_svm_score)
            {
                useHighRankedGlobalHypothesis(maxima);
            }
        }
        else if(merge_function == 3) // this method's name in the phd thesis: fm1
        {
            // type 3: take global class if it is among the top classes
            useHighRankedGlobalHypothesis(maxima);
        }
        else if(merge_function == 4) // this method's name in the phd thesis: fm3
        {
            // type 4: upweight consistent results by fixed factor
            for(VotingMaximum &max : maxima)
            {
                if(max.classId == max.globalHypothesis.classId)
                {
                    if(max.globalHypothesis.classWeight == 0)
                        max.weight = 0;
                    else
                        max.weight *= m_weight_factor;
                }
                if(max.instanceId == max.globalHypothesis.instanceId)
                {
                    if(max.globalHypothesis.instanceWeight == 0)
                        max.instanceWeight = 0;
                    else
                        max.instanceWeight *= m_weight_factor;
                }
            }
        }
        else if(merge_function == 5) // this method's name in the phd thesis: fm4
        {
            // type 5: upweight consistent results depending on weight
            for(VotingMaximum &max : maxima)
            {
                if(max.classId == max.globalHypothesis.classId)
                    max.weight *= 1 + max.globalHypothesis.classWeight;
                if(max.instanceId == max.globalHypothesis.instanceId)
                    max.instanceWeight *= 1 + max.globalHypothesis.instanceWeight;
            }
        }
        else if(merge_function == 6)
        {
            // type 6: multiply weights
            for(VotingMaximum &max : maxima)
            {
                if(max.classId == max.globalHypothesis.classId)
                    max.weight *= max.globalHypothesis.classWeight;
                if(max.instanceId == max.globalHypothesis.instanceId)
                    max.instanceWeight *= max.globalHypothesis.instanceWeight;
            }
        }
        else if(merge_function == 7)  // this method's name in the phd thesis: fm5
        {
            // type 7: apply intermediate T-conorm: S(a,b) = a+b-ab
            for(VotingMaximum &max : maxima)
            {
                if(max.classId == max.globalHypothesis.classId)
                {
                    float w1 = max.weight;
                    float w2 = max.globalHypothesis.classWeight;
                    max.weight = w1+w2 - w1*w2;

                    if(max.instanceId == max.globalHypothesis.instanceId)
                    {
                        w1 = max.instanceWeight;
                        w2 = max.globalHypothesis.instanceWeight;
                        max.instanceWeight = w1+w2 - w1*w2;
                    }
                }
            }
        }
        else
        {
            LOG_ERROR("Invalid merging function specified: " << merge_function <<"! Not merging local and global hypotheses.");
        }
    }

    void GlobalClassifier::useHighRankedGlobalHypothesis(std::vector<VotingMaximum> &maxima)
    {
        float top_weight = maxima.at(0).weight;
        int global_class = maxima.at(0).globalHypothesis.classId;

        // check if global class is among the top classes
        for(int i = 0; i < maxima.size(); i++)
        {
            float cur_weight = maxima.at(i).weight;
            int cur_class = maxima.at(i).classId;

            if(cur_weight >= top_weight * m_rate_limit && cur_class == global_class)
            {
                maxima.at(0).classId = maxima.at(0).globalHypothesis.classId;
                maxima.at(0).instanceId = maxima.at(0).globalHypothesis.instanceId;
                break;
            }
            else if(cur_weight < top_weight * m_rate_limit)
            {
                break;
            }
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
