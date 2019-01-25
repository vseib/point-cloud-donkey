/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */


#include "custom_SVM.h"
#include <iostream>
#include <iomanip>
#include <set>
#include "../utils/utils.h"

CustomSVM::CustomSVM(std::string output_file_name)
{    
    m_output_file_name = output_file_name;
}

CustomSVM::~CustomSVM()
{
}

void CustomSVM::setData(std::vector< std::vector<float> > &training_data, std::vector<int> &labels)
{
    m_training_data = training_data;
    m_labels = labels;
    std::set<int> unique;

    for(float lab : labels)
        unique.insert(lab);

    m_num_classes = unique.size();
}

int CustomSVM::prepareTrainingData(int run_id, int num_runs)
{
    // get dimensions
    m_num_features = m_training_data.size();
    int dim_feature;
    if(m_num_features > 1)
    {
        dim_feature = (m_training_data.at(0)).size();
    }
    else
    {
        LOG_ERROR("No data to train SVM available - no global features?");
    }

    // determine label for training this iteration
    int train_label = 0;
    if(num_runs != 1) // 1 vs all training
    {
        int distinct_label_counter = 0;
        int cur_label = -1;

        for(int lab : m_labels)
        {
            // if label changes, increment label counter
            if(distinct_label_counter == 0 || lab != cur_label)
            {
                cur_label = lab;
                distinct_label_counter++;
            }

            if(distinct_label_counter == run_id)
            {
                train_label = lab;
                break;
            }
        }
    }

    // convert to SVM format
    int labels_arr[m_num_features];
    for(int i = 0; i < m_num_features; i++)
    {
        if(num_runs != 1 && m_labels.at(i) != train_label) // only do this check in 1 vs all training
            labels_arr[i] = -1;
        else
            labels_arr[i] = m_labels.at(i);
    }
    cv::Mat labelsMat(m_num_features, 1, CV_32SC1, labels_arr);

    float training_data_arr[m_num_features][dim_feature];
    for(int i = 0; i < m_num_features; i++)
    {
        for(int j = 0; j < dim_feature; j++)
        {
            training_data_arr[i][j] = (float)(m_training_data.at(i)).at(j);
        }
    }
    cv::Mat trainingDataMat(m_num_features, dim_feature, CV_32FC1, training_data_arr);

    m_labels_mat = labelsMat.clone();
    m_train_data_mat = trainingDataMat.clone();

    return train_label;
}


void CustomSVM::trainSimple(cv::SVMParams svm_params, bool one_vs_all)
{
    if(m_training_data.size() == 0)
    {
        LOG_INFO("No training data set! Training not possible!");
    }

    int num_runs = 1; // for normal training
    if(one_vs_all)
    {
        num_runs = m_num_classes; // for 1 vs all
    }

    std::vector<std::string> saved_files;

    for(int i = 0; i < num_runs; i++)
    {
        LOG_INFO("training loop " << (i+1) << " of " << num_runs);
        clear();
        int train_label = prepareTrainingData(i+1, num_runs);

        train(m_train_data_mat, m_labels_mat, cv::Mat(), cv::Mat(), svm_params);

        // save trained file
        std::stringstream sstr;
        std::string complete_name;
        if(num_runs == 1)
            sstr << "";
        else
            sstr << "_" << std::setfill('0') << std::setw(5) << std::to_string(train_label);

        complete_name = m_output_file_name + sstr.str() + ".svm";
        saved_files.push_back(complete_name);
        save(complete_name.c_str());
    }

    // if several files were generated ...
    if(saved_files.size() > 1)
    {
        // ... put them into a .tar
        std::stringstream sstr;
        sstr << "tar -czf " << m_output_file_name + ".svm.tar.gz";
        for(std::string s : saved_files)
            sstr << " " << s;
        int ignored = std::system(sstr.str().c_str());
        // ... then delete the individual files
        for(std::string s : saved_files)
            ignored = std::system(("rm "+ s).c_str());
    }
}


void CustomSVM::trainAutomatically(cv::SVMParams svm_params, int k_fold, bool one_vs_all)
{
    if(m_training_data.size() == 0)
    {
        LOG_ERROR("No training data set! Training not possible!");
    }

    int num_runs = 1; // for normal training
    if(one_vs_all)
    {
        num_runs = m_num_classes; // for 1 vs all
    }

    std::vector<std::string> saved_files;

    //#pragma omp parallel for
    for(int i = 0; i < num_runs; i++)
    {
        LOG_INFO("training loop " << (i+1) << " of " << num_runs);
        clear();
        int train_label = prepareTrainingData(i+1, num_runs);

        // set up search grid for parameter optimization
        CvParamGrid c_grid = get_default_grid(CvSVM::C);
        c_grid.min_val = 0.00001;
        c_grid.max_val = 4096;
        c_grid.step = 2;
        // if using too many features increase step size
        if(m_num_features > 1000)
        {
            c_grid.min_val = 0.001;
            c_grid.max_val = 1000;
            c_grid.step = 10;
        }

        CvParamGrid gamma_grid = get_default_grid(CvSVM::GAMMA);
        gamma_grid.min_val = 0.000001;
        gamma_grid.max_val = 8;
        gamma_grid.step = sqrt(2);
        // if using too many features increase step size
        if(m_num_features > 1000)
        {
            gamma_grid.min_val = 0.0001;
            gamma_grid.max_val = 10;
            gamma_grid.step = 10;
        }

        train_auto(m_train_data_mat, m_labels_mat, cv::Mat(), cv::Mat(), svm_params, k_fold, c_grid, gamma_grid);

        cv::SVMParams best_params = get_params();
        LOG_INFO("    SVM best params are: C: " << best_params.C << ", gamma: " << best_params.gamma);

        // ------------ use finer grid ------------
        LOG_INFO("auto-training SVM with finer grid");

        c_grid.min_val = best_params.C / (c_grid.step * c_grid.step);
        if(c_grid.min_val < 0.00001)
        {
            LOG_INFO("    c grid min too small: " << c_grid.min_val);
            c_grid.min_val = 0.00001;
        }
        c_grid.max_val = best_params.C * (c_grid.step * c_grid.step);
        c_grid.step = sqrt(c_grid.step);

        gamma_grid.min_val = best_params.gamma / (gamma_grid.step * gamma_grid.step);
        if(gamma_grid.min_val < 0.0001)
        {
            LOG_INFO("    gamma grid min too small: " << gamma_grid.min_val);
            gamma_grid.min_val = 0.0001;
        }
        gamma_grid.max_val = best_params.gamma * (gamma_grid.step * gamma_grid.step);
        gamma_grid.step = sqrt(gamma_grid.step);

        LOG_INFO("    C grid (min/max/step): " << c_grid.min_val << " " << c_grid.max_val << " " << c_grid.step);
        LOG_INFO("    Gamma grid (min/max/step): " << gamma_grid.min_val << " " << gamma_grid.max_val << " " << gamma_grid.step);

        // only refine grid with valid parameters
        if(c_grid.min_val < c_grid.max_val && gamma_grid.min_val < gamma_grid.max_val)
        {
            train_auto(m_train_data_mat, m_labels_mat, cv::Mat(), cv::Mat(), svm_params, k_fold, c_grid, gamma_grid);

            best_params = get_params();
            LOG_INFO("    SVM best params after fine grid are: C: " << best_params.C << ", gamma: " << best_params.gamma);
        }
        else
        {
            LOG_INFO("    skipping grid refinement ...");
        }

        // save trained file
        std::stringstream sstr;
        std::string complete_name;
        if(num_runs == 1)
            sstr << "";
        else
            sstr << "_" << std::setfill('0') << std::setw(5) << std::to_string(train_label);

        complete_name = m_output_file_name+sstr.str()+".svm";
        //#pragma omp critical
        {
            saved_files.push_back(complete_name);
        }
        save(complete_name.c_str());
    }

    // if several files were generated ...
    if(saved_files.size() > 1)
    {
        // ... put them into a .tar
        std::stringstream sstr;
        sstr << "tar -czf " << m_output_file_name + ".svm.tar.gz";
        for(std::string s : saved_files)
            sstr << " " << s;
        int ignored = std::system(sstr.str().c_str());
        // ... then delete the individual files
        for(std::string s : saved_files)
            ignored = std::system(("rm "+ s).c_str());
    }
}

CustomSVM::SVMResponse CustomSVM::predictUnifyScore(cv::Mat test_data, std::vector<std::string> &svm_files)
{
    CustomSVM::SVMResponse response;

    if(svm_files.size() > 1)
    {
        // OpenCV multiple SVMs to simulate an 1 vs all SVM
        response = predictWithScore(test_data, svm_files); // the lower the score, the better

        // switch sign and normalize to [0|1] to make score compatible with other score
        for(float &sc : response.all_scores)
        {
            sc = (sc * (-1) + 1) * 0.5;
        }
        response.score = response.all_scores[response.label];
    }
    else if(svm_files.size() == 1)
    {
        // OpenCV SVM with additional score
        response = predictWithScore(test_data, svm_files[0]);
    }
    else
        LOG_ERROR("no svm files provided for SVM classification!");

    return response;
}



// NOTE: see https://github.com/opencv/opencv/blob/master/modules/ml/src/svm.cpp
CustomSVM::SVMResponse CustomSVM::predictWithScore(cv::Mat test_data, std::vector<std::string> &svm_files)
{
    // call each 2 class svm
    m_num_classes = svm_files.size();
    std::pair<int, float> best_result = {-1, 1};
    m_scores.resize(m_num_classes);
    for(std::string svm_string : svm_files)
    {
        // predict result
        clear();
        load(svm_string.c_str());
        float score = predict(test_data, true);
        // get label
        int start = svm_string.find_last_of('_');
        int end = svm_string.find_last_of('.');
        std::string label_string = svm_string.substr(start+1, end);
        int label = std::atoi(label_string.c_str());
        // store result
        if(score < best_result.second) // the smaller the score, the better; positive score means: object not recognized
        {
            best_result = {label, score};
        }
        m_scores.at(label) = score;
    }

    // create result object
    CustomSVM::SVMResponse svm_response;
    svm_response.label = best_result.first;
    svm_response.score = best_result.second;
    svm_response.all_scores = m_scores;
    return svm_response;
}


// NOTE: see https://github.com/opencv/opencv/blob/master/modules/ml/src/svm.cpp
CustomSVM::SVMResponse CustomSVM::predictWithScore(cv::Mat test_data, std::string &svm_file)
{
    // load svm
    clear();
    load(svm_file.c_str());

    // get important parameters
    int feature_dim = test_data.cols; // size of descriptor
    m_num_classes = class_labels->cols;
    int num_sv = get_support_vector_count();
    float gamma = -params.gamma;

    // calc RBF-kernel result vector
    std::vector<float> kernel_vector(num_sv, 0); // holds weights for each support vector

    for(int i = 0; i < num_sv; i++)
    {
        float s = 0;
        const float *supportVector = get_support_vector(i);

        for(int j = 0; j < feature_dim; j++)
        {
            float t = supportVector[j] - test_data.at<float>(0,j);
            s += t*t;
        }
        kernel_vector.at(i) = std::exp(s * gamma);
    }

    // calc votes and distances
    std::vector<int> class_votes(m_num_classes, 0);
    m_sums_of_sigmoids = std::vector<float>(m_num_classes, 0);

    int dfi = 0;
    for(int i = 0; i < m_num_classes; i++)
    {
        for(int j = i+1; j < m_num_classes; j++, dfi++)
        {
            const CvSVMDecisionFunc df = decision_func[dfi];
            double sum = -df.rho;

            int sv_count = df.sv_count;
            double *alpha = df.alpha;
            int* sv_index = df.sv_index;

            for(int k = 0; k < sv_count; k++)
            {
                sum += alpha[k] * kernel_vector.at(sv_index[k]);
            }

            // vote for class
            if(sum > 0)
                class_votes.at(i)++;
            else
                class_votes.at(j)++;

            // store confidence for classes
            m_sums_of_sigmoids.at(i) += sigmoid(sum);
            m_sums_of_sigmoids.at(j) += sigmoid(-sum);
        }
    }

    // find class index with most votes
    int k = 0;
    std::vector<int> same_votes_idx; // holds all indices with the same (highest) number of votes
    for(int i = 0; i < m_num_classes; i++)
    {
        if(class_votes.at(i) > class_votes.at(k))
        {
            k = i;
            same_votes_idx.clear();
            same_votes_idx.push_back(i);
        }
        else if(class_votes.at(i) == class_votes.at(k))
        {
            same_votes_idx.push_back(i);
        }
    }

    // finally the result (this is the result of the standard OpenCV SVM)
    cv::Mat labels(class_labels);
    int result_label_std_svm = labels.at<int>(k);

    // create result object
    CustomSVM::SVMResponse svm_response;
    computeScores();
    svm_response.label = result_label_std_svm;
    svm_response.score = getScore(result_label_std_svm);  // the higher the score, the better
    svm_response.all_scores = m_scores;

    return svm_response;
}

void CustomSVM::computeScores()
{
    m_scores.clear();

    for(int i = 0; i < m_sums_of_sigmoids.size(); i++)
    {
        float avg_of_sigm = m_sums_of_sigmoids.at(i) / ((float)m_num_classes - 1.0f);
        m_scores.push_back(avg_of_sigm);
    }
}

float CustomSVM::getScore(int class_id)
{
    if(m_scores.size() == 0)
    {
        LOG_ERROR("before calling the function \"getScore\" you must call the function \"predictWithDistance\"!");
        return 0;
    }
    if(class_id == -1)
    {
        LOG_ERROR("invalid class id " << class_id);
        return 9999;
    }
    return m_scores.at(class_id);
}

float CustomSVM::sigmoid(float x)
{
    return 1.0 / (1.0 + std::exp(-x));
}
