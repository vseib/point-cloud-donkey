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
#include <fstream>
#include "../utils/utils.h"

CustomSVM::CustomSVM(std::string output_file_name)
{    
    // omit the path, keep only the file name
    m_output_file_name = output_file_name.substr(output_file_name.find_last_of('/')+1);
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
    cv::Mat labelsMat(m_num_features, 1, CV_32S, labels_arr);

    float training_data_arr[m_num_features][dim_feature];
    for(int i = 0; i < m_num_features; i++)
    {
        for(int j = 0; j < dim_feature; j++)
        {
            training_data_arr[i][j] = (float)(m_training_data.at(i)).at(j);
        }
    }
    cv::Mat trainingDataMat(m_num_features, dim_feature, CV_32F, training_data_arr);

    m_labels_mat = labelsMat.clone();
    m_train_data_mat = trainingDataMat.clone();

    return train_label;
}


void CustomSVM::trainSimple(double param_gamma, double param_c, bool one_vs_all)
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
        // Set up SVM's parameters
        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
        svm->setType(cv::ml::SVM::C_SVC);
        svm->setKernel(cv::ml::SVM::RBF);
        svm->setGamma(param_gamma);
        svm->setC(param_c);
        svm->setDegree(1);
        svm->setCoef0(0.1);
        //svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));

        int train_label = prepareTrainingData(i+1, num_runs);
        cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(m_train_data_mat, cv::ml::ROW_SAMPLE, m_labels_mat);
        svm->train(td);

        // save trained file
        std::stringstream sstr;
        std::string complete_name;
        if(num_runs == 1)
            sstr << "";
        else
            sstr << "_" << std::setfill('0') << std::setw(5) << std::to_string(train_label);

        complete_name = m_output_file_name + sstr.str() + ".svm";
        saved_files.push_back(complete_name);
        svm->save(complete_name.c_str());
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


void CustomSVM::trainAutomatically(double param_gamma, double param_c, int k_fold, bool one_vs_all)
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
        // Set up SVM's parameters
        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
        svm->setType(cv::ml::SVM::C_SVC);
        svm->setKernel(cv::ml::SVM::RBF);
        svm->setGamma(param_gamma);
        svm->setC(param_c);
        svm->setDegree(1);
        svm->setCoef0(0.1);
        //svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));

        int train_label = prepareTrainingData(i+1, num_runs);
        cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(m_train_data_mat, cv::ml::ROW_SAMPLE, m_labels_mat);

        // set up search grid for parameter optimization
        cv::ml::ParamGrid c_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::C);
        c_grid.minVal = 0.00001;
        c_grid.maxVal = 4096;
        c_grid.logStep = 2;
        // if using too many features increase step size
        if(m_num_features > 1000)
        {
            c_grid.minVal = 0.001;
            c_grid.maxVal = 1000;
            c_grid.logStep = 10;
        }

        cv::ml::ParamGrid gamma_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA);
        gamma_grid.minVal = 0.000001;
        gamma_grid.maxVal = 8;
        gamma_grid.logStep = sqrt(2);
        // if using too many features increase step size
        if(m_num_features > 1000)
        {
            gamma_grid.minVal = 0.0001;
            gamma_grid.maxVal = 10;
            gamma_grid.logStep = 10;
        }

        svm->trainAuto(td, k_fold, c_grid, gamma_grid);

        double best_c = svm->getC();
        double best_gamma = svm->getGamma();
        LOG_INFO("    SVM best params are: C: " << best_c << ", gamma: " << best_gamma);

        // ------------ use finer grid ------------
        LOG_INFO("auto-training SVM with finer grid");

        c_grid.minVal = best_c / (c_grid.logStep * c_grid.logStep);
        if(c_grid.minVal < 0.00001)
        {
            LOG_INFO("    c grid min too small: " << c_grid.minVal);
            c_grid.minVal = 0.00001;
        }
        c_grid.maxVal = best_c * (c_grid.logStep * c_grid.logStep);
        c_grid.logStep = sqrt(c_grid.logStep);

        gamma_grid.minVal = best_gamma / (gamma_grid.logStep * gamma_grid.logStep);
        if(gamma_grid.minVal < 0.0001)
        {
            LOG_INFO("    gamma grid min too small: " << gamma_grid.minVal);
            gamma_grid.minVal = 0.0001;
        }
        gamma_grid.maxVal = best_gamma * (gamma_grid.logStep * gamma_grid.logStep);
        gamma_grid.logStep = sqrt(gamma_grid.logStep);

        LOG_INFO("    C grid (min/max/step): " << c_grid.minVal << " " << c_grid.maxVal << " " << c_grid.logStep);
        LOG_INFO("    Gamma grid (min/max/step): " << gamma_grid.minVal << " " << gamma_grid.maxVal << " " << gamma_grid.logStep);

        // only refine grid with valid parameters
        if(c_grid.minVal < c_grid.maxVal && gamma_grid.minVal < gamma_grid.maxVal)
        {
            svm->trainAuto(td, k_fold, c_grid, gamma_grid);

            best_c = svm->getC();
            best_gamma = svm->getGamma();
            LOG_INFO("    SVM best params after fine grid are: C: " << best_c << ", gamma: " << best_gamma);
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
        svm->save(complete_name.c_str());
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
        // OpenCV multiple two-class SVMs to simulate an 1 vs all SVM
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
        // OpenCV multi-class SVM with additional score
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
        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(svm_string.c_str());
        float score = svm->predict(test_data, cv::noArray(), cv::ml::StatModel::Flags::RAW_OUTPUT);
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
    // TODO VS: refactor: SVM is loaded again and again for each object - load only once!
    // load svm
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(svm_file.c_str());

    // get important parameters
    int feature_dim = test_data.cols; // size of descriptor
    // manually read number of classes because OpenCV does not provide a function for this
    std::ifstream svm_input_file(svm_file);
    std::string line;
    while(std::getline(svm_input_file, line))
    {
        int pos = line.find_first_of(':');
        if(line.substr(0, pos+1) == "   class_count:")
        {
            m_num_classes = std::atoi(line.substr(pos+1).c_str());
            break;
        }
    }
    svm_input_file.close();

    cv::Mat support_vectors = svm->getSupportVectors();
    int num_sv = support_vectors.rows;
    float gamma = -1 * svm->getGamma();

    // calc RBF-kernel result vector
    std::vector<float> kernel_vector(num_sv, 0); // holds weights for each support vector

    for(int i = 0; i < num_sv; i++)
    {
        float s = 0;
        cv::Mat sv = support_vectors.row(i);

        for(int j = 0; j < feature_dim; j++)
        {
            float t = sv.at<float>(j) - test_data.at<float>(0,j);
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
            std::vector<double> alpha_vec;
            std::vector<int> sv_index_vec;
            double rho = svm->getDecisionFunction(dfi, alpha_vec, sv_index_vec);
            double sum = -rho;

            for(int k = 0; k < sv_index_vec.size(); k++)
            {
                sum += alpha_vec[k] * kernel_vector.at(sv_index_vec[k]);
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
    for(int i = 0; i < m_num_classes; i++)
    {
        if(class_votes.at(i) > class_votes.at(k))
        {
            k = i;
        }
    }

    // finally the result (this is the result of the standard OpenCV SVM)
    int result_label_std_svm = k;

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
        // compute average of sigmoids
        // NOTE: while there are n*(n-1) decision functions, each classes sums of sigmoids was only updated (n-1) times - once for each of the other classes
        float avg_of_sigm = m_sums_of_sigmoids.at(i) / (m_num_classes - 1);
        m_scores.push_back(avg_of_sigm);
    }
}

float CustomSVM::getScore(int class_id)
{
    if(m_scores.size() == 0)
    {
        LOG_ERROR("before calling the function \"getScore\" you must call the function \"predictWithScore\"!");
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
