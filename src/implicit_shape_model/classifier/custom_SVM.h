/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef CUSTOM_SVM_H
#define CUSTOM_SVM_H

#include <opencv2/ml.hpp>

class CustomSVM
{

public:

    struct SVMResponse
    {
        int label;
        float score;
        std::vector<float> all_scores;
    };

    CustomSVM(std::string output_file_name = "");
    virtual ~CustomSVM();

    void setData(std::vector< std::vector<float> > &training_data, std::vector<int> &labels);

    void trainSimple(double param_gamma, double param_c, bool one_vs_all);
    void trainAutomatically(double param_gamma, double param_c, int k_fold, bool one_vs_all);

    SVMResponse predictUnifyScore(cv::Mat test_data, std::vector<std::string> &svm_files);
    SVMResponse predictWithScore(cv::Mat test_data, std::vector<std::string> &svm_files); // 1 vs all SVM (simulated by multiple 2 class SVMs)
    SVMResponse predictWithScore(cv::Mat test_data, std::string &svm_file); // pairwise 1 vs 1 SVM

private:

    // training
    int prepareTrainingData(int run_id, int num_runs);

    std::vector< std::vector<float> > m_training_data;
    std::vector<int> m_labels;
    cv::Mat m_labels_mat;
    cv::Mat m_train_data_mat;
    int m_num_features;
    std::string m_output_file_name;

    // prediction
    float getScore(int class_id);
    float sigmoid(float x);
    void computeScores();

    int m_num_classes;
    std::vector<float> m_sums_of_sigmoids;
    std::vector<float> m_scores;
};

#endif // CUSTOM_SVM_H
