/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2021, Viktor Seib
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 * * list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */


#include <iostream>
#include <fstream>
#include <boost/timer/timer.hpp>
#include "hough3d.h"
#include "../../eval_tool/eval_helpers.h"

/**
 * Evaluation pipeline for the approach described in
 *
 * F. Tombari, L. Di Stefano:
 *     Object recognition in 3D scenes with occlusions and clutter by Hough voting.
 *     2010, Fourth Pacific-Rim Symposium on Image and Video Technology
 *
 */

int main (int argc, char** argv)
{
    if(argc != 3)
    {
        std::cout << std::endl << "Usage:" << std::endl << std::endl;
        std::cout << argv[0] << " [dataset file] [model name]" << std::endl << std::endl;
        std::cout << "Example:" << std::endl << std::endl;
        std::cout << argv[0] << " train_files.txt trained_model.m" << std::endl << std::endl;
        std::cout << "The dataset file must contain the string '# train' or '# test' " << std::endl;
        std::cout << "in the first line. The following lines contain a filename " << std::endl;
        std::cout << "and class label per line, separated by a space." << std::endl;
        std::cout << "Objects of the same class should all come consecutively." << std::endl;
        std::cout << "The model name will be used to write the output file in " << std::endl;
        std::cout << "training and to load an existing model in testing." << std::endl;
        std::cout << "Example for a dataset file: " << std::endl;
        std::cout << "# train " << std::endl;
        std::cout << "object01.pcd 0" << std::endl;
        std::cout << "object02.pcd 1" << std::endl;
        exit(1);
    }

    // input data
    std::string dataset = argv[1];
    std::string model = argv[2];

    // parse input
    std::vector<std::string> filenames;
    std::vector<unsigned> class_labels;
    std::vector<unsigned> instance_labels;
    std::string mode;

    label_usage = parseFileList(dataset, filenames, class_labels, instance_labels, mode);

    // if both, class and instance labels given, use instances
    // usually, this leads to better accuracy
    if(label_usage == LabelUsage::BOTH_GIVEN)
    {
        label_usage = LabelUsage::INSTANCE_PRIMARY;
    }

    // process training or testing
    std::shared_ptr<Hough3d> hough3d(new Hough3d());

    if(mode == "train")
    {
        if(label_usage == LabelUsage::CLASS_ONLY)
        {
            // instance ids must be filled even if training with class labels only
            instance_labels = class_labels;
        }
        else if(label_usage == LabelUsage::INSTANCE_PRIMARY)
        {
            // use instance labels for classes
            class_labels = instance_labels;
        }

        std::cout << "Started training!" << std::endl;
        hough3d->setLabels(class_labels_rmap, instance_labels_rmap, instance_to_class_map);
        hough3d->train(filenames, class_labels, instance_labels, model);
    }
    else if(mode == "test")
    {
        boost::timer::cpu_timer timer;
        std::cout << "Started testing!" << std::endl;

        if(hough3d->loadModel(model))
        {
            class_labels_rmap = hough3d->getClassLabels();
            instance_labels_rmap = hough3d->getInstanceLabels();
            instance_to_class_map = hough3d->getInstanceClassMap();

            // if both, class and instance labels given, use instances
            if(label_usage == LabelUsage::BOTH_GIVEN)
            {
                label_usage = LabelUsage::INSTANCE_PRIMARY;
            }

            if(label_usage == LabelUsage::CLASS_ONLY)
            {
                // instance ids must be filled even if testing with class labels only
                instance_labels = class_labels;
            }

            int num_correct_classes = 0;
            int num_correct_instances = 0;
            int num_total = 0;
            // used to compute average per class accuracy
            std::map<unsigned, std::pair<unsigned, unsigned>> averageAccuracyHelper; // maps class id to pair <correct, total>


            std::string outputname = model.substr(0, model.find_last_of('.')) + ".txt";
            std::ofstream outfile("output_tombari_"+outputname);

            for(std::string filename : filenames)
            {
                std::cout << "Processing file " << filename << std::endl;

                std::vector<std::pair<unsigned, float>> results;
                results = hough3d->classify(filename);

                // lookup real class ids if instances were used as primary labels
                std::vector<unsigned> result_instance_labels;
                if(label_usage == LabelUsage::INSTANCE_PRIMARY)
                {
                    for(int i = 0; i < results.size(); i++)
                    {
                        std::pair<unsigned, float> res = results.at(i);
                        result_instance_labels.push_back(res.first);
                        results.at(i).first = instance_to_class_map[res.first];
                    }
                }
                else // need to fill it in, but it won't be displayed
                {
                    if(results.size() > 0)
                    {
                        result_instance_labels.push_back(results.at(0).first);
                    }
                    else
                    {
                        result_instance_labels.push_back(-1);
                    }
                }

                // print results
                for(int i = 0; i < results.size(); i++)
                {
                    std::pair<unsigned, float> res = results.at(i);
                    std::cout << i << ": label: " << res.first << ", score: " << res.second << std::endl;
                }

                int result_label = -1;
                if(results.size() > 0)
                    result_label = results.at(0).first;

                outfile << (num_total+1) << ". file: " << filename << ", gt label: " << class_labels.at(num_total)
                           << ", classified as: " << result_label << std::endl;

                // evaluate results
                if(results.size() > 0)
                {
                    unsigned trueID = class_labels.at(num_total);
                    if(trueID == results.at(0).first)
                    {
                        // for overall accuracy
                        num_correct_classes++;
                        // for average per class accuracy
                        if(averageAccuracyHelper.find(trueID) != averageAccuracyHelper.end())
                        {
                            // pair <correct, total>
                            std::pair<unsigned, unsigned> &res = averageAccuracyHelper.at(trueID);
                            res.first++;
                            res.second++;
                        }
                        else
                        {
                            averageAccuracyHelper.insert({trueID, {1,1}});
                        }
                    }
                    else
                    {
                        // wrong classification
                        if(averageAccuracyHelper.find(trueID) != averageAccuracyHelper.end())
                        {
                            // pair <correct, total>
                            std::pair<unsigned, unsigned> &res = averageAccuracyHelper.at(trueID);
                            res.second++;
                        }
                        else
                        {
                            averageAccuracyHelper.insert({trueID, {0,1}});
                        }
                    }
                    if(instance_labels.at(num_total) == result_instance_labels.at(0))
                    {
                        num_correct_instances++;
                    }
                }
                num_total++;
            }

            float avg_pc_acc = 0;
            for(const auto &elem : averageAccuracyHelper)
            {
                const std::pair<unsigned, unsigned> &p = elem.second;
                avg_pc_acc += (float)p.first/p.second;
            }
            avg_pc_acc /= averageAccuracyHelper.size();

            std::cout << "Classified " << num_correct_classes << " of " << num_total << " (" << (num_correct_classes/((float)num_total))*100 << " %) files correctly." << std::endl;
            outfile << std::endl << std::endl << "Average per Class accuracy: " << (avg_pc_acc*100) << " %" << std::endl;
            outfile << "  Classes: " << num_correct_classes << " of " << num_total
                       << " (" << (num_correct_classes/((float)num_total))*100 << " %) files classified correctly." << std::endl;

            if(label_usage == LabelUsage::INSTANCE_PRIMARY)
            {
                outfile << "Instances: " << num_correct_instances << " of " << num_total
                           << " (" << (num_correct_instances/((float)num_total))*100 << " %) files classified correctly." << std::endl;
            }

            outfile << std::endl << "Total processing time: " << timer.format(4, "%w") << " seconds \n";
            outfile.close();
        }
        else
        {
            std::cerr << "ERROR: could not load model from file " << model << "!" << std::endl;
        }
    }
    else
    {
        std::cerr << "ERROR: wrong mode specified: " << mode << "! Must be train or test!" << std::endl;
    }

    return (0);
}
