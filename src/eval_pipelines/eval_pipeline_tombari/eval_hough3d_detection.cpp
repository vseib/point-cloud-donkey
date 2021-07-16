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
#include "hough3d.h"
#include <boost/timer/timer.hpp>
#include "../../eval_tool/eval_helpers_detection.h"

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
    if(argc != 5)
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

    float bin = atof(argv[3]);
    float th = atof(argv[4]);

    // parse input
    std::vector<std::string> filenames;
    std::vector<std::string> gt_filenames;
    std::vector<unsigned> class_labels;
    std::vector<unsigned> instance_labels;
    std::string mode;

    // find dataset name from input file
    std::string datasetname;
    int pos1 = dataset.find_first_of('_');
    std::string str1 = dataset.substr(0, pos1);
    int pos2 = dataset.substr(pos1+1).find_first_of('_');
    std::string str2 = dataset.substr(pos1+1, pos2);

    if(str1 == "train" || str1 == "test" || str1 == "training" || str1 == "testing")
    {
        datasetname = str2;
    }
    else
    {
        datasetname = str1;
    }

    std::shared_ptr<Hough3d> hough3d(new Hough3d(datasetname,bin, th));

    // workaround to set "mode"
    {
        std::ifstream infile(dataset);
        std::string file;
        std::string class_label;
        // special treatment of first line: determine mode
        infile >> file;         // in the first line: #
        infile >> class_label;  // in the first line: the mode ("train" or "test")
        if(file == "#" && (class_label == "train" || class_label == "test"))
        {
            mode = class_label;
        }
    }

    if(mode == "train")
    {
        label_usage = parseFileListDetectionTrain(dataset, filenames, class_labels, instance_labels, "train");

        // if both, class and instance labels given, use instances
        // usually, this leads to better accuracy
        if(label_usage == LabelUsage::BOTH_GIVEN)
        {
            label_usage = LabelUsage::INSTANCE_PRIMARY;
        }

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

        parseFileListDetectionTest(dataset, filenames, gt_filenames);

        if(hough3d->loadModel(model))
        {
            class_labels_rmap = hough3d->getClassLabels();
            instance_labels_rmap = hough3d->getInstanceLabels();
            instance_to_class_map = hough3d->getInstanceClassMap();
            // populate maps with loaded data
            for(auto &elem : class_labels_rmap)
            {
                class_labels_map.insert({elem.second, elem.first});
            }
            for(auto &elem : instance_labels_rmap)
            {
                instance_labels_map.insert({elem.second, elem.first});
            }

            // determine label_usage: empty mapping means that no instance labels were given
            if(instance_to_class_map.size() == 0)
            {
                label_usage = LabelUsage::CLASS_ONLY;
                // instance ids must be filled even if testing with class labels only
                instance_labels = class_labels;
            }
            else
            {
                // determine label_usage: compare all instance and class labels
                bool all_equal = true;
                for(auto elem : class_labels_rmap)
                {
                    std::string label1 = elem.second;
                    std::string label2 = instance_labels_rmap[elem.first];
                    if(label1 != label2)
                    {
                        all_equal = false;
                        break;
                    }
                }

                if(all_equal)
                {
                    // instances used as primary labels, classes determined over mapping
                    label_usage = LabelUsage::INSTANCE_PRIMARY;
                }
                else
                {
                    std::cerr << "Mismatch in instance label usage!" << std::endl;
                    return 1;
                }
            }

            std::vector<DetectionObject> gt_objects;
            std::vector<DetectionObject> detected_objects;

            std::string outputname = model.substr(0, model.find_last_of('.')) + ".txt";
            std::ofstream summaryFile("output_tombari_"+std::to_string(bin)+"_"+std::to_string(th)+"_"+outputname);

            for(unsigned i = 0; i < filenames.size(); i++)
            {
                // detect
                std::string pointCloud = filenames.at(i);
                std::string gt_file = gt_filenames.at(i);
                std::vector<ism3d::VotingMaximum> maxima;

                std::cout << "Processing file " << pointCloud << std::endl;

                bool useHypothesisVerification = true;
                bool useSingleVotingSpace = true;
                maxima = hough3d->detect(pointCloud, useHypothesisVerification, useSingleVotingSpace);

                // collect all gt objects
                std::vector<DetectionObject> gt_objects_from_file = parseGtFile(gt_file);
                gt_objects.insert(gt_objects.end(), gt_objects_from_file.begin(), gt_objects_from_file.end());
                // collect all detections
                for (int i = 0; i < (int)maxima.size(); i++)
                {
                    DetectionObject detected_obj = convertMaxToObj(maxima[i], gt_file);
                    detected_objects.push_back(std::move(detected_obj));
                }
            }

            // maps a class label id to list of objects
            std::map<std::string, std::vector<DetectionObject>> gt_class_map;
            std::map<std::string, std::vector<DetectionObject>> det_class_map;

            rearrangeObjects(gt_objects, gt_class_map);
            rearrangeObjects(detected_objects, det_class_map);

            // max. allowed distance to ground truth position to count the detection as correct
            // TODO VS depends on dataset/cloud resolution --> make param?
            float dist_threshold = 0.05;

            // collect all metrics
            // combined detection - primary metrics
            std::vector<float> ap_per_class(gt_class_map.size(), 0.0);
            std::vector<float> precision_per_class(gt_class_map.size(), 0.0);
            std::vector<float> recall_per_class(gt_class_map.size(), 0.0);

            summaryFile << "  class       num gt   tp    fp   precision  recall   AP";
            summaryFile << std::endl;

            // these variables sum over the whole dataset
            int num_gt_dataset = 0;
            int cumul_tp_dataset = 0;
            int cumul_fp_dataset = 0;

            for(auto item : gt_class_map)
            {
                std::string class_label = item.first;
                unsigned class_id = class_labels_map[class_label];
                std::vector<DetectionObject> class_objects_gt = item.second;
                // these variables sum over each class
                int num_gt = int(class_objects_gt.size());
                int cumul_tp, cumul_fp;

                // if there are no detections for this class
                if(det_class_map.find(class_label) == det_class_map.end())
                {
                    ap_per_class[class_id] = 0;
                    precision_per_class[class_id] = 0;
                    recall_per_class[class_id] = 0;
                }
                else
                {
                    std::vector<DetectionObject> class_objects_det = det_class_map.at(class_label);

                    float precision, recall, ap;
                    std::tie(precision, recall, ap, cumul_tp, cumul_fp) = computeMetrics(class_objects_gt,
                                                                                     class_objects_det,
                                                                                     dist_threshold);
                    precision_per_class[class_id] = precision;
                    recall_per_class[class_id] = recall;
                    ap_per_class[class_id] = ap;
                }

                // log class to summary
                float ap = ap_per_class[class_id];
                float precision = precision_per_class[class_id];
                float recall = recall_per_class[class_id];

                summaryFile << std::setw(3) << std::right << class_id << " "
                            << std::setw(13) << std::left << class_label
                            << std::setw(3) << std::right << num_gt
                            << std::setw(5) << cumul_tp
                            << std::setw(6) << cumul_fp << "   "
                            << std::setw(11) << std::left << std::round(precision*10000.0f)/10000.0f
                            << std::setw(9) << std::round(recall*10000.0f)/10000.0f
                            << std::setw(10) << std::round(ap*10000.0f)/10000.0f;
                summaryFile << std::endl;

                // accumulate values of complete dataset
                num_gt_dataset += num_gt;
                cumul_tp_dataset += cumul_tp;
                cumul_fp_dataset += cumul_fp;
            }

            // store sums
            summaryFile << "-------------------------------------------------------------" << std::endl;
            summaryFile << "Sums:" << std::setw(15) << std::right << num_gt_dataset
                        << std::setw(5) << std::right << cumul_tp_dataset
                        << std::setw(6) << std::right << cumul_fp_dataset << std::endl;

            // compute average metrics
            float mAP = 0;
            float mPrec = 0;
            float mRec = 0;
            for(int idx = 0; idx < ap_per_class.size(); idx++)
            {
                mAP += ap_per_class[idx];
                mPrec += precision_per_class[idx];
                mRec += recall_per_class[idx];
            }
            mAP /= ap_per_class.size();
            mPrec /= ap_per_class.size();
            mRec /= ap_per_class.size();
            summaryFile << std::endl << std::endl;
            summaryFile << "main metrics:" << std::endl;
            summaryFile << "       mAP:            " << std::setw(7) << std::round(mAP*10000.0f)/10000.0f    << " (" << std::round(mAP*10000.0f)/100.0f   << " %)" << std::endl;
            summaryFile << "       mean precision: " << std::setw(7) << std::round(mPrec*10000.0f)/10000.0f  << " (" << std::round(mPrec*10000.0f)/100.0f << " %)" << std::endl;
            summaryFile << "       mean recall:    " << std::setw(7) << std::round(mRec*10000.0f)/10000.0f   << " (" << std::round(mRec*10000.0f)/100.0f  << " %)" << std::endl << std::endl;

            // complete and close summary file
            summaryFile << "total processing time: " << timer.format(4, "%w") << " seconds \n";
            summaryFile.close();
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
