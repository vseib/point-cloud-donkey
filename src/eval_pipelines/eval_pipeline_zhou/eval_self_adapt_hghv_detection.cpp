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
#include "self_adapt_hghv.h"
#include <boost/timer/timer.hpp>
#include "../../eval_tool/eval_helpers_detection.h"
#include "../../eval_tool/logging_to_files.h"

/**
 * Evaluation pipeline for the approach described in
 *
 * W. Zhou, C. Ma, A. Kuijper:
 *     Hough-space-based hypothesis generation and hypothesis verification for 3D
 *     object recognition and 6D pose estimation.
 *     2018, Computers & Graphics
 *
 */


int main (int argc, char** argv)
{
    if(argc != 6)
    {
        std::cout << std::endl << "Usage:" << std::endl << std::endl;
        std::cout << argv[0] << " [object file] [scene description file]" << std::endl << std::endl;
        std::cout << "Example:" << std::endl << std::endl;
        std::cout << argv[0] << " object_files.txt scene_001.txt" << std::endl << std::endl;
        std::cout << "The object file must contain a filename " << std::endl;
        std::cout << "and class label per line, separated by a space." << std::endl;
        std::cout << "The pcd files from the object file will be search for and localized" << std::endl;
        std::cout << "in the scene file contained in the scene description." << std::endl;
        std::cout << "The scene description file contains the name of a scene pcd " << std::endl;
        std::cout << "and locations and class labels of objects in that scene." << std::endl;
        exit(1);
    }

    // input data
    std::string dataset = argv[1];
    std::string model = argv[2];
    float bin = atof(argv[3]);
    float th = atof(argv[4]);
    int count = atof(argv[5]);

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

    std::shared_ptr<SelfAdaptHGHV> sa_hghv(new SelfAdaptHGHV(datasetname, bin, th, count));

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
        sa_hghv->setLabels(class_labels_rmap, instance_labels_rmap, instance_to_class_map);
        sa_hghv->train(filenames, class_labels, instance_labels, model);
    }
    else if(mode == "test")
    {
        boost::timer::cpu_timer timer;
        std::cout << "Started testing!" << std::endl;

        parseFileListDetectionTest(dataset, filenames, gt_filenames);

        if(sa_hghv->loadModel(model))
        {
            class_labels_rmap = sa_hghv->getClassLabels();
            instance_labels_rmap = sa_hghv->getInstanceLabels();
            instance_to_class_map = sa_hghv->getInstanceClassMap();
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
            outputname = "output_zhou_"+std::to_string(bin)+"_"+std::to_string(th)+"_"+std::to_string(count)+"_"+outputname;
            std::ofstream summaryFile;

            for(unsigned i = 0; i < filenames.size(); i++)
            {
                // detect
                std::string pointCloud = filenames.at(i);
                std::string gt_file = gt_filenames.at(i);
                std::vector<ism3d::VotingMaximum> maxima;

                std::cout << "Processing file " << pointCloud << std::endl;
                maxima = sa_hghv->detect(pointCloud);

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

            // create common object to manage all metrics
            MetricsCollection mc;
            rearrangeObjects(gt_objects, mc.gt_class_map);
            rearrangeObjects(detected_objects, mc.det_class_map);
            mc.resizeVectors();

            // max. allowed distance to ground truth position to count the detection as correct
            std::map<unsigned, float> class_radii = sa_hghv->getDetectionThresholds();

            std::string command = "mkdir " + outputname;
            std::ignore = std::system(command.c_str());
            filelog::writeDetectionSummaryHeader(summaryFile, outputname, false);

            // these variables sum over the whole dataset
            int num_gt_dataset = 0;
            int cumul_tp_dataset = 0;
            int cumul_fp_dataset = 0;

            for(auto item : mc.gt_class_map)
            {
                std::string class_label = item.first;
                unsigned class_id = class_labels_map[class_label];
                float dist_threshold = class_radii[class_id];
                std::vector<DetectionObject> class_objects_gt = item.second;

                filelog::computeAndWriteNextClassSummary(summaryFile, mc, class_label, class_id,
                                                class_objects_gt, false,
                                                dist_threshold, num_gt_dataset,
                                                cumul_tp_dataset, cumul_fp_dataset);
            }


            float overall_ap;
            std::tie(std::ignore, std::ignore, overall_ap) = computePrecisionRecallForPlotting(mc.det_class_map, mc.gt_class_map,
                                                                                          mc.tps_per_class, mc.fps_per_class);
            std::map<std::string, double> times; // dummy empty map
            // store sums and overall metrics
            std::string total_time = timer.format(4, "%w");
            filelog::computeAndWriteFinalMetrics(summaryFile, mc, num_gt_dataset,
                                                 cumul_tp_dataset, cumul_fp_dataset,
                                                 overall_ap, times, total_time,
                                                 false);
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
