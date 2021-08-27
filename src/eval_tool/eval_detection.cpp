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

#include <boost/program_options.hpp>
#include <boost/program_options/errors.hpp>
#include "../implicit_shape_model/implicit_shape_model.h"
#include "eval_helpers_detection.h"

#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/basicconfigurator.h>


int main(int argc, char **argv)
{
    boost::program_options::options_description generic("Generic options");
    boost::program_options::options_description training("Training");
    boost::program_options::options_description detection("Detection");

    generic.add_options()
            ("help,h", "Display this help message")
            ("output,o", boost::program_options::value<std::string>(), "The output folder (created automatically) for ism files after training or the detection log after detection")
            ("inputfile,f", boost::program_options::value<std::string>(), "Input file (for training or testing) containing the input clouds and their corresponding labels (replaces m and c in training and p and g in testing");

    training.add_options()
            ("train,t", boost::program_options::value<std::string>(), "Train an implicit shape model")
            ("inplace,i", "Overwrite the loaded ism file");

    detection.add_options()
            ("detect,d", boost::program_options::value<std::string>(), "Detect using a trained implicit shape model");

    boost::program_options::options_description desc;
    desc.add(generic).add(training).add(detection);

    // parse command line arguments
    boost::program_options::variables_map variables;
    try {
        boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).run(), variables);
        boost::program_options::notify(variables);
    }
    catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    // show help
    if (variables.count("help") || variables.size() == 0) {
        std::cout << desc << std::endl;
        return 1;
    }
    else
    {
        try
        {
            std::vector<std::string> filenames;
            std::vector<std::string> gt_filenames;
            std::vector<unsigned> class_labels;
            std::vector<unsigned> instance_labels;

            // train the ISM
            if (variables.count("train"))
            {
                if(variables.count("inputfile"))
                {
                    // manually parse input file
                    std::string input_file_name = variables["inputfile"].as<std::string>();
                    label_usage = parseFileListDetectionTrain(input_file_name, filenames, class_labels, instance_labels, "train");
                }

                std::cout << "starting the training process" << std::endl;

                std::string ismFile;
                try // allows to use -t or -d for ism-files when input file with dataset is specified with -f
                {
                    ismFile = variables["train"].as<std::string>();
                }
                catch(std::exception& e)
                {
                    ismFile = variables["detect"].as<std::string>();
                }

                // try to read the file
                ism3d::ImplicitShapeModel ism;
                ism.setLogging(log_info);
                ism.setSignalsState(false); // disable signals since we are using command line, no GUI

                if(!ism.readObject(ismFile, true))
                {
                    std::cerr << "could not read ism from file, training stopped: " << ismFile << std::endl;
                    return 1;
                }

                // if both, class and instance labels given, check which is primary
                if(label_usage == LabelUsage::BOTH_GIVEN)
                {
                    if(ism.isInstancePrimaryLabel())
                    {
                        label_usage = LabelUsage::INSTANCE_PRIMARY;
                    }
                    else
                    {
                        label_usage = LabelUsage::CLASS_PRIMARY;
                    }
                }

                // set output filename
                if(variables.count("output"))
                {
                    std::string outFilename = variables["output"].as<std::string>();
                    ism.setOutputFilename(outFilename);
                }

                // add the training models to the ism
                if(filenames.size() > 0 && class_labels.size() > 0)
                {
                    std::vector<std::string> models;
                    std::vector<unsigned> class_ids;
                    std::vector<unsigned> instance_ids;

                    if(filenames.size() > 0) // input inside file given on command line
                    {
                        models = filenames;
                        if(label_usage == LabelUsage::CLASS_ONLY)
                        {
                            // instance ids must be filled even if training with class labels only
                            class_ids = class_labels;
                            instance_ids = class_labels;
                        }
                        else if(label_usage == LabelUsage::CLASS_PRIMARY)
                        {
                            class_ids = class_labels;
                            instance_ids = instance_labels;
                        }
                        else if(label_usage == LabelUsage::INSTANCE_PRIMARY)
                        {
                            class_ids = instance_labels;
                            instance_ids = instance_labels;
                        }
                        else
                        {
                            std::cerr << "Label usage not defined or not supported! ("
                                      << static_cast<std::underlying_type<LabelUsage>::type>(label_usage) << ")" << std::endl;
                            return 1;
                        }
                    }

                    if (models.size() == class_ids.size())
                    {
                        for (int i = 0; i < (int)models.size(); i++)
                        {
                            std::string filename = models[i];
                            unsigned class_id = class_ids[i];
                            unsigned instance_id = instance_ids[i];

                            // add the training model to the ISM
                            if (!ism.addTrainingModel(filename, class_id, instance_id))
                            {
                                std::cerr << "could not add training model: " << filename << ", class " << class_id << std::endl;
                                return 1;
                            }
                        }
                    }
                    else
                    {
                        std::cerr << "number of models does not match the number of class ids" << std::endl;
                        return 1;
                    }
                }

                // train
                ism.train();

                // store maps in the model object file
                ism.setLabels(class_labels_rmap, instance_labels_rmap, instance_to_class_map);

                // write the ism data
                if (variables.count("inplace"))
                {
                    if (!ism.writeObject(ismFile, ismFile + "d"))
                    {
                        std::cerr << "could not write ism" << std::endl;
                        return 1;
                    }
                }
                else if (variables.count("output"))
                {
                    std::string outFilename = variables["output"].as<std::string>();
                    if (!ism.writeObject(outFilename))
                    {
                        std::cerr << "could not write ism" << std::endl;
                        return 1;
                    }
                }
                else
                {
                    std::cerr << "the trained ism is not saved" << std::endl;
                    return 1;
                }
            }

            // use ISM for detection
            if (variables.count("detect"))
            {
                if(variables.count("inputfile"))
                {
                    // manually parse input file
                    std::string input_file_name = variables["inputfile"].as<std::string>();
                    parseFileListDetectionTest(input_file_name, filenames, gt_filenames);
                }
                else
                {
                    std::cerr << "No input file provided! You need to provide an input file with -f" << std::endl;
                    return 1;
                }

                std::cout << "starting the detection process" << std::endl;

                std::string ismFile;
                try // allows to use -t or -d for ism-files when input file with dataset is specified with -f
                {
                    ismFile = variables["detect"].as<std::string>();
                }
                catch(std::exception& e)
                {
                    ismFile = variables["train"].as<std::string>();
                }

                // try to read the file
                ism3d::ImplicitShapeModel ism;
                ism.setLogging(log_info);
                ism.setSignalsState(false); // disable signals since we are using command line, no GUI

                if (!ism.readObject(ismFile))
                {
                    std::cerr << "could not read ism from file, detection stopped: " << ismFile << std::endl;
                    return 1;
                }
                else if (filenames.size() > 0 && gt_filenames.size() > 0)  // load pointclouds and groundtruth
                {
                    std::vector<std::string> pointClouds;
                    // load label information from training
                    class_labels_rmap = ism.getClassLabels();
                    instance_labels_rmap = ism.getInstanceLabels();
                    instance_to_class_map = ism.getInstanceClassMap();
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

                        if(all_equal && ism.isInstancePrimaryLabel())
                        {
                            // instances used as primary labels, classes determined over mapping
                            label_usage = LabelUsage::INSTANCE_PRIMARY;
                        }
                        else if(!all_equal && !ism.isInstancePrimaryLabel())
                        {
                            // both labels used, class labels as primary
                            label_usage = LabelUsage::CLASS_PRIMARY;
                        }
                        else
                        {
                            std::cerr << "Mismatch in instance label usage between config file (.ism) and trained file (.ismd)!" << std::endl;
                            std::cerr << "Config file has InstanceLabelsPrimary as " << ism.isInstancePrimaryLabel() << ", while trained file has " << !ism.isInstancePrimaryLabel() << std::endl;
                            return 1;
                        }
                    }

                    if(filenames.size() > 0) // input inside file given on command line
                    {
                        pointClouds = filenames;
                    }

                    // prepare summary
                    std::ofstream summaryFile;
                    if (variables.count("output"))
                    {
                        // create folder for output
                        std::string command = "mkdir ";
                        std::string folder = variables["output"].as<std::string>();
                        command.append(folder);
                        std::ignore = std::system(command.c_str());
                        sleep(1);

                        // summary file
                        std::string outFile = variables["output"].as<std::string>();
                        std::string outFileName = outFile;
                        outFileName.append("/summary.txt");
                        summaryFile.open(outFileName.c_str(), std::ios::out);
                    }
                    else
                    {
                        std::cerr << "no output file specified, detected maxima will not be saved" << std::endl;
                    }

                    std::vector<DetectionObject> gt_objects;
                    std::vector<DetectionObject> detected_objects;

                    boost::timer::cpu_timer timer;
                    std::map<std::string, double> times;
                    for(unsigned i = 0; i < pointClouds.size(); i++)
                    {
                        // detect
                        std::string pointCloud = pointClouds.at(i);
                        std::string gt_file = gt_filenames.at(i);
                        std::vector<ism3d::VotingMaximum> maxima;

                        std::cout << "Processing file: " << pointCloud << std::endl;
                        if (!ism.detect(pointCloud, maxima, times))
                        {
                            std::cerr << "detection failed" << std::endl;
                            return 1;
                        }
                        else
                        {
                            // write detected maxima to detection log file
                            if (variables.count("output"))
                            {
                                // collect all gt objects
                                std::vector<DetectionObject> gt_objects_from_file = parseGtFile(gt_file);
                                gt_objects.insert(gt_objects.end(), gt_objects_from_file.begin(), gt_objects_from_file.end());
                                // collect all detections
                                for (int i = 0; i < (int)maxima.size(); i++)
                                {
                                    DetectionObject detected_obj = convertMaxToObj(maxima[i], gt_file);
                                    detected_objects.push_back(std::move(detected_obj));
                                }
                                if(write_log_to_files)
                                {
                                    unsigned tmp = pointCloud.find_last_of('/');
                                    if(tmp == std::string::npos) tmp = 0;
                                    std::string fileWithoutFolder = pointCloud.substr(tmp+1);

                                    std::cout << "writing detection log" << std::endl;
                                    std::string outFile = variables["output"].as<std::string>();
                                    std::string outFileName = outFile;
                                    outFileName.append("/");
                                    outFileName.append(fileWithoutFolder);
                                    outFileName.append(".txt");

                                    std::ofstream file;
                                    file.open(outFileName.c_str(), std::ios::out);
                                    file << "ISM3D detection log, filename: " << ismFile << ", point cloud: " << pointCloud
                                         << ", ground truth file: " << gt_file << std::endl;
                                    file << "number, classID, weight, instanceID, instance weight, num-votes, position X Y Z, bounding box size X Y Z, bounding Box rotation quaternion w x y z" << std::endl;

                                    for (int i = 0; i < (int)maxima.size(); i++)
                                    {
                                        const ism3d::VotingMaximum& maximum = maxima[i];

                                        file << i << ", ";
                                        file << maximum.classId << ", ";
                                        file << maximum.weight << ", ";
                                        file << maximum.instanceId << ", ";
                                        file << maximum.instanceWeight << ", ";
                                        file << maximum.votes.size() << ", ";
                                        file << maximum.position[0] << ", ";
                                        file << maximum.position[1] << ", ";
                                        file << maximum.position[2] << ", ";
                                        file << maximum.boundingBox.size[0] << ", ";
                                        file << maximum.boundingBox.size[1] << ", ";
                                        file << maximum.boundingBox.size[2] << ", ";
                                        file << maximum.boundingBox.rotQuat.R_component_1() << ", ";
                                        file << maximum.boundingBox.rotQuat.R_component_2() << ", ";
                                        file << maximum.boundingBox.rotQuat.R_component_3() << ", ";
                                        file << maximum.boundingBox.rotQuat.R_component_4() << std::endl;
                                    }

                                    file.close();
                                }
                            }
                        }
                    }

                    // maps a class label id to list of objects
                    std::map<std::string, std::vector<DetectionObject>> gt_class_map;
                    std::map<std::string, std::vector<DetectionObject>> det_class_map;
                    std::map<std::string, std::vector<DetectionObject>> det_class_map_global;

                    rearrangeObjects(gt_objects, gt_class_map);
                    rearrangeObjects(detected_objects, det_class_map);
                    bool report_global_metrics = ism.isUsingGlobalFeatures();
//                    report_global_metrics = true;
                    if(report_global_metrics)
                        rearrangeObjects(detected_objects, det_class_map_global, true);

                    float dist_threshold = ism.getDetectionThreshold();

                    // collect all metrics
                    // combined detection - primary metrics
                    std::vector<float> ap_per_class(gt_class_map.size(), 0.0);
                    std::vector<float> precision_per_class(gt_class_map.size(), 0.0);
                    std::vector<float> recall_per_class(gt_class_map.size(), 0.0);
                    // metrics for the global classifier (if used)
                    std::vector<float> global_ap_per_class(gt_class_map.size(), 0.0);
                    std::vector<float> global_precision_per_class(gt_class_map.size(), 0.0);
                    std::vector<float> global_recall_per_class(gt_class_map.size(), 0.0);

                    summaryFile << "  class       num gt   tp    fp   precision  recall   AP";
                    if(report_global_metrics)
                                summaryFile << "        | global tp    fp   precision  recall   AP";
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
                        int global_cumul_tp, global_cumul_fp;

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

                        if(report_global_metrics)
                        {
                            // if there are no detections for this class in global detector
                            if(det_class_map_global.find(class_label) == det_class_map_global.end())
                            {
                                global_ap_per_class[class_id] = 0;
                                global_precision_per_class[class_id] = 0;
                                global_recall_per_class[class_id] = 0;
                            }
                            else
                            {
                                std::vector<DetectionObject> class_objects_det = det_class_map_global.at(class_label);

                                float precision, recall, ap;
                                std::tie(precision, recall, ap, global_cumul_tp, global_cumul_fp) = computeMetrics(class_objects_gt,
                                                                                                 class_objects_det,
                                                                                                 dist_threshold);
                                global_precision_per_class[class_id] = precision;
                                global_recall_per_class[class_id] = recall;
                                global_ap_per_class[class_id] = ap;
                            }
                        }

                        // log class to summary
                        float ap = ap_per_class[class_id];
                        float precision = precision_per_class[class_id];
                        float recall = recall_per_class[class_id];
                        float global_ap = global_ap_per_class[class_id];
                        float global_precision = global_precision_per_class[class_id];
                        float global_recall = global_recall_per_class[class_id];

                        summaryFile << std::setw(3) << std::right << class_id << " "
                                    << std::setw(13) << std::left << class_label
                                    << std::setw(3) << std::right << num_gt
                                    << std::setw(5) << cumul_tp
                                    << std::setw(6) << cumul_fp << "   "
                                    << std::setw(11) << std::left << std::round(precision*10000.0f)/10000.0f
                                    << std::setw(9) << std::round(recall*10000.0f)/10000.0f
                                    << std::setw(10) << std::round(ap*10000.0f)/10000.0f;
                        if(report_global_metrics)
                        {
                            summaryFile << "| "
                                        << std::setw(9) << std::right << global_cumul_tp
                                        << std::setw(6) << global_cumul_fp << "   "
                                        << std::setw(11) << std::left << std::round(global_precision*10000.0f)/10000.0f
                                        << std::setw(9) << std::round(global_recall*10000.0f)/10000.0f
                                        << std::setw(10) << std::round(global_ap*10000.0f)/10000.0f;
                        }
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

                    // write processing time details to summary
                    double time_sum = 0;
                    for(auto it : times)
                    {
                        if(it.first == "complete") continue;
                        time_sum += (it.second / 1000);
                    }
                    summaryFile << "\n\ncomplete time: " << times["complete"] / 1000 << " [s]" << ", sum all steps: " << time_sum << " [s]" << std::endl;
                    summaryFile << "times per step:\n";
                    summaryFile << "create flann index: " << std::setw(10) << std::setfill(' ') << times["flann"] / 1000 << " [s]" << std::endl;
                    summaryFile << "compute normals:    " << std::setw(10) << std::setfill(' ') << times["normals"] / 1000 << " [s]" << std::endl;
                    summaryFile << "compute keypoints:  " << std::setw(10) << std::setfill(' ') << times["keypoints"] / 1000 << " [s]" << std::endl;
                    summaryFile << "compute features:   " << std::setw(10) << std::setfill(' ') << times["features"] / 1000 << " [s]" << std::endl;
                    summaryFile << "cast votes:         " << std::setw(10) << std::setfill(' ') << times["voting"] / 1000 << " [s]" << std::endl;
                    summaryFile << "find maxima:        " << std::setw(10) << std::setfill(' ') << times["maxima"] / 1000 << " [s]" << std::endl;

                    // compute average metrics
                    float mAP = 0;
                    float mPrec = 0;
                    float mRec = 0;
                    float global_mAP = 0;
                    float global_mPrec = 0;
                    float global_mRec = 0;
                    for(int idx = 0; idx < ap_per_class.size(); idx++)
                    {
                        mAP += ap_per_class[idx];
                        mPrec += precision_per_class[idx];
                        mRec += recall_per_class[idx];
                        global_mAP += global_ap_per_class[idx];
                        global_mPrec += global_precision_per_class[idx];
                        global_mRec += global_recall_per_class[idx];
                    }
                    mAP /= ap_per_class.size();
                    mPrec /= ap_per_class.size();
                    mRec /= ap_per_class.size();
                    float fscore = 2*mPrec*mRec / (mPrec+mRec);
                    global_mAP /= ap_per_class.size();
                    global_mPrec /= ap_per_class.size();
                    global_mRec /= ap_per_class.size();
                    float global_fscore = 2*global_mPrec*global_mRec / (global_mPrec+global_mRec);

                    if(report_global_metrics)
                    {
                        summaryFile << std::endl << std::endl;
                        summaryFile << "global detector metrics:" << std::endl;
                        summaryFile << "global mAP:            " << std::setw(7) << std::round(global_mAP*10000.0f)/10000.0f    << " (" << std::round(global_mAP*10000.0f)/100.0f   << " %)" << std::endl;
                        summaryFile << "global mean precision: " << std::setw(7) << std::round(global_mPrec*10000.0f)/10000.0f  << " (" << std::round(global_mPrec*10000.0f)/100.0f << " %)" << std::endl;
                        summaryFile << "global mean recall:    " << std::setw(7) << std::round(global_mRec*10000.0f)/10000.0f   << " (" << std::round(global_mRec*10000.0f)/100.0f  << " %)" << std::endl;
                        summaryFile << "global f-score:        " << std::setw(7) << std::round(global_fscore*10000.0f)/10000.0f << " (" << std::round(global_fscore*10000.0f)/100.0f<< " %)" << std::endl << std::endl;
                    }
                    summaryFile << std::endl << std::endl;
                    summaryFile << "main metrics:" << std::endl;
                    summaryFile << "       mAP:            " << std::setw(7) << std::round(mAP*10000.0f)/10000.0f    << " (" << std::round(mAP*10000.0f)/100.0f   << " %)" << std::endl;
                    summaryFile << "       mean precision: " << std::setw(7) << std::round(mPrec*10000.0f)/10000.0f  << " (" << std::round(mPrec*10000.0f)/100.0f << " %)" << std::endl;
                    summaryFile << "       mean recall:    " << std::setw(7) << std::round(mRec*10000.0f)/10000.0f   << " (" << std::round(mRec*10000.0f)/100.0f  << " %)" << std::endl;
                    summaryFile << "       f-score:        " << std::setw(7) << std::round(fscore*10000.0f)/10000.0f << " (" << std::round(fscore*10000.0f)/100.0f<< " %)" << std::endl << std::endl;

                    // complete and close summary file
                    summaryFile << "total processing time: " << timer.format(4, "%w") << " seconds \n";
                    summaryFile.close();
                }
                else
                {
                    std::cerr << "number of point clouds does not match the number of groundtruth files or is zero" << std::endl;
                    return 1;
                }
            }
        }
        catch (const ism3d::Exception& e) {
            std::cerr << e.what() << std::endl;
            return 1;
        }
        catch (...) {
            std::cerr << "an exception occurred" << std::endl;
            return 1;
        }
    }

    return 0;
}
