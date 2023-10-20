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
#include "../implicit_shape_model/utils/debug_utils.h"
#include "eval_helpers_detection.h"
#include "logging_to_files.h"

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

    // init logging
    log4cxx::AppenderList appl = log4cxx::Logger::getRootLogger()->getAllAppenders();
    if(appl.size() == 0)
    {
        log4cxx::LayoutPtr layout(new log4cxx::PatternLayout("[\%d{HH:mm:ss}] \%p: \%m\%n"));
        log4cxx::ConsoleAppender* consoleAppender = new log4cxx::ConsoleAppender(layout);
        log4cxx::BasicConfigurator::configure(log4cxx::AppenderPtr(consoleAppender));
        log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());
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
            std::vector<std::string> filenames;       // filenames of point cloud scenes
            std::vector<std::string> annot_filenames; // annotations for point clouds (objects and poses)
            std::vector<unsigned> class_labels;
            std::vector<unsigned> instance_labels;

            // train the ISM
            if (variables.count("train"))
            {
                if(variables.count("inputfile"))
                {
                    // manually parse input file
                    std::string input_file_name = variables["inputfile"].as<std::string>();
                    label_usage = parseFileListDetectionTrain(input_file_name, filenames, class_labels, instance_labels, annot_filenames, "train");
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
                if((filenames.size() > 0 && class_labels.size() > 0)
                    || (filenames.size() > 0 && annot_filenames.size() > 0))
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

                    // standard case where everything is read out from the given file
                    if (models.size() == class_ids.size() && class_ids.size() > 0)
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
                    // special case where additional annotation files are used in training
                    else if(annot_filenames.size() > 0)
                    {
                        for(unsigned cloud_file_idx = 0; cloud_file_idx < filenames.size(); cloud_file_idx++)
                        {
                            std::string point_cloud_filename = filenames.at(cloud_file_idx);
                            std::string gt_file = annot_filenames.at(cloud_file_idx);
                            std::vector<DetectionObject> gt_objects_from_file = parseAnnotationFile(gt_file, point_cloud_filename);

                            // add all objects from the current annotation file
                            std::vector<unsigned> class_ids;
                            std::vector<unsigned> instance_ids;
                            std::vector<std::string> cloud_paths;
                            std::vector<ism3d::Utils::BoundingBox> boxes;

                            pcl::PointCloud<ism3d::PointNormalT>::Ptr pc(new pcl::PointCloud<ism3d::PointNormalT>());
                            pcl::io::loadPCDFile(point_cloud_filename, *pc);

                            for(auto gt_obj : gt_objects_from_file)
                            {
                                class_ids.push_back(convertClassLabel(gt_obj.class_label));
                                instance_ids.push_back(convertClassLabel(gt_obj.instance_label));
                                cloud_paths.push_back(gt_obj.cloud_filepath);
                                // convert eigen quaternion to boost quaternion
                                boost::math::quaternion<float> rot_quat = boost::math::quaternion<float>(gt_obj.bb_orientation.w(),
                                             gt_obj.bb_orientation.x(), gt_obj.bb_orientation.y(), gt_obj.bb_orientation.z());
                                ism3d::Utils::BoundingBox bounding_box(gt_obj.position, ism3d::Utils::normalizeQuat(rot_quat), gt_obj.bb_extent);
                                boxes.push_back(bounding_box);

//                                // TODO VS debug tests
//                                pcl::PointCloud<ism3d::PointNormalT>::Ptr corners = ism3d::DebugUtils::getBoxCorners(bounding_box, 50);
//                                for(auto & corner : corners->points)
//                                {
//                                    pc->push_back(corner);
//                                }
                            }
//                            // TODO VS debug tests
//                            std::string name = "/home/vseib/Desktop/" + point_cloud_filename;
//                            pcl::io::savePCDFile(name, *pc);

                            // add all training objects of the scene with class ids, instance ids and boxes
                            ism.addTrainingModelsWithBoxes(point_cloud_filename, class_ids, instance_ids, boxes);
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
                    parseFileListDetectionTest(input_file_name, filenames, annot_filenames);
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
                else if (filenames.size() > 0 && annot_filenames.size() > 0)  // load pointclouds and groundtruth
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

                    // init label usage: class/instance or both
                    initLabelUsage(ism.isInstancePrimaryLabel());

                    if(filenames.size() > 0) // input inside file given on command line
                    {
                        pointClouds = filenames;
                    }

                    // prepare output
                    if (variables.count("output"))
                    {
                        // create folder for output
                        std::string command = "mkdir ";
                        std::string folder = variables["output"].as<std::string>();
                        command.append(folder);
                        std::ignore = std::system(command.c_str());
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
                        std::string point_cloud = pointClouds.at(i);
                        std::string gt_file = annot_filenames.at(i);
                        std::vector<ism3d::VotingMaximum> maxima;

                        std::cout << "Processing file: " << point_cloud << std::endl;
                        if (!ism.detect(point_cloud, maxima, times))
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
                                std::vector<DetectionObject> gt_objects_from_file = parseAnnotationFile(gt_file, point_cloud);
                                gt_objects.insert(gt_objects.end(), gt_objects_from_file.begin(), gt_objects_from_file.end());
                                // collect all detections
                                for (int i = 0; i < (int)maxima.size(); i++)
                                {
                                    DetectionObject detected_obj = convertMaxToObj(maxima[i], gt_file);
                                    detected_objects.push_back(std::move(detected_obj));
                                }
                                if(write_log_to_files)
                                {
                                    std::string out_path = variables["output"].as<std::string>();
                                    filelog::writeLogPerCloud(point_cloud, ismFile, gt_file, out_path, maxima);
                                }
                            }
                        }
                    }

                    // create common object to manage all metrics
                    MetricsCollection mc;

                    rearrangeObjects(gt_objects, mc.gt_class_map);
                    rearrangeObjects(detected_objects, mc.det_class_map);
                    bool report_global_metrics = ism.isUsingGlobalFeatures();
                    if(report_global_metrics)
                        rearrangeObjects(detected_objects, mc.det_class_map_global, true);

                    mc.resizeVectors();

                    // summary file
                    std::ofstream summaryFile;
                    if (variables.count("output"))
                    {
                        std::string outFile = variables["output"].as<std::string>();
                        filelog::writeDetectionSummaryHeader(summaryFile, outFile, report_global_metrics);
                    }

                    // these variables sum over the whole dataset
                    int num_gt_dataset = 0;
                    int cumul_tp_dataset = 0;
                    int cumul_fp_dataset = 0;

                    std::map<unsigned, float> dist_thresholds = ism.getDetectionThreshold();
                    for(auto item : mc.gt_class_map)
                    {
                        std::string class_label = item.first;
                        unsigned class_id = class_labels_map[class_label];
                        float dist_threshold = dist_thresholds[class_id];
                        std::vector<DetectionObject> class_objects_gt = item.second;

                        filelog::computeAndWriteNextClassSummary(summaryFile, mc, class_label, class_id,
                                                        class_objects_gt, report_global_metrics,
                                                        dist_threshold, num_gt_dataset,
                                                        cumul_tp_dataset, cumul_fp_dataset);
                    }

                    // compute and log values for precision-recall curves
                    std::string outFile = variables["output"].as<std::string>();
                    float overall_ap;
                    filelog::computeAndWritePrecisionRecall(outFile, mc, overall_ap);

                    // store sums and overall metrics
                    std::string total_time = timer.format(4, "%w");
                    filelog::computeAndWriteFinalMetrics(summaryFile, mc, num_gt_dataset,
                                                         cumul_tp_dataset, cumul_fp_dataset,
                                                         overall_ap, times, total_time,
                                                         report_global_metrics);
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
