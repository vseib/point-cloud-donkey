/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2018, Viktor Seib, Norman Link
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
#include <vector>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/program_options/errors.hpp>
#include "../implicit_shape_model/implicit_shape_model.h"


bool write_log_to_files = false;
bool log_info = true;

// determines how labels are used
enum class LabelUsage
{
    CLASS_ONLY,        // only class labels, no instance labels
    BOTH_GIVEN,        // both labels given, needs to be decided which is primary
    CLASS_PRIMARY,     // use class labels as primary labels, instance labels are secondary
    INSTANCE_PRIMARY   // use instance labels as primary labels, store class labels in look-up table
};

// mappings from real labels (string) to label ids and back
std::map<std::string, unsigned> class_labels_map;
std::map<std::string, unsigned> instance_labels_map;
std::map<unsigned, std::string> class_labels_rmap;
std::map<unsigned, std::string> instance_labels_rmap;
// mapping from instance label ids to class label ids
std::map<unsigned, unsigned> instance_to_class_map;
LabelUsage label_usage;



// represents one object instance, either detected or ground truth
struct DetectionObject
{
    std::string class_label;
    std::string instance_label;
    std::string global_class_label; // only for detection, not in ground truth
    Eigen::Vector3f position;
    float visibility_ratio;     // only in ground truth, used to filter detections
    float confidence;           // only for detection, not in ground truth
    // using path because some datasets use repeating names in subfolders
    std::string filepath;       // filename of gt annotations, not the point cloud

    DetectionObject(std::string class_label, std::string instance_label, std::string global_class_label, Eigen::Vector3f position,
                    float visibility_ratio, float confidence, std::string filepath)
        : class_label(class_label), instance_label(instance_label), global_class_label(global_class_label), position(position),
          visibility_ratio(visibility_ratio), confidence(confidence), filepath(filepath) {}

    void print()
    {
        LOG_INFO("Object from " << filepath);
        LOG_INFO("    class: " << class_label << ", instance: " << instance_label << ", visible: " << visibility_ratio << " at position: (" <<
                 position.x() << ", " << position.y() << ", " << position.z() << "), confidence: " << confidence);
    }
};


std::tuple<float,float>
get_precision_recall(std::vector<int> &true_positives, std::vector<int> &false_positives, int num_gt)
{
    int tp = 0;
    for(unsigned i = 0; i < true_positives.size(); i++)
    {
        tp += true_positives[i];
    }

    int fp = 0;
    for(unsigned i = 0; i < false_positives.size(); i++)
    {
        fp += false_positives[i];
    }

    float precision = tp / float(fp + tp);
    float recall = num_gt == 0 ? 0 : float(tp) / num_gt;

    return {precision, recall};
}

std::tuple<std::vector<int>,std::vector<int>>
match_gt_objects(std::vector<DetectionObject> &class_objects_gt,
                    std::vector<DetectionObject> &class_objects_det,
                    float distance_threshold)
{
    // sort objects by confidence then loop over sorted list, starting with hightest conf
    std::sort(class_objects_det.begin(), class_objects_det.end(), [](DetectionObject &obj1, DetectionObject &obj2)
    {
       return obj1.confidence > obj2.confidence;
    });

    std::vector<bool> used_gt(class_objects_gt.size(), false);
    std::vector<int> tp(class_objects_det.size(), 0);
    std::vector<int> fp(class_objects_det.size(), 0);

    for(unsigned det_idx = 0; det_idx < class_objects_det.size(); det_idx++)
    {
        float best_dist = 0.0f;
        int best_index = -1;

        // find matching gt object with smallest distance
        const DetectionObject &det_obj = class_objects_det[det_idx];
        for(unsigned gt_idx = 0; gt_idx < class_objects_gt.size(); gt_idx++)
        {
            const DetectionObject &gt_obj = class_objects_gt[gt_idx];
            if(det_obj.filepath != gt_obj.filepath)
            {
                continue;
            }

            float distance = (gt_obj.position - det_obj.position).norm();
            // record index and smallest dist if this gt object was not used before
            if(distance > best_dist && !used_gt[gt_idx])
            {
                best_dist = distance;
                best_index = int(gt_idx);
            }
        }

        if(best_dist > distance_threshold || best_index == -1)
        {
            fp[det_idx] = 1;
        }
        else
        {
            tp[det_idx] = 1;
            used_gt[unsigned(best_index)] = true;
        }
    }

    return {tp, fp};
}

void rearrangeObjects(std::vector<DetectionObject> &source_list,
                      std::map<std::string, std::vector<DetectionObject>> &target_map)
{
    for(const auto &object : source_list)
    {
        if(target_map.find(object.class_label) != target_map.end())
        {
            target_map.at(object.class_label).push_back(object);
        }
        else // class id not yet in map
        {
            target_map.insert({object.class_label,{object}});
        }
    }
}

DetectionObject convertMaxToObj(const ism3d::VotingMaximum& max, std::string &filename)
{
    std::string class_name, instance_name, global_class;
    // lookup real class ids if instances were used as primary labels
    if(label_usage == LabelUsage::INSTANCE_PRIMARY)
    {
        class_name = class_labels_rmap[instance_to_class_map[max.classId]];
        instance_name = instance_labels_rmap[max.classId]; // class id actually has the instance label
        global_class = class_labels_rmap[instance_to_class_map[max.globalHypothesis.classId]];
    }
    else
    {
        class_name = class_labels_rmap[max.classId];
        instance_name = instance_labels_rmap[max.instanceId];
        global_class = class_labels_rmap[max.globalHypothesis.classId];
    }

    float confidence = max.weight; // TODO VS make conf not weight!
    Eigen::Vector3f position(max.position[0], max.position[1], max.position[2]);
    // create object
    return DetectionObject{class_name, instance_name, global_class, position, -1.0f, confidence, filename};
}


std::vector<DetectionObject> parseGtFile(std::string &filename)
{
    std::vector<DetectionObject> objects;

    std::ifstream file(filename);
    std::string line;
    while(std::getline(file, line))
    {
        if(line == "")
            continue;

        std::stringstream iss(line);
        std::string item;
        std::vector<std::string> tokens;
        while(std::getline(iss, item, ' '))
        {
            if(item == "")
                continue;
            tokens.push_back(item);
        }

        if(tokens.size() == 4 || tokens.size() == 5 || tokens.size() == 6)
        {
            std::string class_name = tokens[0];
            std::string instance_name = class_name; // overwrite later if available
            int offset = 0;

            if(tokens.size() == 6) // gt annotation with visibility ratio and class and instance labels
            {
                instance_name = tokens[1];
                offset += 1;
            }
            float visibility = 1.0f;
            if(tokens.size() == 5) // visibility ratio given, no instance labels
            {
                std::string visibility_str = tokens[1+offset];
                visibility_str = visibility_str.substr(1, visibility_str.find_first_of(')')-1);
                visibility = std::stof(visibility_str);
                offset += 1;
            }
            Eigen::Vector3f position(std::stof(tokens[1+offset]), std::stof(tokens[2+offset]), std::stof(tokens[3+offset]));

            // create object
            objects.emplace_back(class_name, instance_name, class_name, position, visibility, 1.0f, filename);
        }
        else
        {
            LOG_ERROR("Something is wrong, tokens has size: " << tokens.size() << ", expected: 4, 5 or 6!");
            exit(1);
        }
    }
    return objects;
}


unsigned convertLabel(std::string& label,
                      std::map<std::string, unsigned>& labels_map,
                      std::map<unsigned, std::string>& labels_rmap)
{
    if(labels_map.find(label) != labels_map.end())
    {
        return labels_map[label];
    }
    else
    {
        size_t cur_size = labels_map.size();
        labels_map.insert({label, cur_size});
        labels_rmap.insert({cur_size, label});
        return cur_size;
    }
}

void updateInstanceClassMapping(unsigned instance_label, unsigned class_label)
{
    if(instance_to_class_map.find(instance_label) != instance_to_class_map.end())
    {
        // already added do nothing
    }
    else
    {
        instance_to_class_map.insert({instance_label, class_label});
    }
}

void parseFileListDetectionTest(std::string &input_file_name,
                                      std::vector<std::string> &filenames,
                                      std::vector<std::string> &gt_files)
{
    // parse input
    std::ifstream infile(input_file_name);
    std::string file;
    std::string gt_file;
    std::string additional_flag;
    std::string additional_flag_2;

    // special treatment of first line: determine mode
    infile >> file;     // in the first line: #
    infile >> gt_file;  // in the first line: the mode ("train" or "test") - here: must be "test"
    infile >> additional_flag; // in the first line mandatory: "detection"
    infile >> additional_flag_2; // optional: "inst"

    if(file == "#" && (gt_file == "train" || gt_file == "test"))
    {
        if("test" != gt_file)
        {
            LOG_ERROR("ERROR: Check your command line arguments! You specified to '"<< "test" << "', but your input file says '" << gt_file << "'!");
            exit(1);
        }
        if (additional_flag != "detection")
        {
            LOG_ERROR("ERROR: You are using a classification data set with the detection eval_tool! Use the binary 'eval_tool' instead.");
            exit(1);
        }
        if (additional_flag_2 == "inst")
        {
            // don't care here
        }

        // if no instances are used, the first filename has already been read into variable "additional_flag_2"
        file = additional_flag_2;
        filenames.push_back(file);
        infile >> gt_file;
        gt_files.push_back(gt_file);
        // read remaining lines
        while(infile >> file >> gt_file)
        {
            if (file[0] == '#') continue; // allows to comment out lines
            filenames.push_back(file);
            gt_files.push_back(gt_file);
        }
    }
}


LabelUsage parseFileList(std::string &input_file_name,
                   std::vector<std::string> &filenames,
                   std::vector<unsigned> &class_labels,
                   std::vector<unsigned> &instance_labels,
                   std::string mode)
{
    // parse input
    std::ifstream infile(input_file_name);
    std::string file;
    std::string class_label;
    std::string additional_flag;
    std::string additional_flag_2;
    std::string instance_label;
    bool using_instances = false;

    // special treatment of first line: determine mode
    infile >> file;         // in the first line: #
    infile >> class_label;  // in the first line: the mode ("train" or "test")
    infile >> additional_flag; // in the first line mandatory: "detection"
    infile >> additional_flag_2; // optional: "inst"

    if(file == "#" && (class_label == "train" || class_label == "test"))
    {
        if(mode != class_label)
        {
            LOG_ERROR("ERROR: Check your command line arguments! You specified to '"<< mode << "', but your input file says '" << class_label << "'!");
            exit(1);
        }
        if (additional_flag != "detection")
        {
            LOG_ERROR("ERROR: You are using a classification data set with the detection eval_tool! Use the binary 'eval_tool' instead.");
            exit(1);
        }
        if (additional_flag_2 == "inst")
        {
            using_instances = true;
        }
    }

    // process remaining lines
    if (using_instances)
    {
        // other lines contain a filename, a class label and an instance label
        while(infile >> file >> class_label >> instance_label)
        {
            if (file[0] == '#') continue; // allows to comment out lines
            filenames.push_back(file);
            unsigned converted_class_label = convertLabel(class_label, class_labels_map, class_labels_rmap);
            unsigned converted_instance_label = convertLabel(instance_label, instance_labels_map, instance_labels_rmap);
            updateInstanceClassMapping(converted_instance_label, converted_class_label);
            class_labels.push_back(converted_class_label);
            instance_labels.push_back(converted_instance_label);
        }
    }
    else
    {
        // if no instances are used, the first filename has already been read into variable "additional_flag_2"
        file = additional_flag_2;
        infile >> class_label;

        filenames.push_back(file);
        unsigned converted_class_label = convertLabel(class_label, class_labels_map, class_labels_rmap);

        class_labels.push_back(converted_class_label);
        // read remaining lines
        while(infile >> file >> class_label)
        {
            if (file[0] == '#') continue; // allows to comment out lines
            filenames.push_back(file);
            unsigned converted_class_label = convertLabel(class_label, class_labels_map, class_labels_rmap);
            class_labels.push_back(converted_class_label);
        }
    }

    if(using_instances)
    {
        return LabelUsage::BOTH_GIVEN; // instance labels given, decide later which is primary
    }
    else
    {
        return LabelUsage::CLASS_ONLY; // no instance labels given, use only class labels
    }
}


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
                    label_usage = parseFileList(input_file_name, filenames, class_labels, instance_labels, "train");
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

                // TODO VS: check if this really works in both eval_tools, need to save info to recover label_usage for detection
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
                    LOG_ERROR("No input file provided! You need to provide an input file with -f");
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
                    // TODO VS: check in both eval tools if this really works
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
                            LOG_ERROR("Mismatch in instance label usage between config file (.ism) and trained file (.ismd)!");
                            LOG_ERROR("Config file has InstanceLabelsPrimary as " << ism.isInstancePrimaryLabel() << ", while trained file has " << !ism.isInstancePrimaryLabel());
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
                                        file << maximum.voteIndices.size() << ", ";
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

                    rearrangeObjects(gt_objects, gt_class_map);
                    rearrangeObjects(detected_objects, det_class_map);

                    float dist_threshold = ism.getDetectionThreshold();

                    // collect all metrics
                    // combined detection - primary metrics
                    std::vector<float> ap_per_class(gt_class_map.size(), 0.0);
                    std::vector<float> precision_per_class(gt_class_map.size(), 0.0);
                    std::vector<float> recall_per_class(gt_class_map.size(), 0.0);
//                    // metrics for instance labels (if available)
//                    std::vector<float> instance_ap_per_class(gt_class_map.size(), 0.0);
//                    std::vector<float> instance_precision_per_class(gt_class_map.size(), 0.0);
//                    std::vector<float> instance_recall_per_class(gt_class_map.size(), 0.0);
//                    // metrics for the global classifier (if used)
//                    std::vector<float> global_ap_per_class(gt_class_map.size(), 0.0);
//                    std::vector<float> global_precision_per_class(gt_class_map.size(), 0.0);
//                    std::vector<float> global_recall_per_class(gt_class_map.size(), 0.0);

                    for(auto item : gt_class_map)
                    {
                        std::string class_label = item.first;
                        unsigned class_id = class_labels_map[class_label];
                        std::vector<DetectionObject> class_objects_gt = item.second;

                        // if there are no detections for this class
                        if(det_class_map.find(class_label) == det_class_map.end())
                        {
                            ap_per_class[class_id] = 0;
                            precision_per_class[class_id] = 0;
                            recall_per_class[class_id] = 0;
//                            instance_ap_per_class[class_id] = 0;
//                            instance_precision_per_class[class_id] = 0;
//                            instance_recall_per_class[class_id] = 0;
//                            global_ap_per_class[class_id] = 0;
//                            global_precision_per_class[class_id] = 0;
//                            global_recall_per_class[class_id] = 0;
                            continue;
                        }
                        std::vector<DetectionObject> class_objects_det = det_class_map.at(class_label);

                        // match detections and ground truth to get list of tp and fp
                        std::vector<int> tp, fp;
                        std::tie(tp, fp) = match_gt_objects(class_objects_gt, class_objects_det, dist_threshold);

                        // compute precision and recall
                        int num_gt = int(class_objects_gt.size());
                        double precision, recall;
                        std::tie(precision, recall) = get_precision_recall(tp, fp, num_gt);
                        precision_per_class[class_id] = precision;
                        recall_per_class[class_id] = recall;

                        // compute average precision metric
                        float ap = 0.0;
                        float cumul_tp = 0.0;
                        for(unsigned i = 0; i < tp.size(); i++)
                        {
                            if(tp[i] == 1)
                            {
                                cumul_tp += 1;
                                ap += (cumul_tp / (i+1)) * (1.0/num_gt);
                            }
                        }
                        ap_per_class[class_id] = ap;

                        int cumul_fp = 0;
                        for(auto elem : fp)
                        {
                            cumul_fp += elem;
                        }

                        // log class to summary
                        summaryFile << "class " << class_id << ": " << class_label
                                    << std::setw(10) << std::setfill(' ') << " num gt: " << num_gt
                                    << std::setw(10) << std::setfill(' ') << " tp: " << int(cumul_tp)
                                    << std::setw(10) << std::setfill(' ') << " fp: " << cumul_fp
                                    << std::setw(10) << std::setfill(' ') << " precision: " << precision
                                    << std::setw(10) << std::setfill(' ') << " recall: " << recall
                                    << std::setw(10) << std::setfill(' ') << " AP: " << ap << std::endl;
                    }

                    // write processing time details to summary
                    double time_sum = 0;
                    for(auto it : times)
                    {
                        if(it.first == "complete") continue;
                        time_sum += (it.second / 1000);
                    }
                    summaryFile << "\n\n\ncomplete time: " << times["complete"] / 1000 << " [s]" << ", sum all steps: " << time_sum << " [s]" << std::endl;
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
                    for(int idx = 0; idx < ap_per_class.size(); idx++)
                    {
                        mAP += ap_per_class[idx];
                        mPrec += precision_per_class[idx];
                        mRec += recall_per_class[idx];
                    }
                    mAP /= ap_per_class.size();
                    mPrec /= ap_per_class.size();
                    mRec /= ap_per_class.size();

                    // complete and close summary file
                    summaryFile << std::endl << std::endl;
                    summaryFile << "mAP: " << mAP << " (" << (mAP * 100.0f) << "%)" << std::endl;
                    summaryFile << "mean precision: " << mPrec << " (" << (mPrec * 100.0f) << "%)" << std::endl;
                    summaryFile << "mean recall: " << mRec << " (" << (mRec * 100.0f) << "%)" << std::endl << std::endl;

                    summaryFile << "Total processing time: " << timer.format(4, "%w") << " seconds \n";
                    summaryFile.close();
                }
                else
                {
                    std::cerr << "number of point clouds does not match the number of groundtruth files" << std::endl;
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
