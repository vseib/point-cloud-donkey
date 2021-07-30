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

#ifndef EVALHELPERSDETECTION_H
#define EVALHELPERSDETECTION_H

#include <Eigen/Core>
#include "eval_helpers.h"
#include "../implicit_shape_model/voting/voting_maximum.h"

// represents one object instance, either detected or ground truth
struct DetectionObject
{
    std::string class_label;
    std::string instance_label;
    std::string global_class_label; // only for detection, not in ground truth
    Eigen::Vector3f position;
    float occlusion_ratio;     // only in ground truth, used to filter detections
    float confidence;           // only for detection, not in ground truth
    // using path because some datasets use repeating names in subfolders
    std::string filepath;       // filename of gt annotations, not the point cloud

    DetectionObject(std::string class_label, std::string instance_label, std::string global_class_label, Eigen::Vector3f position,
                    float occlusion_ratio, float confidence, std::string filepath)
        : class_label(class_label), instance_label(instance_label), global_class_label(global_class_label), position(position),
          occlusion_ratio(occlusion_ratio), confidence(confidence), filepath(filepath) {}

    void print()
    {
        LOG_INFO("Object from " << filepath);
        LOG_INFO("    class: " << class_label << ", instance: " << instance_label << ", visible: " << occlusion_ratio << " at position: (" <<
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


std::tuple<float, float, float, int, int>
computeMetrics(std::vector<DetectionObject> &class_objects_gt,
                    std::vector<DetectionObject> &class_objects_det,
                    float dist_threshold)
{
    // match detections and ground truth to get list of tp and fp
    std::vector<int> tp, fp;
    std::tie(tp, fp) = match_gt_objects(class_objects_gt, class_objects_det, dist_threshold);

    // compute precision and recall
    int num_gt = int(class_objects_gt.size());
    double precision, recall;
    std::tie(precision, recall) = get_precision_recall(tp, fp, num_gt);

    // compute average precision metric
    float ap = 0.0;
    int cumul_tp = 0;
    for(unsigned i = 0; i < tp.size(); i++)
    {
        if(tp[i] == 1)
        {
            cumul_tp += 1;
            ap += (float(cumul_tp) / (i+1)) * (1.0/num_gt);
        }
    }

    int cumul_fp = 0;
    for(auto elem : fp)
    {
        cumul_fp += elem;
    }

    return {float(precision), float(recall), ap, cumul_tp, cumul_fp};
}


void rearrangeObjects(const std::vector<DetectionObject> &source_list,
                      std::map<std::string, std::vector<DetectionObject>> &target_map,
                      bool use_global_class_label = false)
{
    for(const auto &object : source_list)
    {
        if(!use_global_class_label)
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
        else
        {
            if(target_map.find(object.global_class_label) != target_map.end())
            {
                target_map.at(object.global_class_label).push_back(object);
            }
            else // class id not yet in map
            {
                target_map.insert({object.global_class_label,{object}});
            }
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

    float confidence = max.weight;
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

        if(tokens.size() == 5 || tokens.size() == 12)
        {
            std::string class_name = tokens[0];
            std::string instance_name = class_name; // overwrite later if available


            float occlusion = 0.0f;
            std::string occlusion_str = tokens[1];
            occlusion_str = occlusion_str.substr(1, occlusion_str.find_first_of(')')-1);
            occlusion = std::stof(occlusion_str);

            Eigen::Vector3f position(std::stof(tokens[2]), std::stof(tokens[3]), std::stof(tokens[4]));

            if(tokens.size() == 12)
            {
                // TODO VS add these to eval
                Eigen::Vector3f box(std::stof(tokens[5]), std::stof(tokens[6]), std::stof(tokens[7]));
                Eigen::Quaternionf quat(std::stof(tokens[8]), std::stof(tokens[9]), std::stof(tokens[10]), std::stof(tokens[11]));
            }

            // create object
            objects.emplace_back(class_name, instance_name, class_name, position, occlusion, 1.0f, filename);
        }
        else
        {
            LOG_ERROR("Something is wrong, tokens has size: " << tokens.size() << ", expected: 5 or 12!");
            exit(1);
        }
    }
    return objects;
}



LabelUsage parseFileListDetectionTrain(std::string &input_file_name,
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


#endif // EVALHELPERSDETECTION_H
