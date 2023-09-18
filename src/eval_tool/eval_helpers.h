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

#ifndef EVALHELPERS_H
#define EVALHELPERS_H

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <map>

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

unsigned convertLabel(std::string &label, std::map<std::string, unsigned> &labels_map, std::map<unsigned, std::string> &labels_rmap);

unsigned convertClassLabel(std::string& class_label)
{
    return convertLabel(class_label, class_labels_map, class_labels_rmap);
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

LabelUsage parseFileList(std::string &input_file_name,
                   std::vector<std::string> &filenames,
                   std::vector<unsigned> &class_labels,
                   std::vector<unsigned> &instance_labels,
                   std::string &mode)
{
    // parse input
    std::ifstream infile(input_file_name);
    std::string file;
    std::string class_label;
    std::string instance_label;
    bool using_instances = false;

    // special treatment of first line: determine mode
    infile >> file;         // in the first line: #
    infile >> class_label;  // in the first line: the mode ("train" or "test")
    infile >> instance_label; // in the first line: "inst" or first element of second line

    if(file == "#" && (class_label == "train" || class_label == "test"))
    {
        mode = class_label;
        if (instance_label == "inst")
        {
            using_instances = true;
        }
        if (instance_label == "detection")
        {
            std::cerr << "ERROR: You are using a detection data set with the classification eval_tool! Use the binary 'eval_tool_detection' instead." << std::endl;
            exit(1);
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
        // if no instances are used, the first filename has already been read into variable "instance_label"
        file = instance_label;
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
            // initialize with identity mapping, just in case ... (e.g. pipeline_knopp is using it)
            updateInstanceClassMapping(converted_class_label, converted_class_label);
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


#endif // EVALHELPERS_H
