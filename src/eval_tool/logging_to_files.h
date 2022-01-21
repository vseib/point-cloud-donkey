/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2022, Viktor Seib
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

#pragma once

#include <fstream>
#include "eval_helpers_detection.h"
#include "../implicit_shape_model/voting/voting_maximum.h"

namespace filelog
{

void writeLogPerCloud(const std::string& point_cloud,
                      const std::string& ismFile,
                      const std::string& gt_file,
                      const std::string& out_path,
                      const std::vector<ism3d::VotingMaximum>& maxima)
{
    unsigned tmp = point_cloud.find_last_of('/');
    if(tmp == std::string::npos) tmp = 0;
    std::string fileWithoutFolder = point_cloud.substr(tmp+1);

    std::cout << "writing detection log" << std::endl;
    std::string outFileName = out_path;
    outFileName.append("/");
    outFileName.append(fileWithoutFolder);
    outFileName.append(".txt");

    std::ofstream file;
    file.open(outFileName.c_str(), std::ios::out);
    file << "ISM3D detection log, filename: " << ismFile << ", point cloud: " << point_cloud
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


void writeDetectionSummaryHeader(std::ofstream &summaryFile,
                                 const std::string& outFile,
                                 const bool report_global_metrics)
{

    std::string outFileName = outFile;
    outFileName.append("/summary.txt");
    summaryFile.open(outFileName.c_str(), std::ios::out);

    summaryFile << "  class       num gt   tp    fp   precision  recall   AP      f-score";
    if(report_global_metrics)
                summaryFile << "        | global tp    fp   precision  recall   AP      f-score";
    summaryFile << std::endl;
}

void computeAndWriteNextClassSummary(std::ofstream &summaryFile,
                                     MetricsCollection &mc,
                                     const std::string &class_label,
                                     const unsigned class_id,
                                     const std::vector<DetectionObject>& class_objects_gt,
                                     const bool report_global_metrics,
                                     const float dist_threshold,
                                     int &num_gt_dataset,
                                     int &cumul_tp_dataset,
                                     int &cumul_fp_dataset)
{
    // these variables sum over each class
    int num_gt = int(class_objects_gt.size());
    int cumul_tp = 0;
    int cumul_fp = 0;
    int global_cumul_tp = 0;
    int global_cumul_fp = 0;

    // if there are no detections for this class
    if(mc.det_class_map.find(class_label) == mc.det_class_map.end())
    {
        mc.ap_per_class[class_id] = 0;
        mc.precision_per_class[class_id] = 0;
        mc.recall_per_class[class_id] = 0;
    }
    else
    {
        std::vector<DetectionObject> &class_objects_det = mc.det_class_map.at(class_label);

        float precision, recall, ap;
        std::vector<int> tp_list, fp_list;
        std::tie(precision, recall, ap, cumul_tp, cumul_fp, tp_list, fp_list) = computeAllMetrics(class_objects_gt,
                                                                         class_objects_det,
                                                                         dist_threshold);
        mc.tps_per_class.insert({class_label, tp_list});
        mc.fps_per_class.insert({class_label, fp_list});

        mc.precision_per_class[class_id] = precision;
        mc.recall_per_class[class_id] = recall;
        mc.ap_per_class[class_id] = ap;
    }

    if(report_global_metrics)
    {
        // if there are no detections for this class in global detector
        if(mc.det_class_map_global.find(class_label) == mc.det_class_map_global.end())
        {
            mc.global_ap_per_class[class_id] = 0;
            mc.global_precision_per_class[class_id] = 0;
            mc.global_recall_per_class[class_id] = 0;
        }
        else
        {
            std::vector<DetectionObject> class_objects_det = mc.det_class_map_global.at(class_label);

            float precision, recall, ap;
            std::tie(precision, recall, ap, global_cumul_tp, global_cumul_fp, std::ignore, std::ignore) = computeAllMetrics(class_objects_gt,
                                                                             class_objects_det,
                                                                             dist_threshold);
            mc.global_precision_per_class[class_id] = precision;
            mc.global_recall_per_class[class_id] = recall;
            mc.global_ap_per_class[class_id] = ap;
        }
    }

    // log class to summary
    float ap = mc.ap_per_class[class_id];
    float precision = mc.precision_per_class[class_id];
    float recall = mc.recall_per_class[class_id];
    float global_ap = mc.global_ap_per_class[class_id];
    float global_precision = mc.global_precision_per_class[class_id];
    float global_recall = mc.global_recall_per_class[class_id];
    float fscore = 0.0f;
    if((precision+recall) > 0.0f)
        fscore = 2*precision*recall/(precision+recall);
    float global_fscore = 0.0f;
    if((global_precision+global_recall) > 0.0f)
        global_fscore = 2*global_precision*global_recall/(global_precision+global_recall);

    summaryFile << std::setw(3) << std::right << class_id << " "
                << std::setw(13) << std::left << class_label
                << std::setw(3) << std::right << num_gt
                << std::setw(5) << cumul_tp
                << std::setw(6) << cumul_fp << "   "
                << std::setw(11) << std::left << std::round(precision*10000.0f)/10000.0f
                << std::setw(9) << std::round(recall*10000.0f)/10000.0f
                << std::setw(8) << std::round(ap*10000.0f)/10000.0f
                << std::setw(10) << std::round(fscore*10000.0f)/10000.0f;
    if(report_global_metrics)
    {
        summaryFile << "| "
                    << std::setw(9) << std::right << global_cumul_tp
                    << std::setw(6) << global_cumul_fp << "   "
                    << std::setw(11) << std::left << std::round(global_precision*10000.0f)/10000.0f
                    << std::setw(9) << std::round(global_recall*10000.0f)/10000.0f
                    << std::setw(8) << std::round(global_ap*10000.0f)/10000.0f
                    << std::setw(10) << std::round(global_fscore*10000.0f)/10000.0f;
    }
    summaryFile << std::endl;

    // accumulate values of complete dataset
    num_gt_dataset += num_gt;
    cumul_tp_dataset += cumul_tp;
    cumul_fp_dataset += cumul_fp;
}


void computeAndWritePrecisionRecall(const std::string &out_file,
                                    MetricsCollection &mc,
                                    float &overall_ap)
{
    std::vector<float> precisions;
    std::vector<float> recalls;
    std::tie(precisions, recalls, overall_ap) = computePrecisionRecallForPlotting(mc.det_class_map, mc.gt_class_map,
                                                                                  mc.tps_per_class, mc.fps_per_class);
    std::string plot_filename = out_file;
    plot_filename.append("/precision-recall.txt");
    std::ofstream plot_file;
    plot_file.open(plot_filename.c_str(), std::ios::out);
    plot_file << "# recall precision" << std::endl;
    for(unsigned ppos = 0; ppos < precisions.size(); ppos++)
    {
        plot_file  << recalls[ppos] << " " << precisions[ppos] << std::endl;
    }
    plot_file.close();
}

void computeAndWriteFinalMetrics(std::ofstream &summaryFile,
                                 const MetricsCollection &mc,
                                 const int &num_gt_dataset,
                                 const int &cumul_tp_dataset,
                                 const int &cumul_fp_dataset,
                                 const float &overall_ap,
                                 std::map<std::string, double> &times,
                                 const std::string &total_time,
                                 const bool report_global_metrics)
{
    // store sums
    float overall_precision = cumul_tp_dataset / float(cumul_tp_dataset+cumul_fp_dataset);
    float overall_recall = cumul_tp_dataset / float(num_gt_dataset);
    float overall_fscore = 2*overall_precision*overall_recall/(overall_precision+overall_recall);;
    summaryFile << "---------------------------------------------------------------------" << std::endl;
    summaryFile << "Overall:" << std::setw(12) << std::right << num_gt_dataset
                << std::setw(5) << std::right << cumul_tp_dataset
                << std::setw(6) << std::right << cumul_fp_dataset << "   "
                << std::setw(11) << std::left << std::round(overall_precision*10000.0f)/10000.0f
                << std::setw(9) << std::round(overall_recall*10000.0f)/10000.0f
                << std::setw(8) << std::round(overall_ap*10000.0f)/10000.0f
                << std::setw(10) << std::round(overall_fscore*10000.0f)/10000.0f;

    // compute average metrics
    float mAP = 0;
    float mPrec = 0;
    float mRec = 0;
    float global_mAP = 0;
    float global_mPrec = 0;
    float global_mRec = 0;
    for(int idx = 0; idx < mc.ap_per_class.size(); idx++)
    {
        mAP += mc.ap_per_class[idx];
        mPrec += mc.precision_per_class[idx];
        mRec += mc.recall_per_class[idx];
        global_mAP += mc.global_ap_per_class[idx];
        global_mPrec += mc.global_precision_per_class[idx];
        global_mRec += mc.global_recall_per_class[idx];
    }
    mAP /= mc.ap_per_class.size();
    mPrec /= mc.ap_per_class.size();
    mRec /= mc.ap_per_class.size();
    float fscore = 2*mPrec*mRec / (mPrec+mRec);
    global_mAP /= mc.ap_per_class.size();
    global_mPrec /= mc.ap_per_class.size();
    global_mRec /= mc.ap_per_class.size();
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
    // write processing time details to summary
    double time_sum = 0;
    for(auto it : times)
    {
        if(it.first == "complete") continue;
        time_sum += (it.second / 1000);
    }
    summaryFile << std::endl;
    summaryFile << "complete time: " << times["complete"] / 1000 << " [s]" << ", sum all steps: " << time_sum << " [s]" << std::endl;
    summaryFile << "times per step:\n";
    summaryFile << "create flann index: " << std::setw(10) << std::setfill(' ') << times["flann"] / 1000 << " [s]" << std::endl;
    summaryFile << "compute normals:    " << std::setw(10) << std::setfill(' ') << times["normals"] / 1000 << " [s]" << std::endl;
    summaryFile << "compute keypoints:  " << std::setw(10) << std::setfill(' ') << times["keypoints"] / 1000 << " [s]" << std::endl;
    summaryFile << "compute features:   " << std::setw(10) << std::setfill(' ') << times["features"] / 1000 << " [s]" << std::endl;
    summaryFile << "cast votes:         " << std::setw(10) << std::setfill(' ') << times["voting"] / 1000 << " [s]" << std::endl;
    summaryFile << "find maxima:        " << std::setw(10) << std::setfill(' ') << times["maxima"] / 1000 << " [s]" << std::endl << std::endl;

    summaryFile << "total processing time: " << total_time << " seconds \n";
    summaryFile.close();
}

} // namespace filelog
