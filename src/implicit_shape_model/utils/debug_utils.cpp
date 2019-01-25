/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "debug_utils.h"

#include <fstream>

namespace ism3d
{

void DebugUtils::writeToFile(std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr>> &features,
                            std::string filename)
{
    std::ofstream ofs, ofsl;
    ofs.open((filename+".txt").c_str(), std::ofstream::out | std::ofstream::trunc);
    ofsl.open((filename+"_labels.txt").c_str(), std::ofstream::out | std::ofstream::trunc);

    for(auto it : features)
    {
        unsigned label = it.first;
        for(auto cloud : it.second)
        {
            for(ISMFeature feat : cloud->points)
            {
                for(float val : feat.descriptor)
                {
                    ofs << val << " ";
                }
                ofs << std::endl;
                ofsl << label << std::endl;
            }
        }
    }

    ofs.close();
    ofsl.close();
}


void DebugUtils::writeOutForDebug(const std::map<unsigned, std::vector<std::pair<int, float> > > &sorted_list,
                                 const std::string &type)
{
    // NOTE: this is only for debug to write out additional information about selected features; check absolute path below
    bool debug_flag_write_out = false;

    if(!debug_flag_write_out) return;

    std::ofstream wfs, ifs;
    wfs.open("/home/vseib/Desktop/cwids/" + type + "_scores_sorted.txt", std::ofstream::out);
    ifs.open("/home/vseib/Desktop/cwids/" + type + "_indices_sorted.txt", std::ofstream::out);

    wfs << "scoring type: " << type << ", num classes: " << sorted_list.size() << std::endl << std::endl;
    ifs << "scoring type: " << type << ", num classes: " << sorted_list.size() << std::endl << std::endl;

    wfs << "features per class:" << std::endl;
    ifs << "features per class:" << std::endl;

    for(int c = 0; c < sorted_list.size(); c++)
    {
        wfs << c << ": " << sorted_list.at(c).size() << std::endl;
        ifs << c << ": " << sorted_list.at(c).size() << std::endl;
    }

    wfs << std::endl << std::endl;
    wfs << "scores:" << std::endl;
    ifs << std::endl << std::endl;
    ifs << "indices:" << std::endl;

    for(int c = 0; c < sorted_list.size(); c++)
    {
        wfs << "class " << c << ":" << std::endl;
        ifs << "class " << c << ":" << std::endl;
        auto class_list = sorted_list.at(c);

        int quarter_step = 0.25 * class_list.size();
        for(int e = 0; e < class_list.size(); e++)
        {
            if(e % 25 == 0)
            {
                wfs << std::endl;
                ifs << std::endl;
            }

            if(e % quarter_step == 0)
            {
                wfs << " <--|--> " << std::endl;
                ifs << " <--|--> " << std::endl;
            }

            wfs << class_list.at(e).second << " ";
            ifs << class_list.at(e).first << " ";
        }
        wfs << std::endl << std::endl;
        ifs << std::endl << std::endl;
    }

    wfs.close();
    ifs.close();
}
}
