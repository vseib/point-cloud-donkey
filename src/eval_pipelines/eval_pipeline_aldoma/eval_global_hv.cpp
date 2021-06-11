#include <iostream>
#include <fstream>
#include "pcl_orcg.h"

int main (int argc, char** argv)
{
    if(argc != 3)
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
    std::string scene = argv[2]; // TODO VS: parse filename from scene description

    // parse input
    std::vector<std::string> filenames;
    std::vector<std::string> labels;

    std::ifstream infile(dataset);
    std::string file;
    std::string label;

    while(infile >> file >> label)
    {
        // special treatment of first file: determine mode
        if(file == "#" && (label == "train" || label == "test"))
        {
            // skip: this is for compatibility with other train and test object files
        }
        else // other files contain a filename and a label
        {
            filenames.push_back(file);
            labels.push_back(label);
        }
    }

    std::shared_ptr<Orcg> orcg(new Orcg());

    std::cout << "Started matching!" << std::endl;

    if(orcg->prepareScene(scene))
    {
        int num_correct = 0;
        int num_total = 0;

        std::ofstream outfile("output_"+scene);

        for(std::string filename : filenames)
        {
            std::cout << "Processing file " << filename << std::endl;

            std::vector<std::pair<unsigned, float>> results;
            results = orcg->findObjectInScene(filename);

            // TODO VS: implement result evaluation (this current version is not for localization, but for classification)

            // print results
            for(int i = 0; i < results.size(); i++)
            {
                std::pair<unsigned, float> res = results.at(i);
                std::cout << i << ": label: " << res.first << ", score: " << -res.second << std::endl;
            }

            int result_label = -1;
            if(results.size() > 0)
                result_label = results.at(0).first;

            outfile << (num_total+1) << ". file: " << filename << ", gt label: " << labels.at(num_total)
                       << ", classified as: " << result_label << std::endl;

            // evaluate results
            if(results.size() > 0)
            {
                if(labels.at(num_total) == std::to_string(results.at(0).first))
                {
                    num_correct++;
                }
            }
            num_total++;
        }

        std::cout << "Classified " << num_correct << " of " << num_total << " (" << (num_correct/((float)num_total))*100 << " %) files correctly." << std::endl;
        outfile << std::endl << std::endl << "Classified " << num_correct << " of " << num_total
                   << " (" << (num_correct/((float)num_total))*100 << " %) files correctly." << std::endl;

        outfile.close();
    }
    else
    {
        std::cerr << "ERROR: could not load model from file " << scene << "!" << std::endl;
    }

    return (0);
}
