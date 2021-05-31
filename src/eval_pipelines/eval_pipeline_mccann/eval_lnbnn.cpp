#include <iostream>
#include <fstream>
#include "lnbnn.h"

int main (int argc, char** argv)
{
    if(argc != 3)
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

    // parse input
    std::vector<std::string> filenames;
    std::vector<std::string> labels;
    std::string mode;

    std::ifstream infile(dataset);
    std::string file;
    std::string label;

    while(infile >> file >> label)
    {
        // special treatment of first line: determine mode
        if(file == "#" && (label == "train" || label == "test"))
        {
            mode = label;
        }
        else // other lines contain a filename and a label
        {
            filenames.push_back(file);
            labels.push_back(label);
        }
    }

    // process training or testing
    std::shared_ptr<Lnbnn> lnbnn(new Lnbnn());

    if(mode == "train")
    {
        std::cout << "Started training!" << std::endl;
        lnbnn->train(filenames, labels, model);
    }
    else if(mode == "test")
    {
        std::cout << "Started testing!" << std::endl;

        if(lnbnn->loadModel(model))
        {
            int num_correct = 0;
            int num_total = 0;

            std::ofstream outfile("output_"+model);

            for(std::string filename : filenames)
            {
                std::cout << "Processing file " << filename << std::endl;

                std::vector<std::pair<unsigned, float>> results;
                results = lnbnn->classify(filename);

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
            std::cerr << "ERROR: could not load model from file " << model << "!" << std::endl;
        }
    }
    else
    {
        std::cerr << "ERROR: wrong mode specified: " << mode << "! Must be train or test!" << std::endl;
    }

    return (0);
}
