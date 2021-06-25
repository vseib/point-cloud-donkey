#ifndef LNBNN_H
#define LNBNN_H

#include <vector>
#include <string>
#include <flann/flann.hpp>
#include <pcl/features/feature.h>
#include "../../implicit_shape_model/utils/ism_feature.h"

using namespace ism3d;

typedef pcl::PointXYZRGB PointT;

class Lnbnn
{

public:

    Lnbnn();

    virtual ~Lnbnn()
    {
    }

    void train(const std::vector<std::string> &filenames,
               const std::vector<unsigned> &class_labels,
               const std::vector<unsigned> &instance_labels,
               const std::string &output_file) const;

    std::vector<std::pair<unsigned, float>> classify(const std::string &filename) const;

    bool loadModel(std::string &filename);

    void setLabels(std::map<unsigned, std::string> &class_labels,
                   std::map<unsigned, std::string> &instance_labels,
                   std::map<unsigned, unsigned> &instance_to_class_map)
    {
        m_class_labels = class_labels;
        m_instance_labels = instance_labels;
        m_instance_to_class_map = instance_to_class_map;
    }

    std::map<unsigned, std::string> getClassLabels()
    {
        return m_class_labels;
    }

    std::map<unsigned, std::string> getInstanceLabels()
    {
        return m_instance_labels;
    }

    std::map<unsigned, unsigned> getInstanceClassMap()
    {
        return m_instance_to_class_map;
    }

private:

    flann::Matrix<float> createFlannDataset();

    std::vector<std::pair<unsigned, float>> accumulateClassDistances(const pcl::PointCloud<ISMFeature>::Ptr& features) const;

    bool saveModelToFile(std::string &filename, std::map<unsigned, pcl::PointCloud<ISMFeature>::Ptr> &all_features) const;

    bool loadModelFromFile(std::string& filename);


    template<typename T>
    static bool containsValue(const std::vector<T> &vec, const T &val)
    {
        for(T elem : vec)
            if (elem == val) return true;

        return false;
    }

    std::map<unsigned, std::string> m_class_labels;
    std::map<unsigned, std::string> m_instance_labels;
    std::map<unsigned, unsigned> m_instance_to_class_map;

    int m_number_of_classes;
    std::vector<unsigned> m_class_lookup;
    pcl::PointCloud<ISMFeature>::Ptr m_features;

    flann::Index<flann::L2<float>> m_flann_index;
};

#endif // LNBNN_H
