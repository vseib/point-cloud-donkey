#ifndef PCL_ISM_H
#define PCL_ISM_H

#include <vector>
#include <string>
#include <pcl/features/feature.h>
#include <pcl/features/fpfh.h>

//#include <pcl/recognition/implicit_shape_model.h>
#include "implicit_shape_model.h"

typedef pcl::PointXYZ PointT;

class PclIsm
{

public:

    PclIsm();

    virtual ~PclIsm()
    {
    }

    void train(const std::vector<std::string> &filenames,
               const std::vector<unsigned> &labels,
               const std::string &output_file) const;

    std::vector<std::pair<unsigned, float>> classify(const std::string &filename) const;

    bool loadModel(std::string &filename);

    void setClassLabels(const std::vector<unsigned> &class_labels);

private:

    pcl::ism::ImplicitShapeModelEstimation<153, PointT, pcl::Normal>::ISMModelPtr m_model;

    std::vector<unsigned> m_class_labels;

    std::string m_feature_type;

    float m_normal_radius;
    float m_feature_radius;
    float m_keypoint_sampling_radius;
};

#endif // PCL_ISM_H
