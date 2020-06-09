/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_FLANN_HELPER_H
#define ISM3D_FLANN_HELPER_H

#include <vector>
#include <flann/flann.hpp>
#include "distance.h"
#include "../codebook/codeword.h"
#include "ism_feature.h"

#include "utils.h"

namespace ism3d
{
// flann
class FlannHelper
{
public:

    FlannHelper(int descriptor_size, int num_codewords) :
        dataset(new float[descriptor_size * num_codewords], num_codewords, descriptor_size)
    {
        m_index_created = false;
    }

    ~FlannHelper();

    void createDataset(const std::vector<std::shared_ptr<Codeword> > &codewords);

    void createDataset(pcl::PointCloud<ISMFeature>::Ptr global_features);

    void buildIndex(std::string dist_type, int kd_trees);

    std::string getDistType()
    {
        return m_dist_type;
    }

    bool m_index_created;
    flann::Matrix<float> dataset;

    // TODO VS make this nicer
    std::shared_ptr<flann::Index<flann::L2<float>>> getIndexL2();
    std::shared_ptr<flann::Index<flann::ChiSquareDistance<float>>> getIndexChi();

    std::shared_ptr<void> m_index;

    std::string m_dist_type;
};
}

#endif // ISM3D_FLANN_HELPER_H
