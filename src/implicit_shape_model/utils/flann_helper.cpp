/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "flann_helper.h"

namespace ism3d
{

FlannHelper::~FlannHelper()
{
    delete[] dataset.ptr();
}

void FlannHelper::createDataset(const std::vector<std::shared_ptr<Codeword>> &codewords)
{
    // build dataset
    for(int i = 0; i < (int)codewords.size(); i++)
    {
        const std::shared_ptr<Codeword>& codeword = codewords[i];
        if (!codeword.get())
        {
            LOG_WARN("invalid codeword, this might lead to errors");
            continue;
        }

        std::vector<float> descriptor = codeword->getData();
        for(int j = 0; j < (int)descriptor.size(); j++)
        {
            dataset[i][j] = descriptor.at(j);
        }
    }
}

void FlannHelper::createDataset(pcl::PointCloud<ISMFeature>::Ptr global_features)
{
    // build dataset
    for(int i = 0; i < (int)global_features->size(); i++)
    {
        const ISMFeature feat = global_features->at(i);
        std::vector<float> descriptor = feat.descriptor;
        for(int j = 0; j < (int)descriptor.size(); j++)
        {
            dataset[i][j] = descriptor.at(j);
        }
    }
}

void FlannHelper::buildIndex(std::string dist_type, int kd_trees)
{
    if(dist_type == "Euclidean")
    {
        m_index = std::make_shared<flann::Index<flann::L2<float>>>(dataset, flann::KDTreeIndexParams(kd_trees));
        (std::static_pointer_cast<flann::Index<flann::L2<float>>>(m_index))->buildIndex();
    }
    if(dist_type == "ChiSquared")
    {
        m_index = std::make_shared<flann::Index<flann::ChiSquareDistance<float>>>(dataset, flann::KDTreeIndexParams(kd_trees));
        (std::static_pointer_cast<flann::Index<flann::ChiSquareDistance<float>>>(m_index))->buildIndex();
    }

    m_index_created = true;
    m_dist_type = dist_type;
}

std::shared_ptr<flann::Index<flann::L2<float>>> FlannHelper::getIndexL2()
{
    return std::static_pointer_cast<flann::Index<flann::L2<float>>>(m_index);
}

std::shared_ptr<flann::Index<flann::ChiSquareDistance<float>>> FlannHelper::getIndexChi()
{
    LOG_WARN("----------- flann helper return, index created: " << m_index_created);
    return std::static_pointer_cast<flann::Index<flann::ChiSquareDistance<float>>>(m_index);
}

}
