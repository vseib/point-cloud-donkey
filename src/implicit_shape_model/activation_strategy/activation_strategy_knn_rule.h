/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2021, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_ACTIVATIONSTRATEGYKNNRULE_H
#define ISM3D_ACTIVATIONSTRATEGYKNNRULE_H

#include "activation_strategy.h"

#include "../codebook/codebook.h"
#include "../utils/utils.h"
#include "../utils/distance.h"

namespace ism3d
{
/**
     * @brief The ActivationStrategyKnnRule class
     * Activates the 1 best matching codeword for the feature according to some rule of thumb
     */
class ActivationStrategyKnnRule
        : public ActivationStrategy
{
public:
    ActivationStrategyKnnRule();
    ~ActivationStrategyKnnRule();

    static std::string getTypeStatic();
    std::string getType() const;

    int getK() const
    {
        return m_k;
    }

    template<typename T>
    std::vector<std::shared_ptr<Codeword>> activateKNN(const ISMFeature& feature,
                                                        const std::vector<std::shared_ptr<Codeword> >& codewords,
                                                        const flann::Index<T> &index,
                                                        const bool flann_exact_match) const
    {
        std::vector<std::shared_ptr<Codeword>> activatedCodewords;

        // no need to do anything
        if ((int)codewords.size() <= m_k) {
            for (int i = 0; i < (int)codewords.size(); i++)
                activatedCodewords.push_back(codewords[i]);
            return activatedCodewords;
        }

        // insert the query point
        flann::Matrix<float> query(new float[feature.descriptor.size()], 1, feature.descriptor.size());
        for(int i = 0; i < feature.descriptor.size(); i++)
        {
            query[0][i] = feature.descriptor.at(i);
        }

        // prepare results
        std::vector<std::vector<int>> indices;
        std::vector<std::vector<float>> distances;
        flann::SearchParams params = flann_exact_match ? flann::SearchParams(-1) : flann::SearchParams(128);
        std::cout << "-------------- m_k is  " << m_k << std::endl;
        index.knnSearch(query, indices, distances, 3, params);

        delete[] query.ptr();


        // for now use normal knn with k = 1 during training
        if(!m_is_detection)
        {
            activatedCodewords.push_back(codewords[indices[0][0]]);
        }
        else // apply rule during detection
        {
            std::array<int, 3> class_ids = {codewords.at(indices[0][0])->getClassId(),
                                            codewords.at(indices[0][1])->getClassId(),
                                            codewords.at(indices[0][2])->getClassId()};

            activatedCodewords.clear();

            if(class_ids[0] == class_ids[1] && class_ids[0] == class_ids[2])
            {
                // if all are same, accept match
                activatedCodewords.push_back(codewords[indices[0][0]]);
                std::cout << " ---- case 1" << std::endl;
            }
            else if(class_ids[0] == class_ids[1] && class_ids[0] != class_ids[2])
            {
                // if k1 and k2 are same class
                if(distances[0][1] / distances[0][2] < m_distance_ratio_threshold)
                {
                    // accept if valid distance ratio between k2 and k3 (TODO VS: or k1 and k3?)
                    activatedCodewords.push_back(codewords[indices[0][0]]);
                    std::cout << " ---- case 2" << std::endl;
                }
            }
            else if(class_ids[0] != class_ids[1] && class_ids[1] == class_ids[2])
            {
                // if k2 and k3 are same, but different from k1
                if(distances[0][0] / distances[0][1] >= m_distance_ratio_threshold)
                {
                    // accept __k2__ if distance ratio k1-k2 INvalid? (TODO VS: AND k2-k3 valid?)
                    activatedCodewords.push_back(codewords[indices[0][1]]);
                    std::cout << " ---- case 3" << std::endl;
                }
            }
            else if(class_ids[0] != class_ids[1] && class_ids[1] != class_ids[2])
            {
                // if all are different or k2 different from k1 and k3
                if(distances[0][0] / distances[0][1] < m_distance_ratio_threshold)
                {
                    // accept if valid distance ratio between k1 and k2
                    activatedCodewords.push_back(codewords[indices[0][0]]);
                    std::cout << " ---- case 4" << std::endl;
                }
            }
        }


        //        // TODO VS: fix using binary descriptors with flann, look for flann::LshIndexParams
        //        // convert query to bitset for b-shot
        //        std::bitset<352> queryb;
        //        queryb.reset();
        //        for(int i = 0; i < feature.descriptor.size(); i++)
        //        {
        //            if(feature.descriptor.at(i) == 1.0) queryb.set(i);
        //        }
        //        // iterate over all features (here codewords are used instead of the flann index)
        //        int min_dist = 999;
        //        int min_idx = -1;
        //        for(int c = 0; c < (int)codewords.size(); c++)
        //        {
        //            // convert each codeword descriptor
        //            std::vector<float> descr = codewords.at(c)->getData();
        //            std::bitset<352> temp;
        //            temp.reset();
        //            for(int i = 0; i < descr.size(); i++)
        //            {
        //                if(descr.at(i) == 1.0) temp.set(i);
        //            }
        //            // compute distance on bitset
        //            int dist = (int)(queryb ^ temp).count();
        //            if(dist < min_dist)
        //            {
        //                min_dist = dist;
        //                min_idx = c;
        //            }
        //        }
        //        activatedCodewords.push_back(codewords[min_idx]);

        return activatedCodewords;
    }


protected:

    std::vector<std::shared_ptr<Codeword>> activate(const ISMFeature&,
                                                     const std::vector<std::shared_ptr<Codeword> >& ) const
    {
        std::vector<std::shared_ptr<Codeword> > activatedCodewords;
        return activatedCodewords;
    }

private:
    int m_k;
};
}

#endif // ISM3D_ACTIVATIONSTRATEGYKNNRULE_H
