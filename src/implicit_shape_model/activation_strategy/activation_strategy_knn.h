/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_ACTIVATIONSTRATEGYKNN_H
#define ISM3D_ACTIVATIONSTRATEGYKNN_H

#include "activation_strategy.h"

#include "../codebook/codebook.h"
#include "../utils/utils.h"
#include "../utils/distance.h"

namespace ism3d
{
/**
     * @brief The ActivationStrategyKNN class
     * Activates the k best matching codewords for the feature.
     */
class ActivationStrategyKNN
        : public ActivationStrategy
{
public:
    ActivationStrategyKNN();
    ~ActivationStrategyKNN();

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
        std::vector<std::vector<int> > indices;
        std::vector<std::vector<float> > distances;
        flann::SearchParams params = flann_exact_match ? flann::SearchParams(-1) : flann::SearchParams(128);
        index.knnSearch(query, indices, distances, m_k, params);

        delete[] query.ptr();

        // create output data
        for (int i = 0; i < (int)indices[0].size(); i++)
        {
            activatedCodewords.push_back(codewords[indices[0][i]]);
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

    std::vector<std::shared_ptr<Codeword> > activate(const ISMFeature&,
                                                     const std::vector<std::shared_ptr<Codeword> >& ) const
    {
        std::vector<std::shared_ptr<Codeword> > activatedCodewords;
        return activatedCodewords;
    }

private:
    int m_k;
};
}

#endif // ISM3D_ACTIVATIONSTRATEGYKNN_H
