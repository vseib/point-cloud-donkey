/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_ACTIVATIONSTRATEGYINN_H
#define ISM3D_ACTIVATIONSTRATEGYINN_H

#include "activation_strategy.h"

#include "../codebook/codebook.h"
#include "../utils/utils.h"
#include "../utils/distance.h"

namespace ism3d
{
/**
     * @brief The ActivationStrategyINN class
     * Activates the 1 best matching codeword using iterative nearest neighbors
     */
class ActivationStrategyINN
        : public ActivationStrategy
{
public:
    ActivationStrategyINN();
    ~ActivationStrategyINN();

    static std::string getTypeStatic();
    std::string getType() const;

    template<typename T>
    std::vector<std::shared_ptr<Codeword> > activateINN(const ISMFeature& feature,
                                                          const std::vector<std::shared_ptr<Codeword> >& codewords,
                                                          const flann::Index<T> index,
                                                          const bool flann_exact_match) const
    {
        std::vector<std::shared_ptr<Codeword> > activatedCodewords;

        // insert the query point
        flann::Matrix<float> query(new float[feature.descriptor.size()], 1, feature.descriptor.size());
        for(int i = 0; i < feature.descriptor.size(); i++)
        {
            query[0][i] = feature.descriptor.at(i);
        }

        // prepare results
        std::vector<std::vector<int> > indices;
        std::vector<std::vector<float> > distances;

        for(int it = 0; it < m_num_iterations; it++)
        {
            // INN: identification step
            flann::SearchParams params = flann_exact_match ? flann::SearchParams(-1) : flann::SearchParams(128);
            index.knnSearch(query, indices, distances, 1, params);

            // INN: estimation step
            std::vector<float> neighbor = codewords[indices[0][0]]->getData();
            float factor = 0;
            for(int i = 0; i < feature.descriptor.size(); i++)
            {
                factor += neighbor[i] * query[0][i];
            }
            // INN: update step
            for(int i = 0; i < feature.descriptor.size(); i++)
            {
                query[0][i] = query[0][i] + m_residual_weight*(query[0][i] - factor*neighbor[i]);
            }
        }

        delete[] query.ptr();

        // create output data
        for (int i = 0; i < (int)indices[0].size(); i++)
        {
            activatedCodewords.push_back(codewords[indices[0][i]]);
        }

        return activatedCodewords;
    }

protected:

    int m_num_iterations;
    float m_residual_weight;

    std::vector<std::shared_ptr<Codeword> > activate(const ISMFeature&,
                                                       const std::vector<std::shared_ptr<Codeword> >& ) const
    {
        std::vector<std::shared_ptr<Codeword> > activatedCodewords;
        return activatedCodewords;
    }

private:

};
}

#endif // ISM3D_ACTIVATIONSTRATEGYINN_H
