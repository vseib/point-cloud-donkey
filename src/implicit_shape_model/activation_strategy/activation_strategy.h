/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_ACTIVATIONSTRATEGY_H
#define ISM3D_ACTIVATIONSTRATEGY_H

#include <list>

#include "../utils/ism_feature.h"
#include "../utils/json_object.h"
#include <flann/flann.h>

#include "../codebook/codebook.h"
#include "../codebook/codeword.h"

namespace ism3d
{
    class Codebook;
    class Codeword;
    class Distance;

    /**
     * @brief The ActivationStrategy class
     * The activation strategy matches a feature against codewords and returns those codewords
     * that have been determined as similar to the feature according to the chosen strategy.
     */
    class ActivationStrategy
            : public JSONObject
    {
    public:
        virtual ~ActivationStrategy();

        /**
         * @brief Match the feature against a list of codewords
         * @param feature the feature to match against the codebook
         * @param codewords a list of codewords
         * @param distance the distance measure which determines the distance between feature and codeword
         * @return a list of activated codewords
         */
        std::vector<std::shared_ptr<Codeword> > operate(const ISMFeature& feature,
                                                             const std::vector<std::shared_ptr<Codeword> >& codewords,
                                                             const Distance* distance);

        void setIsDetection()
        {
            m_is_detection = true;
        }

    protected:
        ActivationStrategy();

        virtual std::vector<std::shared_ptr<Codeword> > activate(const ISMFeature&,
                                                                   const std::vector<std::shared_ptr<Codeword> >&) const = 0;
        const Distance& distance() const;
        const Distance* m_distance;

        bool m_use_distance_ratio;
        float m_distance_ratio_threshold;
        bool m_is_detection; // indicates whether training or detection/testing is running
    };
}

#endif // ISM3D_ACTIVATIONSTRATEGY_H
