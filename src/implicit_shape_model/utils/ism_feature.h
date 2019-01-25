/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_ISMFEATURE_H
#define ISM3D_ISMFEATURE_H

#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>
#include <pcl/point_representation.h>

namespace ism3d
{
    // ISM Feature point type
    struct ISMFeature
    {
        ISMFeature();

        // point structure
        PCL_ADD_POINT4D;

        // descriptor data
        pcl::ReferenceFrame referenceFrame;
        std::vector<float> descriptor;
        float centerDist; // distance of keypoint that produced this feature to object's centroid

        // only used for global descriptors, -1 otherwise
        // learned in training, needed to estimate object size in cluttered regions during recognition
        float globalDescriptorRadius;

        // used for:
        //   1) global features during detection
        //   2) local features during training (weights computation)
        int classId;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    } EIGEN_ALIGN16;

    // ISM Feature point representation used to create custom k-D tree
    class ISMFeaturePointRepresentation : public pcl::PointRepresentation <ISMFeature>
    {
      using pcl::PointRepresentation<ISMFeature>::nr_dimensions_;
    public:
      ISMFeaturePointRepresentation ()
      {
        // Define the number of dimensions
        nr_dimensions_ = 1;
      }

      ISMFeaturePointRepresentation (int dim)
      {
          nr_dimensions_ = dim;
      }

      // Override the copyToFloatArray method to define our feature vector
      virtual void copyToFloatArray (const ISMFeature &p, float * out) const
      {
        for(int i = 0; i < nr_dimensions_; i++)
        {
            out[i] = p.descriptor.at(i);
        }
      }
    };
}

POINT_CLOUD_REGISTER_POINT_STRUCT(
    ism3d::ISMFeature,
    (float, x, x)
    (float, y, y)
    (float, z, z)
)

#endif // ISM3D_ISMFEATURE_H
