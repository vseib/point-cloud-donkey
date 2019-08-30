/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */


#include "features_cgf.h"

#define PCL_NO_PRECOMPILE
#include <pcl/common/centroid.h>

#include "../third_party/cgf/cgf.h"
#include "../third_party/cnpy/cnpy.h"

namespace ism3d
{
    FeaturesCGF::FeaturesCGF()
    {
        addParameter(m_radius, "Radius", 0.1);
    }

    FeaturesCGF::~FeaturesCGF()
    {
    }

    pcl::PointCloud<ISMFeature>::Ptr FeaturesCGF::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                         pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                         pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                                                       pcl::search::Search<PointT>::Ptr search)
    {

        // NOTE: this block of code generates and computes the CGF
        // (three steps: generate raw features, convert to npy format and apply embedding)
        // using the code in  third_party/cgf/cgf.cpp   provided at    https://github.com/marckhoury/CGF
        //
        // NOTE: in order to use this feature descriptor you need to download the trained weights from the above repository

        // generate raw features
        std::string raw_output = "raw_cgf.lzf";
        std::vector<std::string> argv = {
          "",
          "-s", std::to_string(m_radius),       // feature support
          "-m", std::to_string(m_radius*0.05),  // smallest subdivision
          "-l", std::to_string(m_radius*0.75),  // lrf radius
          "-t", std::to_string(8),              // number of threads
          "-o", raw_output                      // output file
        };
        cgf_main(argv, pointCloudWithoutNaNNormals, keypoints);

        // convert raw features to npy format
        std::string compressed_output = "raw_compressed_cgf";
        std::string command_compress = "python compress.py " + raw_output + " 2244 " + compressed_output;
        int unused = std::system(command_compress.c_str());

        // apply feature embedding
        std::string feature_file = "features_cgf";
        std::string model_file_with_path = "embed_model_910000.ckpt";
        std::string command_embed = "python embedding.py --evaluate=True --checkpoint_model="+model_file_with_path
                +" --output_file="+ feature_file + " " + compressed_output + ".npz";
        unused = std::system(command_embed.c_str());

        // read computed features
        std::map<std::string, cnpy::NpyArray> arrs = cnpy::npz_load(feature_file+".npz");
        cnpy::NpyArray arr = arrs["data"];
        std::vector<float> loaded_data = arr.as_vec<float>();
        int num_features = arr.shape[0];
        int num_dims = arr.shape[1];

        Eigen::Vector4d centroid;
        pcl::compute3DCentroid(*pointCloudWithoutNaNNormals, centroid);

        // create descriptor point cloud
        pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
        features->resize(num_features);

        for (int i = 0; i < num_features; i++)
        {
            ISMFeature& feature = features->at(i);

            // store the descriptor
            feature.descriptor.resize(num_dims);
            for (int j = 0; j < feature.descriptor.size(); j++)
                feature.descriptor[j] = loaded_data[i*num_dims + j];

            // store distance to centroid
            feature.centerDist = (keypoints->at(i).getVector3fMap() - Eigen::Vector3f(centroid.x(), centroid.y(), centroid.z())).norm();
        }

        // delete generated files
        std::string cleanup_command = "rm " + raw_output + " " + compressed_output + ".npz" + " " + feature_file + ".npz";
        unused = std::system(cleanup_command.c_str());

        return features;
    }

    std::string FeaturesCGF::getTypeStatic()
    {
        return "CGF";
    }

    std::string FeaturesCGF::getType() const
    {
        return FeaturesCGF::getTypeStatic();
    }
}
