/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2018, Viktor Seib, Norman Link
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 * * list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef ISM3D_IMPLICITSHAPEMODEL_H
#define ISM3D_IMPLICITSHAPEMODEL_H

#include <map>
#include <vector>
#include <tuple>
#include <boost/shared_ptr.hpp>
#include <boost/signals2.hpp>
#include <boost/timer/timer.hpp>

#include "utils/json_object.h"
#include "utils/flann_helper.h"
#include "codebook/codebook.h"
#include "utils/ism_feature.h"
#include "utils/point_cloud_resizing.h"
#include "keypoints/keypoints.h"
#include "features/features.h"
#include "feature_ranking/feature_ranking.h"
#include "clustering/clustering.h"
#include "voting/voting.h"

#define PCL_NO_PRECOMPILE
#include <pcl/surface/mls.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/search.h>
#include <Eigen/Core>

namespace ism3d
{
    class Vote;
    class VotingMaximum;
    class Distance;

    /**
     * @brief The ImplicitShapeModel class
     * This class is the main entry point for the implicit shape model detection algorithm.
     * Use the functions addTrainingModel() to add training objects and train() to actually
     * train the implicit shape model.
     * Use the detect() function for analyzing an unclassified point cloud for object occurrences
     * and returning a list of object hypotheses.
     * Use the inherited functions readObject() and writeObject() to read and write the object
     * to a file. The implicit shape model representation contains two files, the .ism file
     * containing the parameter set and the .ismd file containing the actual training data. Change
     * the parameters in the first file. For an initial parameter set, write the empty model
     * first and then change the parameters.
     */
    class ImplicitShapeModel
            : public JSONObject
    {
    public:
        ImplicitShapeModel();
        ~ImplicitShapeModel();

        /**
         * @brief Clear all data.
         */
        void clear();

        /**
         * @brief Add a new training model for a specified class id.
         * @param filename the filename to the training model to add
         * @param class_id the class id for the training model
         * @param instance_id the instance id for the training model
         * @return false if the training model has already been added, true if successful
         */
        bool addTrainingModel(const std::string& filename, unsigned class_id, unsigned instance_id);

        /**
         * @brief Train the implicit shape model using all objects added before
         */
        void train();

        /**
         * @brief add_normals Computes normals for the filename specified and saves the cloud
         * @param filename filename of the object to add_normals
         * @param folder folder to save the cloud with normals to
         * @return true if no errors occured, false otherwise
         */
        bool add_normals(const std::string& filename, const std::string& folder);

        /**
         * @brief Detect unknown object instances using the implicit shape model.
         * @param pointCloud the point cloud in which objects should be detected
         * @return a tuple with list of detected object positions, time measurements
         */
        std::tuple<std::vector<VotingMaximum>, std::map<std::string, double> > detect(pcl::PointCloud<PointT>::ConstPtr pointCloud);

        /**
         * @brief Detect unknown object instances using the implicit shape model.
         * @param pointCloud the point cloud in which objects should be detected
         * @param hasNormals specify whether the input point cloud contains normal information
         * @return a tuple with list of detected object positions, time measurements
         */
        std::tuple<std::vector<VotingMaximum>, std::map<std::string, double> > detect(pcl::PointCloud<PointNormalT>::ConstPtr pointCloud, bool hasNormals = true);

        /**
         * @brief Detect unknown object instances using the implicit shape model.
         * @param filename the filename to the point cloud in which objects should be detected
         * @param maxima return paramerter: a list of detected object positions
         * @param times map for time measurements
         * @return true if no error occured
         */
        bool detect(const std::string& filename, std::vector<VotingMaximum>& maxima, std::map<std::string, double> &times);

        /**
         * @brief Get the codebook for this implicit shape model.
         * @return the codebook for this implicit shape model
         */
        const Codebook* getCodebook() const;

        /**
         * @brief Get the voting class.
         * @return the voting class
         */
        const Voting* getVoting() const;

        /**
         * @brief Used to enable and disable signals (disable to speed up command line evaluation, enable for GUI)
         * @param s - new state (true: enabled, false: disabled) (default in constructor: true)
         */
        void setSignalsState(bool s)
        {
            m_enable_signals = s;
        }

        /**
         * @brief setLogging Whether or not INFO should be logged.
         * @param l if true logger level will be INFO, otherwise logger level will be WARN
         */
        void setLogging(bool l);

        /**
         * @brief setLabels Sets class and instance labels as members to be safed for later use
         * @param class_labels the map with class labels used during training
         * @param instance_labels the map with instance labels used during training
         */
        void setLabels(std::map<unsigned, std::string> &class_labels, std::map<unsigned, std::string> &instance_labels)
        {
            m_class_labels = class_labels;
            m_instance_labels = instance_labels;
        }

        std::map<unsigned, std::string> getClassLabels()
        {
            return m_class_labels;
        }

        std::map<unsigned, std::string> getInstanceLabels()
        {
            return m_instance_labels;
        }


        // signals
        boost::signals2::signal<void(pcl::PointCloud<PointT>::ConstPtr)> m_signalPointCloud;
        boost::signals2::signal<void(const Utils::BoundingBox&)> m_signalBoundingBox;
        boost::signals2::signal<void(pcl::PointCloud<PointT>::ConstPtr, pcl::PointCloud<pcl::Normal>::ConstPtr)> m_signalNormals;
        boost::signals2::signal<void(pcl::PointCloud<ISMFeature>::ConstPtr)> m_signalFeatures;
        boost::signals2::signal<void(const Codebook&)> m_signalCodebook;
        boost::signals2::signal<void(std::vector<VotingMaximum>)> m_signalMaxima;

    protected:
        Json::Value iChildConfigsToJson() const;
        bool iChildConfigsFromJson(const Json::Value&);
        void iSaveData(boost::archive::binary_oarchive &oa) const;
        bool iLoadData(boost::archive::binary_iarchive &ia);
        void iPostInitConfig();

    private:
        void init();
        pcl::PointCloud<PointNormalT>::Ptr loadPointCloud(const std::string& filename);

        // the tuple is: local features, global features, points without NAN, normals without NAN
        std::tuple<pcl::PointCloud<ISMFeature>::ConstPtr, pcl::PointCloud<ISMFeature>::ConstPtr,
                    pcl::PointCloud<PointT>::ConstPtr, pcl::PointCloud<pcl::Normal>::ConstPtr >
            computeFeatures(pcl::PointCloud<PointNormalT>::ConstPtr, bool, boost::timer::cpu_timer&, boost::timer::cpu_timer &timer_keypoints,
                            bool compute_global);

        void computeNormals(pcl::PointCloud<PointT>::ConstPtr,
                            pcl::PointCloud<pcl::Normal>::Ptr&,
                            pcl::search::Search<PointT>::Ptr) const;

        void filterNormals(pcl::PointCloud<PointT>::ConstPtr model,
                           pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                           pcl::PointCloud<PointT>::Ptr& modelWithoutNaN,
                           pcl::PointCloud<pcl::Normal>::Ptr& normalsWithoutNaN);

        void writeFeaturesToDisk(std::string file_name,
                                 const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &features,
                                 const std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &globalFeatures,
                                 const std::map<unsigned, std::vector<Utils::BoundingBox> > &boundingBoxes);

        void readFeaturesFromDisk(std::string file_name,
                                  std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &features,
                                  std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &globalFeatures,
                                  std::map<unsigned, std::vector<Utils::BoundingBox> > &boundingBoxes);

        // removes all features with NAN in the given input; output: filtered list
        pcl::PointCloud<ISMFeature>::Ptr removeNaNFeatures(pcl::PointCloud<ISMFeature>::ConstPtr modelFeatures);

        std::map<unsigned, pcl::PointCloud<PointT>::Ptr > analyzeVotingSpacesForDebug
                            (const std::map<unsigned, std::vector<Voting::Vote> > &all_votes,
                             pcl::PointCloud<PointNormalT>::Ptr points);

        void addMaximaForDebug(std::map<unsigned, pcl::PointCloud<PointT>::Ptr> &all_votings, std::vector<VotingMaximum> &positions);

        void trainSVM(std::map<unsigned, std::vector<pcl::PointCloud<ISMFeature>::Ptr> > &features);

        pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> m_mls_smoothing;
        pcl::StatisticalOutlierRemoval<PointNormalT> m_stat_outlier_rem;
        pcl::RadiusOutlierRemoval<PointNormalT> m_radius_outlier_rem;
        pcl::VoxelGrid<PointNormalT> m_voxel_filtering;
        Codebook* m_codebook;
        Keypoints* m_keypoints_detector;
        Features* m_feature_descriptor;
        Features* m_global_feature_descriptor;
        Clustering* m_clustering;
        Voting* m_voting;
        FeatureRanking* m_feature_ranking;

        std::map<unsigned, std::vector<std::string>> m_training_objects_filenames;
        std::map<unsigned, std::vector<unsigned>> m_training_objects_instance_ids;
        std::map<unsigned, std::vector<bool>> m_training_objects_has_normals;
        Distance* m_distance;
        std::string m_distanceType;

        // parameters for preprocessing
        bool m_use_smoothing;
        int m_polynomial_order;
        float m_smoothing_radius;
        bool m_use_statistical_outlier_removal;
        int m_som_mean_k;
        float m_som_std_dev_mul;
        bool m_use_radius_outlier_removal;
        int m_ror_min_neighbors;
        float m_ror_radius;
        bool m_use_voxel_viltering;
        float m_voxel_leaf_size;

        float m_normal_radius;
        int m_consistent_normals_k;
        int m_consistent_normals_method;
        int m_num_threads;
        std::string m_bb_type;
        bool m_set_color_to_zero;
        bool m_enable_voting_analysis;
        std::string m_voting_analysis_output_path;
        bool m_svm_auto_train;
        double m_svm_param_c;
        double m_svm_param_gamma;
        int m_svm_param_k_fold;
        bool m_single_object_mode; // remains here for backward-compatible error throwing

        int m_num_kd_trees;
        bool m_flann_exact_match;

        std::map<int, std::pair<std::string, std::string> > m_id_objects_map; // maps class ids to pairs of <class_name, instance_name>

        bool m_enable_signals;

        std::shared_ptr<FlannHelper> m_flann_helper;
        bool m_index_created;

        std::map<unsigned, std::string> m_class_labels;
        std::map<unsigned, std::string> m_instance_labels;

        // TODO VS temp
        static int m_counter;
        double getElapsedTime(boost::timer::cpu_timer timer, std::string format);
        std::map<std::string, double> m_processing_times;
    };
}

#endif // ISM3D_IMPLICITSHAPEMODEL_H
