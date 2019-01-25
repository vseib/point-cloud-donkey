cmake_minimum_required(VERSION 2.8.3)
project(implicit_shape_model_2)

# set build type
set(CMAKE_BUILD_TYPE Release)

# set optional components
set(USE_VCGLIB false)

#find boost
find_package(Boost REQUIRED COMPONENTS system timer signals date_time program_options serialization)

# find pcl
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

if(PCL_FOUND)
    message(STATUS "Found PCL version " ${PCL_VERSION})
    if(${PCL_VERSION} VERSION_GREATER 1.8.1)
        add_definitions(-DWITH_PCL_GREATER_1_8)
        #add_compile_definitions(WITH_PCL_GREATER_1_8)
    else()
        message(STATUS "  NOT using new PCL features: FLARE reference frame will not be available!")
    endif()
endif()

# find vtk
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# optional vcglib
if(USE_VCGLIB)
    message(STATUS "Using VCGLIB")
    include_directories("/usr/local/include/vcglib-1.0.1")
    add_definitions(-DUSE_VCGLIB)
    #add_compile_definitions(USE_VCGLIB)
else()
    message(STATUS "NOT using VCGLIB: EMST will not be available for normal's orientation!")
endif()

# find opencv
find_package(OpenCV REQUIRED)

# find and activate OpenMP multicore processing standard
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "Found OpenMP")
endif()

# find zlib (required for third_party/cnpy - which is required by the CGF feature)
find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

# implicit_shape_model
add_library(implicit_shape_model
    # cpp-files
    activation_strategy/activation_strategy.cpp
    activation_strategy/activation_strategy_threshold.cpp
    activation_strategy/activation_strategy_knn.cpp
    activation_strategy/activation_strategy_inn.cpp
    classifier/custom_SVM.cpp
    clustering/clustering.cpp
    clustering/clustering_agglomerative.cpp
    clustering/clustering_kmeans.cpp
    clustering/clustering_kmeans_count.cpp
    clustering/clustering_kmeans_factor.cpp
    clustering/clustering_kmeans_thumb_rule.cpp
    clustering/clustering_kmeans_hartigan.cpp
    clustering/clustering_none.cpp
    codebook/codebook.cpp
    codebook/codeword.cpp
    codebook/codeword_distribution.cpp
    features/features.cpp
    features/features_shot.cpp
    features/features_bshot.cpp
    features/features_shot_global.cpp
    features/features_short_shot.cpp
    features/features_cshot.cpp
    features/features_cshot_global.cpp
    #features/features_short_cshot.cpp
    features/features_pfh.cpp
    features/features_fpfh.cpp
    features/features_rift.cpp
    features/features_esf.cpp
    features/features_esf_local.cpp
    features/features_3dsc.cpp
    features/features_usc.cpp
    features/features_usc_global.cpp
    features/features_rsd.cpp
    features/features_grsd.cpp
    features/features_spin_image.cpp
    features/features_rops.cpp
    features/features_narf.cpp
    features/features_vfh.cpp
    features/features_cvfh.cpp
    features/features_ourcvfh.cpp
    features/features_dummy.cpp
    features/features_cgf.cpp
    feature_ranking/feature_ranking.cpp
    feature_ranking/ranking_uniform.cpp
    feature_ranking/ranking_naive_bayes.cpp
    feature_ranking/ranking_incremental.cpp
    feature_ranking/ranking_knn_activation.cpp
    feature_ranking/ranking_strangeness.cpp
    implicit_shape_model.cpp
    keypoints/keypoints.cpp
    keypoints/keypoints_harris3d.cpp
    keypoints/keypoints_iss3d.cpp
    keypoints/keypoints_voxel_grid.cpp
    keypoints/keypoints_sift3d.cpp
    utils/debug_utils.cpp
    utils/distance.cpp
    utils/ism_feature.cpp
    utils/json_parameter_base.cpp
    utils/json_object.cpp
    utils/exception.cpp
    utils/utils.cpp
    utils/normal_orientation.cpp
    utils/point_cloud_resizing.cpp
    voting/voting.cpp
    voting/voting_hough_3d.cpp
    voting/voting_mean_shift.cpp
    # h-files
    activation_strategy/activation_strategy_factory.h
    clustering/clustering_factory.h
    codebook/codebook_factory.h
    codebook/codeword_factory.h
    codebook/codeword_distribution_factory.h
    features/features_factory.h
    feature_ranking/ranking_factory.h
    utils/factory.h
    utils/flann_helper.cpp
    utils/json_parameter_traits.h
    keypoints/keypoints_factory.h
    voting/voting_factory.h
    # external
    third_party/libgdiam-1.3/gdiam.cpp
    third_party/liblzf-3.6/lzf_c.c
    third_party/liblzf-3.6/lzf_d.c
    third_party/cnpy/cnpy.cpp
    third_party/cgf/cgf.cpp
    third_party/pcl_shot_na_lrf/shot_na_lrf.cpp
    third_party/pcl_shot_na_lrf/shot_na_lrf.hpp
    third_party/pcl_shot_na_lrf/shot_na_lrf.h
)

# link library with dependencies
target_link_libraries(implicit_shape_model ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${VTK_LIBRARIES} ${ZLIB_LIBRARIES}
    gomp blitz jsoncpp log4cxx)
