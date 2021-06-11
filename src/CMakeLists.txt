cmake_minimum_required(VERSION 3.1)
project(ism_3d_new)

# set build type
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

# find pcl
find_package(PCL 1.10 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

if (PCL_FOUND)
    message(STATUS "Found PCL version " ${PCL_VERSION})
endif()

#find boost
find_package(Boost REQUIRED COMPONENTS system date_time program_options timer)

#set output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# find and activate OpenMP multicore processing standard
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "Found OpenMP")
endif()

###########
## Build ##
###########

# include and build implicit_shape_model library
add_subdirectory(implicit_shape_model)
file(COPY implicit_shape_model/third_party/cgf/compress.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/)
file(COPY implicit_shape_model/third_party/cgf/embedding.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/)
# TODO VS make this work later
#file(COPY ../config/default_config_kinect.ism DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/)

add_subdirectory(vtk_utils)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt5 COMPONENTS Core Gui REQUIRED)
include_directories(
    SYSTEM
    ${QT_INCLUDE_DIR}
)


qt5_wrap_cpp(TRAINING_MOCS_HDRS
    training_gui/training_gui.h
    training_gui/model_dlg.h
    training_gui/ism_worker.h
)
add_executable(training_gui
    training_gui/main.cpp
    training_gui/training_gui.cpp
    training_gui/model_dlg.cpp
    training_gui/ism_worker.cpp
    ${TRAINING_MOCS_HDRS}
)
target_link_libraries(training_gui vtk_utils implicit_shape_model ${QT_LIBRARIES} ${VTK_LIBRARIES} ${Boost_LIBRARIES})


qt5_wrap_cpp(MODEL_MOCS_HDRS
    model_gui/model_gui.h
)
add_executable(model_gui
    model_gui/main.cpp
    model_gui/model_gui.cpp
    ${MODEL_MOCS_HDRS}
)
target_link_libraries(model_gui vtk_utils ${QT_LIBRARIES} ${PCL_LIBRARIES} ${VTK_LIBRARIES} ${Boost_LIBRARIES})

#ISM ground truth gui
qt5_wrap_cpp(GROUND_TRUTH_MOCS_HDRS
    ground_truth_gui/ground_truth_gui.h
)
add_executable(ground_truth_gui
    ground_truth_gui/main.cpp
    ground_truth_gui/ground_truth_gui.cpp
    ${GROUND_TRUTH_MOCS_HDRS}
)
target_link_libraries(ground_truth_gui implicit_shape_model vtk_utils ${QT_LIBRARIES} ${VTK_LIBRARIES} ${Boost_LIBRARIES})


#Tools to compare with other approaches
add_executable(eval_pipeline_knopp
    eval_pipelines/eval_pipeline_knopp/eval_pcl_ism.cpp
    eval_pipelines/eval_pipeline_knopp/pcl_ism.cpp
    eval_pipelines/eval_pipeline_knopp/implicit_shape_model.hpp
)
target_link_libraries(eval_pipeline_knopp ${Boost_LIBRARIES} ${PCL_LIBRARIES})

add_executable(eval_pipeline_mccann
    eval_pipelines/eval_pipeline_mccann/eval_lnbnn.cpp
    eval_pipelines/eval_pipeline_mccann/lnbnn.cpp
)
target_link_libraries(eval_pipeline_mccann implicit_shape_model ${Boost_LIBRARIES} ${PCL_LIBRARIES})

add_executable(eval_pipeline_tombari
    eval_pipelines/eval_pipeline_tombari/eval_hough3d.cpp
    eval_pipelines/eval_pipeline_tombari/hough3d.cpp
)
target_link_libraries(eval_pipeline_tombari implicit_shape_model ${Boost_LIBRARIES} ${PCL_LIBRARIES})

add_executable(eval_pipeline_tombari_detection
    eval_pipelines/eval_pipeline_tombari/eval_hough3d_detection.cpp
    eval_pipelines/eval_pipeline_tombari/hough3d.cpp
)
target_link_libraries(eval_pipeline_tombari_detection implicit_shape_model ${Boost_LIBRARIES} ${PCL_LIBRARIES})

add_executable(eval_pipeline_aldoma_detection
    eval_pipelines/eval_pipeline_aldoma/eval_global_hv_detection.cpp
    eval_pipelines/eval_pipeline_aldoma/global_hv.cpp
)
target_link_libraries(eval_pipeline_aldoma_detection implicit_shape_model ${Boost_LIBRARIES} ${PCL_LIBRARIES})
set_property(TARGET eval_pipeline_aldoma_detection PROPERTY CXX_STANDARD 14)


#ISM evaluation tools
add_executable(eval_tool
    eval_tool/eval_classification.cpp
)
target_link_libraries(eval_tool implicit_shape_model ${Boost_LIBRARIES})

add_executable(eval_tool_detection
    eval_tool/eval_detection.cpp
)
target_link_libraries(eval_tool_detection implicit_shape_model ${Boost_LIBRARIES})


#ISM add normals tool
#add_executable(add_normals_tool
#    add_normals_tool/main.cpp
#)
#target_link_libraries(add_normals_tool implicit_shape_model ${Boost_LIBRARIES})
