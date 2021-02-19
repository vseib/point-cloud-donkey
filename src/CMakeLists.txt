cmake_minimum_required(VERSION 2.8.3)
project(ism_3d_new)

# set build type
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# find pcl
find_package(PCL REQUIRED)
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

#add_subdirectory(vtk_utils)
#
#find_package(VTK REQUIRED)
#include(${VTK_USE_FILE})
#
#set(CMAKE_INCLUDE_PATH "/usr/share/qt4/include")
#find_package(Qt4 COMPONENTS QtCore QtGui REQUIRED)
#include(${QT_USE_FILE})
#
#include_directories(
#    SYSTEM
#    ${QT_INCLUDE_DIR}
#)


#ISM training gui #TODO VS do not build now - include later when ROS re-integration starts
#qt4_wrap_cpp(TRAINING_MOCS_HDRS
#    training_gui/training_gui.h
#    training_gui/model_dlg.h
#    training_gui/ism_worker.h
#)
#add_executable(training_gui
#    training_gui/main.cpp
#    training_gui/training_gui.cpp
#    training_gui/model_dlg.cpp
#    training_gui/ism_worker.cpp
#    ${TRAINING_MOCS_HDRS}
#)
#target_link_libraries(training_gui vtk_utils implicit_shape_model
#    ${QT_LIBRARIES} ${VTK_LIBRARIES} QVTK ${Boost_LIBRARIES})


#ISM model gui #TODO VS do not build now - include later when ROS re-integration starts
#qt4_wrap_cpp(MODEL_MOCS_HDRS
#    model_gui/model_gui.h
#)
#add_executable(model_gui
#    model_gui/main.cpp
#    model_gui/model_gui.cpp
#    ${MODEL_MOCS_HDRS}
#)
# target_link_libraries(model_gui vtk_utils ${QT_LIBRARIES} ${PCL_LIBRARIES} ${VTK_LIBRARIES} QVTK ${Boost_LIBRARIES})


#ISM ground truth gui
#qt4_wrap_cpp(GROUND_TRUTH_MOCS_HDRS
#    ground_truth_gui/ground_truth_gui.h
#)
#add_executable(ground_truth_gui
#    ground_truth_gui/main.cpp
#    ground_truth_gui/ground_truth_gui.cpp
#    ${GROUND_TRUTH_MOCS_HDRS}
#)
#target_link_libraries(ground_truth_gui implicit_shape_model vtk_utils ${QT_LIBRARIES} ${VTK_LIBRARIES} QVTK ${Boost_LIBRARIES})


#Tools to compare with other approaches
#add_executable(eval_pcl_ism
#    eval_comparison/pcl_ism/eval_pcl_ism.cpp
#    eval_comparison/pcl_ism/pcl_ism.cpp
#    eval_comparison/pcl_ism/implicit_shape_model.hpp
#)
#target_link_libraries(eval_pcl_ism ${PCL_LIBRARIES})

#add_executable(eval_lnbnn
#    eval_comparison/lnbnn/eval_lnbnn.cpp
#    eval_comparison/lnbnn/lnbnn.cpp
#)
#target_link_libraries(eval_lnbnn ${PCL_LIBRARIES} implicit_shape_model ${Boost_LIBRARIES})

#add_executable(eval_pcl_orcg
#    eval_comparison/pcl_orcg/eval_pcl_orcg.cpp
#    eval_comparison/pcl_orcg/pcl_orcg.cpp
#)
#target_link_libraries(eval_pcl_orcg ${PCL_LIBRARIES} implicit_shape_model ${Boost_LIBRARIES})




#ISM evaluation tools
add_executable(eval_tool
    eval_tool/eval_classification.cpp
)
target_link_libraries(eval_tool implicit_shape_model ${Boost_LIBRARIES})


#ISM add normals tool
#add_executable(add_normals_tool
#    add_normals_tool/main.cpp
#)
#target_link_libraries(add_normals_tool implicit_shape_model ${Boost_LIBRARIES})


#ISM stanford tool
#add_executable(stanford_tool
#    stanford_tool/main.cpp
#)
#target_link_libraries(stanford_tool implicit_shape_model ${PCL_LIBRARIES} ${Boost_LIBRARIES})
