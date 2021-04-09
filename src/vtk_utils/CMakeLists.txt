cmake_minimum_required(VERSION 2.8.3)
project(vtk_utils)

# set build type
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

#find boost
find_package(Boost REQUIRED COMPONENTS system timer)

# find pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

set(CMAKE_INCLUDE_PATH "/usr/share/qt4/include")
find_package(Qt4 COMPONENTS QtCore QtGui REQUIRED)
include(${QT_USE_FILE})

qt4_wrap_cpp(MOCS_HDRS
    render_view.h
)

# implicit_shape_model
add_library(vtk_utils
    interactor_ortho.cpp
    interactor_scene.cpp
    render_view.cpp
    ${MOCS_HDRS}
)

# link library with dependencies
target_link_libraries(vtk_utils ${PCL_LIBRARIES} ${Boost_LIBRARIES})
