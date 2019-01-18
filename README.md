
# PointCloudDonkey
A Hough-Space-based Nearest Neighbor Object Recognition Pipeline for Point Clouds

![alt text](images/complete_pipeline.png "Visualization of the Pipeline")

The maneki-neko (lucky cat) mesh model is intellectual property of user bs3 (taken from <a href="https://www.thingiverse.com/thing:923097" target="_blank">here</a> and converted to a point cloud).

---
## Content

[Description](#Description) <br>
[Acknowledgments](#Acknowledgments) <br>
[News](#News) <br>
[Installation](#Installation) <br>
[Documentation](#Documentation) <br>
[Citing](#Citing) <br>
[License](#License) <br>
[References](#References) <br>


## Description

PointCloudDonkey is a customizable pipeline based on the <a href="http://pointclouds.org" target="_blank">Point Cloud Library (PCL)</a> for point cloud classification. The development started as an adaptation of the Implicit Shape Model (ISM) [1] algorithm to point cloud data. There are still many references to ISM in the code. Over time, the framework was extended and became a general, local feature, Hough-Space voting approach for point cloud classification. It allows to classify point cloud objects and localize them amongst clutter (the latter part still being in development).

The following table summarizes classification results on some datasets.
* TODO include table


## Acknowledgments

The development of this framework started during the Master's thesis of Norman Link, that I supervised. I am very thankful to Norman for his contributions. Please check out other projects of Norman on GitHub in [Norman's repository](https://github.com/Norman0406).

Further, I would like to thank the developers of third party libraries used in this project: 
* [Compact Geometric Features](https://marckhoury.github.io/CGF/)
* [cnpy Library](https://github.com/rogersce/cnpy)
* [gdiam Library](https://sarielhp.org/research/papers/00/diameter/diam_prog.html)
* [lzf Library](http://oldhome.schmorp.de/marc/liblzf.html)

## News

| Date | Update |
| ---  | ---    |
| 2019-01-xx | Initial code commit, ReadMe added |


## Installation

This framework was tested with Ubuntu 14.04 LTS and Ubuntu 16.04 LTS.

### Ubuntu 14.04 
TODO: add commit ID or branch for Ubuntu 14
#### Install Dependencies

    sudo apt install libeigen3-dev libboost-all-dev qt5-default libvtk5.8 libvtk5-dev libvtk5.8-qt4 \
         libvtk5-qt4-dev libgomp1 libopencv-core-dev libopencv-dev zlib1g zlib1g-dev libblitz0-dev \
         libjsoncpp0 libjsoncpp-dev liblog4cxx10 liblog4cxx10-dev

#### Install PCL
PointCloudDonkey requires at least PCL version 1.8.0. However, only PCL 1.7.2 is available in Ubuntu packages for 14.04. Therefore, PCL version 1.8.0 must be installed from source.
See the [official PCL instructions](http://www.pointclouds.org/documentation/tutorials/compiling_pcl_posix.php) or follow the steps below:
* If you want to reproduce the classification results reported here, download PCL release 1.8.0 from [GitHub](https://github.com/PointCloudLibrary/pcl/releases), otherwise use PCL release 1.8.1
* unpack the archive
```
    tar xvfj pcl-pcl-1.8.0.tar.gz
```
* fix a bug in SHOT initialization
  * open the file `pcl-pcl-1.8.0/features/include/pcl/features/shot.h`
  * change line 97 to be `maxAngularSectors_ (32),`
* execute the following commands:
``` 
    cd pcl-pcl-1.8.0 && mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j4
    sudo make -j4 install
```
#### Optionally: Install VCG Library
You can find instructions for installation on the [VCG Library webpage](http://vcg.isti.cnr.it/vcglib/). To include VCG in the compilation process
* open the file `PointCloudDonkey/src/implicit_shape_model/CMakeLists.txt`
* go to line 8 and set the `USE_VCG` flag to `true`
#### Compiling
Execute the following commands to compile the framework:
``` 
    cd PointCloudDonkey && mkdir build && cd build
    cmake ../src
    make -j4
```
The binaries will be placed in the folder `PointCloudDonkey/build/bin`.

## Quick Start
This guide shows how to use the software with a prepared example.
* Download example point cloud objects from the [PCL repository](https://github.com/PointCloudLibrary/data/tree/master/tutorials) <br> (five files for training `ism_train_*.pcd` and five files for testing `ism_test_*.pcd`)
* in the following steps I will assume that these files are downloaded and placed in the folder `PointCloudDonkey/example_clouds`

### Using the Command Line
For this example I will use the executable `eval_tool` which is build from the code located in `PointCloudDonkey/src/eval_tool`. The `eval_tool` provides two interfaces: one that allows to specify the input data _as arguments_ on the command line and one that allows to specify _a file_ with input data.
#### Training
1. Data as arguments
    * Run the following commands from the `PointCloudDonkey` folder for training
    ```
        ./build/bin/eval_tool -t ../configs/qs_input_config.ism -o output_config.ism \ 
        -m example_clouds/ism_train_cat.pcd example_clouds/ism_train_horse.pcd example_clouds/ism_train_lioness.pcd \ 
        example_clouds/ism_train_michael.pcd example_clouds/ism_train_wolf.pcd \
        -c 0 1 2 3 4
      ```
      * The executable takes an input config file for training (`-t`), a name for the output config file (`-o`), a list of point clouds (`-m`) and a list of class labels in the same order as the provided point clouds (`-c`).
      
2. Data in a file (recommended interface)
    *  Run the following commands from the `PointCloudDonkey` folder for training
    ```
        ./build/bin/eval_tool -t ../config/qs_input_config.ism -o output_config.ism -f qs_train_list.txt
      ```
   * As before, the executable takes an input config file for training (`-t`) and a name for the output config file (`-o`). However, the objects and corresponding classes are specified inside a text file (`-f`).

* some examples with provided config files
* maybe include some point clouds? or just link to clouds from pcl

## Documentation

* how to convert datasets to point clouds (include script)
* extensive config file documentation

## Citing

* include papers to cite

## License

PointCloudDonkey is released under the BSD-3-Clause license. See [LICENSE](LICENSE) for details.
PointCloudDonkey also includes some 3rd party code which might be subject to other licenses. Please see TODO for details.

## References

[1] Leibe, Bastian and Leonardis, Ales and Schiele, Bernt; 
"Combined Object Categorization and Segmentation with an Implicit Shape Model",
    Workshop on statistical learning in computer vision, ECCV, 2004

