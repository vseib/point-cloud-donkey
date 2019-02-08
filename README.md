



# PointCloudDonkey
A Hough-Space-based Nearest Neighbor Object Recognition Pipeline for Point Clouds

![alt text](images/complete_pipeline.png "Visualization of the Pipeline")

The maneki-neko (lucky cat) mesh model is intellectual property of user bs3 (taken from [thingiverse](https://www.thingiverse.com/thing:923097") and converted to a point cloud).

---
## Content

[Description](#Description) <br>
[Acknowledgments](#Acknowledgments) <br>
[News](#News) <br>
[Installation](#Installation) <br>
[Quick Start](#Quick-Start) <br>
[Documentation](#Documentation) <br>
[Citing](#Citing) <br>
[License](#License) <br>
[References](#References) <br>


## Description

PointCloudDonkey is a customizable pipeline based on the <a href="http://pointclouds.org" target="_blank">Point Cloud Library (PCL)</a> for point cloud classification. The development started as an adaptation of the Implicit Shape Model (ISM) [1] algorithm to point cloud data. There are still many references to ISM in the code. Over time, the framework was extended and became a general, local feature, Hough-Space voting approach for point cloud classification. It allows to classify point cloud objects and localize them amongst clutter (the latter part still being in development).

The following table summarizes classification results on some datasets. For more information please refer to the publications linked in the [Citing](#Citing) section.

| Dataset | Base Pipeline<sup>[2](#fnbasep)</sup> | Base Pipeline<sup>[3](#fnbasep2)</sup> | Extended Pipeline<sup>[4](#fnbasepext)</sup> |
| ---  | ---    | --- | --- |
| Aim@Shape<sup>[1](#fnaim)</sup> | 85.0 | 90.0 | 93.5 |
| [McGill 3D Shape Benchmark](http://www.cim.mcgill.ca/~shape/benchMark/) | - |85.2 | 86.6 |
| [Princeton Shape Benchmark](http://shape.cs.princeton.edu/benchmark/) | 61.7 | 67.0 | 68.4 |
| [Shrec 2012](https://www.itl.nist.gov/iad/vug/sharp/contest/2012/Generic3D/) | -| 70.2 | 74.5
| [ModelNet 10](http://modelnet.cs.princeton.edu/) | - | 62.4 | 83.8 |
| [ModelNet 40](http://modelnet.cs.princeton.edu/) | - |71.9 | 75.4 |

<a name="fnaim">1</a>: Dataset no longer online <br>
<a name="fnbasep">2</a>: Pipeline **excluding** steps marked with a red star in the image above <br>
<a name="fnbasep">3</a>: Optimized parameters in training <br>
<a name="fnbasep">4</a>: Pipeline **including** orange and green steps marked with a red star in the image above <br>

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
| 2019-02-08 | Added benchmark results and citing section |
| 2019-01-25 | Initial code commit, ReadMe with quick start examples added |


## Installation

This framework was tested with Ubuntu 14.04 LTS and Ubuntu 16.04 LTS.

### Ubuntu 14.04 
TODO: add commit ID or branch for Ubuntu 14

Note: You need to have OpenCV 2.4 installed to be able to use this code with Ubuntu 14. If you have different versions installed, make sure that `cmake` finds the OpenCV version 2.4 (I recommend using `ccmake` to configure paths).

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

### Ubuntu 16.04

TODO

## Quick Start
This guide shows how to use the software to classify isolated point clouds.
* Download example point cloud objects from the [PCL repository](https://github.com/PointCloudLibrary/data/tree/master/tutorials) <br> (five files for training `ism_train_*.pcd` and five files for testing `ism_test_*.pcd`).
* in the following steps I will assume that these files are downloaded and placed in the folder `PointCloudDonkey/example_clouds`.

### Using the Command Line
For this example I will use the executable `eval_tool` which is build from the code located in `PointCloudDonkey/src/eval_tool`. The `eval_tool` provides two interfaces: one that allows to specify the input data _as arguments_ on the command line and one that allows to specify _a file_ with input data.
#### Training
1. Data as arguments
    * Run the following command from the `PointCloudDonkey` folder for training
    ```
        ./build/bin/eval_tool -t config/qs_input_config.ism -o output_config.ism  \
        -m example_clouds/ism_train_cat.pcd example_clouds/ism_train_horse.pcd example_clouds/ism_train_lioness.pcd \
        example_clouds/ism_train_michael.pcd example_clouds/ism_train_wolf.pcd -c 0 1 2 3 4
      ```
      * The executable takes an input config file for training (`-t`), a name for the output config file (`-o`), a list of point clouds (`-m`) and a list of class labels in the same order as the provided point clouds (`-c`).
      
2. Data in a file (recommended interface)
    *  Run the following command from the `PointCloudDonkey` folder for training
    ```
        ./build/bin/eval_tool -t config/qs_input_config.ism -o output_config.ism -f data/qs_train_list.txt
      ```
   * As before, the executable takes an input config file for training (`-t`) and a name for the output config file (`-o`). However, the objects and corresponding class labels are specified inside a text file (`-f`).

After running either of the above commands you will see two new files: `output_config.ism` and `output_config.ismd`. The first file contains the parameters used for training and a reference to the second file. The second file is stored in binary format and contains all the data extracted from the dataset during training.


#### Testing

1. Data as arguments
    * Run the following command from the `PointCloudDonkey` folder for testing (classification)
    ```
        ./build/bin/eval_tool -d output_config.ism -o output_folder -p example_clouds/ism_test_cat.pcd \
        example_clouds/ism_test_horse.pcd example_clouds/ism_test_lioness.pcd example_clouds/ism_test_michael.pcd \
        example_clouds/ism_test_wolf.pcd -g 0 1 2 3 4
      ```
      * The executable takes an input config file for testing (`-d`), a name for the output folder for logging (`-o`), a list of point clouds (`-p`) and a list of ground truth class labels in the same order as the provided point clouds (`-g`).
      
2. Data in a file (recommended interface)
    *  Run the following command from the `PointCloudDonkey` folder for testing (classification)
    ```
        ./build/bin/eval_tool -d output_config.ism -o output_folder -f data/qs_test_list.txt
      ```
   * As before, the executable takes an input config file for testing (`-d`) and a name for the output folder (`-o`). However, the objects and corresponding ground truth labels are specified inside a text file (`-f`).

After running either of the above commands you will see a new folder: `output_folder`. This folder contains detailed statistics about the classification process. It contains a textfile for each of the objects in the classification dataset with information about found maxima, their locations, class label and bounding box coordinates. Further, there is a file `summary.txt`that summarizes the classification of each object, the time taken for each step and the overall classification accuracy.


### Using the GUI
TODO

## Documentation

### Preparing Point Clouds
### Preparing a File List
### Preparing a Config File
### Detailed Config File Documentation

## Citing

This sections lists all publications related to this repository. If you are using this repository for academic work, please consider citing one of the following publications. If you are unsure what to cite, please choose the "Extended Pipeline" paper.

TODO add missing bibtex info

### Extended Pipeline
The extended pipeline is the generic point cloud processing pipeline depicted in the image on top of this page including the orange and green steps marked with a red star.

#### Boosting 3D Shape Classification with Global Verification and Redundancy-Free Codebooks
* accepted for publication at VISAPP 2019
* PDF: TODO

### Short-SHOT Feature Descriptor
This descriptor is a modification to the original SHOT descriptor [2]. In short: it is based on the number of points in a grid bin in contrast to the orientations of their normals.

#### A LOW-DIMENSIONAL FEATURE TRANSFORM FOR KEYPOINT MATCHING AND CLASSIFICATION OF POINT CLOUDS WITHOUT NORMAL COMPUTATION
* Paper introducing and evaluating the short version of the SHOT descriptor
* PDF: [Seib2018ALF](http://www.uni-koblenz.de/~agas/Documents/Seib2018ALF.pdf)
```
@inproceedings{Seib2018ALF,
   author = {Seib, Viktor and Paulus, Dietrich},
   title = {A Low-Dimensional Feature Transform for Keypoint Matching and Classification of Point Clouds without Normal Computation},
   booktitle = {2018 25th IEEE International Conference on Image Processing (ICIP)},
   pages = {2949-2953},
   month = {10},
   year = {2018},
   isbn = {978-1-4799-7061-2}
}
```

### Base Pipeline
The base pipeline is the generic point cloud processing pipeline depicted in the image on top of this page excluding all steps marked with a red star.

#### Pose Estimation and Shape Retrieval with Hough Voting in a Continuous Voting Space
* Paper presenting additional experiments and modifications to the base pipeline
* PDF: [Seib2015PEA](http://www.uni-koblenz.de/~agas/Documents/Seib2015PEA.pdf)
```
@inproceedings{Seib2015PEA,
   author = {Seib, Viktor and Link, Norman and Paulus, Dietrich},
   editor = {Gall, Juergen and Gehler, Peter and Leibe, Bastian},
   title = {Pose Estimation and Shape Retrieval with Hough Voting in a Continuous Voting Space},
   booktitle = {Pattern Recognition},
   publisher = {Springer International Publishing},
   volume = {9358},
   series = {LNCS},
   pages = {458-469},
   year = {2015},
   isbn = {978-3-319-24946-9},
}
```

#### Implicit Shape Models For 3D Shape Classification With a Continuous Voting Space
* Paper with the original publication of a 3D ISM and a continuous voting space
* PDF: [Seib2015ISM](http://www.uni-koblenz.de/~agas/Documents/Seib2015ISM.pdf)
```
@inproceedings{Seib2015ISM,
   author = {Seib, Viktor and Link, Norman and Paulus, Dietrich},
   editor = {Braz, and Battiato, Sebastiano and Imai, Francisco H.},
   title = {Implicit Shape Models For 3D Shape Classification With a Continuous Voting Space},
   booktitle = {{VISAPP} 2015 - Proceedings of the 10th International Conference on Computer Vision Theory and Applications},
   publisher = {SciTePress},
   volume = {2},
   pages = {33-43},
   year = {2015},
   isbn = {978-989-758-090-1},
}
```


## License

PointCloudDonkey is released under the BSD-3-Clause license. See [LICENSE](LICENSE) for details.
PointCloudDonkey also includes some 3rd party code which might be subject to other licenses.

## References

[1] Leibe, Bastian and Leonardis, Ales and Schiele, Bernt; 
"Combined Object Categorization and Segmentation with an Implicit Shape Model",
Workshop on statistical learning in computer vision, ECCV, 2004
    
[2] Tombari, Federico and Salti, Samuele and Di Stefano, Luigi;
"Unique signatures of histograms for local surface description",
European conference on computer vision, ECCV, 2010

