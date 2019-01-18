
# PointCloudDonkey
A Hough-Space-based Nearest Neighbor Object Recognition Pipeline for Point Clouds

![alt text](images/complete_pipeline.png "Visualization of the Pipeline")

The maneki-neko (lucky cat) mesh model is intellectual property of user bs3 (taken from <a href="https://www.thingiverse.com/thing:923097" target="_blank">here</a> and converted to a point cloud).

---

## Description

PointCloudDonkey is a customizable pipeline based on the <a href="http://pointclouds.org" target="_blank">Point Cloud Library (PCL)</a> for point cloud classification. The development started as an adaptation of the Implicit Shape Model (ISM) [1] algorithm to point cloud data. Over time, the framework was extended and became a general, local feature, Hough-Space voting approach for point cloud classification. It allows to classify point cloud objects and localize them amongst clutter (the latter part still being in development).

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
#### Optionally: Install XY

## Quick Start

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

