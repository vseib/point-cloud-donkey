



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
* [B-SHOT Descriptor](https://github.com/saimanoj18/iros_bshot)

## News

| Date | Update |
| ---  | ---    |
| 2019-02-24 | Added instructions for Ubuntu 16, moved information to the Wiki |
| 2019-02-08 | Added benchmark results and citing section |
| 2019-01-25 | Initial code commit, ReadMe with quick start examples added |


## Installation

This framework was tested with Ubuntu 14.04 LTS and Ubuntu 16.04 LTS.
The installation instructions can be found on the Wiki pages.

[Ubuntu 14.04 LTS](https://github.com/vseib/PointCloudDonkey/wiki/Installation_Ubuntu_14.04_LTS)

[Ubuntu 16.04 LTS](https://github.com/vseib/PointCloudDonkey/wiki/Installation_Ubuntu_16.04_LTS)


## Quick Start

To quickly start training a classifier to classify isolated point clouds, refer to the instructions on the following Wiki pages.

[Quick Start Using the Command Line](https://github.com/vseib/PointCloudDonkey/wiki/Quick_Start_Using_Command_Line)

TODO: [Quick Start Using the GUI]

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
* Paper introducing codebook cleaning and global verification
* PDF: [Seib2019B3S](http://www.uni-koblenz.de/~agas/Documents/Seib2019B3S.pdf)
```
@inproceedings{Seib2019B3S,
   author = {Seib, Viktor and Theisen, Nick and Paulus, Dietrich},
   editor = {Tremeau, Alain and Farinella, Giovanni Maria and Braz, Jose},
   title = {Boosting 3D Shape Classification with Global Verification and Redundancy-free Codebooks},
   booktitle = {Proceedings of the 14th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications},
   publisher = {SciTePress},
   volume = {5},
   pages = {257 - 264},
   year = {2019},
   url = {http://www.uni-koblenz.de/~agas/Documents/Seib2019B3S.pdf},
   isbn = {978-989-758-354-4},
}
```

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

