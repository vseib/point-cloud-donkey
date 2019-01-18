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

* link to Norman's repo
* thanks to 3rd party libs (also with links?)

## News

* todo

## Installation

* mention required libs
* what to do on ubuntu 14 / 16 / 18 (all lts)

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

```
[1] Leibe, Bastian and Leonardis, Ales and Schiele, Bernt, Combined Object Categorization and Segmentation with an Implicit Shape Model, Workshop on statistical learning in computer vision, ECCV, 2004
```
