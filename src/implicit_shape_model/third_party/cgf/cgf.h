/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2019, Viktor Seib
 * All rights reserved.
 *
 */


#ifndef CGF_H
#define CGF_H

#include <vector>
#include <pcl/common/common_headers.h>

void usage(const char* program);

std::vector<std::vector<double> > compute_intensities(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints,
                                            int num_bins_radius,
                                            int num_bins_polar,
                                            int num_bins_azimuth,
                                            double search_radius,
                                            double lrf_radius,
                                            double rmin,
                                            int num_threads);

int cgf_main(std::vector<std::string> &argvector,
                                     pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints);


#endif // CGF_H
