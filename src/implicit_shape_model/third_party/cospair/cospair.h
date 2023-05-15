/*
 * COSPAIR.h
 *
 *  Created on: Feb 12, 2017
 *      Author: berker
 */

#ifndef COSPAIR_H_
#define COSPAIR_H_

#include "../../utils/utils.h"

#include <iostream>
#include <string>
#include <algorithm>
#include <boost/thread/thread.hpp>
#include <vector>

//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/common/eigen.h>
#include <pcl/point_types.h>
#include <pcl/common/angles.h>
#include <pcl/features/pfh.h>
#include <pcl/features/shot.h>
#include <pcl/kdtree/kdtree_flann.h>

//OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>




class COSPAIR {
public:
	COSPAIR();
	virtual ~COSPAIR();

    std::vector<std::vector<float>> ComputeCOSPAIR(pcl::PointCloud<ism3d::PointT>::ConstPtr cloud,
                                                   pcl::PointCloud<ism3d::PointT>::Ptr keypoints,
                                                   pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                   double radius,
                                                   int num_levels,
                                                   int num_bins,
                                                   int rgb_type,
                                                   int num_rgb_bins);
	float SafeAcos (float x);
	void RGB2HSV(float r, float g, float b, float &h, float &s, float &v);
	double computeCloudResolution (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);
	double computeCloudSize (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);


private:
	float harris_radius;

};

#endif /* COSPAIR_H_ */
