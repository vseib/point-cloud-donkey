#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <utility>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot.h>
#include <pcl/features/shot_lrf.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/shot_lrf_omp.h>
#include <pcl/features/board.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>

#include "cgf.h"

#include "../../third_party/liblzf-3.6/lzf_c.c"
#include "../../third_party/liblzf-3.6/lzf_d.c"

/*
 * NOTE:
 * This is a modified version of the file https://github.com/marckhoury/CGF/blob/master/src/main.cpp
 * Copyright of the original file by Marc Khoury.
 *
 * See
 *      https://marckhoury.github.io/CGF/
 * and
 *      https://github.com/marckhoury/CGF
 *
 * Any modifications I made to this file are under the
 * BSD 3-Clause License
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * As of March 1st, 2019, the author of the above mentioned GitHub repository did not add a license.
 * If you are reading this, please check if a license was added. If this is the case, please notify me,
 * especially if the added license is in conflict with the BSD 3-Clause License.
 *
 * Thank you very much in advance.
 * Viktor Seib
 */


void usage(const char* program)
{
    cout << "Usage: " << program << " [options] <input.pcd>" << endl << endl;
    cout << "Options: " << endl;
    cout << "--relative If selected, scale is relative to the diameter of the model (-d). Otherwise scale is absolute." << endl;
    cout << "-r R Number of subdivisions in the radial direction. Default 17." << endl;
    cout << "-p P Number of subdivisions in the elevation direction. Default 11." << endl;
    cout << "-a A Number of subdivisions in the azimuth direction. Default 12." << endl;
    cout << "-s S Radius of sphere around each point. Default 1.18 (absolute) or 17\% of diameter (relative)." << endl;
    cout << "-d D Diameter of full model. Must be provided for relative scale." << endl;
    cout << "-m M Smallest radial subdivision. Default 0.1 (absolute) or 1.5\% of diameter (relative)." << endl;
    cout << "-l L Search radius for local reference frame. Default 0.25 (absolute) or 2\% of diameter (relative)." << endl;
    cout << "-t T Number of threads. Default 16." << endl;
    cout << "-o Output file name." << endl;
    cout << "-h Help menu." << endl;
}

std::vector<std::vector<double> > compute_intensities(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints,
                                            int num_bins_radius, 
                                            int num_bins_polar,
                                            int num_bins_azimuth,
                                            double search_radius,
                                            double lrf_radius, 
                                            double rmin,
                                            int num_threads)
{
    std::vector<std::vector<double>> intensities;
    intensities.resize(keypoints->points.size());

    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
    tree->setInputCloud(cloud);

    pcl::PointCloud<pcl::ReferenceFrame>::Ptr frames(new pcl::PointCloud<pcl::ReferenceFrame>());
    pcl::SHOTLocalReferenceFrameEstimation<pcl::PointXYZRGB>::Ptr lrf_estimator(new pcl::SHOTLocalReferenceFrameEstimation<pcl::PointXYZRGB>());
    lrf_estimator->setRadiusSearch(lrf_radius);
    lrf_estimator->setInputCloud(keypoints);
    lrf_estimator->setSearchSurface(cloud);
    lrf_estimator->compute(*frames);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setInputCloud(keypoints);
    ne.setSearchSurface(cloud);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_normals (new pcl::search::KdTree<pcl::PointXYZRGB>());
    ne.setSearchMethod(tree_normals);
    ne.setRadiusSearch(search_radius*0.1);
    ne.compute(*normals);

    pcl::StopWatch watch_intensities;

    double ln_rmin = log(rmin);
    double ln_rmax_rmin = log(search_radius/rmin);
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(int i = 0; i < keypoints->points.size(); i++)
    {
        std::vector<int> indices;
        std::vector<float> distances;
        std::vector<double> intensity;
        int sum = 0;
        intensity.resize(num_bins_radius * num_bins_polar * num_bins_azimuth);
 
        pcl::ReferenceFrame current_frame = (*frames)[i];
        Eigen::Vector4f current_frame_x (current_frame.x_axis[0], current_frame.x_axis[1], current_frame.x_axis[2], 0);
        Eigen::Vector4f current_frame_y (current_frame.y_axis[0], current_frame.y_axis[1], current_frame.y_axis[2], 0);
        Eigen::Vector4f current_frame_z (current_frame.z_axis[0], current_frame.z_axis[1], current_frame.z_axis[2], 0);

        if(isnan(current_frame_x[0]) || isnan(current_frame_x[1]) || isnan(current_frame_x[2]) ) {
            current_frame_x[0] = 1, current_frame_x[1] = 0, current_frame_x[2] = 0;
            current_frame_y[0] = 0, current_frame_y[1] = 1, current_frame_y[2] = 0;
            current_frame_z[0] = 0, current_frame_z[1] = 0, current_frame_z[2] = 1;
        } else {
            float nx = normals->points[i].normal_x, ny = normals->points[i].normal_y, nz = normals->points[i].normal_z;
            Eigen::Vector4f n(nx, ny, nz, 0);
            if(current_frame_z.dot(n) < 0) {
                current_frame_x = -current_frame_x;
                current_frame_y = -current_frame_y;
                current_frame_z = -current_frame_z;
            }
        }
    
        fill(intensity.begin(), intensity.end(), 0);
        tree->radiusSearch(keypoints->points[i], search_radius, indices, distances);
        for(int j = 1; j < indices.size(); j++) {
            if(distances[j] > 1E-15) {
                Eigen::Vector4f v = cloud->points[indices[j]].getVector4fMap() - keypoints->points[i].getVector4fMap();
                double x_l = (double)v.dot(current_frame_x);
                double y_l = (double)v.dot(current_frame_y);
                double z_l = (double)v.dot(current_frame_z);
                
                double r = sqrt(x_l*x_l + y_l*y_l + z_l*z_l);
                double theta = pcl::rad2deg(acos(z_l / r));
                double phi = pcl::rad2deg(atan2(y_l, x_l));
                int bin_r = int((num_bins_radius - 1) * (log(r) - ln_rmin) / ln_rmax_rmin + 1);
                int bin_theta = int(num_bins_polar * theta / 180);
                int bin_phi = int(num_bins_azimuth * (phi + 180) / 360);

                bin_r = bin_r >= 0 ? bin_r : 0;
                bin_r = bin_r < num_bins_radius ? bin_r : num_bins_radius - 1;
                bin_theta = bin_theta < num_bins_polar ? bin_theta : num_bins_polar - 1;
                bin_phi = bin_phi < num_bins_azimuth ? bin_phi : num_bins_azimuth - 1;
                int idx = bin_r + bin_theta * num_bins_radius + bin_phi * num_bins_radius * num_bins_polar;
                intensity[idx] += 1;
                sum += 1;
            }
        }
        if(sum > 0) {
            for(int j = 0; j < intensity.size(); j++) {
                intensity[j] /= sum;
            }
        }
      intensities[i] = intensity;
    }
    //pcl::console::print_highlight("Raw Spherical Histograms Time: %f (s)\n", watch_intensities.getTimeSeconds());
    return intensities;
}

int cgf_main(std::vector<std::string> &argvector,
                                     pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints)
{
    // convert arguments to argc and argv
    int argc = argvector.size();
    char* argv[argc];
    for(int i = 0; i < argvector.size(); i++)
    {
        argv[i] = const_cast<char*>(argvector[i].c_str());
    }

    int num_bins_radius = 17, num_bins_polar = 11, num_bins_azimuth = 12;
    int num_threads = 16;
    double search_radius = 1.18, lrf_radius = 0.25;
    double diameter = 4*sqrt(3);
    double rmin = 0.1;
    std::string output_file;
 
    if(pcl::console::find_argument(argc, argv, "-h") >= 0 || argc == 1) {
        usage(argv[0]);
        return 1;
    }

    pcl::console::parse_argument(argc, argv, "-r", num_bins_radius);
    pcl::console::parse_argument(argc, argv, "-p", num_bins_polar);
    pcl::console::parse_argument(argc, argv, "-a", num_bins_azimuth);
    pcl::console::parse_argument(argc, argv, "-s", search_radius);
    pcl::console::parse_argument(argc, argv, "-m", rmin);
    pcl::console::parse_argument(argc, argv, "-l", lrf_radius);
    pcl::console::parse_argument(argc, argv, "-o", output_file);
    pcl::console::parse_argument(argc, argv, "-t", num_threads);
    pcl::console::parse_argument(argc, argv, "-d", diameter);

    bool relative_scale = pcl::console::find_argument(argc, argv, "--relative") >= 0;
    //std::cout << relative_scale << std::endl;
    if(relative_scale) {
        search_radius = 0.17 * diameter;
        lrf_radius = 0.02 * diameter;
        rmin = 0.015 * diameter; 
    }

    std::ofstream fout;
    fout.open(output_file.c_str(), ios::binary);

    std::vector<double> intensities_flat;
    std::vector<std::vector<double>> intensities = compute_intensities(cloud, keypoints,
                                                              num_bins_radius, num_bins_polar, num_bins_azimuth, 
                                                              search_radius, lrf_radius, 
                                                              rmin, num_threads);

    // the rest of the code is only needed if an output is generated
    for(int i = 0; i < intensities.size(); i++)
    {
        for(int j = 0; j < intensities[i].size(); j++)
        {
            intensities_flat.push_back(intensities[i][j]);
        }
    }

    if(intensities_flat.size()*sizeof(double) > 4294967293)
    {
        std::cout << "Warning: More than 4294967293 bytes allocated. Compression may not work as expected." << std::endl;
    }

    std::vector<unsigned char> intensities_bytes;
    intensities_bytes.resize(intensities_flat.size()*sizeof(double));
    memcpy(&intensities_bytes[0], &intensities_flat[0], intensities_flat.size()*sizeof(double));
    std::vector<unsigned char> intensities_compressed;
    intensities_compressed.resize(intensities_bytes.size());
    size_t compressed_len = lzf_compress(&intensities_bytes[0], intensities_bytes.size(), &intensities_compressed[0], intensities_bytes.size());

    fout.write(reinterpret_cast<const char*>(&intensities_compressed[0]), compressed_len);
    fout.close();
    return 0;
}
