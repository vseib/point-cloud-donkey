/*
 * COSPAIR.cpp
 *
 *  Created on: Feb 12, 2017
 *      Author: berker
 */

#include "cospair.h"

COSPAIR::COSPAIR()
{
}

COSPAIR::~COSPAIR()
{
}

std::vector<std::vector<float>> COSPAIR::ComputeCOSPAIR(pcl::PointCloud<ism3d::PointT>::ConstPtr cloud,
                                                        pcl::PointCloud<ism3d::PointT>::Ptr keypoints,
                                                        pcl::PointCloud<pcl::Normal>::ConstPtr cloud_normals,
                                                        double radius,
                                                        int num_levels,
                                                        int num_bins,
                                                        int rgb_type,
                                                        int num_rgb_bins)
{
    std::vector<std::vector<float>> COSPAIR_features;
    pcl::KdTreeFLANN<ism3d::PointT>::Ptr tree(new pcl::KdTreeFLANN<ism3d::PointT>());
    tree->setInputCloud(cloud);

    int histsize = (num_levels * num_bins * 3);
    int levelsize = num_bins * 3 ;
    int levelsearch[num_levels];

    int levelsize_rgb = num_rgb_bins * 3;
    int histsize_rgb = (num_levels * num_rgb_bins * 3);

	int levelsize_total = levelsize + levelsize_rgb;
	int histsize_total = histsize + histsize_rgb;

    std::cout << "keypoint size: " << keypoints->size() << std::endl;

	//For each keypoint calculate SPAIR and push to SPFH_features
    for(unsigned key_idx = 0; key_idx < keypoints->size(); key_idx++)
	{
        std::vector<float> spair_features(histsize_total,0);
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        float f1,f2,f3,f4;
        float deg_f1,deg_f2,deg_f3;
        int bin_f1,bin_f2,bin_f3;
        int searchstart = 1;
        for(int l=1;l<=num_levels;l++)
        {
            //std::vector<float> level_features(levelsize,0);
            float level_features[levelsize];
            float level_features_rgb[levelsize_rgb];
            for(int z=0;z<levelsize;z++)
                level_features[z] = 0.0;
            for(int z=0;z<levelsize_rgb;z++)
                level_features_rgb[z] = 0.0;
            double r;
            int searchsize;
            unsigned int levelpaircount = 0;
            r = ((l*1.0)/num_levels)*radius;
            // NOTE: keypoints must be part of the original cloud with its indices
            // look up closest point to keypoint from cloud and get its index
            std::vector<int> temp_indices;
            std::vector<float> temp_distances;
            tree->nearestKSearch(keypoints->points[key_idx], 1, temp_indices, temp_distances);
            int keypoint_indice = temp_indices[0];
            //Number of points inside radius r
            searchsize = tree->radiusSearch(keypoint_indice, r, pointIdxRadiusSearch, pointRadiusSquaredDistance);
            //std::cout<<j<<"; in level "<<l<<" "<<searchsize<<std::endl;
            levelsearch[l-1] = searchsize;
            if(l!=1)
                searchstart = levelsearch[l-2];

            // Iterate over all the points in the neighborhood
            for (int i_idx = searchstart; i_idx < searchsize; ++i_idx)
            {
                    // If the 3D points are invalid, don't bother estimating, just continue
                    if (pcl::isFinite(cloud->points[pointIdxRadiusSearch[i_idx]])  && pcl::isFinite(cloud_normals->points[pointIdxRadiusSearch[i_idx]]))
                    {
                        levelpaircount++;
                        int p1, p2;
                        p1 = pointIdxRadiusSearch[i_idx];
                        p2 = keypoint_indice;
                        //std::cout<<j<<" "<<l<<" "<<searchsize<<std::endl;
                        //std::cout<<i_idx<<" "<<p1<<" "<<p2<<" "<<pointRadiusSquaredDistance[i_idx]<<std::endl;
                        //std::cout<<pcl::isFinite(cloud->points[p1])<<" "<<pcl::isFinite(cloud_normals->points[p1])<<" "<<pcl::isFinite(cloud->points[p2])<<" "<<pcl::isFinite(cloud_normals->points[p2])<< std::endl;

                        //Calculate the pair features (angles)
                        pcl::computePairFeatures(cloud->points[p2].getVector4fMap(),cloud_normals->points[p2].getNormalVector4fMap(),
                                                 cloud->points[p1].getVector4fMap(),cloud_normals->points[p1].getNormalVector4fMap(),f1,f2,f3,f4);

                        deg_f1 = pcl::rad2deg(f1) + 180;
                        deg_f2 = pcl::rad2deg(SafeAcos(f2)) ;
                        deg_f3 = pcl::rad2deg(SafeAcos(f3));
                        bin_f1 = int(floor(deg_f1 / (360.0/num_bins)));
                        bin_f2 = int(floor(deg_f2 / (180.0/num_bins)));
                        bin_f3 = int(floor(deg_f3 / (180.0/num_bins)));
                        //std::cout<<f1<<" "<<f2<<" "<<f3<<" ; "<<deg_f1<<" "<<deg_f2<<" "<<deg_f3<<" ; "<<bin_f1<<" "<<bin_f2<<" "<<bin_f3<<std::endl;
                        //std::cout<<" in level "<<l<<std::endl;
                        //std::cout<<bin_f1<<" "<<1*bins + bin_f2<<" "<<2*bins + bin_f3<<std::endl;
                        //std::cout<<opfh_features[(levelsize*(l-1)) + bin_f1]<<" "<<opfh_features[(levelsize*(l-1)) + 1*bins + bin_f2]<<" "<<opfh_features[(levelsize*(l-1)) + 2*bins + bin_f3]<<std::endl;

                        level_features[bin_f1] = level_features[bin_f1] + 1.0;
                        level_features[1*num_bins + bin_f2] = level_features[1*num_bins + bin_f2] + 1.0;
                        level_features[2*num_bins + bin_f3] = level_features[2*num_bins + bin_f3] + 1.0;


                        //RGB-COLOR features

                        //std::cout<<cloud->points[p1].r<<" "<<r<<" "<<cloud->points[p1].g<<" "<<g<<" "<<cloud->points[p1].b<<" "<<b<<std::endl;

                        if(rgb_type==1) // RGB
                        {
                            float r,g,b;
                            int bin_r, bin_g, bin_b;
                            r = cloud->points[p1].r;
                            g = cloud->points[p1].g;
                            b = cloud->points[p1].b;

                            bin_r = int(floor(r / (255.0/num_rgb_bins)));
                            bin_g = int(floor(g / (255.0/num_rgb_bins)));
                            bin_b = int(floor(b / (255.0/num_rgb_bins)));
                            //std::cout<<bin_r<<" "<<bin_g<<" "<<bin_b<<std::endl;
                            level_features_rgb[bin_r] = level_features_rgb[bin_r] + 1.0;
                            level_features_rgb[1*num_rgb_bins + bin_g] = level_features_rgb[1*num_rgb_bins + bin_g] + 1.0;
                            level_features_rgb[2*num_rgb_bins + bin_b] = level_features_rgb[2*num_rgb_bins + bin_b] + 1.0;
                        }

                        if(rgb_type==2)  // RGB - L1
                        {
                            float r,g,b;
                            int bin_r, bin_g, bin_b;
                            r = cloud->points[p1].r;
                            g = cloud->points[p1].g;
                            b = cloud->points[p1].b;

                            float r_k,g_k,b_k;
                            r_k = cloud->points[p2].r;
                            g_k = cloud->points[p2].g;
                            b_k = cloud->points[p2].b;

                            //std::cout<<cloud->points[p1].r<<" "<<r<<" "<<cloud->points[p1].g<<" "<<g<<" "<<cloud->points[p1].b<<" "<<b<<std::endl;
                            //std::cout<<r_k<<" "<<g_k<<" "<<b_k<<std::endl;
                            bin_r = int(floor( abs(r-r_k) / (255.0/num_rgb_bins)));
                            bin_g = int(floor( abs(g-g_k) / (255.0/num_rgb_bins)));
                            bin_b = int(floor( abs(b-b_k) / (255.0/num_rgb_bins)));
                            //std::cout<<bin_r<<" "<<bin_g<<" "<<bin_b<<std::endl;
                            level_features_rgb[bin_r] = level_features_rgb[bin_r] + 1.0;
                            level_features_rgb[1*num_rgb_bins + bin_g] = level_features_rgb[1*num_rgb_bins + bin_g] + 1.0;
                            level_features_rgb[2*num_rgb_bins + bin_b] = level_features_rgb[2*num_rgb_bins + bin_b] + 1.0;


                        }
                        if(rgb_type==3) // HSV
                        {
                            float r,g,b;
                            int bin_r, bin_g, bin_b;
                            r = cloud->points[p1].r;
                            g = cloud->points[p1].g;
                            b = cloud->points[p1].b;

                            float h,s,v;
                            r = r/255.0;
                            g = g/255.0;
                            b = b/255.0;
                            RGB2HSV(r,g,b,h,s,v);
                            //std::cout<<r<<" "<<g<<" "<<b<<std::endl;
                            //std::cout<<h<<" "<<s<<" "<<v<<std::endl;

                            bin_r = int(floor(h / (1.0/num_rgb_bins)));
                            bin_g = int(floor(s / (1.0/num_rgb_bins)));
                            bin_b = int(floor(v / (1.0/num_rgb_bins)));
                            //std::cout<<bin_r<<" "<<bin_g<<" "<<bin_b<<std::endl;
                            level_features_rgb[bin_r] = level_features_rgb[bin_r] + 1.0;
                            level_features_rgb[1*num_rgb_bins + bin_g] = level_features_rgb[1*num_rgb_bins + bin_g] + 1.0;
                            level_features_rgb[2*num_rgb_bins + bin_b] = level_features_rgb[2*num_rgb_bins + bin_b] + 1.0;
                        }
                        if(rgb_type==4) // HSV-L1
                        {
                            float r,g,b;
                            int bin_r, bin_g, bin_b;
                            r = cloud->points[p1].r;
                            g = cloud->points[p1].g;
                            b = cloud->points[p1].b;

                            float h,s,v;
                            r = r/255.0;
                            g = g/255.0;
                            b = b/255.0;
                            RGB2HSV(r,g,b,h,s,v);
                            //std::cout<<r<<" "<<g<<" "<<b<<std::endl;
                            //std::cout<<h<<" "<<s<<" "<<v<<std::endl;
                            float r_k,g_k,b_k;
                            r_k = cloud->points[p2].r;
                            g_k = cloud->points[p2].g;
                            b_k = cloud->points[p2].b;

                            r_k = r_k/255.0;
                            g_k = g_k/255.0;
                            b_k = b_k/255.0;

                            float h_k,s_k,v_k;
                            RGB2HSV(r_k,g_k,b_k,h_k,s_k,v_k);

                            bin_r = int(floor(abs(h-h_k) / (1.0/num_rgb_bins)));
                            bin_g = int(floor(abs(s-s_k) / (1.0/num_rgb_bins)));
                            bin_b = int(floor(abs(v-v_k) / (1.0/num_rgb_bins)));
                            //std::cout<<bin_r<<" "<<bin_g<<" "<<bin_b<<std::endl;
                            level_features_rgb[bin_r] = level_features_rgb[bin_r] + 1.0;
                            level_features_rgb[1*num_rgb_bins + bin_g] = level_features_rgb[1*num_rgb_bins + bin_g] + 1.0;
                            level_features_rgb[2*num_rgb_bins + bin_b] = level_features_rgb[2*num_rgb_bins + bin_b] + 1.0;
                        }
                        if(rgb_type==5) // CIELab (recommened and used in paper)
                        {
                            float l,a,b;
                            pcl::SHOTColorEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> shot;
                            shot.RGB2CIELAB(cloud->points[p1].r,cloud->points[p1].g,cloud->points[p1].b,l,a,b);
                            //std::cout<<int(cloud->points[p1].r)<<" "<<int(cloud->points[p1].g)<<" "<<int(cloud->points[p1].b)<<std::endl;
                            //std::cout<<l<<" "<<a<<" "<<b<<std::endl;
                            l = 1.0 * l / 100;
                            a = 1.0 * (a + 86.185) / 184.439;
                            b = 1.0 * (b + 107.863) / 202.345;
                            int bin_r, bin_g, bin_b;
                            bin_r = int(floor(l / (1.0/num_rgb_bins)));
                            bin_g = int(floor(a / (1.0/num_rgb_bins)));
                            bin_b = int(floor(b / (1.0/num_rgb_bins)));
                            //std::cout<<bin_r<<" "<<bin_g<<" "<<bin_b<<std::endl;
                            level_features_rgb[bin_r] = level_features_rgb[bin_r] + 1.0;
                            level_features_rgb[1*num_rgb_bins + bin_g] = level_features_rgb[1*num_rgb_bins + bin_g] + 1.0;
                            level_features_rgb[2*num_rgb_bins + bin_b] = level_features_rgb[2*num_rgb_bins + bin_b] + 1.0;
                        }
                        if(rgb_type==6)  // CIELab-L1
                        {
                            float l,a,b;
                            pcl::SHOTColorEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> shot;
                            shot.RGB2CIELAB(cloud->points[p1].r,cloud->points[p1].g,cloud->points[p1].b,l,a,b);
                            //std::cout<<int(cloud->points[p1].r)<<" "<<int(cloud->points[p1].g)<<" "<<int(cloud->points[p1].b)<<std::endl;
                            //std::cout<<l<<" "<<a<<" "<<b<<std::endl;
                            l = 1.0 * l / 100;
                            a = 1.0 * (a + 86.185) / 184.439;
                            b = 1.0 * (b + 107.863) / 202.345;

                            float l_k,a_k,b_k;
                            shot.RGB2CIELAB(cloud->points[p2].r,cloud->points[p2].g,cloud->points[p2].b,l_k,a_k,b_k);
                            l_k = 1.0 * l_k / 100;
                            a_k = 1.0 * (a_k + 86.185) / 184.439;
                            b_k = 1.0 * (b_k + 107.863) / 202.345;


                            int bin_r, bin_g, bin_b;
                            bin_r = int(floor(abs(l-l_k) / (1.0/num_rgb_bins)));
                            bin_g = int(floor(abs(a-a_k) / (1.0/num_rgb_bins)));
                            bin_b = int(floor(abs(b-b_k) / (1.0/num_rgb_bins)));
                            //std::cout<<bin_r<<" "<<bin_g<<" "<<bin_b<<std::endl;
                            level_features_rgb[bin_r] = level_features_rgb[bin_r] + 1.0;
                            level_features_rgb[1*num_rgb_bins + bin_g] = level_features_rgb[1*num_rgb_bins + bin_g] + 1.0;
                            level_features_rgb[2*num_rgb_bins + bin_b] = level_features_rgb[2*num_rgb_bins + bin_b] + 1.0;
                        }


                    }

            }
            //std::cout<<levelpaircount<<" pairs in level"<<std::endl;
            if(levelpaircount!=0){
                for(int n=0;n<levelsize;n++)
                {
                    //std::cout<<level_features[n]<<" "<<level_features[n]/levelpaircount<<std::endl;
                    level_features[n] = (level_features[n] / levelpaircount) * l;//(levels - (l-1));
                    //std::cout<<(l-1)*levelsize + n<<" "<<level_features[n]<<std::endl;
                    spair_features[(l-1)*levelsize_total + n] = level_features[n];
                }
            }

            if(levelpaircount!=0){
                for(int n=0;n<levelsize_rgb;n++)
                {
                    //std::cout<<level_features[n]<<" "<<level_features[n]/levelpaircount<<std::endl;
                    level_features_rgb[n] = (level_features_rgb[n] / levelpaircount) * l;// (levels - (l-1));
                    //std::cout<<(l-1)*levelsize + n<<" "<<level_features[n]<<std::endl;
                    spair_features[(l-1)*levelsize_total + levelsize + n] = level_features_rgb[n];
                }
            }
        }
        COSPAIR_features.push_back(spair_features);
	}

	return COSPAIR_features;
}


double COSPAIR::computeCloudResolution (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<pcl::PointXYZRGB> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size(); i++)
  {
    if (pcl::isFinite(cloud->points[i]) == false)
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2)
    {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  return res;
}

double COSPAIR::computeCloudSize (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
	double cloudsize;
	long int numofpoints = cloud->size();
	std::vector<int> indices;//(numofpoints);
	std::vector<float> sqr_distances;//(numofpoints);
	pcl::search::KdTree<pcl::PointXYZRGB> tree;
	tree.setInputCloud(cloud);
	double maxsize = 0;
	double localmax;

	for (size_t i = 0; i < cloud->size(); i++)
	{
		if (pcl::isFinite(cloud->points[i]) == true)
		{

			//tree.radiusSearch(i,0.02,indices,sqr_distances);
			tree.nearestKSearch (i, numofpoints, indices, sqr_distances);
			std::vector<float>::const_iterator it;
			it = std::max_element(sqr_distances.begin(), sqr_distances.end());
			//it = sqr_distances.end();
			localmax = sqrt(*it);
			//std::cout << " the max is " << localmax << std::endl;
			if(localmax > maxsize){
				maxsize = localmax;
			}
		}
	}
	cloudsize = maxsize;
	return cloudsize;
}

float COSPAIR::SafeAcos (float x)
  {
  if (x < -1.0) x = -1.0 ;
  else if (x > 1.0) x = 1.0 ;
  return acos (x) ;
  }

void COSPAIR::RGB2HSV(float r, float g, float b, float &h, float &s, float &v)
{
    float K = 0.f;

    if (g < b)
    {
        std::swap(g, b);
        K = -1.f;
    }

    if (r < g)
    {
        std::swap(r, g);
        K = -2.f / 6.f - K;
    }

    float chroma = r - std::min(g, b);
    h = fabs(K + (g - b) / (6.f * chroma + 1e-20f));
    s = chroma / (r + 1e-20f);
    v = r;
}
