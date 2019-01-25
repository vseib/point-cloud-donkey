/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "normal_orientation.h"

#include <pcl/features/shot_lrf_omp.h>
#include <pcl/features/normal_3d_omp.h>

#ifdef USE_VCGLIB
    #include <vcg/complex/complex.h>
    #include <vcg/complex/algorithms/pointcloud_normal.h>
#endif

namespace ism3d
{
    NormalOrientation::NormalOrientation()
        : m_k(10), m_radius(0)
    {

    }

    NormalOrientation::NormalOrientation(int k)
        : m_k(k), m_radius(0)
    {
    }

    NormalOrientation::NormalOrientation(int k, float radius)
        : m_k(k), m_radius(radius)
    {
    }

    NormalOrientation::NormalOrientation(float radius)
        : m_k(-1), m_radius(radius)
    {
    }

    NormalOrientation::~NormalOrientation()
    {
    }

    bool NormalOrientation::processSHOTLRF(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                    pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                    pcl::PointCloud<pcl::Normal>::Ptr& orientedNormals,
                                           pcl::search::Search<PointT>::Ptr searchTree)
    {
        LOG_ASSERT(pointCloud->size() == normals->size());

        pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames(new pcl::PointCloud<pcl::ReferenceFrame>());
        pcl::SHOTLocalReferenceFrameEstimationOMP<PointT, pcl::ReferenceFrame> refEst;

        refEst.setRadiusSearch(m_radius); // using normal radius for reference frame, since we only want normals
        refEst.setInputCloud(pointCloud);
        refEst.setSearchSurface(pointCloud);
        refEst.setSearchMethod(searchTree);
        refEst.compute(*referenceFrames);

        std::vector<int> invalid_normals_indices;

        // compute normals as inverted z-axis of SHOT reference frame
        for (int i = 0; i < (int)referenceFrames->size(); i++)
        {
            const pcl::ReferenceFrame& frame = referenceFrames->at(i);
            if (pcl_isfinite (frame.x_axis[0]) && pcl_isfinite (frame.y_axis[0]) && pcl_isfinite (frame.z_axis[0]))
            {
                // use inverted z-axis of reference frame as normal
                float curv = normals->at(i).curvature;
                orientedNormals->at(i) = pcl::Normal(-frame.z_axis[0], -frame.z_axis[1], -frame.z_axis[2]);
                orientedNormals->at(i).curvature = curv;
            }
            else
            {
                invalid_normals_indices.push_back(i);
            }
        }

        // compute missing normals
        if(invalid_normals_indices.size() > 0)
        {
            LOG_WARN("invalid normals: " << invalid_normals_indices.size());

            pcl::NormalEstimationOMP<PointT, pcl::Normal> normalEst;

            pcl::KdTreeFLANN<PointT> kdtree;
            kdtree.setInputCloud (pointCloud);
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;

            // use each point index without valid normal
            for(int idx = 0; idx < invalid_normals_indices.size(); idx++)
            {
                // manually compute normals where missing
                pointIdxRadiusSearch.clear();
                pointRadiusSquaredDistance.clear();
                kdtree.radiusSearch(pointCloud->at(idx), m_radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
                float x, y, z, curv;
                normalEst.computePointNormal(*pointCloud, pointIdxRadiusSearch, x, y, z, curv);
                // add computed normal
                orientedNormals->at(idx) = pcl::Normal(x, y, z);
            }
        }

        return true;
    }

#ifdef USE_VCGLIB
    void NormalOrientation::computeUsingEMST(pcl::PointCloud<PointT>::ConstPtr model,
                                             pcl::PointCloud<pcl::Normal>::Ptr normals)
    {
        normals->points.resize(model->size());
        LOG_ASSERT(model->size() == normals->size());

        class MyVertex; class MyEdge; class MyFace;
        struct MyUsedTypes : public vcg::UsedTypes<vcg::Use<MyVertex>::AsVertexType, vcg::Use<MyEdge>::AsEdgeType, vcg::Use<MyFace>::AsFaceType>{};

        class MyVertex  : public vcg::Vertex< MyUsedTypes, vcg::vertex::Coord3f, vcg::vertex::Normal3f, vcg::vertex::BitFlags  >{};
        class MyFace    : public vcg::Face<   MyUsedTypes, vcg::face::FFAdj,  vcg::face::VertexRef, vcg::face::BitFlags > {};
        class MyEdge    : public vcg::Edge<   MyUsedTypes> {};
        class MyMesh : public vcg::tri::TriMesh< std::vector<MyVertex>, std::vector<MyFace> > {};

        MyMesh m;
        MyMesh::VertexIterator vi = vcg::tri::Allocator<MyMesh>::AddVertices(m, model->size());

        for(int i = 0; i < model->size(); i++)
        {
            PointT point = model->at(i);
            vi->P() = MyMesh::CoordType(point.x, point.y, point.z);
            vi++;
        }

        vcg::tri::PointCloudNormal<MyMesh>::Param p;
        p.fittingAdjNum = m_k;
        p.smoothingIterNum = m_k;
        vcg::tri::PointCloudNormal<MyMesh>::Compute(m, p, NULL);

        int i = 0;
        for(MyMesh::VertexIterator fi = m.vert.begin(); fi != m.vert.end(); ++fi)
        {
            MyVertex::NormalType nv = fi->N();
            float nx = nv[0];
            float ny = nv[1];
            float nz = nv[2];

            normals->at(i) = pcl::Normal(-nx, -ny, -nz);
            i++;
        }
    }
#endif
}
