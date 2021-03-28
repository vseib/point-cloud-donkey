/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#ifndef PCL_NORMAL_3D_OMP_MOD_H_
#define PCL_NORMAL_3D_OMP_MOD_H_

#include <pcl/features/feature.h>
#include <pcl/common/centroid.h>

namespace pcl
{
    /** \brief determines the eigenvector and eigenvalue of the smallest eigenvalue of the symmetric positive semi definite input matrix
      * \param[in] mat symmetric positive semi definite input matrix
      * \param[out] eigenvalues eigenvalue of the input matrix in ascending order
      * \param[out] eigenvector the eigenvector corresponding to the smallest eigenvalue
      * \note if the smallest eigenvalue is not unique, this function may return any eigenvector that is consistent to the eigenvalue.
      * \ingroup common
      */
    template <typename Matrix, typename Vector> void
    eigen33Mod (const Matrix &mat, Vector &eigenvalues, Vector &eigenvector);
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Matrix, typename Vector> inline void
pcl::eigen33Mod (const Matrix& mat, Vector& eigenvalues, Vector& eigenvector)
{
  typedef typename Matrix::Scalar Scalar;
  // Scale the matrix so its entries are in [-1,1].  The scaling is applied
  // only when at least one matrix entry has magnitude larger than 1.

  Scalar scale = mat.cwiseAbs ().maxCoeff ();
  if (scale <= std::numeric_limits < Scalar > ::min ())
    scale = Scalar (1.0);

  Matrix scaledMat = mat / scale;

  Vector eigenvalues_temp;
  computeRoots (scaledMat, eigenvalues_temp);

  eigenvalues = eigenvalues_temp * scale;

  scaledMat.diagonal ().array () -= eigenvalues_temp (0);

  Vector vec1 = scaledMat.row (0).cross (scaledMat.row (1));
  Vector vec2 = scaledMat.row (0).cross (scaledMat.row (2));
  Vector vec3 = scaledMat.row (1).cross (scaledMat.row (2));

  Scalar len1 = vec1.squaredNorm ();
  Scalar len2 = vec2.squaredNorm ();
  Scalar len3 = vec3.squaredNorm ();

  if (len1 >= len2 && len1 >= len3)
    eigenvector = vec1 / std::sqrt (len1);
  else if (len2 >= len1 && len2 >= len3)
    eigenvector = vec2 / std::sqrt (len2);
  else
    eigenvector = vec3 / std::sqrt (len3);
}

namespace pcl
{
    /** \brief Compute the Least-Squares plane fit for a given set of points, using their indices,
      * and return the estimated plane parameters together with the surface curvature.
      * \param cloud the input point cloud
      * \param indices the point cloud indices that need to be used
      * \param plane_parameters the plane parameters as: a, b, c, d (ax + by + cz + d = 0)
      * \param eigen_values the eigenvalues in ascending order
      * \param curvature the estimated surface curvature as a measure of
      * \f[
      * \lambda_0 / (\lambda_0 + \lambda_1 + \lambda_2)
      * \f]
      * \ingroup features
      */
    template <typename PointT> inline bool
    computePointNormalMod (const pcl::PointCloud<PointT> &cloud, const std::vector<int> &indices,
                        Eigen::Vector4f &plane_parameters, Eigen::Vector3f &eigen_values, float &curvature)
    {
      // Placeholder for the 3x3 covariance matrix at each surface patch
      EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
      // 16-bytes aligned placeholder for the XYZ centroid of a surface patch
      Eigen::Vector4f xyz_centroid;
      if (indices.size () < 3 ||
          computeMeanAndCovarianceMatrix (cloud, indices, covariance_matrix, xyz_centroid) == 0)
      {
        plane_parameters.setConstant (std::numeric_limits<float>::quiet_NaN ());
        curvature = std::numeric_limits<float>::quiet_NaN ();
        return false;
      }

      // Get the plane normal and surface curvature
      // NOTE: the following code was encapsulated in the "solvePlaneParameters" methods (see feature.hpp in PCL)
      EIGEN_ALIGN16 Eigen::Vector3f eigen_vector; // will hold the smallest eigenvector
      pcl::eigen33Mod (covariance_matrix, eigen_values, eigen_vector);

      // smallest eigenvector is the normal
      plane_parameters [0] = eigen_vector [0];
      plane_parameters [1] = eigen_vector [1];
      plane_parameters [2] = eigen_vector [2];

      // Compute the curvature surface change
      float eig_sum = covariance_matrix.coeff (0) + covariance_matrix.coeff (4) + covariance_matrix.coeff (8);
      if (eig_sum != 0)
        curvature = fabsf (eigen_values(0) / eig_sum);
      else
        curvature = 0;

      plane_parameters[3] = 0;
      // Hessian form (D = nc . p_plane (centroid here) + p)
      plane_parameters[3] = -1 * plane_parameters.dot (xyz_centroid);

      return true;
    }

    /** \brief Flip (in place) the estimated normal of a point towards a given viewpoint
      * \param point a given point
      * \param vp_x the X coordinate of the viewpoint
      * \param vp_y the X coordinate of the viewpoint
      * \param vp_z the X coordinate of the viewpoint
      * \param nx the resultant X component of the plane normal
      * \param ny the resultant Y component of the plane normal
      * \param nz the resultant Z component of the plane normal
      * \ingroup features
      */
    template <typename PointT> inline void
    flipNormalTowardsViewpointMod (const PointT &point, float vp_x, float vp_y, float vp_z,
                                float &nx, float &ny, float &nz)
    {
      // See if we need to flip any plane normals
      vp_x -= point.x;
      vp_y -= point.y;
      vp_z -= point.z;

      // Dot product between the (viewpoint - point) and the plane normal
      float cos_theta = (vp_x * nx + vp_y * ny + vp_z * nz);

      // Flip the plane normal
      if (cos_theta < 0)
      {
        nx *= -1;
        ny *= -1;
        nz *= -1;
      }
    }


  /** \brief NormalEstimationOMP estimates local surface properties at each 3D point, such as surface normals and
    * curvatures, in parallel, using the OpenMP standard.
    * \author Radu Bogdan Rusu
    * \ingroup features
    */
  template <typename PointInT, typename PointOutT>
  class NormalEstimationOMPWithEigVals: public Feature<PointInT, PointOutT>
  {
    public:
      typedef boost::shared_ptr<NormalEstimationOMPWithEigVals<PointInT, PointOutT> > Ptr;
      typedef boost::shared_ptr<const NormalEstimationOMPWithEigVals<PointInT, PointOutT> > ConstPtr;
      using Feature<PointInT, PointOutT>::feature_name_;
      using Feature<PointInT, PointOutT>::getClassName;
      using Feature<PointInT, PointOutT>::indices_;
      using Feature<PointInT, PointOutT>::input_;
      using Feature<PointInT, PointOutT>::surface_;
      using Feature<PointInT, PointOutT>::k_;
      using Feature<PointInT, PointOutT>::search_radius_;
      using Feature<PointInT, PointOutT>::search_parameter_;

      typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
      typedef typename Feature<PointInT, PointOutT>::PointCloudConstPtr PointCloudConstPtr;


      /** \brief Initialize the scheduler and set the number of threads to use.
        * \param nr_threads the number of hardware threads to use (0 sets the value back to automatic)
        */
      NormalEstimationOMPWithEigVals (unsigned int nr_threads = 0)
      : vpx_ (0)
      , vpy_ (0)
      , vpz_ (0)
      , use_sensor_origin_ (true)
      , eigen_values_(new pcl::PointCloud<PointInT>())
      {
        feature_name_ = "NormalEstimationOMPWithEigVals";
        setNumberOfThreads(nr_threads);
      };

      /** \brief Empty destructor */
      virtual ~NormalEstimationOMPWithEigVals () {}

      /** \brief Initialize the scheduler and set the number of threads to use.
        * \param nr_threads the number of hardware threads to use (0 sets the value back to automatic)
        */
      void
      setNumberOfThreads (unsigned int nr_threads = 0);

      /** \brief Set the viewpoint.
        * \param vpx the X coordinate of the viewpoint
        * \param vpy the Y coordinate of the viewpoint
        * \param vpz the Z coordinate of the viewpoint
        */
      inline void
      setViewPoint (float vpx, float vpy, float vpz)
      {
        vpx_ = vpx;
        vpy_ = vpy;
        vpz_ = vpz;
        use_sensor_origin_ = false;
      }

      /** \brief Get the viewpoint.
        * \param [out] vpx x-coordinate of the view point
        * \param [out] vpy y-coordinate of the view point
        * \param [out] vpz z-coordinate of the view point
        * \note this method returns the currently used viewpoint for normal flipping.
        * If the viewpoint is set manually using the setViewPoint method, this method will return the set view point coordinates.
        * If an input cloud is set, it will return the sensor origin otherwise it will return the origin (0, 0, 0)
        */
      inline void
      getViewPoint (float &vpx, float &vpy, float &vpz)
      {
        vpx = vpx_;
        vpy = vpy_;
        vpz = vpz_;
      }

      /** \brief sets whether the sensor origin or a user given viewpoint should be used. After this method, the
        * normal estimation method uses the sensor origin of the input cloud.
        * to use a user defined view point, use the method setViewPoint
        */
      inline void
      useSensorOriginAsViewPoint ()
      {
        use_sensor_origin_ = true;
        if (input_)
        {
          vpx_ = input_->sensor_origin_.coeff (0);
          vpy_ = input_->sensor_origin_.coeff (1);
          vpz_ = input_->sensor_origin_.coeff (2);
        }
        else
        {
          vpx_ = 0;
          vpy_ = 0;
          vpz_ = 0;
        }
      }

    protected:
      /** \brief The number of threads the scheduler should use. */
      unsigned int threads_;

    /** \brief Estimate normals for all points given in <setInputCloud (), setIndices ()> using the surface in
      * setSearchSurface () and the spatial locator in setSearchMethod ()
      * \note In situations where not enough neighbors are found, the normal and curvature values are set to NaN.
      * \param output the resultant point cloud model dataset that contains surface normals and curvatures
      */
    void
    computeFeature (PointCloudOut &output);

    /** \brief Values describing the viewpoint ("pinhole" camera model assumed). For per point viewpoints, inherit
      * from NormalEstimation and provide your own computeFeature (). By default, the viewpoint is set to 0,0,0. */
    float vpx_, vpy_, vpz_;

    /** whether the sensor origin of the input cloud or a user given viewpoint should be used.*/
    bool use_sensor_origin_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /** eigen values in descending order */
    typename pcl::PointCloud<PointInT>::Ptr eigen_values_;
  };
}

#ifdef PCL_NO_PRECOMPILE
#include "normal_3d_omp_with_eigenvalues.hpp"
#endif

#endif  //#ifndef PCL_NORMAL_3D_OMP_MOD_H_
