
#include "features_short_cshot.h"

namespace ism3d
{
FeaturesSHORTCSHOT::FeaturesSHORTCSHOT() :
    nr_shape_bins_ (10), nr_color_bins_ (30),
    sqradius_ (0), radius3_4_ (0), radius1_4_ (0), radius1_2_ (0),
    nr_grid_sector_ (32), maxAngularSectors_ (32), descLength_ (0)
{
    addParameter(m_radius, "Radius", 0.1);
}

FeaturesSHORTCSHOT::~FeaturesSHORTCSHOT()
{
}

pcl::PointCloud<ISMFeature>::Ptr FeaturesSHORTCSHOT::iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr pointCloud,
                                                                        pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                                        pcl::PointCloud<PointT>::ConstPtr pointCloudWithoutNaNNormals,
                                                                        pcl::PointCloud<pcl::Normal>::ConstPtr normalsWithoutNaN,
                                                                        pcl::PointCloud<pcl::ReferenceFrame>::Ptr referenceFrames,
                                                                        pcl::PointCloud<PointT>::Ptr keypoints,
                                                                        pcl::search::Search<PointT>::Ptr search)
{
    // init params
    if (pointCloud->isOrganized())
        surface_ = pointCloud;
    else
        surface_ = pointCloudWithoutNaNNormals;

    input_ = keypoints;
    frames_ = referenceFrames;
    search_radius_ = m_radius;
    search_ = search;
    search_->setInputCloud(surface_);

    // init indices
    indices_.reset (new std::vector<int>);
    indices_->resize (input_->points.size ());
    for (size_t i = 0; i < indices_->size (); ++i)
    {
        (*indices_)[i] = static_cast<int>(i);
    }

    // this is the "computeFeature" method
    pcl::PointCloud<pcl::SHOT1344>::Ptr output(new pcl::PointCloud<pcl::SHOT1344>());
    output->resize(input_->size());

    descLength_ = nr_grid_sector_ * (nr_shape_bins_+1);
    descLength_ += nr_grid_sector_ * (nr_color_bins_+1);

    sqradius_ = search_radius_ * search_radius_;
    radius3_4_ = (search_radius_*3) / 4;
    radius1_4_ = search_radius_ / 4;
    radius1_2_ = search_radius_ / 2;

    output->is_dense = true;
    // Iterating over the entire index vector
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(m_numThreads)
    #endif
    for (size_t idx = 0; idx < indices_->size (); ++idx)
    {
        Eigen::VectorXf shot;
        shot.setZero (descLength_);

        std::vector<int> nn_indices;
        std::vector<float> nn_dists;

        bool lrf_is_nan = false;
        const pcl::ReferenceFrame& current_frame = (*frames_)[idx];
        if (!pcl_isfinite (current_frame.x_axis[0]) ||
                !pcl_isfinite (current_frame.y_axis[0]) ||
                !pcl_isfinite (current_frame.z_axis[0]))
        {
            LOG_WARN("The local reference frame is not valid! Aborting description of point with index " << (*indices_)[idx])
                    lrf_is_nan = true;
        }

        if (lrf_is_nan || search_->radiusSearch((*input_)[(*indices_)[idx]], search_radius_, nn_indices, nn_dists) == 0)
        {
            // Copy into the resultant cloud
            for (int d = 0; d < descLength_; ++d)
                output->points[idx].descriptor[d] = std::numeric_limits<float>::quiet_NaN ();
            for (int d = 0; d < 9; ++d)
                output->points[idx].rf[d] = std::numeric_limits<float>::quiet_NaN ();

            output->is_dense = false;
            continue;
        }

        // Estimate the SHOT descriptor at each patch
        computePointSHOT (static_cast<int> (idx), nn_indices, nn_dists, shot);

        // Copy into the resultant cloud
        for (int d = 0; d < descLength_; ++d)
            output->points[idx].descriptor[d] = shot[d];
        for (int d = 0; d < 3; ++d)
        {
            output->points[idx].rf[d + 0] = frames_->points[idx].x_axis[d];
            output->points[idx].rf[d + 3] = frames_->points[idx].y_axis[d];
            output->points[idx].rf[d + 6] = frames_->points[idx].z_axis[d];
        }
    }

    // create descriptor point cloud
    pcl::PointCloud<ISMFeature>::Ptr features(new pcl::PointCloud<ISMFeature>());
    features->resize(output->size());

    for (int i = 0; i < (int)output->size(); i++)
    {
        ISMFeature& feature = features->at(i);
        const pcl::SHOT1344& shot = output->at(i);

        // store the shape descriptor: only one value per histogram
        feature.descriptor.resize(32);
        for (int j = 0; j < feature.descriptor.size(); j++)
            feature.descriptor[j] = shot.descriptor[5+j*11];

        // store the color descriptor: only one value per histogram
        feature.descriptor.resize(64); // TODO VS: verify that the relevant value is in the central bin: try default value for binDistanceColor
        for (int j = 32; j < feature.descriptor.size(); j++)
            feature.descriptor[j] = shot.descriptor[352+(j-32)*31];

//            // store the complete color descriptor
//            feature.descriptor.resize(1344-352+32);
//            for (int j = 32; j < feature.descriptor.size(); j++)
//                feature.descriptor[j] = shot.descriptor[352+(j-32)];
    }

    return features;
}


void FeaturesSHORTCSHOT::computePointSHOT(
        const int index, const std::vector<int> &indices, const std::vector<float> &sqr_dists, Eigen::VectorXf &shot)
{
    // Clear the resultant shot
    shot.setZero ();
    std::vector<double> binDistanceColor;
    size_t nNeighbors = indices.size ();

    //Skip the current feature if the number of its neighbors is not sufficient for its description
    if (indices.size () < 5)
    {
        LOG_WARN("Warning! Neighborhood has less than 5 vertexes. Aborting description of point with index " << (*indices_)[index]);
        shot.setConstant(descLength_, 1, std::numeric_limits<float>::quiet_NaN () );
        return;
    }


    // compute binDistanceColor
    binDistanceColor.resize (nNeighbors);

    unsigned char redRef = input_->points[(*indices_)[index]].r;
    unsigned char greenRef = input_->points[(*indices_)[index]].g;
    unsigned char blueRef = input_->points[(*indices_)[index]].b;

    float LRef, aRef, bRef;

    RGB2CIELAB (redRef, greenRef, blueRef, LRef, aRef, bRef);
    LRef /= 100.0f;
    aRef /= 120.0f;
    bRef /= 120.0f;    //normalized LAB components (0<L<1, -1<a<1, -1<b<1)

    for (size_t i_idx = 0; i_idx < indices.size (); ++i_idx)
    {
      unsigned char red = surface_->points[indices[i_idx]].r;
      unsigned char green = surface_->points[indices[i_idx]].g;
      unsigned char blue = surface_->points[indices[i_idx]].b;

      float L, a, b;

      RGB2CIELAB (red, green, blue, L, a, b);
      L /= 100.0f;
      a /= 120.0f;
      b /= 120.0f;   //normalized LAB components (0<L<1, -1<a<1, -1<b<1)

      double colorDistance = (fabs (LRef - L) + ((fabs (aRef - a) + fabs (bRef - b)) / 2)) /3;

      if (colorDistance > 1.0)
        colorDistance = 1.0;
      if (colorDistance < 0.0)
        colorDistance = 0.0;

      binDistanceColor[i_idx] = colorDistance * nr_color_bins_;
    }

    // Interpolate
    interpolateDoubleChannel (indices, sqr_dists, index, binDistanceColor, nr_shape_bins_, nr_color_bins_, shot);

    // Normalize the final histogram
    normalizeHistogram (shot, descLength_);
}


void FeaturesSHORTCSHOT::interpolateDoubleChannel(const std::vector<int> &indices,
  const std::vector<float> &sqr_dists,
  const int index,
  std::vector<double> &binDistanceColor,
  const int nr_bins_shape,
  const int nr_bins_color,
  Eigen::VectorXf &shot)
{
  const Eigen::Vector4f &central_point = (*input_)[(*indices_)[index]].getVector4fMap ();
  const pcl::ReferenceFrame& current_frame = (*frames_)[index];

  int shapeToColorStride = nr_grid_sector_*(nr_bins_shape+1);

  Eigen::Vector4f current_frame_x (current_frame.x_axis[0], current_frame.x_axis[1], current_frame.x_axis[2], 0);
  Eigen::Vector4f current_frame_y (current_frame.y_axis[0], current_frame.y_axis[1], current_frame.y_axis[2], 0);
  Eigen::Vector4f current_frame_z (current_frame.z_axis[0], current_frame.z_axis[1], current_frame.z_axis[2], 0);

  for (size_t i_idx = 0; i_idx < indices.size (); ++i_idx)
  {
    // binDistanceShape becomes a constant without normals
    double binDistanceShape = 5.0;

    Eigen::Vector4f delta = surface_->points[indices[i_idx]].getVector4fMap () - central_point;
    delta[3] = 0;

    // Compute the Euclidean norm
    double distance = sqrt (sqr_dists[i_idx]);

    if (areEquals (distance, 0.0))
      continue;

    double xInFeatRef = delta.dot (current_frame_x);
    double yInFeatRef = delta.dot (current_frame_y);
    double zInFeatRef = delta.dot (current_frame_z);

    // To avoid numerical problems afterwards
    if (fabs (yInFeatRef) < 1E-30)
      yInFeatRef  = 0;
    if (fabs (xInFeatRef) < 1E-30)
      xInFeatRef  = 0;
    if (fabs (zInFeatRef) < 1E-30)
      zInFeatRef  = 0;

    unsigned char bit4 = ((yInFeatRef > 0) || ((yInFeatRef == 0.0) && (xInFeatRef < 0))) ? 1 : 0;
    unsigned char bit3 = static_cast<unsigned char> (((xInFeatRef > 0) || ((xInFeatRef == 0.0) && (yInFeatRef > 0))) ? !bit4 : bit4);

    assert (bit3 == 0 || bit3 == 1);

    int desc_index = (bit4<<3) + (bit3<<2);

    desc_index = desc_index << 1;

    if ((xInFeatRef * yInFeatRef > 0) || (xInFeatRef == 0.0))
      desc_index += (fabs (xInFeatRef) >= fabs (yInFeatRef)) ? 0 : 4;
    else
      desc_index += (fabs (xInFeatRef) > fabs (yInFeatRef)) ? 4 : 0;

    desc_index += zInFeatRef > 0 ? 1 : 0;

    // 2 RADII
    desc_index += (distance > radius1_2_) ? 2 : 0;

    int step_index_shape = static_cast<int>(floor (binDistanceShape +0.5));
    int step_index_color = static_cast<int>(floor (binDistanceColor[i_idx] +0.5));

    int volume_index_shape = desc_index * (nr_bins_shape+1);
    int volume_index_color = shapeToColorStride + desc_index * (nr_bins_color+1);

    //Interpolation on the cosine (adjacent bins in the histrogram)
    binDistanceShape -= step_index_shape;
    binDistanceColor[i_idx] -= step_index_color;

    double intWeightShape = (1- fabs (binDistanceShape));
    double intWeightColor = (1- fabs (binDistanceColor[i_idx]));

    if (binDistanceShape > 0)
      shot[volume_index_shape + ((step_index_shape + 1) % nr_bins_shape)] += static_cast<float> (binDistanceShape);
    else
      shot[volume_index_shape + ((step_index_shape - 1 + nr_bins_shape) % nr_bins_shape)] -= static_cast<float> (binDistanceShape);

    if (binDistanceColor[i_idx] > 0)
      shot[volume_index_color + ((step_index_color+1) % nr_bins_color)] += static_cast<float> (binDistanceColor[i_idx]);
    else
      shot[volume_index_color + ((step_index_color - 1 + nr_bins_color) % nr_bins_color)] -= static_cast<float> (binDistanceColor[i_idx]);

    //Interpolation on the distance (adjacent husks)

    if (distance > radius1_2_)   //external sphere
    {
      double radiusDistance = (distance - radius3_4_) / radius1_2_;

      if (distance > radius3_4_) //most external sector, votes only for itself
      {
        intWeightShape += 1 - radiusDistance; //weight=1-d
        intWeightColor += 1 - radiusDistance; //weight=1-d
      }
      else  //3/4 of radius, votes also for the internal sphere
      {
        intWeightShape += 1 + radiusDistance;
        intWeightColor += 1 + radiusDistance;
        shot[(desc_index - 2) * (nr_bins_shape+1) + step_index_shape] -= static_cast<float> (radiusDistance);
        shot[shapeToColorStride + (desc_index - 2) * (nr_bins_color+1) + step_index_color] -= static_cast<float> (radiusDistance);
      }
    }
    else    //internal sphere
    {
      double radiusDistance = (distance - radius1_4_) / radius1_2_;

      if (distance < radius1_4_) //most internal sector, votes only for itself
      {
        intWeightShape += 1 + radiusDistance;
        intWeightColor += 1 + radiusDistance; //weight=1-d
      }
      else  //3/4 of radius, votes also for the external sphere
      {
        intWeightShape += 1 - radiusDistance; //weight=1-d
        intWeightColor += 1 - radiusDistance; //weight=1-d
        shot[(desc_index + 2) * (nr_bins_shape+1) + step_index_shape] += static_cast<float> (radiusDistance);
        shot[shapeToColorStride + (desc_index + 2) * (nr_bins_color+1) + step_index_color] += static_cast<float> (radiusDistance);
      }
    }

    //Interpolation on the inclination (adjacent vertical volumes)
    double inclinationCos = zInFeatRef / distance;
    if (inclinationCos < - 1.0)
      inclinationCos = - 1.0;
    if (inclinationCos > 1.0)
      inclinationCos = 1.0;

    double inclination = acos (inclinationCos);

    assert (inclination >= 0.0 && inclination <= PST_RAD_180);

    if (inclination > PST_RAD_90 || (fabs (inclination - PST_RAD_90) < 1e-30 && zInFeatRef <= 0))
    {
      double inclinationDistance = (inclination - PST_RAD_135) / PST_RAD_90;
      if (inclination > PST_RAD_135)
      {
        intWeightShape += 1 - inclinationDistance;
        intWeightColor += 1 - inclinationDistance;
      }
      else
      {
        intWeightShape += 1 + inclinationDistance;
        intWeightColor += 1 + inclinationDistance;
        assert ((desc_index + 1) * (nr_bins_shape+1) + step_index_shape >= 0 && (desc_index + 1) * (nr_bins_shape+1) + step_index_shape < descLength_);
        assert (shapeToColorStride + (desc_index + 1) * (nr_bins_color+ 1) + step_index_color >= 0 && shapeToColorStride + (desc_index + 1) * (nr_bins_color+1) + step_index_color < descLength_);
        shot[(desc_index + 1) * (nr_bins_shape+1) + step_index_shape] -= static_cast<float> (inclinationDistance);
        shot[shapeToColorStride + (desc_index + 1) * (nr_bins_color+1) + step_index_color] -= static_cast<float> (inclinationDistance);
      }
    }
    else
    {
      double inclinationDistance = (inclination - PST_RAD_45) / PST_RAD_90;
      if (inclination < PST_RAD_45)
      {
        intWeightShape += 1 + inclinationDistance;
        intWeightColor += 1 + inclinationDistance;
      }
      else
      {
        intWeightShape += 1 - inclinationDistance;
        intWeightColor += 1 - inclinationDistance;
        assert ((desc_index - 1) * (nr_bins_shape+1) + step_index_shape >= 0 && (desc_index - 1) * (nr_bins_shape+1) + step_index_shape < descLength_);
        assert (shapeToColorStride + (desc_index - 1) * (nr_bins_color+ 1) + step_index_color >= 0 && shapeToColorStride + (desc_index - 1) * (nr_bins_color+1) + step_index_color < descLength_);
        shot[(desc_index - 1) * (nr_bins_shape+1) + step_index_shape] += static_cast<float> (inclinationDistance);
        shot[shapeToColorStride + (desc_index - 1) * (nr_bins_color+1) + step_index_color] += static_cast<float> (inclinationDistance);
      }
    }

    if (yInFeatRef != 0.0 || xInFeatRef != 0.0)
    {
      //Interpolation on the azimuth (adjacent horizontal volumes)
      double azimuth = atan2 (yInFeatRef, xInFeatRef);

      int sel = desc_index >> 2;
      double angularSectorSpan = PST_RAD_45;
      double angularSectorStart = - PST_RAD_PI_7_8;

      double azimuthDistance = (azimuth - (angularSectorStart + angularSectorSpan*sel)) / angularSectorSpan;
      assert ((azimuthDistance < 0.5 || areEquals (azimuthDistance, 0.5)) && (azimuthDistance > - 0.5 || areEquals (azimuthDistance, - 0.5)));
      azimuthDistance = (std::max)(- 0.5, std::min (azimuthDistance, 0.5));

      if (azimuthDistance > 0)
      {
        intWeightShape += 1 - azimuthDistance;
        intWeightColor += 1 - azimuthDistance;
        int interp_index = (desc_index + 4) % maxAngularSectors_;
        assert (interp_index * (nr_bins_shape+1) + step_index_shape >= 0 && interp_index * (nr_bins_shape+1) + step_index_shape < descLength_);
        assert (shapeToColorStride + interp_index * (nr_bins_color+1) + step_index_color >= 0 && shapeToColorStride + interp_index * (nr_bins_color+1) + step_index_color < descLength_);
        shot[interp_index * (nr_bins_shape+1) + step_index_shape] += static_cast<float> (azimuthDistance);
        shot[shapeToColorStride + interp_index * (nr_bins_color+1) + step_index_color] += static_cast<float> (azimuthDistance);
      }
      else
      {
        int interp_index = (desc_index - 4 + maxAngularSectors_) % maxAngularSectors_;
        intWeightShape += 1 + azimuthDistance;
        intWeightColor += 1 + azimuthDistance;
        assert (interp_index * (nr_bins_shape+1) + step_index_shape >= 0 && interp_index * (nr_bins_shape+1) + step_index_shape < descLength_);
        assert (shapeToColorStride + interp_index * (nr_bins_color+1) + step_index_color >= 0 && shapeToColorStride + interp_index * (nr_bins_color+1) + step_index_color < descLength_);
        shot[interp_index * (nr_bins_shape+1) + step_index_shape] -= static_cast<float> (azimuthDistance);
        shot[shapeToColorStride + interp_index * (nr_bins_color+1) + step_index_color] -= static_cast<float> (azimuthDistance);
      }
    }

    assert (volume_index_shape + step_index_shape >= 0 &&  volume_index_shape + step_index_shape < descLength_);
    assert (volume_index_color + step_index_color >= 0 &&  volume_index_color + step_index_color < descLength_);
    shot[volume_index_shape + step_index_shape] += static_cast<float> (intWeightShape);
    shot[volume_index_color + step_index_color] += static_cast<float> (intWeightColor);
  }
}


float FeaturesSHORTCSHOT::sRGB_LUT[256] = {- 1};

float FeaturesSHORTCSHOT::sXYZ_LUT[4000] = {- 1};

void FeaturesSHORTCSHOT::RGB2CIELAB (unsigned char R, unsigned char G, unsigned char B,
                                     float &L, float &A, float &B2)
{
  if (sRGB_LUT[0] < 0)
  {
    for (int i = 0; i < 256; i++)
    {
      float f = static_cast<float> (i) / 255.0f;
      if (f > 0.04045)
        sRGB_LUT[i] = powf ((f + 0.055f) / 1.055f, 2.4f);
      else
        sRGB_LUT[i] = f / 12.92f;
    }

    for (int i = 0; i < 4000; i++)
    {
      float f = static_cast<float> (i) / 4000.0f;
      if (f > 0.008856)
        sXYZ_LUT[i] = static_cast<float> (powf (f, 0.3333f));
      else
        sXYZ_LUT[i] = static_cast<float>((7.787 * f) + (16.0 / 116.0));
    }
  }

  float fr = sRGB_LUT[R];
  float fg = sRGB_LUT[G];
  float fb = sRGB_LUT[B];

  // Use white = D65
  const float x = fr * 0.412453f + fg * 0.357580f + fb * 0.180423f;
  const float y = fr * 0.212671f + fg * 0.715160f + fb * 0.072169f;
  const float z = fr * 0.019334f + fg * 0.119193f + fb * 0.950227f;

  float vx = x / 0.95047f;
  float vy = y;
  float vz = z / 1.08883f;

  vx = sXYZ_LUT[int(vx*4000)];
  vy = sXYZ_LUT[int(vy*4000)];
  vz = sXYZ_LUT[int(vz*4000)];

  L = 116.0f * vy - 16.0f;
  if (L > 100)
    L = 100.0f;

  A = 500.0f * (vx - vy);
  if (A > 120)
    A = 120.0f;
  else if (A <- 120)
    A = -120.0f;

  B2 = 200.0f * (vy - vz);
  if (B2 > 120)
    B2 = 120.0f;
  else if (B2<- 120)
    B2 = -120.0f;
}


void FeaturesSHORTCSHOT::normalizeHistogram(Eigen::VectorXf &shot, int desc_length)
{
    // Normalization is performed by considering the L2 norm
    // and not the sum of bins, as reported in the ECCV paper.
    // This is due to additional experiments performed by the authors after its pubblication,
    // where L2 normalization turned out better at handling point density variations.
    double acc_norm = 0;
    for (int j = 0; j < desc_length; j++)
        acc_norm += shot[j] * shot[j];
    acc_norm = sqrt (acc_norm);
    for (int j = 0; j < desc_length; j++)
        shot[j] /= static_cast<float> (acc_norm);
}


std::string FeaturesSHORTCSHOT::getTypeStatic()
{
    return "SHORT_CSHOT";
}

std::string FeaturesSHORTCSHOT::getType() const
{
    return FeaturesSHORTCSHOT::getTypeStatic();
}
}
