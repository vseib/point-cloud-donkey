#ifndef ISM3D_FEATURESSHORTCSHOT_H
#define ISM3D_FEATURESSHORTCSHOT_H

#include "features.h"

// Useful constants.
#define PST_PI 3.1415926535897932384626433832795
#define PST_RAD_45 0.78539816339744830961566084581988
#define PST_RAD_90 1.5707963267948966192313216916398
#define PST_RAD_135 2.3561944901923449288469825374596
#define PST_RAD_180 PST_PI
#define PST_RAD_360 6.283185307179586476925286766558
#define PST_RAD_PI_7_8 2.7488935718910690836548129603691


namespace ism3d
{
/**
     * @brief The FeaturesSHORTCSHOT class
     * Computes features using the signature of histograms of orientations descriptor with color,
     * but only retains a small portion of the descriptor
     */
class FeaturesSHORTCSHOT
        : public Features
{
public:
    FeaturesSHORTCSHOT();
    ~FeaturesSHORTCSHOT();

    static std::string getTypeStatic();
    std::string getType() const;

protected:
    pcl::PointCloud<ISMFeature>::Ptr iComputeDescriptors(pcl::PointCloud<PointT>::ConstPtr,
                                                         pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                         pcl::PointCloud<PointT>::ConstPtr,
                                                         pcl::PointCloud<pcl::Normal>::ConstPtr,
                                                         pcl::PointCloud<pcl::ReferenceFrame>::Ptr,
                                                         pcl::PointCloud<PointT>::Ptr,
                                                         pcl::search::Search<PointT>::Ptr);

    inline bool
    areEquals (double val1, double val2, double zeroDoubleEps = 1E-15)
    {
      return (fabs (val1 - val2)<zeroDoubleEps);
    }

    inline bool
    areEquals (float val1, float val2, float zeroFloatEps = 1E-8f)
    {
      return (fabs (val1 - val2)<zeroFloatEps);
    }

    void computePointSHOT(const int index, const std::vector<int> &indices, const std::vector<float> &sqr_dists, Eigen::VectorXf &shot);

    void interpolateDoubleChannel (const std::vector<int> &indices,
      const std::vector<float> &sqr_dists, const int index, std::vector<double> &binDistanceColor,
      const int nr_bins_shape, const int nr_bins_color, Eigen::VectorXf &shot);

    void RGB2CIELAB (unsigned char R, unsigned char G, unsigned char B, float &L, float &A, float &B2);

    void normalizeHistogram(Eigen::VectorXf &shot, int desc_length);


private:

    static float sRGB_LUT[256];
    static float sXYZ_LUT[4000];

    double m_radius;

    double search_radius_;
    pcl::PointCloud<PointT>::ConstPtr surface_;
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr frames_;
    pcl::PointCloud<PointT>::Ptr input_;
    pcl::IndicesPtr indices_;
    pcl::search::Search<PointT>::Ptr search_;

    pcl::PointCloud<pcl::Normal>::ConstPtr normals_;

    int nr_shape_bins_;
    int nr_color_bins_;
    float sqradius_;
    float radius3_4_;
    float radius1_4_;
    float radius1_2_;
    int nr_grid_sector_;
    int maxAngularSectors_;
    int descLength_;
};
}

#endif // ISM3D_FEATURESSHORTCSHOT_H
