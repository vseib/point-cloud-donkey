#ifndef ISM3D_FEATURESCGF_H
#define ISM3D_FEATURESCGF_H

#include "features.h"

namespace ism3d
{
    /**
     * @brief The FeaturesCGF class
     * Computes features using the Compact Geometric Features
     * see https://marckhoury.github.io/CGF/ and https://github.com/marckhoury/CGF
     */
    class FeaturesCGF
            : public Features
    {
    public:
        FeaturesCGF();
        ~FeaturesCGF();

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

    private:

        double m_radius;
    };
}

#endif // ISM3D_FEATURESCGF_H
