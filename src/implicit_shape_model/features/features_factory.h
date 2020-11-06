/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_FEATURESFACTORY_H
#define ISM3D_FEATURESFACTORY_H

#include "../utils/factory.h"
#include "features_fpfh.h"
#include "features_pfh.h"
#include "features_bshot.h"
#include "features_shot.h"
#include "features_shot_global.h"
#include "features_short_shot_pcl.h"
#include "features_short_shot.h"
#include "features_short_shot_global.h"
#include "features_cshot.h"
#include "features_cshot_global.h"
//#include "features_short_cshot.h"  TODO VS: re-implement short c-shot
#include "features_rift.h"
#include "features_esf.h"
#include "features_3dsc.h"
#include "features_usc.h"
#include "features_rsd.h"
#include "features_grsd.h"
#include "features_spin_image.h"
#include "features_rops.h"
#include "features_narf.h"
#include "features_vfh.h"
#include "features_cvfh.h"
#include "features_ourcvfh.h"
#include "features_usc_global.h"
#include "features_esf_local.h"
#include "features_cgf.h"
#include "features_gasd.h"
#include "features_dummy.h"

namespace ism3d
{
    template <>
    Features* Factory<Features>::createByType(const std::string& type)
    {
        if (type == FeaturesFPFH::getTypeStatic())
            return new FeaturesFPFH();
        else if (type == FeaturesPFH::getTypeStatic())
            return new FeaturesPFH();
        else if (type == FeaturesSHOT::getTypeStatic())
            return new FeaturesSHOT();
        else if (type == FeaturesBSHOT::getTypeStatic())
            return new FeaturesBSHOT();
        else if (type == FeaturesSHORTSHOTPCL::getTypeStatic())
            return new FeaturesSHORTSHOTPCL();
        else if (type == FeaturesSHORTSHOT::getTypeStatic())
            return new FeaturesSHORTSHOT();
        else if (type == FeaturesCSHOT::getTypeStatic())
            return new FeaturesCSHOT();
        else if (type == FeaturesCGF::getTypeStatic())
            return new FeaturesCGF();
        else if (type == FeaturesSHORTCSHOT::getTypeStatic())
            return new FeaturesSHORTCSHOT();
        else if (type == FeaturesRIFT::getTypeStatic())
            return new FeaturesRIFT(); // works only with color data!
        else if (type == Features3DSC::getTypeStatic())
            return new Features3DSC();
        else if (type == FeaturesSpinImage::getTypeStatic())
            return new FeaturesSpinImage();
        else if (type == FeaturesROPS::getTypeStatic())
            return new FeaturesROPS();
        else if (type == FeaturesNARF::getTypeStatic())
            return new FeaturesNARF(); // works only with organized point clouds from a sensor
        else if (type == FeaturesUSC::getTypeStatic())
            return new FeaturesUSC();
        else if (type == FeaturesRSD::getTypeStatic())
            return new FeaturesRSD();
        else if (type == FeaturesESFLocal::getTypeStatic())
            return new FeaturesESFLocal();
        else if (type == FeaturesGRSD::getTypeStatic())
            return new FeaturesGRSD(); // global feature!
        else if (type == FeaturesESF::getTypeStatic())
            return new FeaturesESF(); // global feature!
        else if (type == FeaturesVFH::getTypeStatic())
            return new FeaturesVFH(); // global feature!
        else if (type == FeaturesCVFH::getTypeStatic())
            return new FeaturesCVFH(); // global feature!
        else if (type == FeaturesOURCVFH::getTypeStatic())
            return new FeaturesOURCVFH(); // global feature!
        else if (type == FeaturesSHOTGlobal::getTypeStatic())
            return new FeaturesSHOTGlobal(); // global feature!
        else if (type == FeaturesCSHOTGlobal::getTypeStatic())
            return new FeaturesCSHOTGlobal(); // global feature!
        else if (type == FeaturesSHORTSHOTGlobal::getTypeStatic())
            return new FeaturesSHORTSHOTGlobal(); // global feature!
        else if (type == FeaturesUSCGlobal::getTypeStatic())
            return new FeaturesUSCGlobal(); // global feature!
        else if (type == FeaturesGASD::getTypeStatic())
            return new FeaturesGASD(); // global feature, min. PCL 1.9!
        else if (type == FeaturesDummy::getTypeStatic())
            return new FeaturesDummy(); // global feature dummy to be able to use old config files without global features
        else
            return 0;
    }
}

#endif // ISM3D_FEATURESFACTORY_H
