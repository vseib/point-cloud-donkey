/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2021, Viktor Seib
 * All rights reserved.
 *
 */

#include "color_conversion.h"
#include <math.h>

ism3d::ColorConversion ism3d::ColorConversionStatic::color_conversion;

namespace ism3d
{

    ColorConversion::ColorConversion()
    {
        // init look-up tables, code taken from PCL (Point Cloud Library)
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

    // NOTE: next block taken from PCL (Point Cloud Library): cshot.hpp
    void ColorConversion::RgbToCieLabNormalized (unsigned char R, unsigned char G, unsigned char B,
                                         float &L, float &A, float &B2) const
    {

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

        //normalized LAB components (0<L<1, -1<a<1, -1<b<1)
        L /= 100.0f;
        A /= 120.0f;
        B2 /= 120.0f;
    }

    float ColorConversion::getColorDistance(const float L, const float a, const float b,
                                            const float LRef, const float aRef, const float bRef) const
    {
        double color_distance = (fabs (LRef - L) + ((fabs (aRef - a) + fabs (bRef - b)) / 2)) /3;
        if (color_distance > 1.0)
            color_distance = 1.0;
        if (color_distance < 0.0)
            color_distance = 0.0;
        return float(color_distance);
    }
}
