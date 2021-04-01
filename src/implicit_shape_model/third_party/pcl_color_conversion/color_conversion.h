/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_COLOR_CONVERSION_H
#define ISM3D_COLOR_CONVERSION_H

#include <array>

namespace ism3d
{
    class ColorConversion
    {
    public:
        ColorConversion();
        virtual ~ColorConversion() = default;

        void RgbToCieLabNormalized (unsigned char R, unsigned char G, unsigned char B,
                                    float &L, float &A, float &B2) const;

        float getColorDistance(const float L, const float a, const float b,
                               const float LRef, const float aRef, const float bRef) const;

        std::array<float, 256> sRGB_LUT;
        std::array<float, 4000> sXYZ_LUT;
    };

    class ColorConversionStatic
    {
    public:
        static ColorConversion& getColorConversion()
        {
            return color_conversion;
        }

    private:
        static ColorConversion color_conversion;
    };
}

#endif // ISM3D_COLOR_CONVERSION_H
