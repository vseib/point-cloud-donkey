/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_JSONPARAMETERBASE_H
#define ISM3D_JSONPARAMETERBASE_H

#include <string>
#include <jsoncpp/json/json.h>

namespace ism3d
{
    class JSONParameterBase
    {
    public:
        JSONParameterBase(std::string);
        virtual ~JSONParameterBase();

        const std::string& getName() const;
        void toJson(Json::Value&) const;
        bool fromJson(const Json::Value&);

    protected:
        virtual Json::Value iToJson() const = 0;
        virtual bool iFromJson(const Json::Value&) = 0;
        virtual void assignDefault() = 0;

    private:
        std::string m_name;
    };
}

#endif // ISM3D_JSONPARAMETERBASE_H
