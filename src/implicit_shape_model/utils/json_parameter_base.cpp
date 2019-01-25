/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "json_parameter_base.h"
#include "utils.h"

namespace ism3d
{
    JSONParameterBase::JSONParameterBase(std::string name)
        : m_name(name)
    {
    }

    JSONParameterBase::~JSONParameterBase()
    {
    }

    const std::string& JSONParameterBase::getName() const
    {
        return m_name;
    }

    void JSONParameterBase::toJson(Json::Value& object) const
    {
        object[m_name] = iToJson();
    }

    bool JSONParameterBase::fromJson(const Json::Value& object)
    {
        if (!object.isMember(m_name)) {
            LOG_WARN("json object does not have a member " << m_name << ", assigning default value");
            assignDefault();
        }
        else
            return iFromJson(object[m_name]);

        return true;
    }
}
