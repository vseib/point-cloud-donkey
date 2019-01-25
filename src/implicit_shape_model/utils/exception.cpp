/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "exception.h"
#include "utils.h"

namespace ism3d
{
    Exception::Exception(std::string message)
        : m_message(message)
    {
        LOG_FATAL("Exception: " + m_message);
    }

    Exception::~Exception() throw()
    {
    }

    const char* Exception::what() const throw()
    {
        return m_message.c_str();
    }

    JSONException::JSONException(std::string message)
        : Exception(message)
    {
    }

    JSONException::JSONException(std::string message, Json::Value json)
        : Exception(message + ", JSON = \"" + json.toStyledString() + "\"")
    {
    }

    RuntimeException::RuntimeException(std::string message)
        : Exception(message)
    {
    }

    BadParamException::BadParamException(std::string message)
        : Exception(message)
    {
    }
}
