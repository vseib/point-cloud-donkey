/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_EXCEPTION_H
#define ISM3D_EXCEPTION_H

#include <exception>
#include <string>
#include <sstream>
#include <jsoncpp/json/value.h>

namespace ism3d
{
    class Exception
            : public std::exception
    {
    public:
        virtual ~Exception() throw();
        virtual const char* what() const throw();

    protected:
        Exception(std::string);

    private:
        std::string m_message;
    };

    class JSONException
            : public Exception
    {
    public:
        JSONException(std::string);
        JSONException(std::string, Json::Value);
    };

    class RuntimeException
            : public Exception
    {
    public:
        RuntimeException(std::string);
    };

    class BadParamException
            : public Exception
    {
    public:
        BadParamException(std::string);
    };

    template <typename T>
    class BadParamExceptionType
            : public BadParamException
    {
    public:
        BadParamExceptionType(std::string message, T value)
            : BadParamException(getMessage(message, value)), m_value(value)
        {
        }

        ~BadParamExceptionType() throw()
        {
        }

    protected:
        std::string getMessage(std::string message, T value)
        {
            std::string valueStr = getValue(value);
            if (valueStr.length() > 0)
                message += " (Value: " + valueStr + ")";
            return message;
        }

        std::string getValue(T value) const {
            std::stringstream sstr;
            sstr << value;
            return sstr.str();
        }

    private:
        T m_value;
    };
}

#endif // ISM3D_EXCEPTION_H
