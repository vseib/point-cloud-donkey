#ifndef ISM3D_JSONPARAMETER_H
#define ISM3D_JSONPARAMETER_H

#include "json_parameter_base.h"
#include "json_parameter_traits.h"
#include "exception.h"

namespace ism3d
{
    template <typename T>
    class JSONParameter
            : public JSONParameterBase
    {
    public:
        JSONParameter(T& value, std::string name, T defaultValue)
            : JSONParameterBase(name), m_value(value), m_defaultValue(defaultValue) {
            assignDefault();
        }

    protected:
        Json::Value iToJson() const
        {
            return JSONParameterTraits<T>::toJson(m_value);
        }

        bool iFromJson(const Json::Value& object)
        {
            if (!JSONParameterTraits<T>::check(object))
                throw JSONException("invalid json object type", object);

            m_value = JSONParameterTraits<T>::fromJson(object);
            return true;
        }

        void assignDefault()
        {
            m_value = m_defaultValue;
        }

    private:
        T& m_value;
        T m_defaultValue;
    };
}

#endif // ISM3D_JSONPARAMETER_H
