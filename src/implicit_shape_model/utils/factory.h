/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_FACTORY_H
#define ISM3D_FACTORY_H

namespace ism3d
{
    /**
     * @brief The Factory class
     * The factory is used to create and initialize instances of JSONObject
     */
    template <typename TClass>
    class Factory
    {
    public:
        static TClass* create(Json::Value object) {
            if (object.isNull())
                return 0;

            std::string typeStr = "";

            if (object.isMember("Type")) {
                Json::Value type = object["Type"];
                if (!type.isString())
                    return 0;

                typeStr = type.asString();
            }

            TClass* newInstance = createByType(typeStr);

            if (!newInstance || !newInstance->configFromJson(object)) {
                delete newInstance;
                throw RuntimeException("could not create object of type: \"" + typeStr + "\"");
            }

            return newInstance;
        }

    private:
        Factory();
        ~Factory();

        static TClass* createByType(const std::string& type);
    };
}

#endif // ISM3D_FACTORY_H
