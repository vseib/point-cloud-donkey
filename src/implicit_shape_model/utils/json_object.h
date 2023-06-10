/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#ifndef ISM3D_JSONOBJECT_H
#define ISM3D_JSONOBJECT_H

#include <boost/filesystem.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>

#include "json_parameter.h"

namespace ism3d
{
    /**
     * @brief The JSONObject class
     * The base class for object which can be (de-)serialized to JSON. The object
     * can have a list of parameters which are automatically restored at loading,
     * as well as arbitrary data. When writing, the object gets serialized into
     * two distinct files, one storing the parameter configuration and the other
     * the object data.
     */
    class JSONObject
    {
    public:
        JSONObject();
        virtual ~JSONObject();

        /**
         * @brief The object instance is created with a factory pattern, thus the type identifies
         * the right object class.
         * @return the object type string
         */
        virtual std::string getType() const;

        /**
         * @brief Write the object to the disk.
         * @param file the configuration filename
         * @return true if successful
         */
        bool writeObject(std::string file);

        /**
         * @brief Write the object to the disk.
         * @param file the configuration filename
         * @param fileData the data filename
         * @return true if successful
         */
        bool writeObject(std::string file, std::string fileData);

        /**
         * @brief Read the object from the disk.
         * @param file the configuration filename, which also stores a reference to the appropriate
         * data filename.
         * @param training indicates if file is loaded during training - in that case a possible stored data file is not loaded
         * @return  true if successful
         */
        bool readObject(std::string file, bool training = false);

        Json::Value configToJson() const;
        bool configFromJson(const Json::Value&);

        void saveData(boost::archive::binary_oarchive &oa) const;
        bool loadData(boost::archive::binary_iarchive &ia);

        void setOutputFilename(std::string file);

    protected:
        template <typename T>
        void addParameter(T& param, std::string name, T defaultValue) {
            m_params.push_back(new JSONParameter<T>(param, name, defaultValue));
        }

        virtual Json::Value iChildConfigsToJson() const;
        virtual bool iChildConfigsFromJson(const Json::Value&);

        virtual void iSaveData(boost::archive::binary_oarchive &oa) const;
        virtual bool iLoadData(boost::archive::binary_iarchive &ia);

        virtual void iPostInitConfig();

        bool m_use_svm;
        bool m_svm_1_vs_all_train;
        std::string m_svm_path;
        std::string m_output_file_name;
        std::string m_input_config_file;

    private:
        bool write(const Json::Value&, const std::string&, bool) const;
        Json::Value read(const std::string&);
        std::string toJsonString(const Json::Value&, bool) const;
        Json::Value fromJsonString(const std::string&);

        std::vector<JSONParameterBase*> m_params;
    };
}

#endif // ISM3D_JSONOBJECT_H
