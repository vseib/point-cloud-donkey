/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "json_object.h"
#include "utils.h"
#include <fstream>


namespace ism3d
{
    JSONObject::JSONObject()
    {
    }

    JSONObject::~JSONObject()
    {
        for (int i = 0; i < (int)m_params.size(); i++)
            delete m_params[i];
        m_params.clear();
    }

    std::string JSONObject::getType() const
    {
        return "";
    }

    void JSONObject::setOutputFilename(std::string file)
    {
        // remove the file extension "ism" from output name
        int pos = file.find_last_of('.');
        m_output_file_name = file.substr(0, pos);
    }

    bool JSONObject::writeObject(std::string file)
    {
        boost::filesystem::path filePath(file);
        if (!filePath.has_extension())
            file += ".ism";

        return writeObject(file, file + "d");
    }

    bool JSONObject::writeObject(std::string file, std::string fileData)
    {
        LOG_INFO("writing object to files: " << file << ", " << fileData);

        // get relative path from ism file to data file
        int pos = fileData.find_last_of('/');
        std::string fileDataNoPath;
        if(pos != fileData.npos)
            fileDataNoPath = fileData.substr(pos+1);
        else
            fileDataNoPath = fileData;

        // create JSON config object
        Json::Value configJson(Json::objectValue);
        configJson["ObjectConfig"] = configToJson();
        configJson["ObjectData"] = fileDataNoPath; //relative.string();
        if(m_use_svm)
        {
            // this is to be easily accessible in the config file
            if(m_svm_1_vs_all_train)
                configJson["ObjectDataSVM"] = Json::Value(m_output_file_name+".svm.tar.gz");
            else
                configJson["ObjectDataSVM"] = Json::Value(m_output_file_name+".svm");
            // NOTE: in case this name is changed in config, it needs to be forwarded to data while loading
        }

        // create boost data object
        std::ofstream ofs(fileData);
        boost::archive::binary_oarchive oa(ofs);
        saveData(oa);
        ofs.close();

        if (!write(configJson, file, true))
            return false;

        LOG_INFO("writing successful");

        return true;
    }

    bool JSONObject::readObject(std::string file, bool training)
    {
        LOG_INFO("reading object configuration from file: " << file);

        // read configuration
        Json::Value configJson = read(file);
        if (configJson.isNull())
        {
            LOG_ERROR("Json Config is NULL!");
            return false;
        }

        // create object hierarchy
        if (configJson.isMember("ObjectConfig"))
        {
            if (!configFromJson(configJson["ObjectConfig"]))
            {
                LOG_ERROR("Json ObjectConfig is NULL, could not create object hierarchy!");
                return false;
            }
        }

        // read SVM path
        Json::Value svm_path;
        if (configJson.isMember("ObjectDataSVM") && !training)
        {
            // make SVM path available in voting.cpp
            svm_path = configJson["ObjectDataSVM"];
            m_svm_path = svm_path.asString();
        }
        else
        {
            m_svm_path = "";
        }

        // read data
        if (configJson.isMember("ObjectData") && !training)
        {
            // get the filename
            Json::Value jsonFileData = configJson["ObjectData"];
            if (!jsonFileData.isString())
                return false;

            // read the file
            std::string fileData = jsonFileData.asString();

            boost::filesystem::path filePath(file);
            filePath = filePath.parent_path();
            filePath /= fileData;
            fileData = filePath.string();

            if (fileData.empty())
            {
                LOG_ERROR("Json ObjectData is empty!");
                return false;
            }

            // read boost data object
            std::ifstream ifs(fileData);
            if(ifs)
            {
                boost::archive::binary_iarchive ia(ifs);
                loadData(ia);
                ifs.close();
            }
            else
            {
                LOG_ERROR("Error opening file: " << fileData);
                return false;
            }
        }
        else if(!training)
        {
            LOG_ERROR("Config file " << file << " has only parameters, but no trained object data!");
            return false;
        }

        LOG_INFO("reading successful");

        return true;
    }

    Json::Value JSONObject::configToJson() const
    {
        Json::Value object(Json::objectValue);

        // get config
        Json::Value parameters(Json::nullValue);
        for (int i = 0; i < (int)m_params.size(); i++) {
            if (parameters.isNull())
                parameters = Json::Value(Json::objectValue);

            const JSONParameterBase* param = m_params[i];
            param->toJson(parameters);
        }

        // append object parameters
        if (!parameters.isNull())
            object["Parameters"] = parameters;

        // append child configs
        Json::Value children = iChildConfigsToJson();
        if (!children.isNull())
            object["Children"] = children;

        // add a type attribute
        std::string type = getType();
        if (type.length() > 0) {
            object["Type"] = Json::Value(type);
        }

        return object;
    }

    bool JSONObject::configFromJson(const Json::Value& object)
    {
        // read parameters
        Json::Value parameters = object["Parameters"];
        if (parameters.isObject()) {
            for (int i = 0; i < (int)m_params.size(); i++)
            {
                JSONParameterBase* param = m_params[i];
                if (!param->fromJson(parameters))
                    return false;
            }
        }

        iPostInitConfig();

        // read children
        Json::Value children = object["Children"];
        if (children.isObject() && !iChildConfigsFromJson(children))
            return false;

        return true;
    }

    void JSONObject::saveData(boost::archive::binary_oarchive &oa) const
    {
        iSaveData(oa);
    }

    bool JSONObject::loadData(boost::archive::binary_iarchive &ia)
    {
        return iLoadData(ia);
    }

    Json::Value JSONObject::iChildConfigsToJson() const
    {
        Json::Value object(Json::nullValue);
        return object;
    }

    bool JSONObject::iChildConfigsFromJson(const Json::Value&)
    {
        return true;
    }

    void JSONObject::iSaveData(boost::archive::binary_oarchive &oa) const
    {
        // do nothing - all functionality (if needed) is implemented in derived classes
    }

    bool JSONObject::iLoadData(boost::archive::binary_iarchive &ia)
    {
        // do nothing - all functionality (if needed) is implemented in derived classes
        return true;
    }



    void JSONObject::iPostInitConfig()
    {
    }

    bool JSONObject::write(const Json::Value& json, const std::string& filename, bool styled) const
    {
        std::ofstream file;
        file.open(filename.c_str(), std::ios_base::out);

        if (!file.is_open())
            return false;

        // get json string
        std::string jsonString = toJsonString(json, styled);
        if (jsonString.length() == 0)
            return false;

        // write to file
        file << jsonString;
        file.close();

        return true;
    }

    Json::Value JSONObject::read(const std::string& filename)
    {
        std::ifstream file;
        file.open(filename.c_str(), std::ios_base::in);

        if (!file.is_open())
            return Json::Value(Json::nullValue);

        std::string jsonString;

        // get file string length
        file.seekg(0, std::ios::end);
        jsonString.reserve(file.tellg());
        file.seekg(0, std::ios::beg);

        // retrieve string from file
        jsonString.assign((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());

        file.close();

        // parse json
        return fromJsonString(jsonString);
    }

    std::string JSONObject::toJsonString(const Json::Value& json, bool styled) const
    {
        // make sure the decimal mark is a point
        setlocale(LC_NUMERIC, "C");

        // get json string
        if (json.isNull())
            return "";

        if (styled) {
            Json::StyledWriter writer;
            return writer.write(json);
        } else {
            Json::FastWriter writer;
            return writer.write(json);
        }
    }

    Json::Value JSONObject::fromJsonString(const std::string& jsonString)
    {
        // make sure the decimal mark is a point
        setlocale(LC_NUMERIC, "C");

        // try to parse json
        Json::Value root;
        Json::Reader reader;

        if (!reader.parse(jsonString, root, false))
        {
            throw JSONException("Could not parse json:\n" +
                                reader.getFormattedErrorMessages());
        }

        return root;
    }
}
