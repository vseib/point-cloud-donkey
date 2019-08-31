/*
 * BSD 3-Clause License
 *
 * Full text: https://opensource.org/licenses/BSD-3-Clause
 *
 * Copyright (c) 2018, Viktor Seib
 * All rights reserved.
 *
 */

#include "codeword.h"
#include "codeword_factory.h"

namespace ism3d
{
    int Codeword::m_maxId = 0;

    Codeword::Codeword(const std::vector<float>& data, int numFeatures, float weight)
    {
        m_data = std::vector<float>(data);
        m_numFeatures = numFeatures;
        m_weight = weight;

        m_id = m_maxId;
        m_maxId++;
    }

    Codeword::Codeword()
    {
        m_id = m_maxId;
        m_numFeatures = 0;
        m_weight = 1.0f;
        m_maxId++;
    }

    Codeword::~Codeword()
    {
    }

    void Codeword::setData(const std::vector<float>& data, int numFeatures, float weight)
    {
        m_data = std::vector<float>(data);
        m_numFeatures = numFeatures;
        m_weight = weight;
    }

    const std::vector<float>& Codeword::getData() const
    {
        return m_data;
    }

    int Codeword::getId() const
    {
        return m_id;
    }

    int Codeword::getNumFeatures() const
    {
        return m_numFeatures;
    }

    float Codeword::getWeight() const
    {
        return m_weight;
    }

    void Codeword::addFeature(unsigned classId)
    {
        m_featureClasses.push_back(classId);
    }

    const std::vector<unsigned>& Codeword::getFeatureClasses() const
    {
        return m_featureClasses;
    }

    void Codeword::iSaveData(boost::archive::binary_oarchive &oa) const
    {
        oa << m_id;
        oa << m_numFeatures;
        oa << m_weight;

        m_maxId = m_id + 1; // TODO VS: remove this

        oa << m_featureClasses;
        oa << m_data;
    }

    bool Codeword::iLoadData(boost::archive::binary_iarchive &ia)
    {
        ia >> m_id;
        ia >> m_numFeatures;
        ia >> m_weight;
        ia >> m_featureClasses;
        ia >> m_data;

        return true;
    }

    Json::Value Codeword::iDataToJson() const
    {
        Json::Value object(Json::objectValue);
        object["ID"] = m_id;
        object["NumFeatures"] = m_numFeatures;
        object["Weight"] = m_weight;

        m_maxId = m_id + 1;

        Json::Value classes(Json::arrayValue);
        classes.resize(m_featureClasses.size());
        for (int i = 0; i < (int)m_featureClasses.size(); i++)
            classes[i] = m_featureClasses[i];
        object["Classes"] = classes;

        Json::Value dataArray(Json::arrayValue);
        dataArray.resize(m_data.size());
        for (int i = 0; i < (int)m_data.size(); i++)
            dataArray[i] = m_data[i];
        object["CodewordArray"] = dataArray;

        return object;
    }

    bool Codeword::iDataFromJson(const Json::Value& object)
    {
        const Json::Value *id = &(object["ID"]);
        const Json::Value *numFeatures = &(object["NumFeatures"]);
        const Json::Value *classes = &(object["Classes"]);
        const Json::Value *dataArray = &(object["CodewordArray"]);
        const Json::Value *weight = &(object["Weight"]);

        if (id->isNull() || !id->isInt() ||
                numFeatures->isNull() || !numFeatures->isInt() ||
                classes->isNull() || !classes->isArray() ||
                dataArray->isNull() || !dataArray->isArray())
            return false;

        if(weight->isNull() || !weight->isDouble())
        {
            m_weight = 1.0f;
        }
        else
        {
            m_weight = weight->asFloat();
        }

        m_id = id->asInt();
        m_numFeatures = numFeatures->asInt();
        m_featureClasses.resize(classes->size());
        m_data.resize(dataArray->size());

        for (int i = 0; i < (int)m_featureClasses.size(); i++) {
            Json::Value arrayVal = (*classes)[i];

            if (arrayVal.isNull() || !arrayVal.isInt())
                return false;

            m_featureClasses[i] = arrayVal.asInt();
        }

        for (int i = 0; i < (int)m_data.size(); i++) {
            Json::Value arrayVal = (*dataArray)[i];

            if (arrayVal.isNull() || !arrayVal.isNumeric())
                return false;

            m_data[i] = arrayVal.asFloat();
        }

        return true;
    }
}
