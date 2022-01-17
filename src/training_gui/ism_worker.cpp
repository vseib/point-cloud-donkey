#include "../implicit_shape_model/implicit_shape_model.h"
#include "ism_worker.h"

ISMWorker::ISMWorker()
    : m_ism(0), m_cloud()
{
}

ISMWorker::~ISMWorker()
{
}

void ISMWorker::setImplicitShapeModel(ism3d::ImplicitShapeModel* ism)
{
    m_ism = ism;
}

void ISMWorker::setDetectionPointCloud(pcl::PointCloud<PointNormalT>::ConstPtr cloud)
{
    m_cloud = cloud;
}

void ISMWorker::train()
{
    if (!m_ism)
        emit trainingFinished(false);

    m_ism->train();

    emit trainingFinished(true);
    emit finished();
}

void ISMWorker::detect()
{
    if (!m_ism || !m_cloud.get())
        emit detectionFinished(false);

    m_ism->detect(m_cloud, false);

    emit detectionFinished(true);
    emit finished();
}
