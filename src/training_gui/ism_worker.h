#ifndef ISMWORKER_H
#define ISMWORKER_H

#include <QObject>

#ifndef Q_MOC_RUN
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#endif // Q_MOC_RUN

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointXYZRGBNormal PointNormalT;

namespace ism3d
{
    class ImplicitShapeModel;
}

class ISMWorker
        : public QObject
{
    Q_OBJECT

public:
    ISMWorker();
    ~ISMWorker();

    void setImplicitShapeModel(ism3d::ImplicitShapeModel*);
    void setDetectionPointCloud(pcl::PointCloud<PointNormalT>::ConstPtr);

public slots:
    void train();
    void detect();

signals:
    void trainingFinished(bool);
    void detectionFinished(bool);
    void finished();

private:
    ism3d::ImplicitShapeModel* m_ism;
    pcl::PointCloud<PointNormalT>::ConstPtr m_cloud;
};

#endif // ISMWORKER_H
