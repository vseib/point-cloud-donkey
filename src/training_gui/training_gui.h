#ifndef TRAININGGUI_H
#define TRAININGGUI_H


#include <boost/signals2.hpp>
// Qt
#include <QWidget>
#include <QTimer>
#include <QVTKWidget.h>
#include <QPushButton>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QCheckBox>
#include <QLineEdit>
#include <QLabel>

// TODO VS temporarily (?) disabling ROS
//// ROS
//#include "ros/ros.h"
//#include "sensor_msgs/PointCloud2.h"
//#include "message_filters/subscriber.h"
//#include "message_filters/sync_policies/approximate_time.h"

// PCL
#ifndef Q_MOC_RUN
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/vtk.h>
#endif // Q_MOC_RUN

#include "../implicit_shape_model/utils/utils.h"

namespace ism3d
{
    class ImplicitShapeModel;
    class VotingMaximum;
    class Codebook;
    class ActivationDistribution;
    class ISMFeature;
}

class RenderView;

class TrainingGUI
        : public QWidget
{
    Q_OBJECT

public:
    TrainingGUI(QWidget* parent = 0);
    ~TrainingGUI();

private slots:
    void spinOnce();
    void pauseResume();
    void reset();
    void addModel();
    void loadScene();
    void load();
    void save();
    void trainModel();
    void trainingFinished(bool);
    void detectISM();
    void detectionFinished(bool);
    void clearISM();

protected:
    // TODO VS temporarily (?) disabling ROS
    //void cbPoints(const sensor_msgs::PointCloud2::ConstPtr& points);

private:
    QGroupBox* createNavigatorApplication();
    QGroupBox* createNavigatorGeneral();
    QGroupBox* createNavigatorISM();

    void draw();
    void drawCloud();
    void updateCameras();

    void signalPointCloud(pcl::PointCloud<ism3d::PointT>::ConstPtr);
    void signalBoundingBox(const ism3d::Utils::BoundingBox&);
    void signalNormals(pcl::PointCloud<ism3d::PointT>::ConstPtr, pcl::PointCloud<pcl::Normal>::ConstPtr);
    void signalFeatures(pcl::PointCloud<ism3d::ISMFeature>::ConstPtr);
    void signalCodebook(const ism3d::Codebook&);
    void signalMaxima(std::vector<ism3d::VotingMaximum>);

    void addBoundingBox(const ism3d::Utils::BoundingBox&);

    // TODO VS temporarily (?) disabling ROS
//    ros::NodeHandle m_node;
//    ros::Subscriber m_subPoints;
    ism3d::ImplicitShapeModel* m_ism;

    bool m_isLoaded;
    bool m_updateCloud;
    bool m_isDetecting;

    // Qt items
    QTimer* m_spinTimer;
    QPushButton* m_btPauseResume;
    QGroupBox* m_implicitShapeModel;
    QCheckBox* m_chkShowAllVotes;
    QCheckBox* m_chkShowBbAndCenters;
    QCheckBox* m_chkShowVotes;
    QCheckBox* m_chkShowFeatures;
    QCheckBox* m_chkInvertNormals;
    QCheckBox* m_chkShowNormals;
    QCheckBox* m_chkShowOnlyBestMaxPerClass;

    QLabel* m_minMaxVotesLabel;
    QLineEdit* m_minMaxVotesToShowLine;

    QLabel* m_onlyShowClassLabel;
    QLineEdit* m_onlyShowClassLine;

    QGroupBox* m_navApplication;
    QGroupBox* m_navGeneral;
    QGroupBox* m_navISM;

    RenderView* m_renderView;

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr m_detectCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_displayCloud;

    vtkSmartPointer<vtkPolyData> m_points;
    vtkSmartPointer<vtkActor> m_pointsActor;

    std::vector<bool> m_maxima_classes; // used to display only the best maximum per class

    // temporary data for visualization
    ism3d::Utils::BoundingBox m_boundingBox;
};

#endif // TRAININGGUI_H
