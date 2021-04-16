#ifndef MODELGUI_H
#define MODELGUI_H

// NOTE temporarily disabling ROS
// ROS
//#include "ros/ros.h"
//#include "sensor_msgs/PointCloud2.h"
//#include "message_filters/subscriber.h"
//#include "message_filters/sync_policies/approximate_time.h"

// Qt
#include <QWidget>
#include <QTimer>
#include <QVTKWidget.h>
#include <QPushButton>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QCheckBox>
#include <QSlider>
#include <QListWidget>

// PCL
#ifndef Q_MOC_RUN
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/vtk.h>
#include <pcl/filters/voxel_grid.h>
#endif // Q_MOC_RUN

class RenderView;

class ModelGUI
        : public QWidget
{
    Q_OBJECT

public:
    ModelGUI(QWidget* parent = 0);
    ~ModelGUI();

private slots:
    // void spinOnce(); // NOTE temporarily disabling ROS
    // void pauseResume(); // NOTE temporarily disabling ROS
    void reset();
    void segment();
    void importCloud();
    void exportCloud();
    void enableSegmentation(bool);
    void indexChanged(int);
    void merge();
    void smooth();
    void filter();
    void downsample();
    void startAnnotation();
    void setLabel();
    void labelSelected(int);
    void removeGroundPlane();

    void moveXY(float, float);
    void moveYZ(float, float);
    void moveXZ(float, float);
    void scaleXY(float, float);
    void scaleYZ(float, float);
    void scaleXZ(float, float);
    void rotateX(float, Eigen::Vector3f);
    void rotateY(float, Eigen::Vector3f);
    void rotateZ(float, Eigen::Vector3f);

protected:
    // NOTE temporarily disabling ROS
    //void cbPoints(const sensor_msgs::PointCloud2::ConstPtr& points);

private:
    QGroupBox* createNavigatorApplication();
    QGroupBox* createNavigatorGeneral();
    QGroupBox* createNavigatorAnnotation();
    QGroupBox* createNavigatorPlaneSegmentation();

    void draw();
    void drawCloud();
    void updateBox();
    void drawCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr);
    void activateCurCloud(bool);
    void updateCameras();
    void resetTransform();

    // NOTE temporarily disabling ROS
    // ros::NodeHandle m_node;
    // ros::Subscriber m_subPoints;

    // indicates that a cloud was loaded manually, sets drawing the cloud at m_pointClouds[0]
    // once a cloud is loaded, sensor clouds will not be drawn anymore until button Reset is pressed
    bool m_isLoaded;
    // set to true each time ROS callback is triggered, will draw m_cloud once per callback
    bool m_drawSensorCloud;
    // whether or not the currently stored cloud in m_cloud will be updated on ROS callback,
    // only valid if m_isLoaded == false
    bool m_updateSensorCloud;

    bool m_annotationMode;
    std::vector<int> m_annotationIDs;
    std::vector<std::string> m_annotationNames;
    std::vector<Eigen::Vector3d> m_annotationColors;
    int m_currentLabelIndex;
    // color for segmentation box and points
    std::vector<int> m_segmentColor;

    // Qt items
    QTimer* m_spinTimer;
    QPushButton* m_btPauseResume;
    QPushButton* m_btSegment;
    QPushButton* m_StartStopAnnotation;
    QPushButton* m_btMerge;
    QCheckBox* m_chkEnableSegmentation;
    QTimer m_drawTimer;
    QListWidget* m_list;
    RenderView* m_renderView;

    // holds the transforms to each cloud in m_pointClouds,
    // transforms are applied to corresponding cloud before drawing
    std::vector<Eigen::Matrix4f> m_transforms;
    // data used to draw when m_isLoaded == true
    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr> m_pointClouds;
    // data used to draw when m_drawSensorCloud == true
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr m_cloud;
    // actually drawn cloud, content switched to m_cloud or m_pointClouds
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr m_displayCloud;

    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr m_annotated_cloud;

    vtkSmartPointer<vtkPolyData> m_points;
    vtkSmartPointer<vtkActor> m_pointsActor;
    vtkSmartPointer<vtkPolyData> m_curPoints;
    vtkSmartPointer<vtkTransform> m_curPointsTransform;
    vtkSmartPointer<vtkActor> m_curPointsActor;
    vtkSmartPointer<vtkCubeSource> m_cube;
    vtkSmartPointer<vtkActor> m_cubeActor;
    vtkSmartPointer<vtkAxesActor> m_cubeAxes;
    vtkSmartPointer<vtkTransform> m_cubeTrans;
    vtkSmartPointer<vtkTransform> m_cubeAxesTrans;
};

#endif // MODELGUI_H
