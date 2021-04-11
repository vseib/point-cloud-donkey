#ifndef GROUNDTRUTHGUI_H
#define GROUNDTRUTHGUI_H

// Qt
#include <QWidget>
#include <QTimer>
#include <QVTKWidget.h>
#include <QPushButton>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QCheckBox>
#include <QSlider>
#include <QLineEdit>

// PCL
#ifndef Q_MOC_RUN
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/vtk.h>
#endif // Q_MOC_RUN

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointXYZRGBNormal PointNormalT;

class RenderView;

class GroundTruthGUI
        : public QWidget
{
    Q_OBJECT

public:
    GroundTruthGUI(QWidget* parent = 0);
    ~GroundTruthGUI();

private slots:
    void reset();
    void loadScene();
    void addModel();
    void removeModel();
    void exportGroundTruth();
    void computeNormals();
    void moveXY(float, float);
    void moveYZ(float, float);
    void moveXZ(float, float);
    void rotateX(float, Eigen::Vector3f);
    void rotateY(float, Eigen::Vector3f);
    void rotateZ(float, Eigen::Vector3f);

private:
    QGroupBox* createNavigatorApplication();
    QGroupBox* createNavigatorGeneral();

    QLineEdit* m_normalRadiusLineEdit;
    QLineEdit* m_normalMethodLineEdit;

    void drawCloud();
    void updateCameras();

    bool m_isLoaded;

    RenderView* m_renderView;

    pcl::PointCloud<PointT>::Ptr m_cloud;
    pcl::PointCloud<PointT>::Ptr m_displayCloud;
    std::vector<pcl::PointCloud<PointT>::ConstPtr> m_models;
    std::vector<vtkSmartPointer<vtkActor>> m_modelActors;
    std::vector<vtkSmartPointer<vtkTransform>> m_transforms;
    std::vector<std::string> m_modelFiles;
    std::string m_sceneFile;

    vtkSmartPointer<vtkPolyData> m_points;
    vtkSmartPointer<vtkActor> m_pointsActor;
};

#endif // GROUNDTRUTHGUI_H
