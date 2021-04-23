#include "model_gui.h"

#include "../vtk_utils/render_view.h"

// Qt
#include <QLayout>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QSlider>
#include <QGroupBox>
#include <QResizeEvent>
#include <QFileDialog>
#include <QMessageBox>
#include <QAction>
#include <QComboBox>

// PCL
#include <pcl/point_types.h>
// NOTE temporarily disabling ROS
//#include <pcl/ros/conversions.h>
//#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/impl/mls.hpp>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/filters/statistical_outlier_removal.h>

// VTK
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkBox.h>

#include <jsoncpp/json/json.h>
#include <jsoncpp/json/writer.h>

#include <Eigen/Geometry>
#include <Eigen/Core>

// Plane Segmentation
// NOTE temporarily disabling PlaneDetection
//#include "PlaneDetection.h" // see: ~/git/homer/experimental/Libraries/PlaneDetection/src/PlaneDetection.h


float deg2rad(float deg)
{
    return deg * (M_PI / 180.0f);
}

ModelGUI::ModelGUI(QWidget* parent)
    : QWidget(parent),
      m_isLoaded(false),
      m_updateSensorCloud(true), // NOTE: with ROS disabled this is unused
      m_drawSensorCloud(false), // NOTE: with ROS disabled will be always false
      m_annotationMode(false),
      m_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>()),
      m_displayCloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>()),
      m_currentLabelIndex(0)
{
    // NOTE temporarily disabling ROS - don't need the timer without ROS
    // init qt related
//    m_spinTimer = new QTimer(this);
//    m_spinTimer->setInterval(30);
//    connect(m_spinTimer, SIGNAL(timeout()), this, SLOT(spinOnce()));
//    m_spinTimer->start();

    // subscribe to ros topic to acquire depth images
    // TODO handle both topics: registered and normal points, registered should be painted in RGB
    // NOTE temporarily disabling ROS
    //m_subPoints = m_node.subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/points", 10, &ModelGUI::cbPoints, this);

    setWindowTitle("ISM3D - Model GUI");

    // init annotation variables, NOTE: all lists must have same length
    // TODO VS: init from config files to avoid recompiling if other labels are used
    m_annotationIDs.push_back(2);
    m_annotationIDs.push_back(6);
    m_annotationIDs.push_back(5);
    m_annotationIDs.push_back(4);
    m_annotationIDs.push_back(3);
    m_annotationNames.push_back("nothing");
    m_annotationNames.push_back("sitting plane");
    m_annotationNames.push_back("backrest");
    m_annotationNames.push_back("armrest");
    m_annotationNames.push_back("headrest");
    m_annotationColors.push_back(Eigen::Vector3d(255, 69, 0));
    m_annotationColors.push_back(Eigen::Vector3d(0, 0, 179));
    m_annotationColors.push_back(Eigen::Vector3d(0, 179, 128));
    m_annotationColors.push_back(Eigen::Vector3d(138, 0, 0));
    m_annotationColors.push_back(Eigen::Vector3d(138, 138, 230));

    // create vtk views
    m_renderView = new RenderView(this);
    m_renderView->getRendererFront()->SetBackground(255,255,255);
    m_renderView->getRendererTop()->SetBackground(255,255,255);
    m_renderView->getRendererSide()->SetBackground(255,255,255);
    m_renderView->getRendererScene()->SetBackground(255,255,255);

    connect(m_renderView, SIGNAL(moveXY(float,float)), SLOT(moveXY(float,float)));
    connect(m_renderView, SIGNAL(moveYZ(float,float)), SLOT(moveYZ(float,float)));
    connect(m_renderView, SIGNAL(moveXZ(float,float)), SLOT(moveXZ(float,float)));
    connect(m_renderView, SIGNAL(scaleXY(float,float)), SLOT(scaleXY(float,float)));
    connect(m_renderView, SIGNAL(scaleYZ(float,float)), SLOT(scaleYZ(float,float)));
    connect(m_renderView, SIGNAL(scaleXZ(float,float)), SLOT(scaleXZ(float,float)));
    connect(m_renderView, SIGNAL(rotateX(float, Eigen::Vector3f)), SLOT(rotateX(float, Eigen::Vector3f)));
    connect(m_renderView, SIGNAL(rotateY(float, Eigen::Vector3f)), SLOT(rotateY(float, Eigen::Vector3f)));
    connect(m_renderView, SIGNAL(rotateZ(float, Eigen::Vector3f)), SLOT(rotateZ(float, Eigen::Vector3f)));

    m_segmentColor = {0, 240, 160};

    // add points
    m_points = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPolyDataMapper> pointsMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    pointsMapper->SetInputData(m_points);
    m_pointsActor = vtkSmartPointer<vtkActor>::New();
    m_pointsActor->SetMapper(pointsMapper);

    // add current points (for the current transformed point cloud)
    m_curPoints = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPolyDataMapper> curPointsMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    curPointsMapper->SetInputData(m_curPoints);
    m_curPointsActor = vtkSmartPointer<vtkActor>::New();
    m_curPointsActor->SetMapper(curPointsMapper);
    m_curPointsTransform = vtkSmartPointer<vtkTransform>::New();
    m_curPointsActor->SetUserTransform(m_curPointsTransform);

    // add segmentation transformations
    m_cubeTrans = vtkSmartPointer<vtkTransform>::New();
    m_cubeAxesTrans = vtkSmartPointer<vtkTransform>::New();

    // add cube
    m_cube = vtkSmartPointer<vtkCubeSource>::New();
    vtkSmartPointer<vtkPolyDataMapper> cubeMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    cubeMapper->SetInputConnection(m_cube->GetOutputPort());
    m_cubeActor = vtkSmartPointer<vtkActor>::New();
    m_cubeActor->SetMapper(cubeMapper);
    m_cubeActor->GetProperty()->SetRepresentationToWireframe();
    m_cubeActor->GetProperty()->SetLighting(false);
    m_cubeActor->SetUserTransform(m_cubeTrans);
    // NOTE: this method expects the color to be between 0 and 1
    m_cubeActor->GetProperty()->SetColor(m_segmentColor[0]/255.0,m_segmentColor[1]/255.0,m_segmentColor[2]/255.0);
    m_cubeActor->GetProperty()->SetLineWidth(2);

    // add cube axes
    m_cubeAxes = vtkSmartPointer<vtkAxesActor>::New();
    m_cubeAxes->AxisLabelsOff();
    m_cubeAxes->SetConeRadius(0.2);
    m_cubeAxes->SetUserTransform(m_cubeAxesTrans);

    m_cubeActor->SetVisibility(false);
    m_cubeAxes->SetVisibility(false);

    //m_renderView->addActorToScene(m_pointsActor);
    m_renderView->addActorToAll(m_pointsActor);
    m_renderView->addActorToAll(m_curPointsActor);
    m_renderView->addActorToAll(m_cubeActor);
    m_renderView->addActorToAll(m_cubeAxes);

    // create navigator panes
    QVBoxLayout* navigatorLayout = new QVBoxLayout();
    navigatorLayout->addWidget(createNavigatorApplication());
    navigatorLayout->addWidget(createNavigatorGeneral());
    navigatorLayout->addWidget(createNavigatorAnnotation());
    // NOTE temporarily disabling PlaneDetection
    //navigatorLayout->addWidget(createNavigatorPlaneSegmentation());
    navigatorLayout->addItem(new QSpacerItem(150, 0, QSizePolicy::Minimum, QSizePolicy::Expanding));

    // put it all together
    QGridLayout* layout = new QGridLayout(this);    // NOTE: produces a warning
    layout->addWidget(m_renderView, 0,  0);
    layout->addLayout(navigatorLayout, 0, 1);
    this->setLayout(layout);

    resetTransform();
    resize(1024, 768);
}

ModelGUI::~ModelGUI()
{
}

QGroupBox* ModelGUI::createNavigatorApplication()
{
    QPushButton* btClose = new QPushButton(this);
    btClose->setText("Close");
    connect(btClose, SIGNAL(clicked()), this, SLOT(close()));

    QVBoxLayout* layout = new QVBoxLayout();
    layout->addWidget(btClose);

    QGroupBox* groupBox = new QGroupBox(this);
    groupBox->setTitle("Application");
    groupBox->setLayout(layout);
    return groupBox;
}

QGroupBox* ModelGUI::createNavigatorGeneral()
{
    // NOTE temporarily disabling ROS
//    m_btPauseResume = new QPushButton(this);
//    connect(m_btPauseResume, SIGNAL(clicked()), SLOT(pauseResume()));
//    m_btPauseResume->setText("Pause");

    QPushButton* btReset = new QPushButton(this);
    connect(btReset, SIGNAL(clicked()), this, SLOT(reset()));
    btReset->setText("Reset");

    QPushButton* btImportCloud = new QPushButton(this);
    connect(btImportCloud, SIGNAL(clicked()), this, SLOT(importCloud()));
    btImportCloud->setText("Import Point-Cloud");

    QPushButton* btExportCloud = new QPushButton(this);
    connect(btExportCloud, SIGNAL(clicked()), this, SLOT(exportCloud()));
    btExportCloud->setText("Export Point-Cloud");

    m_chkEnableSegmentation = new QCheckBox(this);
    connect(m_chkEnableSegmentation, SIGNAL(clicked(bool)), SLOT(enableSegmentation(bool)));
    m_chkEnableSegmentation->setText("Enable Segmentation");
    m_chkEnableSegmentation->setChecked(false);

    m_btSegment = new QPushButton(this);
    connect(m_btSegment, SIGNAL(clicked()), SLOT(segment()));
    m_btSegment->setText("Segment");
    m_btSegment->setEnabled(false);

    QLabel* label = new QLabel(this);
    label->setText("Imported Point-Clouds:");

    m_list = new QListWidget(this);
    m_list->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    m_list->setFixedHeight(100);
    connect(m_list, SIGNAL(currentRowChanged(int)), SLOT(indexChanged(int)));

    m_btMerge = new QPushButton(this);
    connect(m_btMerge, SIGNAL(clicked()), SLOT(merge()));
    m_btMerge->setText("Merge");
    m_btMerge->setEnabled(false);

    QPushButton* btFilter = new QPushButton(this);
    connect(btFilter, SIGNAL(clicked()), SLOT(filter()));
    btFilter->setText("Filter");

    QPushButton* btSmooth = new QPushButton(this);
    connect(btSmooth, SIGNAL(clicked()), SLOT(smooth()));
    btSmooth->setText("Smooth");

    QPushButton* btDownsample = new QPushButton(this);
    connect(btDownsample, SIGNAL(clicked()), SLOT(downsample()));
    btDownsample->setText("Grid Sample");

    QVBoxLayout* layout = new QVBoxLayout();
//    layout->addWidget(m_btPauseResume); // NOTE temporarily disabling ROS
    layout->addWidget(btReset);
    layout->addWidget(btImportCloud);
    layout->addWidget(btExportCloud);
    layout->addWidget(m_chkEnableSegmentation);
    layout->addWidget(m_btSegment);
    layout->addWidget(label);
    layout->addWidget(m_list);
    layout->addWidget(m_btMerge);
    layout->addWidget(btFilter);
    layout->addWidget(btSmooth);
    layout->addWidget(btDownsample);

    QGroupBox* groupBox = new QGroupBox(this);
    groupBox->setTitle("General");
    groupBox->setLayout(layout);
    return groupBox;
}

QGroupBox* ModelGUI::createNavigatorAnnotation()
{
    m_StartStopAnnotation = new QPushButton(this);
    m_StartStopAnnotation->setText("Start Annotation");
    connect(m_StartStopAnnotation, SIGNAL(clicked()), this, SLOT(startAnnotation()));

    QComboBox* annotationBox = new QComboBox(this);
    for(unsigned i = 0; i < m_annotationNames.size(); i++)
    {
        annotationBox->addItem(QString::fromStdString(m_annotationNames.at(i)), QVariant());
    }
    connect(annotationBox, SIGNAL(currentIndexChanged(int)), this, SLOT(labelSelected(int)));

    QPushButton* btSetLabel = new QPushButton(this);
    btSetLabel->setText("Set Selected Label");
    connect(btSetLabel, SIGNAL(clicked()), this, SLOT(setLabel()));

    QVBoxLayout* layout = new QVBoxLayout();
    layout->addWidget(m_StartStopAnnotation);
    layout->addWidget(annotationBox);
    layout->addWidget(btSetLabel);

    QGroupBox* groupBox = new QGroupBox(this);
    groupBox->setTitle("Annotation");
    groupBox->setLayout(layout);
    return groupBox;
}


QGroupBox* ModelGUI::createNavigatorPlaneSegmentation()
{
    QPushButton* btRemoveGroundPlane = new QPushButton(this);
    btRemoveGroundPlane->setText("Remove Ground Plane");
    connect(btRemoveGroundPlane, SIGNAL(clicked()), this, SLOT(removeGroundPlane()));

    QVBoxLayout* layout = new QVBoxLayout();
    layout->addWidget(btRemoveGroundPlane);

    QGroupBox* groupBox = new QGroupBox(this);
    groupBox->setTitle("Plane Segmentation");
    groupBox->setLayout(layout);
    return groupBox;
}

// NOTE temporarily disabling ROS
//void ModelGUI::spinOnce()
//{
//    if (!ros::ok())
//        this->close();

//    ros::spinOnce();

//    if (m_isLoaded)
//        drawCloud();

//    m_renderView->update();
//}

// NOTE temporarily disabling ROS
//void ModelGUI::cbPoints(const sensor_msgs::PointCloud2::ConstPtr& pointCloud)
//{
//    if (m_isLoaded)
//        return;

//    if (m_updateSensorCloud) {
//        pcl::PointCloud<pcl::PointXYZRGB> cloud;
//        pcl::fromROSMsg(*pointCloud, cloud);
//        pcl::copyPointCloud(cloud, *m_cloud);
//        drawCloud();
//        m_drawSensorCloud = true;
//    }
//}

void ModelGUI::drawCloud()
{
    // downsample
    const int downsampling = 1;
    m_displayCloud->clear();
    if (m_drawSensorCloud) // NOTE: never true with ROS disabled
    {
        if (m_cloud->isOrganized()) {
            for (int i = 0; i < (int)m_cloud->width; i += downsampling) {
                for (int j = 0; j < (int)m_cloud->height; j += downsampling) {
                    m_displayCloud->push_back(m_cloud->at(i, j));
                }
            }
        }
        else {
            for (int i = 0; i < (int)m_cloud->size(); i += downsampling) {
                m_displayCloud->push_back(m_cloud->at(i));
            }
        }
    }
    else
    {
        m_cloud->clear();
        for (int i = 0; i < m_pointClouds.size(); i++)
        {
            if (i == m_list->currentRow() && !m_chkEnableSegmentation->isChecked())
                continue;

            pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr pointCloud = m_pointClouds[i];
            const Eigen::Matrix4f& transform = m_transforms[i];
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            pcl::transformPointCloudWithNormals(*pointCloud, *transformed, transform);
            *m_displayCloud += *transformed;
        }
    }

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

    vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);

    // create points from point cloud
    for (size_t i = 0; i < m_displayCloud->size(); i++)
    {
        const pcl::PointXYZRGBNormal& point = m_displayCloud->at(i);

        points->InsertNextPoint(point.x, point.y, point.z);
        colors->InsertNextTuple3(point.r, point.g, point.b);
    }

    vtkSmartPointer<vtkCellArray> conn = vtkSmartPointer<vtkCellArray>::New();
    for (vtkIdType i = 0; i < points->GetNumberOfPoints(); i++)
        conn->InsertNextCell(1, &i);

    m_points->SetPoints(points);
    m_points->GetPointData()->SetScalars(colors);
    m_points->SetVerts(conn);
    m_points->Modified();
    m_renderView->update();

    m_drawSensorCloud = false;
}


void ModelGUI::drawCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud)
{
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

    vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);

    // create points from point cloud
    for (size_t i = 0; i < cloud->size(); i++) {
        const pcl::PointXYZRGBNormal& point = cloud->at(i);

        points->InsertNextPoint(point.x, point.y, point.z);
        colors->InsertNextTuple3(point.r, point.g, point.b);
    }

    vtkSmartPointer<vtkCellArray> conn = vtkSmartPointer<vtkCellArray>::New();
    for (vtkIdType i = 0; i < points->GetNumberOfPoints(); i++)
        conn->InsertNextCell(1, &i);

    m_curPoints->SetPoints(points);
    m_curPoints->GetPointData()->SetScalars(colors);
    m_curPoints->SetVerts(conn);
    m_curPoints->Modified();
    m_renderView->update();
}


void ModelGUI::updateBox()
{
    vtkSmartPointer<vtkLinearTransform> invT;
    if (m_cubeActor->GetVisibility())
        invT = m_cubeTrans->GetLinearInverse();

    vtkSmartPointer<vtkPoints> points = m_points->GetPoints();
    vtkSmartPointer<vtkDataArray> colors = m_points->GetPointData()->GetScalars();

    for (int i = 0; i < points->GetNumberOfPoints(); i++)
    {
        double* point = points->GetPoint(i);
        double* color = colors->GetTuple3(i);

        if (m_cubeActor->GetVisibility() == true) {
            // check for bounding box intersection
            double* transPoint = invT->TransformDoublePoint(point[0], point[1], point[2]);

            if (transPoint[0] > 0.5 ||
                    transPoint[0] < -0.5 ||
                    transPoint[1] > 0.5 ||
                    transPoint[1] < -0.5 ||
                    transPoint[2] > 0.5 ||
                    transPoint[2] < -0.5) {
                color[0] = color[1] = color[2] = 50;
            }
            else
            {
                if(m_annotationMode) // in annotation mode use label color instead of default segmentation color
                {
                    Eigen::Vector3d selected_color = m_annotationColors.at(m_currentLabelIndex);
                    color[0] = selected_color[0];
                    color[1] = selected_color[1];
                    color[2] = selected_color[2];
                }
                else
                {
                    color[0] = m_segmentColor[0];
                    color[1] = m_segmentColor[1];
                    color[2] = m_segmentColor[2];
                }
            }

            colors->SetTuple3(i, color[0], color[1], color[2]);
        }
    }
    m_points->Modified();
    m_renderView->update();
}


void ModelGUI::indexChanged(int index)
{
    if (index >= 0) {
        drawCloud();
        drawCloud(m_pointClouds[index]);

        Eigen::Matrix4f& transform = m_transforms[index];
        vtkSmartPointer<vtkMatrix4x4> tempMat = vtkSmartPointer<vtkMatrix4x4>::New();
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                tempMat->Element[i][j] = (double)transform(i, j);
        m_curPointsTransform->SetMatrix(tempMat);
    }
}


void ModelGUI::activateCurCloud(bool activate)
{
    if (activate) {
        m_renderView->getRendererScene()->AddActor(m_curPointsActor);
        m_renderView->getRendererTop()->AddActor(m_curPointsActor);
        m_renderView->getRendererSide()->AddActor(m_curPointsActor);
        m_renderView->getRendererFront()->AddActor(m_curPointsActor);
        m_renderView->update();
    }
    else {
        m_renderView->getRendererScene()->RemoveActor(m_curPointsActor);
        m_renderView->getRendererTop()->RemoveActor(m_curPointsActor);
        m_renderView->getRendererSide()->RemoveActor(m_curPointsActor);
        m_renderView->getRendererFront()->RemoveActor(m_curPointsActor);
        m_renderView->update();
    }
}

// NOTE temporarily disabling ROS - makes this method unused
//void ModelGUI::pauseResume()
//{
//    if (m_updateSensorCloud) {
//        m_updateSensorCloud = false;
//        m_btPauseResume->setText("Resume");
//    }
//    else {
//        m_updateSensorCloud = true;
//        m_btPauseResume->setText("Pause");
//    }
//}

void ModelGUI::reset()
{
    if (m_isLoaded) {
        m_cloud->clear();
        m_transforms.clear();
        m_pointClouds.clear();
        m_curPoints->Reset();
        m_list->clear();
        drawCloud();
    }

    resetTransform();
    m_isLoaded = false;
    m_updateSensorCloud = true;
    // NOTE temporarily disabling ROS
    //m_btPauseResume->setText("Pause");
    m_renderView->reset();
    m_renderView->update();
}

void ModelGUI::resetTransform()
{
    m_cubeTrans->Identity();
    m_cubeAxesTrans->Identity();
}

void ModelGUI::segment()
{
    vtkSmartPointer<vtkLinearTransform> invT = m_cubeTrans->GetLinearInverse();
    vtkSmartPointer<vtkMatrix4x4> invMat = vtkSmartPointer<vtkMatrix4x4>::New();
    invT->GetMatrix(invMat);
    Eigen::Matrix4f globalTrans;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            globalTrans(i, j) = (float)invMat->Element[i][j];

    for (int i = 0; i < (int)m_pointClouds.size(); i++) {
        const pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr& cloud = m_pointClouds[i];
        const Eigen::Matrix4f& transform = m_transforms[i];

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        for (int j = 0; j < (int)cloud->size(); j++) {
            pcl::PointXYZRGBNormal point = cloud->points[j];

            //float* transPoint = invT->TransformFloatPoint(point.x, point.y, point.z);
            Eigen::Vector4f point4f(point.x, point.y, point.z, 1);
            Eigen::Vector4f transPoint = globalTrans * transform * point4f;

            if (transPoint[0] <= 0.5 &&
                    transPoint[0] >= -0.5 &&
                    transPoint[1] <= 0.5 &&
                    transPoint[1] >= -0.5 &&
                    transPoint[2] <= 0.5 &&
                    transPoint[2] >= -0.5) {
                newCloud->push_back(point);
            }
        }
        m_pointClouds[i] = newCloud;
    }

    invT = m_cubeAxesTrans->GetLinearInverse();
    invT->GetMatrix(invMat);

    Eigen::Matrix4f globalInvTrans;

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            globalInvTrans(i, j) = (float)invMat->Element[i][j];

    for (int i = 0; i < (int)m_pointClouds.size(); i++) {
        Eigen::Matrix4f& transform = m_transforms[i];
        transform = globalInvTrans * transform;
    }

    m_cubeActor->SetVisibility(false);
    m_cubeAxes->SetVisibility(false);

    indexChanged(m_list->currentRow());

    enableSegmentation(false);
    resetTransform();

    m_updateSensorCloud = false;
}

void ModelGUI::merge()
{
    for (int i = 0; i < m_pointClouds.size(); i++) {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr pointCloud = m_pointClouds[i];
        const Eigen::Matrix4f& transform = m_transforms[i];
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        pcl::transformPointCloudWithNormals(*pointCloud, *transformed, transform);
        *m_cloud += *transformed;
    }
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::copyPointCloud(*m_cloud, *newCloud);

    m_pointClouds.clear();
    m_transforms.clear();
    m_list->clear();

    m_pointClouds.push_back(newCloud);
    m_transforms.push_back(Eigen::Matrix4f::Identity());
    m_curPointsTransform->Identity();
    m_list->addItem("merged");
    m_list->setCurrentRow(m_list->count() - 1);
    m_btMerge->setEnabled(true);
}

void ModelGUI::smooth()
{
    // smooth surface of first point cloud
    if (m_pointClouds.size() > 0) {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mlsPoints(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
        pcl::MovingLeastSquares<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> mls;
        mls.setComputeNormals(false);
        mls.setInputCloud(m_pointClouds[0]);
        mls.setPolynomialOrder(1);
        mls.setSearchMethod(tree);
        mls.setSearchRadius(0.01);
        mls.process(*mlsPoints);

        m_pointClouds[0] = mlsPoints;
        indexChanged(0);
    }
}

void ModelGUI::filter()
{
    // filter surface of first point cloud
    if (m_pointClouds.size() > 0) {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filteredPoints(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBNormal> sor;
        sor.setInputCloud(m_pointClouds[0]);
        sor.setMeanK(50);
        sor.setStddevMulThresh(2);
        //sor.setNegative (true);
        sor.filter(*filteredPoints);

        m_pointClouds[0] = filteredPoints;
        indexChanged(0);
    }
}

void ModelGUI::downsample()
{
    // downsample surface of first point cloud
    if (m_pointClouds.size() > 0) {

        // convert to PCL cloud
        pcl::PCLPointCloud2::Ptr inputPoints(new pcl::PCLPointCloud2 ());
        pcl::PCLPointCloud2::Ptr downsampledPoints(new pcl::PCLPointCloud2 ());
        pcl::toPCLPointCloud2(*m_pointClouds[0], *inputPoints);
        // filter point cloud using PCL voxel grid filter
        pcl::VoxelGrid<pcl::PCLPointCloud2> vgFilter;
        vgFilter.setInputCloud(inputPoints);
        float voxelSize = 0.005f;
        vgFilter.setLeafSize(voxelSize, voxelSize, voxelSize);
        vgFilter.filter(*downsampledPoints);
        // convert to templated point cloud
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        pcl::fromPCLPointCloud2(*downsampledPoints, *cloud);

        m_pointClouds[0] = cloud;
        indexChanged(0);
    }
}


void ModelGUI::importCloud()
{
    QString filename = QFileDialog::getOpenFileName(this, "Load Point-Cloud", QString(), tr("PCD-Files (*.pcd);;All Files (*)"));

    if (!filename.isEmpty())
    {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        if (pcl::io::loadPCDFile(filename.toStdString(), *cloud) >= 0)
        {
            m_updateSensorCloud = false;
            m_pointClouds.push_back(cloud);
            // crashes here with c++ standard below 17, needs Eigen aligned allocator
            // see http://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html
            m_transforms.push_back(Eigen::Matrix4f::Identity());
            m_curPointsTransform->Identity();
            m_list->addItem(QFileInfo(filename).fileName());
            m_list->setCurrentRow(m_list->count() - 1);
            m_btMerge->setEnabled(true);
            m_isLoaded = true;
        }
    }
}

void ModelGUI::exportCloud()
{
    bool wasUpdating = m_updateSensorCloud;
    m_updateSensorCloud = false;

    /*if (m_pointClouds.size() == 0) {
        QMessageBox::warning(this, "Warning", "no point clouds loaded");
        return;
    }*/

    QString filename = QFileDialog::getSaveFileName(this, "Save Point-Cloud", QString(), tr("PCD-Files (*.pcd);;All Files (*)"));

    if (!filename.isEmpty())
    {
        if(!filename.contains(QString(".pcd")))
        {
            filename.append(".pcd");
        }
        if (m_isLoaded)
        {
            if(m_annotationMode)
            {
                pcl::io::savePCDFileBinary(filename.toStdString(), *m_annotated_cloud);
            }
            else
            {
                pcl::io::savePCDFileBinary(filename.toStdString(), *m_pointClouds[0]);
            }
        }
        else
        {
            pcl::io::savePCDFileBinary(filename.toStdString(), *m_cloud);
        }
    }

    if (wasUpdating)
        m_updateSensorCloud = true;
}

void ModelGUI::enableSegmentation(bool enabled)
{
    m_chkEnableSegmentation->setChecked(enabled);
    if (enabled) {
        m_cubeActor->SetVisibility(true);
        m_cubeAxes->SetVisibility(true);
        m_btSegment->setEnabled(true);
        m_list->setEnabled(false);
        activateCurCloud(false);
    } else {
        m_cubeActor->SetVisibility(false);
        m_cubeAxes->SetVisibility(false);
        m_btSegment->setEnabled(false);
        m_list->setEnabled(true);
        activateCurCloud(true);
    }
    drawCloud();
    updateBox();
    m_renderView->update();
}

void ModelGUI::moveXY(float value1, float value2)
{
    if (m_chkEnableSegmentation->isChecked()) {
        m_cubeTrans->PostMultiply();
        m_cubeTrans->Translate(value1, value2, 0);
        m_cubeAxesTrans->PostMultiply();
        m_cubeAxesTrans->Translate(value1, value2, 0);
        updateBox();
    }
    else {
        if (m_transforms.size() > 0) {
            Eigen::Matrix4f& transform = m_transforms[m_list->currentRow()];
            Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
            translation.block<3, 1>(0, 3) = Eigen::Vector3f(value1, value2, 0);
            transform = translation * transform;

            vtkSmartPointer<vtkMatrix4x4> tempMat = vtkSmartPointer<vtkMatrix4x4>::New();
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    tempMat->Element[i][j] = (double)transform(i, j);
            m_curPointsTransform->SetMatrix(tempMat);
        }
    }
    m_renderView->update();
}

void ModelGUI::moveYZ(float value1, float value2)
{
    if (m_chkEnableSegmentation->isChecked()) {
        m_cubeTrans->PostMultiply();
        m_cubeTrans->Translate(0, value1, value2);
        m_cubeAxesTrans->PostMultiply();
        m_cubeAxesTrans->Translate(0, value1, value2);
        updateBox();
    }
    else {
        if (m_transforms.size() > 0) {
            Eigen::Matrix4f& transform = m_transforms[m_list->currentRow()];
            Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
            translation.block<3, 1>(0, 3) = Eigen::Vector3f(0, value1, value2);
            transform = translation * transform;

            vtkSmartPointer<vtkMatrix4x4> tempMat = vtkSmartPointer<vtkMatrix4x4>::New();
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    tempMat->Element[i][j] = (double)transform(i, j);
            m_curPointsTransform->SetMatrix(tempMat);
        }
    }
    m_renderView->update();
}

void ModelGUI::moveXZ(float value1, float value2)
{
    if (m_chkEnableSegmentation->isChecked()) {
        m_cubeTrans->PostMultiply();
        m_cubeTrans->Translate(value1, 0, value2);
        m_cubeAxesTrans->PostMultiply();
        m_cubeAxesTrans->Translate(value1, 0, value2);
        updateBox();
    }
    else {
        if (m_transforms.size() > 0) {
            Eigen::Matrix4f& transform = m_transforms[m_list->currentRow()];
            Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
            translation.block<3, 1>(0, 3) = Eigen::Vector3f(value1, 0, value2);
            transform = translation * transform;

            vtkSmartPointer<vtkMatrix4x4> tempMat = vtkSmartPointer<vtkMatrix4x4>::New();
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    tempMat->Element[i][j] = (double)transform(i, j);
            m_curPointsTransform->SetMatrix(tempMat);
        }
    }
    m_renderView->update();
}

void ModelGUI::scaleXY(float value1, float value2)
{
    if (m_chkEnableSegmentation->isChecked()) {
        m_cubeTrans->PreMultiply();
        m_cubeTrans->Scale(1 + value1, 1 + value2, 1);
        updateBox();
    }
    m_renderView->update();
}

void ModelGUI::scaleYZ(float value1, float value2)
{
    if (m_chkEnableSegmentation->isChecked()) {
        m_cubeTrans->PreMultiply();
        m_cubeTrans->Scale(1, 1 + value1, 1 + value2);
        updateBox();
    }
    m_renderView->update();
}

void ModelGUI::scaleXZ(float value1, float value2)
{
    if (m_chkEnableSegmentation->isChecked()) {
        m_cubeTrans->PreMultiply();
        m_cubeTrans->Scale(1 + value1, 1, 1 + value2);
        updateBox();
    }
    m_renderView->update();
}

void ModelGUI::rotateX(float value, Eigen::Vector3f pickPoint)
{
    if (m_chkEnableSegmentation->isChecked()) {
        m_cubeTrans->PostMultiply();
        m_cubeAxesTrans->PostMultiply();
        double* pos = m_cubeTrans->GetPosition();
        m_cubeTrans->Translate(-pos[0], -pos[1], -pos[2]);
        m_cubeTrans->RotateX(value);
        m_cubeTrans->Translate(pos[0], pos[1], pos[2]);
        m_cubeAxesTrans->Translate(-pos[0], -pos[1], -pos[2]);
        m_cubeAxesTrans->RotateX(value);
        m_cubeAxesTrans->Translate(pos[0], pos[1], pos[2]);
        updateBox();
    }
    else {
        if (m_transforms.size() > 0) {
            Eigen::Matrix4f& transform = m_transforms[m_list->currentRow()];
            Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
            Eigen::AngleAxisf angleAxis(deg2rad(value), Eigen::Vector3f(1, 0, 0));
            Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
            translation.block<3, 1>(0, 3) = -pickPoint;
            rotation.block<3, 3>(0, 0) = angleAxis.toRotationMatrix();
            transform = translation.inverse() * rotation * translation * transform;

            vtkSmartPointer<vtkMatrix4x4> tempMat = vtkSmartPointer<vtkMatrix4x4>::New();
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    tempMat->Element[i][j] = (double)transform(i, j);
            m_curPointsTransform->SetMatrix(tempMat);
        }
    }
    m_renderView->update();
}

void ModelGUI::rotateY(float value, Eigen::Vector3f pickPoint)
{
    if (m_chkEnableSegmentation->isChecked()) {
        m_cubeTrans->PostMultiply();
        m_cubeAxesTrans->PostMultiply();
        double* pos = m_cubeTrans->GetPosition();
        m_cubeTrans->Translate(-pos[0], -pos[1], -pos[2]);
        m_cubeTrans->RotateY(value);
        m_cubeTrans->Translate(pos[0], pos[1], pos[2]);
        m_cubeAxesTrans->Translate(-pos[0], -pos[1], -pos[2]);
        m_cubeAxesTrans->RotateY(value);
        m_cubeAxesTrans->Translate(pos[0], pos[1], pos[2]);
        updateBox();
    }
    else {
        if (m_transforms.size() > 0) {
            Eigen::Matrix4f& transform = m_transforms[m_list->currentRow()];
            Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
            Eigen::AngleAxisf angleAxis(deg2rad(value), Eigen::Vector3f(0, 1, 0));
            Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
            translation.block<3, 1>(0, 3) = -pickPoint;
            rotation.block<3, 3>(0, 0) = angleAxis.toRotationMatrix();
            transform = translation.inverse() * rotation * translation * transform;

            vtkSmartPointer<vtkMatrix4x4> tempMat = vtkSmartPointer<vtkMatrix4x4>::New();
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    tempMat->Element[i][j] = (double)transform(i, j);
            m_curPointsTransform->SetMatrix(tempMat);
        }
    }
    m_renderView->update();
}

void ModelGUI::rotateZ(float value, Eigen::Vector3f pickPoint)
{
    if (m_chkEnableSegmentation->isChecked()) {
        m_cubeTrans->PostMultiply();
        m_cubeAxesTrans->PostMultiply();
        double* pos = m_cubeTrans->GetPosition();
        m_cubeTrans->Translate(-pos[0], -pos[1], -pos[2]);
        m_cubeTrans->RotateZ(value);
        m_cubeTrans->Translate(pos[0], pos[1], pos[2]);
        m_cubeAxesTrans->Translate(-pos[0], -pos[1], -pos[2]);
        m_cubeAxesTrans->RotateZ(value);
        m_cubeAxesTrans->Translate(pos[0], pos[1], pos[2]);
        updateBox();
    }
    else {
        if (m_transforms.size() > 0) {
            Eigen::Matrix4f& transform = m_transforms[m_list->currentRow()];
            Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
            Eigen::AngleAxisf angleAxis(deg2rad(value), Eigen::Vector3f(0, 0, 1));
            Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
            translation.block<3, 1>(0, 3) = -pickPoint;
            rotation.block<3, 3>(0, 0) = angleAxis.toRotationMatrix();
            transform = translation.inverse() * rotation * translation * transform;

            vtkSmartPointer<vtkMatrix4x4> tempMat = vtkSmartPointer<vtkMatrix4x4>::New();
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    tempMat->Element[i][j] = (double)transform(i, j);
            m_curPointsTransform->SetMatrix(tempMat);
        }
    }
    m_renderView->update();
}

void ModelGUI::startAnnotation()
{
    if(m_isLoaded && !m_annotationMode)
    {
        m_StartStopAnnotation->setText("Stop Annotation");
        m_annotationMode = true;
        drawCloud(m_pointClouds.at(0));

        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr label_cloud(new pcl::PointCloud<pcl::PointXYZRGBL>);
        for(unsigned i = 0; i < m_pointClouds.at(0)->size(); i++)
        {
            pcl::PointXYZRGBL point;
            point.x = m_pointClouds.at(0)->at(i).x;
            point.y = m_pointClouds.at(0)->at(i).y;
            point.z = m_pointClouds.at(0)->at(i).z;
            point.rgb = m_pointClouds.at(0)->at(i).rgb;
            point.label = 2; // corresponds to default label in affordance annotations
            label_cloud->push_back(point);
        }
        label_cloud->is_dense = m_pointClouds.at(0)->is_dense;
        label_cloud->height = m_pointClouds.at(0)->height;
        label_cloud->width = m_pointClouds.at(0)->width;

        m_annotated_cloud = label_cloud;
        updateBox();
    }
    else if(m_isLoaded && m_annotationMode)
    {
        m_StartStopAnnotation->setText("Start Annotation");
        m_annotationMode = false;
        updateBox();
    }
}

void ModelGUI::labelSelected(int index)
{
    m_currentLabelIndex = index;
    updateBox();
}

void ModelGUI::setLabel()
{
    vtkSmartPointer<vtkLinearTransform> invT = m_cubeTrans->GetLinearInverse();
    vtkSmartPointer<vtkMatrix4x4> invMat = vtkSmartPointer<vtkMatrix4x4>::New();
    invT->GetMatrix(invMat);
    Eigen::Matrix4f globalTrans;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            globalTrans(i, j) = (float)invMat->Element[i][j];

    for (int i = 0; i < (int)m_pointClouds.size(); i++) {
        const pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr& cloud = m_pointClouds[i];
        const Eigen::Matrix4f& transform = m_transforms[i];

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        for (int j = 0; j < (int)cloud->size(); j++)
        {
            pcl::PointXYZRGBNormal point = cloud->points[j];

            //float* transPoint = invT->TransformFloatPoint(point.x, point.y, point.z);
            Eigen::Vector4f point4f(point.x, point.y, point.z, 1);
            Eigen::Vector4f transPoint = globalTrans * transform * point4f;

            if (transPoint[0] <= 0.5 &&
                    transPoint[0] >= -0.5 &&
                    transPoint[1] <= 0.5 &&
                    transPoint[1] >= -0.5 &&
                    transPoint[2] <= 0.5 &&
                    transPoint[2] >= -0.5)
            {
                // assign RGB data of selected label to displayed cloud
                Eigen::Vector3d color = m_annotationColors.at(m_currentLabelIndex);
                point.r = color[0];
                point.g = color[1];
                point.b = color[2];
                // assign label in other annotation point cloud
                m_annotated_cloud->at(j).label = m_annotationIDs.at(m_currentLabelIndex);
            }
            newCloud->push_back(point);
        }
        m_pointClouds[i] = newCloud;
    }

    invT = m_cubeAxesTrans->GetLinearInverse();
    invT->GetMatrix(invMat);

    Eigen::Matrix4f globalInvTrans;

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            globalInvTrans(i, j) = (float)invMat->Element[i][j];

    for (int i = 0; i < (int)m_pointClouds.size(); i++) {
        Eigen::Matrix4f& transform = m_transforms[i];
        transform = globalInvTrans * transform;
    }

    m_cubeActor->SetVisibility(false);
    m_cubeAxes->SetVisibility(false);

    indexChanged(m_list->currentRow());

    enableSegmentation(false);
    resetTransform();

    m_updateSensorCloud = false;
}

void ModelGUI::removeGroundPlane()
{
    std::cout << "------------- NOTE: plane segmentation is currently disabled!" << std::endl;
//    if(m_pointClouds.size() == 0) return;

//    // manually create cloud without normals for plane detection
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
//    for(unsigned i = 0; i < m_pointClouds[0]->size(); i++)
//    {
//        pcl::PointXYZRGBNormal p = m_pointClouds[0]->points.at(i);
//        pcl::PointXYZRGB pn;
//        pn.x = p.x;
//        pn.y = p.y;
//        pn.z = p.z;
//        pn.rgb = p.rgb;
//        newCloud->push_back(pn);
//    }
//    newCloud->height = 1;
//    newCloud->width = newCloud->size();
//    newCloud->is_dense = false;

//    // plane detection params
//    float curvature = 0.01;
//    float distance = 0.01;
//    int num_points = 1000;
//    float angle = 30;
//    angle = angle * M_PI / 180;

    // detect planes
//    std::vector<Plane> plane_list;
//    PlaneDetection *plane_detector = new PlaneDetection(curvature, angle, distance, num_points, false);
//    plane_detector->detectPlanes(newCloud, plane_list);
//    std::cout << "Num detected planes: " << plane_list.size() << std::endl;

//    // find largest plane
//    int max_points = 0;
//    std::vector<int> point_indices;
//    for(unsigned i = 0; i < plane_list.size(); i++)
//    {
//        std::vector<int> point_indices_cand = plane_list.at(i).getPointIndices();
//        if(point_indices_cand.size() > max_points)
//        {
//            max_points = point_indices_cand.size();
//            point_indices = point_indices_cand;
//        }
//    }

//    // set plane points to red (only for visualization)
//    for(unsigned i = 0; i < point_indices.size(); i++)
//    {
//        newCloud->points.at(point_indices.at(i)).r = 255;
//        newCloud->points.at(point_indices.at(i)).g = 0;
//        newCloud->points.at(point_indices.at(i)).b = 0;
//    }

//    // sort indices
//    std::sort(point_indices.begin(), point_indices.end());

//    // convert back to cloud with normals
//    int idx_counter = 0;
//    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr resulting_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
//    for(unsigned i = 0; i < newCloud->size(); i++)
//    {
//        // skip points in the plane
//        if(idx_counter < point_indices.size() && i == point_indices.at(idx_counter))
//        {
//            idx_counter++;
//            continue;
//        }

//        pcl::PointXYZRGB p = newCloud->points.at(i);
//        pcl::PointXYZRGBNormal pn;
//        pn.x = p.x;
//        pn.y = p.y;
//        pn.z = p.z;
//        pn.rgb = p.rgb;
//        resulting_cloud->push_back(pn);
//    }
//    resulting_cloud->height = 1;
//    resulting_cloud->width = newCloud->size();
//    resulting_cloud->is_dense = false;

//    m_pointClouds[0] = resulting_cloud;
//    drawCloud(m_pointClouds.at(0));

//    delete plane_detector;
 // ------------------ detecting planes up to here disabled ----------------------------
}
