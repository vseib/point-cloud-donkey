#include "training_gui.h"

#include "../vtk_utils/render_view.h"

#include "model_dlg.h"
#include "ism_worker.h"

#include "../implicit_shape_model/implicit_shape_model.h"
#include "../implicit_shape_model/codebook/codebook.h"
#include "../implicit_shape_model/codebook/codeword_distribution.h"
#include "../implicit_shape_model/voting/voting.h"
#include "../implicit_shape_model/utils/exception.h"
#include "../implicit_shape_model/utils/utils.h"

// TODO VS temporarily (?) disabling ROS
//#include <pcl_conversions/pcl_conversions.h>

#include <boost/algorithm/string.hpp>

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
#include <QThread>

// PCL
#include <pcl/point_types.h>
// TODO VS temporarily (?) disabling ROS
//#include <pcl/ros/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>

// VTK
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkBox.h>

#include <jsoncpp/json/json.h>
#include <jsoncpp/json/writer.h>

#include <Eigen/Geometry>
#include <Eigen/Core>

std::vector<std::vector<double>> colorTable;

void buildTable(int maxSize)
{
    colorTable.resize(maxSize);
    for (int i = 0; i < maxSize; i++) {
        std::vector<double>& color = colorTable[i];
        color.resize(3);
        color[0] = rand() / (double)RAND_MAX;
        color[1] = rand() / (double)RAND_MAX;
        color[2] = rand() / (double)RAND_MAX;
    }
}

void getColor(int index, double& r, double& g, double& b)
{
    int modIndex = index % colorTable.size();
    r = colorTable[modIndex][0];
    g = colorTable[modIndex][1];
    b = colorTable[modIndex][2];
}

TrainingGUI::TrainingGUI(QWidget* parent)
    : QWidget(parent),
      m_isLoaded(false),
      m_updateCloud(true)
{
    srand(0);

    buildTable(1000);

    colorTable[0][0] = 0;
    colorTable[0][1] = 0;
    colorTable[0][2] = 1;

    // init qt related
    m_spinTimer = new QTimer(this);
    m_spinTimer->setInterval(15);
    connect(m_spinTimer, SIGNAL(timeout()), this, SLOT(spinOnce()));
    m_spinTimer->start();

    // create pcl data
    m_detectCloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    m_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    m_displayCloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());

    // TODO VS temporarily (?) disabling ROS
    // subscribe to ros topic to acquire depth images
    //m_subPoints = m_node.subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/points", 1000, &TrainingGUI::cbPoints, this);

    setWindowTitle("ISM3D - Training GUI");

    // create vtk views
    m_renderView = new RenderView(this);

    // add points
    m_points = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPolyDataMapper> pointsMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    pointsMapper->SetInputData(m_points);
    m_pointsActor = vtkSmartPointer<vtkActor>::New();
    m_pointsActor->SetMapper(pointsMapper);
    m_pointsActor->GetProperty()->SetPointSize(1);

    m_renderView->addActorToScene(m_pointsActor);

    // create navigator panes
    m_navApplication = createNavigatorApplication();
    m_navGeneral = createNavigatorGeneral();
    m_navISM = createNavigatorISM();

    QVBoxLayout* navigatorLayout = new QVBoxLayout();
    navigatorLayout->addWidget(m_navApplication);
    navigatorLayout->addWidget(m_navGeneral);
    navigatorLayout->addWidget(m_navISM);
    navigatorLayout->addItem(new QSpacerItem(150, 0, QSizePolicy::Minimum, QSizePolicy::Expanding));

    // put it all together
    QGridLayout* layout = new QGridLayout(this);    // NOTE: produces a warning
    layout->addWidget(m_renderView, 0,  0);
    layout->addLayout(navigatorLayout, 0, 1);
    this->setLayout(layout);

    // create ism class
    m_ism = new ism3d::ImplicitShapeModel();
    m_ism->m_signalPointCloud.connect(boost::bind(&TrainingGUI::signalPointCloud, this, _1));
    m_ism->m_signalBoundingBox.connect(boost::bind(&TrainingGUI::signalBoundingBox, this, _1));
    m_ism->m_signalNormals.connect(boost::bind(&TrainingGUI::signalNormals, this, _1, _2));
    m_ism->m_signalFeatures.connect(boost::bind(&TrainingGUI::signalFeatures, this, _1));
    //m_ism->m_signalCodebook.connect(boost::bind(&TrainingGUI::signalCodebook, this, _1));
    m_ism->m_signalMaxima.connect(boost::bind(&TrainingGUI::signalMaxima, this, _1));

    resize(1024, 768);
}

TrainingGUI::~TrainingGUI()
{
    delete m_ism;
}

QGroupBox* TrainingGUI::createNavigatorApplication()
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

QGroupBox* TrainingGUI::createNavigatorGeneral()
{
    m_btPauseResume = new QPushButton(this);
    connect(m_btPauseResume, SIGNAL(clicked()), SLOT(pauseResume()));
    m_btPauseResume->setText("Pause");

    QPushButton* btReset = new QPushButton(this);
    connect(btReset, SIGNAL(clicked()), this, SLOT(reset()));
    btReset->setText("Reset");

    QPushButton* btAddModel = new QPushButton(this);
    connect(btAddModel, SIGNAL(clicked()), this, SLOT(addModel()));
    btAddModel->setText("Add training model");

    QPushButton* btLoadScene = new QPushButton(this);
    connect(btLoadScene, SIGNAL(clicked()), this, SLOT(loadScene()));
    btLoadScene->setText("Load Scene");

    QVBoxLayout* layout = new QVBoxLayout();
    layout->addWidget(m_btPauseResume);
    layout->addWidget(btReset);
    layout->addWidget(btAddModel);
    layout->addWidget(btLoadScene);

    QGroupBox* groupBox = new QGroupBox(this);
    groupBox->setTitle("General");
    groupBox->setLayout(layout);
    return groupBox;
}

QGroupBox* TrainingGUI::createNavigatorISM()
{
    QPushButton* btLoad = new QPushButton(this);
    btLoad->setText("Load");
    connect(btLoad, SIGNAL(clicked()), this, SLOT(load()));

    QPushButton* btSave = new QPushButton(this);
    btSave->setText("Save");
    connect(btSave, SIGNAL(clicked()), this, SLOT(save()));

    QPushButton* btClear = new QPushButton(this);
    btClear->setText("Clear");
    connect(btClear, SIGNAL(clicked()), this, SLOT(clearISM()));

    QPushButton* btDetect = new QPushButton(this);
    btDetect->setText("Detect");
    connect(btDetect, SIGNAL(clicked()), this, SLOT(detectISM()));

    QPushButton* btTrain = new QPushButton(this);
    btTrain->setText("Train");
    connect(btTrain, SIGNAL(clicked()), this, SLOT(trainModel()));

    m_chkShowFeatures = new QCheckBox(this);
    m_chkShowFeatures->setText("Show Features");

    m_chkInvertNormals = new QCheckBox(this);
    m_chkInvertNormals->setText("Invert Normals");
    m_chkInvertNormals->setChecked(false);

    m_chkShowNormals = new QCheckBox(this);
    m_chkShowNormals->setText("Show Normals");
    m_chkShowNormals->setChecked(false);

    m_chkShowAllVotes = new QCheckBox(this);
    m_chkShowAllVotes->setText("Show non-max votes");

    m_chkShowVotes = new QCheckBox(this);
    m_chkShowVotes->setText("Show Votes");
    m_chkShowVotes->setChecked(true);

    m_chkShowBbAndCenters = new QCheckBox(this);
    m_chkShowBbAndCenters->setText("Show Results");
    m_chkShowBbAndCenters->setChecked(true);

    m_chkShowOnlyBestMaxPerClass = new QCheckBox(this);
    m_chkShowOnlyBestMaxPerClass->setText("Show only best Max/Class");
    m_chkShowOnlyBestMaxPerClass->setChecked(false);

    m_minMaxVotesLabel = new QLabel(this);
    m_minMaxVotesLabel->setText("Min Votes:");

    m_minMaxVotesToShowLine = new QLineEdit(this);
    m_minMaxVotesToShowLine->setToolTip(QString("Minimum number of votes for displayed maxima"));
    m_minMaxVotesToShowLine->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);

    m_onlyShowClassLabel = new QLabel(this);
    m_onlyShowClassLabel->setText("Only show maxima of classes:");

    m_onlyShowClassLine = new QLineEdit(this);
    m_onlyShowClassLine->setToolTip(QString("Only show detected maxima of these classes, separated by spaces"));
    m_onlyShowClassLine->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(btLoad);
    layout->addWidget(btSave);
    layout->addWidget(btClear);
    layout->addWidget(btDetect);
    layout->addWidget(btTrain);
    layout->addWidget(m_chkInvertNormals);
    layout->addWidget(m_chkShowNormals);
    layout->addWidget(m_chkShowFeatures);
    layout->addWidget(m_chkShowVotes);
    layout->addWidget(m_chkShowAllVotes);
    layout->addWidget(m_chkShowBbAndCenters);
    layout->addWidget(m_chkShowOnlyBestMaxPerClass);
    layout->addWidget(m_minMaxVotesLabel);
    layout->addWidget(m_minMaxVotesToShowLine);
    layout->addWidget(m_onlyShowClassLabel);
    layout->addWidget(m_onlyShowClassLine);

    m_implicitShapeModel = new QGroupBox(this);
    m_implicitShapeModel->setTitle("Implicit Shape Model");
    m_implicitShapeModel->setLayout(layout);

    return m_implicitShapeModel;
}

void TrainingGUI::spinOnce()
{
    if (m_isLoaded)
        drawCloud();

    m_renderView->update();

// TODO VS temporarily (?) disabling ROS
//    if (!ros::ok())
//        this->close();

//    ros::spinOnce();
}

// TODO VS temporarily (?) disabling ROS
//void TrainingGUI::cbPoints(const sensor_msgs::PointCloud2::ConstPtr& pointCloud)
//{
//    if (m_isLoaded)
//        return;

//    if (m_updateCloud) {
//        pcl::fromROSMsg(*pointCloud, *m_cloud);
//        drawCloud();
//    }
//}

void TrainingGUI::drawCloud()
{
    // downsample
    const int downsampling = 1;
    m_displayCloud->clear();
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

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

    vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);

    // create points from point cloud
    for (size_t i = 0; i < m_displayCloud->size(); i++) {
        const pcl::PointXYZRGB& point = m_displayCloud->points[i];
        points->InsertNextPoint(point.x, point.y, point.z);

        if(point.r == 0 && point.g == 0 && point.b == 0)
            colors->InsertNextTuple3(255, 0, 0);
        else
            colors->InsertNextTuple3(point.r, point.g, point.b);
    }

    vtkSmartPointer<vtkCellArray> conn = vtkSmartPointer<vtkCellArray>::New();
    for (vtkIdType i = 0; i < points->GetNumberOfPoints(); i++)
        conn->InsertNextCell(1, &i);

    m_pointsActor->GetProperty()->SetPointSize(5);

    m_points->SetPoints(points);
    m_points->GetPointData()->SetScalars(colors);
    m_points->SetVerts(conn);
    m_points->Modified();
}

void TrainingGUI::pauseResume()
{
    if (m_updateCloud) {
        m_updateCloud = false;
        m_btPauseResume->setText("Resume");
    }
    else {
        m_updateCloud = true;
        m_btPauseResume->setText("Pause");
    }
}

void TrainingGUI::reset()
{
    //m_textStatus->SetVisibility(true);

    if (m_isLoaded) {
        m_cloud->clear();
        drawCloud();
    }

    m_isLoaded = false;
    m_updateCloud = true;
    m_btPauseResume->setText("Pause");

    m_renderView->reset();
}

void TrainingGUI::addModel()
{
    m_updateCloud = false;

    ModelDlg dlg;
    dlg.exec();

    if (dlg.isValid()) {
        // get data
        unsigned classId = dlg.getClassId();
        const QString& filename = dlg.getFilename();

        // load the model
        pcl::PointCloud<PointNormalT>::Ptr model(new pcl::PointCloud<PointNormalT>());

        if (filename.endsWith(".pcd", Qt::CaseInsensitive)) {
            if (pcl::io::loadPCDFile(filename.toStdString(), *model) < 0)
                QMessageBox::warning(this, "Error", "Could not load PCD file!");
            else {
                m_ism->addTrainingModel(filename.toStdString(), classId, classId); // TODO VS: for now setting instance id to class id
                m_cloud->clear();
                m_detectCloud->clear();
                pcl::copyPointCloud(*model, *m_cloud);
                pcl::copyPointCloud(*model, *m_detectCloud);
                m_isLoaded = true;
            }
        }
        else if (filename.endsWith(".ply", Qt::CaseInsensitive)) {
            if (pcl::io::loadPLYFile(filename.toStdString(), *model) < 0)
                QMessageBox::warning(this, "Error", "Could not load PLY file!");
            else {
                m_ism->addTrainingModel(filename.toStdString(), classId, classId); // TODO VS: for now setting instance id to class id
                m_cloud->clear();
                m_detectCloud->clear();
                pcl::copyPointCloud(*model, *m_cloud);
                pcl::copyPointCloud(*model, *m_detectCloud);
                m_isLoaded = true;
            }
        }
        else
            QMessageBox::warning(this, "Error", "Invalid file extension");
    }

    m_updateCloud = true;
}

void TrainingGUI::loadScene()
{
    m_updateCloud = false;

    QString filename = QFileDialog::getOpenFileName(this, "Load Scene", QString(), tr("PCD-Files (*.pcd);;PLY-Files (*.ply);;All Files (*.*)"), 0, QFileDialog::DontUseNativeDialog);

    m_detectCloud->clear();
    if (!filename.isEmpty()) {
        if (filename.endsWith(".pcd", Qt::CaseInsensitive)) {
            if (pcl::io::loadPCDFile(filename.toStdString(), *m_detectCloud) < 0)
                QMessageBox::warning(this, "Error", "Could not load PCD file!");
            {
                m_cloud->clear();
                pcl::copyPointCloud(*m_detectCloud, *m_cloud);
                m_isLoaded = true;
            }
        }
        else if (filename.endsWith(".ply", Qt::CaseInsensitive)) {
            if (pcl::io::loadPLYFile(filename.toStdString(), *m_detectCloud) < 0)
                QMessageBox::warning(this, "Error", "Could not load PLY file!");
            else {
                m_cloud->clear();
                pcl::copyPointCloud(*m_detectCloud, *m_cloud);
                m_isLoaded = true;
            }
        }
        else
            QMessageBox::warning(this, "Error", "Invalid file extension");
    }

    m_updateCloud = true;
}

void TrainingGUI::load()
{
    QString filename = QFileDialog::getOpenFileName(this, "Load ISM", QString(), tr("ISM-Files (*.ism);;All Files (*.*)"), 0, QFileDialog::DontUseNativeDialog);

    if (!filename.isEmpty()) {
        try {
            m_ism->readObject(filename.toStdString());
        }
        catch (const ism3d::Exception& e) {
            QMessageBox::warning(this, "Exception", e.what());
        }
        catch (...) {
            QMessageBox::warning(this, "Exception", "an unhandled exception ocurred");
        }
    }
}

void TrainingGUI::save()
{
    QString filename = QFileDialog::getSaveFileName(this, "Save ISM", QString(), tr("ISM-Files (*.ism);;All Files (*.*)"), 0, QFileDialog::DontUseNativeDialog);

    if (!filename.isEmpty()) {
        try {
            m_ism->writeObject(filename.toStdString());
        }
        catch (const ism3d::Exception& e) {
            QMessageBox::warning(this, "Exception", e.what());
        }
        catch (...) {
            QMessageBox::warning(this, "Exception", "an unhandled exception ocurred");
        }
    }
}

int i = 0;

void TrainingGUI::trainModel()
{
    m_updateCloud = false;

    m_renderView->reset();
    m_renderView->setStatus("Training...");

    m_navGeneral->setEnabled(false);
    m_navISM->setEnabled(false);

    /*Eigen::Affine3f objTransform(Eigen::Translation3f(0, 0, 0));
    objTransform *= Eigen::AngleAxisf(0.5f, Eigen::Vector3f(1, 0, 0));
    objTransform *= Eigen::Translation3f(-0.4, 0.5, 0.5);
    pcl::transformPointCloud(*m_cloud, *m_cloud, objTransform);*/

    m_cloud->clear();

    if (m_isLoaded) {
        // train
        try {
            // start training on a separate thread
            QThread* thread = new QThread;
            ISMWorker* worker = new ISMWorker;
            worker->setImplicitShapeModel(m_ism);
            worker->moveToThread(thread);
            connect(thread, SIGNAL(started()), worker, SLOT(train()));
            connect(worker, SIGNAL(finished()), thread, SLOT(quit()));
            connect(worker, SIGNAL(finished()), worker, SLOT(deleteLater()));
            connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater()));
            connect(worker, SIGNAL(trainingFinished(bool)), this, SLOT(trainingFinished(bool)));
            thread->start();
        }
        catch (const ism3d::Exception& e) {
            QMessageBox::warning(this, "Exception", e.what());
            trainingFinished(false);
        }
        catch (const std::exception& e) {
            QMessageBox::warning(this, "Exception", e.what());
            trainingFinished(false);
        }
        catch (...) {
            QMessageBox::warning(this, "Exception", "an unhandled exception ocurred");
            trainingFinished(false);
        }
    }
    else {
        QMessageBox::warning(this, "Warning", "No training models loaded.");
        trainingFinished(false);
    }

    i = 0;
}

void TrainingGUI::trainingFinished(bool successful)
{
    if (!successful)
        m_renderView->setStatus("Training failed");
    else
        m_renderView->resetStatus();

    m_navGeneral->setEnabled(true);
    m_navISM->setEnabled(true);
}

void TrainingGUI::detectISM()
{
    m_updateCloud = false;

    m_renderView->reset();
    m_renderView->setStatus("Detecting...");

    m_navGeneral->setEnabled(false);
    m_navISM->setEnabled(false);

    /*if (i == 0) {
        for (int i = 0; i < m_cloud->size(); i++) {
            pcl::PointXYZRGB& point = m_cloud->at(i);
            point.x += ((rand() / (float)RAND_MAX) - 0.5f) * 0.005f;
            point.y += ((rand() / (float)RAND_MAX) - 0.5f) * 0.005f;
            point.z += ((rand() / (float)RAND_MAX) - 0.5f) * 0.005f;
        }
    }*/

    /*if (i > 0) {
        float angle = (rand() / (float)RAND_MAX) * 2 * M_PI;
        float x = rand() / (float)RAND_MAX;
        float y = rand() / (float)RAND_MAX;
        float z = rand() / (float)RAND_MAX;

        Eigen::Vector3f axis(x, y, z);
        axis.normalize();

        pcl::PointCloud<PointNormalT>::Ptr newCloud(new pcl::PointCloud<PointNormalT>());

        Eigen::Affine3f objTransform(Eigen::Translation3f(0, 0, 0));
        //objTransform *= Eigen::AngleAxisf(angle, axis);
        objTransform *= Eigen::AngleAxisf(0.2, Eigen::Vector3f(0, 1, 0));
        //objTransform *= Eigen::Translation3f(x, y, z);
        pcl::transformPointCloud(*m_detectCloud, *newCloud, objTransform);

        m_detectCloud->clear();
        m_cloud->clear();
        pcl::copyPointCloud(*newCloud, *m_detectCloud);
        pcl::copyPointCloud(*newCloud, *m_cloud);
    }

    i++;*/

    if (m_isLoaded) {
        // detect object
        try {
            m_isDetecting = true;

            // start detection on a separate thread
            QThread* thread = new QThread;
            ISMWorker* worker = new ISMWorker;
            worker->setImplicitShapeModel(m_ism);

//            pcl::PointCloud<PointNormalT>::Ptr newCloud(new pcl::PointCloud<PointNormalT>());
//            pcl::copyPointCloud(*m_detectCloud, *newCloud);
//            worker->setDetectionPointCloud(newCloud);
            worker->setDetectionPointCloud(m_detectCloud);

            worker->moveToThread(thread);
            connect(thread, SIGNAL(started()), worker, SLOT(detect()));
            connect(worker, SIGNAL(finished()), thread, SLOT(quit()));
            connect(worker, SIGNAL(finished()), worker, SLOT(deleteLater()));
            connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater()));
            connect(worker, SIGNAL(detectionFinished(bool)), this, SLOT(detectionFinished(bool)));
            thread->start();
        }
        catch (const ism3d::Exception& e) {
            QMessageBox::warning(this, "Exception", e.what());
            detectionFinished(false);
        }
        catch (...) {
            QMessageBox::warning(this, "Exception", "an unhandled exception ocurred");
            detectionFinished(false);
        }
    }
    else
        QMessageBox::warning(this, "Warning", "No detection point cloud loaded.");

    //m_cloud->clear();
}

void TrainingGUI::detectionFinished(bool successful)
{
    if (!successful)
        m_renderView->setStatus("Detection failed");
    else
        m_renderView->resetStatus();

    m_navGeneral->setEnabled(true);
    m_navISM->setEnabled(true);
    m_isDetecting = false;
}

void TrainingGUI::clearISM()
{
    m_ism->clear();
}

void TrainingGUI::signalPointCloud(pcl::PointCloud<ism3d::PointT>::ConstPtr pointCloud)
{
    // can cause synchronizing issues
    if (!m_isDetecting) {
        m_cloud->clear();
        pcl::copyPointCloud(*pointCloud, *m_cloud);
    }
}

void TrainingGUI::signalBoundingBox(const ism3d::Utils::BoundingBox& box)
{
    m_boundingBox = box;
    addBoundingBox(box);
}

void TrainingGUI::addBoundingBox(const ism3d::Utils::BoundingBox& box)
{
    // convert inverse quaternion into angle axis representation
    Eigen::Vector3f axis;
    float angle;
    ism3d::Utils::quat2Axis(boost::math::conj(box.rotQuat), axis, angle);

    // create transform with correct position (but without scale yet)
    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->Translate(box.position[0], box.position[1], box.position[2]);
    transform->RotateWXYZ(angle, axis[0], axis[1], axis[2]);

    // create axes and set the scale manually
    vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
    axes->SetUserTransform(transform);
    axes->AxisLabelsOff();
    axes->SetTotalLength(box.size[0] / 2, box.size[1] / 2, box.size[2] / 2);

    // also set the bounding box scale manually
    vtkSmartPointer<vtkCubeSource> bbox = vtkSmartPointer<vtkCubeSource>::New();
    bbox->SetXLength(box.size[0]);
    bbox->SetYLength(box.size[1]);
    bbox->SetZLength(box.size[2]);

    vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transformFilter->SetInputConnection(bbox->GetOutputPort());
    transformFilter->SetTransform(transform);
    transformFilter->Update();

    vtkSmartPointer<vtkPolyDataMapper> bboxMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    bboxMapper->SetInputConnection(transformFilter->GetOutputPort());
    vtkSmartPointer<vtkActor> bboxActor = vtkSmartPointer<vtkActor>::New();
    bboxActor->SetMapper(bboxMapper);
    bboxActor->GetProperty()->SetLighting(false);
    //bboxActor->GetProperty()->SetColor(255, 255, 255);
    bboxActor->GetProperty()->SetColor(0, 0, 255);    // TEMP
    bboxActor->GetProperty()->SetLineWidth(2);
    bboxActor->GetProperty()->SetRepresentationToWireframe();

    m_renderView->lock();
    m_renderView->getRendererScene()->AddActor(bboxActor);
    m_renderView->getRendererTop()->AddActor(bboxActor);
    m_renderView->getRendererSide()->AddActor(bboxActor);
    m_renderView->getRendererFront()->AddActor(bboxActor);
    m_renderView->unlock();
}

void TrainingGUI::signalNormals(pcl::PointCloud<ism3d::PointT>::ConstPtr pointCloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
    vtkSmartPointer<vtkPoints> line = vtkSmartPointer<vtkPoints>::New();

    for (int i = 0; i < normals->size(); i++) {
        const ism3d::PointT& point = pointCloud->at(i);
        const pcl::Normal& normal = normals->at(i);

        Eigen::Vector3f nn = normal.getNormalVector3fMap();
        if(m_chkInvertNormals->isChecked())
        {
            nn *= -1;
        }

        Eigen::Vector3f pos = point.getVector3fMap();
        Eigen::Vector3f posNormal = pos + 0.01f * nn;

        vtkIdType p1 = line->InsertNextPoint(pos[0], pos[1], pos[2]);
        vtkIdType p2 = line->InsertNextPoint(posNormal[0], posNormal[1], posNormal[2]);

        vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();
        polyLine->GetPointIds()->SetNumberOfIds(2);
        polyLine->GetPointIds()->SetId(0, p1);
        polyLine->GetPointIds()->SetId(1, p2);

        cells->InsertNextCell(polyLine);
    }

    // Create a polydata to store everything in
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();

    polyData->SetPoints(line);
    polyData->SetLines(cells);

    // Setup actor and mapper
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polyData);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(0, 0, 255);
    actor->GetProperty()->SetLineWidth(2);
    //actor->GetProperty()->SetColor(0, 0, 0);
    if(m_chkShowNormals->isChecked())
    {
        m_renderView->getRendererScene()->AddActor(actor);
    }
}

void TrainingGUI::signalFeatures(pcl::PointCloud<ism3d::ISMFeature>::ConstPtr features)
{
    m_renderView->lock();

    for (int i = 0; i < (int)features->size(); i++) {
        const ism3d::ISMFeature& feature = features->at(i);

        const pcl::ReferenceFrame& frame = feature.referenceFrame;

        Eigen::Matrix4d rot = Eigen::Matrix4d::Identity();
        rot(0, 0) = frame.x_axis[0];
        rot(0, 1) = frame.x_axis[1];
        rot(0, 2) = frame.x_axis[2];
        rot(1, 0) = frame.y_axis[0];
        rot(1, 1) = frame.y_axis[1];
        rot(1, 2) = frame.y_axis[2];
        rot(2, 0) = frame.z_axis[0];
        rot(2, 1) = frame.z_axis[1];
        rot(2, 2) = frame.z_axis[2];

        Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
        trans(3, 0) = feature.x;
        trans(3, 1) = feature.y;
        trans(3, 2) = feature.z;

        Eigen::Matrix4d mat = rot * trans;

        vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
        transform->SetMatrix(&mat(0, 0));

        vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
        axes->SetUserTransform(transform);
        axes->AxisLabelsOff();
        axes->SetConeRadius(0);
        axes->GetXAxisShaftProperty()->SetLineWidth(2);
        axes->GetYAxisShaftProperty()->SetLineWidth(2);
        axes->GetZAxisShaftProperty()->SetLineWidth(2);
        axes->SetTotalLength(0.1, 0.1, 0.1);

        if(m_chkShowFeatures->isChecked())
        {
            m_renderView->getRendererScene(true)->AddActor(axes);
        }

        // add a sphere
        /*vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New();
        sphere->SetCenter(feature.x, feature.y, feature.z);
        sphere->SetRadius(0.1);
        vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputConnection(sphere->GetOutputPort());
        vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        actor->GetProperty()->SetRepresentationToWireframe();
        actor->GetProperty()->SetLighting(false);
        actor->GetProperty()->SetColor(1, 1, 1);
        m_renderView->getRendererScene()->AddActor(actor);*/

        /*Eigen::Vector3f keyPos(keypoint.x, keypoint.y, keypoint.z);
        Eigen::Vector3f normal(keypoint.normal_x, keypoint.normal_y, keypoint.normal_z);
        Eigen::Vector3f keyNormal = keyPos + 0.1f * normal;

        vtkSmartPointer<vtkPoints> line = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();
        polyLine->GetPointIds()->SetNumberOfIds(2);
        for(unsigned int i = 0; i < 2; i++)
            polyLine->GetPointIds()->SetId(i, i);
        line->InsertNextPoint(keyPos[0], keyPos[1], keyPos[2]);
        line->InsertNextPoint(keyNormal[0], keyNormal[1], keyNormal[2]);

        // Create a cell array to store the lines in and add the lines to it
        vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
        cells->InsertNextCell(polyLine);

        // Create a polydata to store everything in
        vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();

        polyData->SetPoints(line);
        polyData->SetLines(cells);

        // Setup actor and mapper
        vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInput(polyData);

        vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        actor->GetProperty()->SetColor(0, 0, 255);
        m_rendererScene->AddActor(actor);*/
    }

    m_renderView->unlock();
}

void TrainingGUI::signalMaxima(std::vector<ism3d::VotingMaximum> maxima)
{
    m_renderView->lock();

    const std::map<unsigned, std::vector<ism3d::Voting::Vote> >& votes = m_ism->getVoting()->getVotes();
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

    // Setup colors
    vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(4);

    // enable or disable visualization of votes
    if(m_chkShowVotes->isChecked())
    {
        // save vote indices
        std::map<unsigned, std::vector<int> > voteIndices;
        for (int i = 0; i < (int)maxima.size(); i++) {
            const std::vector<int> indices = maxima[i].voteIndices;
            std::vector<int>& mapEntry = voteIndices[maxima[i].classId];
            mapEntry.insert(mapEntry.end(), indices.begin(), indices.end());
        }

        for (std::map<unsigned, std::vector<ism3d::Voting::Vote> >::const_iterator it = votes.begin();
             it != votes.end(); it++) {
            unsigned classId = it->first;
            const std::vector<ism3d::Voting::Vote>& classVotes = it->second;

            for (int i = 0; i < (int)classVotes.size(); i++) {
                ism3d::Voting::Vote vote = classVotes[i];

                double classColor[3];
                getColor(classId, classColor[0], classColor[1], classColor[2]);

                //float alpha = 0.1f + (vote.weight * 0.9f);
                //float alpha = vote.weight;
                float alpha = 1.0f;

                bool isMaximumVote = true;
                std::map<unsigned, std::vector<int> >::const_iterator it = voteIndices.find(classId);
                if (it == voteIndices.end()) {
                    alpha = 0.3f;
                    isMaximumVote = false;
                }
                else {
                    const std::vector<int>& indices = it->second;
                    if (std::find(indices.begin(), indices.end(), i) == indices.end()) {
                        alpha = 0.3f;
                        isMaximumVote = false;
                    }
                }

                if (!m_chkShowAllVotes->isChecked() && !isMaximumVote)
                    continue;

                // create vote point
                points->InsertNextPoint(vote.position[0], vote.position[1], vote.position[2]);

                unsigned char color[4] = {(unsigned char) (classColor[0] * 255), (unsigned char) (classColor[1] * 255),
                                          (unsigned char) (classColor[2] * 255), (unsigned char) (alpha * 255)};
                colors->InsertNextTupleValue(color);

                // create vote line from keypoint to vote position
                vtkSmartPointer<vtkPoints> line = vtkSmartPointer<vtkPoints>::New();
                vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();
                polyLine->GetPointIds()->SetNumberOfIds(2);
                for(unsigned int i = 0; i < 2; i++) {
                    polyLine->GetPointIds()->SetId(i, i);
                }
                line->InsertNextPoint(vote.keypoint[0], vote.keypoint[1], vote.keypoint[2]);
                line->InsertNextPoint(vote.position[0], vote.position[1], vote.position[2]);

                vtkSmartPointer<vtkUnsignedCharArray> lineColors = vtkSmartPointer<vtkUnsignedCharArray>::New();
                lineColors->SetNumberOfComponents(4);
                lineColors->InsertNextTupleValue(color);
                lineColors->InsertNextTupleValue(color);

                // Create a cell array to store the lines in and add the lines to it
                vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
                cells->InsertNextCell(polyLine);

                // Create a polydata to store everything in
                vtkSmartPointer<vtkPolyData> linePolyData = vtkSmartPointer<vtkPolyData>::New();

                linePolyData->SetPoints(line);
                linePolyData->SetLines(cells);
                linePolyData->GetPointData()->SetScalars(lineColors);

                // Setup actor and mapper
                vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
                mapper->SetInputData(linePolyData);

                vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
                actor->SetMapper(mapper);
                actor->GetProperty()->SetLineWidth(1.5);
                m_renderView->getRendererScene(true)->AddActor(actor);
            }
        }
    }

    vtkSmartPointer<vtkPolyData> pointsPolydata = vtkSmartPointer<vtkPolyData>::New();
    pointsPolydata->SetPoints(points);

    vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    vertexFilter->SetInputData(pointsPolydata);
    vertexFilter->Update();

    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->ShallowCopy(vertexFilter->GetOutput());

    polydata->GetPointData()->SetScalars(colors);

    // Visualization
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polydata);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetPointSize(3);

    // clear list of maxima to show
    m_maxima_classes.clear();
    m_maxima_classes.resize(64, false); // FIXME: actually should be number of possible classes, not a fixed value

    // enable / disable bounding box and center visualization
    if(m_chkShowBbAndCenters->isChecked())
    {
        //m_rendererScene->AddActor(actor);
        m_renderView->getRendererScene(true)->AddActor(actor);

        // add bounding boxes to represent object positions
        int min_votes = m_minMaxVotesToShowLine->text().toInt();
        LOG_WARN("Only maxima with at least " << min_votes << " votes will be displayed!");

        // only show maxima of certain classes
        std::string temp_text = m_onlyShowClassLine->text().toStdString();
        std::vector<std::string> result;
        boost::split(result, temp_text, boost::is_any_of(" \n"));

        // convert textline to ints
        std::vector<int> classes;
        for(int i = 0; i < result.size(); i++)
        {
            std::string s = result.at(i);
            classes.push_back(std::atoi(s.c_str()));
        }
        if(temp_text != "")
        {
            LOG_WARN("Only maxima of class IDs " << temp_text << " will be displayed!");
        }
        else
        {
            LOG_WARN("All class IDs will be displayed!");
        }

        for (int i = 0; i <(int)maxima.size(); i++)
        {
            const ism3d::VotingMaximum& max = maxima[i];

            // only show maxima of selected classes
            if(temp_text != "")
            {
                bool found = false;
                for(int i = 0; i < classes.size(); i++)
                {
                    if(max.classId == classes.at(i))
                    {
                        found = true;
                    }
                }
                if(!found) continue;
            }

            // check if only best maximum per class has to be shown
            if(m_chkShowOnlyBestMaxPerClass->isChecked() && max.classId < m_maxima_classes.size())
            {
                if(m_maxima_classes.at(max.classId))
                {
                    continue; // one max of with this class id has been displayed already
                }
                m_maxima_classes.at(max.classId) = true; // mark this class as shown
            }

            if(max.voteIndices.size() < min_votes) // only report maxima with at least X votes
                continue;

            // add a sphere
            vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New();
            sphere->SetCenter(max.position[0], max.position[1], max.position[2]);
            //sphere->SetRadius(0.025);
            sphere->SetRadius(0.01);
            vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
            mapper->SetInputConnection(sphere->GetOutputPort());
            vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
            actor->SetMapper(mapper);
            actor->GetProperty()->SetLighting(false);
            actor->GetProperty()->SetColor(1, 0, 0);

            m_renderView->getRendererScene(true)->AddActor(actor);
            m_renderView->getRendererTop(true)->AddActor(actor);
            m_renderView->getRendererSide(true)->AddActor(actor);
            m_renderView->getRendererFront(true)->AddActor(actor);

            // add an oriented box
            addBoundingBox(max.boundingBox);
        }
    }

    // TEMP STUFF

    // draw trajectories
    /*const std::vector<std::vector<Eigen::Vector3f> >& trajectories = ((VotingMeanShiftBase*)m_ism->getVoting())->trajectories;
    for (int i = 0; i < (int)trajectories.size(); i++) {
        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        const std::vector<Eigen::Vector3f>& path = trajectories[i];
        for (int j = 0; j < (int)path.size(); j++)
            points->InsertNextPoint(&path[j][0]);

        vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();
        polyLine->GetPointIds()->SetNumberOfIds(path.size());
        for(unsigned int i = 0; i < path.size(); i++)
            polyLine->GetPointIds()->SetId(i,i);

        // Create a cell array to store the lines in and add the lines to it
        vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
        cells->InsertNextCell(polyLine);

        // Create a polydata to store everything in
        vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();

        // Add the points to the dataset
        polyData->SetPoints(points);

        // Add the lines to the dataset
        polyData->SetLines(cells);

        // Setup actor and mapper
        vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInput(polyData);

        vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        actor->GetProperty()->SetColor(255, 255, 0);
        m_rendererScene->AddActor(actor);
    }*/

    m_renderView->unlock();
}

void TrainingGUI::signalCodebook(const ism3d::Codebook& codebook)
{
    m_renderView->lock();

    const std::map<int, std::shared_ptr<ism3d::CodewordDistribution> >& distributionEntries =
            codebook.getDistribution();

    // visualize the activation distribution, i.e. for each codeword display activation vectors
    for (std::map<int, std::shared_ptr<ism3d::CodewordDistribution> >::const_iterator it = distributionEntries.begin();
         it != distributionEntries.end(); it++) {
        const std::shared_ptr<ism3d::CodewordDistribution>& distribution = it->second;

        int codewordId = distribution->getCodewordId();

        double r, g, b;
        getColor(codewordId, r, g, b);

        const std::vector<Eigen::Vector3f>& votes = distribution->getOriginalVotes();
        for (int i = 0; i < (int)votes.size(); i++) {
            Eigen::Vector3f direction = votes[i];
            Eigen::Vector3f position = m_boundingBox.position - direction;

            // add sphere
            vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New();
            sphere->SetCenter(position[0], position[1], position[2]);
            sphere->SetRadius(0.03);
            vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
            mapper->SetInputConnection(sphere->GetOutputPort());
            vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
            actor->SetMapper(mapper);
            actor->GetProperty()->SetLighting(false);
            actor->GetProperty()->SetColor(r, g, b);
            m_renderView->getRendererScene(true)->AddActor(actor);
        }
    }

    m_renderView->unlock();
}
