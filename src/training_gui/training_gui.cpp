#include "training_gui.h"

#include "../vtk_utils/render_view.h"

#include "model_dlg.h"
#include "ism_worker.h"

#include "../implicit_shape_model/implicit_shape_model.h"
#include "../implicit_shape_model/codebook/codebook.h"
#include "../implicit_shape_model/codebook/codeword_distribution.h"
#include "../implicit_shape_model/voting/voting.h"
#include "../implicit_shape_model/voting/voting_mean_shift.h"
#include "../implicit_shape_model/utils/exception.h"

// used to color boxes as tp and fp, i.e. dataset info
#include "../eval_tool/eval_helpers_detection.h"


// NOTE temporarily disabling ROS
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
// NOTE temporarily disabling ROS
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
      m_isDetecting(false),
      m_updateSensorCloud(true), // NOTE: with ROS disabled this is unused
      m_detectCloud(new pcl::PointCloud<PointNormalT>()),
      m_cloud(new pcl::PointCloud<PointNormalT>()),
      m_normals(new pcl::PointCloud<pcl::Normal>()),
      m_displayCloud(new pcl::PointCloud<PointNormalT>()),
      m_use_gt_info(false),
      m_dataset_info_added(false),
      m_loaded_scene_path("")
{
    srand(0);
    buildTable(1000);

    colorTable[0][0] = 0;
    colorTable[0][1] = 0;
    colorTable[0][2] = 1;

    // NOTE temporarily disabling ROS - however using the timer to update render views
    // init qt related
    m_spinTimer = new QTimer(this);
    m_spinTimer->setInterval(100);
    connect(m_spinTimer, SIGNAL(timeout()), this, SLOT(spinOnce()));
    m_spinTimer->start();

    // NOTE temporarily disabling ROS
    // subscribe to ros topic to acquire depth images
    //m_subPoints = m_node.subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/points", 1000, &TrainingGUI::cbPoints, this);

    setWindowTitle("ISM3D - Training GUI");

    // create vtk views
    m_renderView = new RenderView(this);
    m_renderView->getRendererFront()->SetBackground(255,255,255);
    m_renderView->getRendererTop()->SetBackground(255,255,255);
    m_renderView->getRendererSide()->SetBackground(255,255,255);

    // add points
    m_points = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPolyDataMapper> pointsMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    pointsMapper->SetInputData(m_points);
    m_pointsActor = vtkSmartPointer<vtkActor>::New();
    m_pointsActor->SetMapper(pointsMapper);
    m_pointsActor->GetProperty()->SetPointSize(1);

//    m_renderView->addActorToScene(m_pointsActor);
    m_renderView->addActorToAll(m_pointsActor);

    // create navigator panes
    m_navApplication = createNavigatorApplication();
    m_navGeneral = createNavigatorTraining();
    m_navISM = createNavigatorDetect();

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
    //m_ism->readObject("default_config_kinect.ism", true);
    m_ism->readObject("ds5_ransac_test.ism", false);
    m_ism->m_signalPointCloud.connect(boost::bind(&TrainingGUI::signalPointCloud, this, _1));
    m_ism->m_signalBoundingBox.connect(boost::bind(&TrainingGUI::signalBoundingBox, this, _1));
    m_ism->m_signalNormals.connect(boost::bind(&TrainingGUI::signalNormals, this, _1, _2));
    m_ism->m_signalFeatures.connect(boost::bind(&TrainingGUI::signalFeatures, this, _1));
    m_ism->m_signalMaxima.connect(boost::bind(&TrainingGUI::signalMaxima, this, _1));
//    m_ism->m_signalCodebook.connect(boost::bind(&TrainingGUI::signalCodebook, this, _1));

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

    QPushButton* btReset = new QPushButton(this);
    connect(btReset, SIGNAL(clicked()), this, SLOT(reset()));
    btReset->setText("Reset");

    QPushButton* btDatasetInfo = new QPushButton(this);
    connect(btDatasetInfo, SIGNAL(clicked()), this, SLOT(addDatasetInfo()));
    btDatasetInfo->setText("Add Dataset Info");

    QVBoxLayout* layout = new QVBoxLayout();
    layout->addWidget(btClose);
    layout->addWidget(btReset);
    layout->addWidget(btDatasetInfo);

    QGroupBox* groupBox = new QGroupBox(this);
    groupBox->setTitle("Application");
    groupBox->setLayout(layout);
    return groupBox;
}

QGroupBox* TrainingGUI::createNavigatorTraining()
{
     // NOTE temporarily disabling ROS
//    m_btPauseResume = new QPushButton(this);
//    connect(m_btPauseResume, SIGNAL(clicked()), SLOT(pauseResume()));
//    m_btPauseResume->setText("Pause");

    QPushButton* btClear = new QPushButton(this);
    btClear->setText("Clear ISM");
    connect(btClear, SIGNAL(clicked()), this, SLOT(clearISM()));

    QPushButton* btLoadConfig = new QPushButton(this);
    connect(btLoadConfig, SIGNAL(clicked()), this, SLOT(loadConfig()));
    btLoadConfig->setText("Load Train Config");

    QPushButton* btAddModel = new QPushButton(this);
    connect(btAddModel, SIGNAL(clicked()), this, SLOT(addModel()));
    btAddModel->setText("Add Cloud");

    QPushButton* btTrain = new QPushButton(this);
    btTrain->setText("Train ISM");
    connect(btTrain, SIGNAL(clicked()), this, SLOT(trainModel()));

    QPushButton* btSave = new QPushButton(this);
    btSave->setText("Save ISM");
    connect(btSave, SIGNAL(clicked()), this, SLOT(save()));

    QVBoxLayout* layout = new QVBoxLayout();
//    layout->addWidget(m_btPauseResume); // NOTE temporarily disabling ROS
    layout->addWidget(btClear);
    layout->addWidget(btLoadConfig);
    layout->addWidget(btAddModel);
    layout->addWidget(btTrain);
    layout->addWidget(btSave);

    QGroupBox* groupBox = new QGroupBox(this);
    groupBox->setTitle("Training");
    groupBox->setLayout(layout);
    return groupBox;
}

QGroupBox* TrainingGUI::createNavigatorDetect()
{
    QPushButton* btLoad = new QPushButton(this);
    btLoad->setText("Load ISM");
    connect(btLoad, SIGNAL(clicked()), this, SLOT(load()));

    QPushButton* btLoadScene = new QPushButton(this);
    connect(btLoadScene, SIGNAL(clicked()), this, SLOT(loadScene()));
    btLoadScene->setText("Load Scene");

    QPushButton* btDetect = new QPushButton(this);
    btDetect->setText("Detect");
    connect(btDetect, SIGNAL(clicked()), this, SLOT(detectISM()));

    m_chkShowFeatures = new QCheckBox(this);
    m_chkShowFeatures->setText("Show Features");
    m_chkShowFeatures->setChecked(false);

    m_chkShowKeypoints = new QCheckBox(this);
    m_chkShowKeypoints->setText("Show Keypoints");
    m_chkShowKeypoints->setChecked(false);

    m_chkShowNormals = new QCheckBox(this);
    m_chkShowNormals->setText("Show Normals");
    m_chkShowNormals->setChecked(false);
    connect(m_chkShowNormals, SIGNAL(clicked(bool)), this, SLOT(updateRenderView(bool)));

    m_chkShowAllVotes = new QCheckBox(this);
    m_chkShowAllVotes->setText("Show non-max votes");

    m_chkShowVotes = new QCheckBox(this);
    m_chkShowVotes->setText("Show Votes");
    m_chkShowVotes->setChecked(false);

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
    layout->addWidget(btLoadScene);
    layout->addWidget(btDetect);
    layout->addWidget(m_chkShowNormals);
    layout->addWidget(m_chkShowFeatures);
    layout->addWidget(m_chkShowKeypoints);
    layout->addWidget(m_chkShowVotes);
    layout->addWidget(m_chkShowAllVotes);
    layout->addWidget(m_chkShowBbAndCenters);
    layout->addWidget(m_chkShowOnlyBestMaxPerClass);
    layout->addWidget(m_minMaxVotesLabel);
    layout->addWidget(m_minMaxVotesToShowLine);
    layout->addWidget(m_onlyShowClassLabel);
    layout->addWidget(m_onlyShowClassLine);

    m_implicitShapeModel = new QGroupBox(this);
    m_implicitShapeModel->setTitle("Detection");
    m_implicitShapeModel->setLayout(layout);

    return m_implicitShapeModel;
}

void TrainingGUI::spinOnce()
{
    if (m_isLoaded)
        drawCloud();

    m_renderView->update();

// NOTE temporarily disabling ROS
//    if (!ros::ok())
//        this->close();

//    ros::spinOnce();
}

// NOTE temporarily disabling ROS
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
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

    vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);

    m_displayCloud->clear();

    // downsample
    const int downsampling = 1;
    if (m_cloud->isOrganized())
    {
        for (int i = 0; i < (int)m_cloud->width; i += downsampling)
        {
            for (int j = 0; j < (int)m_cloud->height; j += downsampling)
            {
                m_displayCloud->points.emplace_back(m_cloud->at(i, j));

                const PointNormalT& point = m_cloud->at(i,j);
                points->InsertNextPoint(point.x, point.y, point.z);
                colors->InsertNextTuple3(point.r, point.g, point.b);
            }
        }
    }
    else
    {
        for (int i = 0; i < (int)m_cloud->size(); i += downsampling)
        {
            m_displayCloud->points.emplace_back(m_cloud->at(i));

            const PointNormalT& point = m_cloud->at(i);
            points->InsertNextPoint(point.x, point.y, point.z);
            colors->InsertNextTuple3(point.r, point.g, point.b);
        }
    }

    vtkSmartPointer<vtkCellArray> conn = vtkSmartPointer<vtkCellArray>::New();
    for (vtkIdType i = 0; i < points->GetNumberOfPoints(); i++)
        conn->InsertNextCell(1, &i);

    m_pointsActor->GetProperty()->SetPointSize(2);

    m_points->SetPoints(points);
    m_points->GetPointData()->SetScalars(colors);
    m_points->SetVerts(conn);
    m_points->Modified();
}

// NOTE temporarily disabling ROS - makes this method unused
//void TrainingGUI::pauseResume()
//{
//    if (m_updateCloud) {
//        m_updateCloud = false;
//        m_btPauseResume->setText("Resume");
//    }
//    else {
//        m_updateCloud = true;
//        m_btPauseResume->setText("Pause");
//    }
//}

void TrainingGUI::reset()
{
    //m_textStatus->SetVisibility(true);

    if (m_isLoaded) {
        m_cloud->clear();
        drawCloud();
    }

    m_isLoaded = false;
    m_updateSensorCloud = true;
    // NOTE temporarily disabling ROS
    //m_btPauseResume->setText("Pause");

    m_renderView->reset();

    m_dataset_mapping.clear();
    m_use_gt_info = false;
    m_dataset_info_added = false;
    m_loaded_scene_path = "";
}


void TrainingGUI::addDatasetInfo()
{
    std::string input_file_name;
    QString filename = QFileDialog::getOpenFileName(this, "Load Dataset File", QString(), tr("TXT-Files (*.txt);;All Files (*.*)"), 0, QFileDialog::DontUseNativeDialog);
    if (!filename.isEmpty() && filename.endsWith(".txt", Qt::CaseInsensitive))
    {
        input_file_name = filename.toStdString();
    }
    else
    {
        LOG_ERROR("Could not load dataset information!");
        return;
    }

    unsigned pos = input_file_name.find_last_of('/');
    std::string path_prefix = input_file_name.substr(0, pos+1);

    // from eval_helpers_detection.h:
    std::vector<std::string> filenames;
    std::vector<std::string> gt_filenames;
    if(checkFileListDetectionTest(input_file_name))
    {
        parseFileListDetectionTest(input_file_name, filenames, gt_filenames);
    }
    else
    {
        QMessageBox::warning(this, "Error", "Only detection dataset files for test mode are supported!");
        LOG_ERROR("Could not load dataset information: wrong file selected");
        return;
    }

    if(filenames.size() == gt_filenames.size())
    {
        m_dataset_mapping.clear();
        for(unsigned i = 0; i < filenames.size(); i++)
        {
            m_dataset_mapping.insert({path_prefix + filenames[i], path_prefix + gt_filenames[i]});
        }
        m_dataset_info_added = true;
    }

    if (filenames.size() > 0 && gt_filenames.size() > 0)
    {
        // load label information from training
        class_labels_rmap = m_ism->getClassLabels();
        instance_labels_rmap = m_ism->getInstanceLabels();
        instance_to_class_map = m_ism->getInstanceClassMap();
        // populate maps with loaded data
        for(auto &elem : class_labels_rmap)
        {
            class_labels_map.insert({elem.second, elem.first});
        }
        for(auto &elem : instance_labels_rmap)
        {
            instance_labels_map.insert({elem.second, elem.first});
        }

        // init label usage: class/instance or both
        initLabelUsage(m_ism->isInstancePrimaryLabel());
    }
    else
    {
        // some kind of error message
        m_dataset_info_added = false;
    }

    loadGTInfoForScene();
}


void TrainingGUI::loadGTInfoForScene()
{
    if(m_loaded_scene_path != "")
    {
        // load ground-truth information for loaded scene
        if(m_dataset_info_added)
        {
            if(m_dataset_mapping.find(m_loaded_scene_path) != m_dataset_mapping.end())
            {
                m_gt_file = m_dataset_mapping[m_loaded_scene_path];
                m_gt_objects = parseGtFile(m_gt_file);
                m_use_gt_info = true;
            }
            else
            {
                LOG_ERROR("Loaded scene path\"" << m_loaded_scene_path << "\" not found in the dataset map!");
                LOG_ERROR("Detected object boxes will NOT be colored as true and false positives!");
                QMessageBox::warning(this, "Error", "Loaded scene does not match any in loaded dataset info!");
                m_use_gt_info = false;
            }
        }
    }
}


void TrainingGUI::addModel()
{
    m_updateSensorCloud = false;

    ModelDlg dlg;
    dlg.exec();

    if (dlg.isValid())
    {
        // get data
        unsigned classId = dlg.getClassId();
        const QString& filename = dlg.getFilename();

        // load the model
        pcl::PointCloud<PointNormalT>::Ptr cloud(new pcl::PointCloud<PointNormalT>());

        if (filename.endsWith(".pcd", Qt::CaseInsensitive)) {
            if (pcl::io::loadPCDFile(filename.toStdString(), *cloud) < 0)
                QMessageBox::warning(this, "Error", "Could not load PCD file!");
            else {
                m_ism->addTrainingModel(filename.toStdString(), classId, classId); // TODO VS: for now setting instance id to class id
                m_cloud->clear();
                m_detectCloud->clear();
                pcl::copyPointCloud(*cloud, *m_cloud);
                pcl::copyPointCloud(*cloud, *m_detectCloud);
                m_isLoaded = true;
            }
        }
        else if (filename.endsWith(".ply", Qt::CaseInsensitive)) {
            if (pcl::io::loadPLYFile(filename.toStdString(), *cloud) < 0)
                QMessageBox::warning(this, "Error", "Could not load PLY file!");
            else {
                m_ism->addTrainingModel(filename.toStdString(), classId, classId); // TODO VS: for now setting instance id to class id
                m_cloud->clear();
                m_detectCloud->clear();
                pcl::copyPointCloud(*cloud, *m_cloud);
                pcl::copyPointCloud(*cloud, *m_detectCloud);
                m_isLoaded = true;
            }
        }
        else
            QMessageBox::warning(this, "Error", "Invalid file extension");
    }

    m_updateSensorCloud = true;
}

void TrainingGUI::loadScene()
{
    m_updateSensorCloud = false;

    QString filename = QFileDialog::getOpenFileName(this, "Load Scene", QString(), tr("PCD-Files (*.pcd);;PLY-Files (*.ply);;All Files (*.*)"), 0, QFileDialog::DontUseNativeDialog);

    m_detectCloud->clear();
    if (!filename.isEmpty()) {
        if (filename.endsWith(".pcd", Qt::CaseInsensitive)) {
            if (pcl::io::loadPCDFile(filename.toStdString(), *m_detectCloud) < 0)
                QMessageBox::warning(this, "Error", "Could not load PCD file!");
            else {
                m_cloud->clear();
                pcl::copyPointCloud(*m_detectCloud, *m_cloud);
                m_isLoaded = true;
                m_loaded_scene_path = filename.toStdString();
            }
        }
        else if (filename.endsWith(".ply", Qt::CaseInsensitive)) {
            if (pcl::io::loadPLYFile(filename.toStdString(), *m_detectCloud) < 0)
                QMessageBox::warning(this, "Error", "Could not load PLY file!");
            else {
                m_cloud->clear();
                pcl::copyPointCloud(*m_detectCloud, *m_cloud);
                m_isLoaded = true;
                m_loaded_scene_path = filename.toStdString();
            }
        }
        else
            QMessageBox::warning(this, "Error", "Invalid file extension");
    }

    m_updateSensorCloud = true;

    // load ground-truth information for loaded scene
    loadGTInfoForScene();
}


void TrainingGUI::loadConfig()
{
    m_updateSensorCloud = false;

    QString filename = QFileDialog::getOpenFileName(this, "Load Train Config", QString(), tr("ISM-Files (*.ism);;All Files (*.*)"), 0, QFileDialog::DontUseNativeDialog);

    if (!filename.isEmpty())
    {
        if (filename.endsWith(".ism", Qt::CaseInsensitive))
        {
            m_ism->readObject(filename.toStdString(), true);
        }
        else
            QMessageBox::warning(this, "Error", "Invalid file extension");
    }

    m_updateSensorCloud = true;
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


void TrainingGUI::trainModel()
{
    m_updateSensorCloud = false;

    m_renderView->reset();
    m_renderView->setStatus("Training...");

    m_navGeneral->setEnabled(false);
    m_navISM->setEnabled(false);
    m_cloud->clear();

    if (m_isLoaded)
    {
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
    else
    {
        QMessageBox::warning(this, "Warning", "No training models loaded.");
        trainingFinished(false);
    }
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
    m_updateSensorCloud = false;

    m_renderView->reset();
    m_renderView->setStatus("Detecting...");

    m_navGeneral->setEnabled(false);
    m_navISM->setEnabled(false);

    if (m_isLoaded)
    {
        // detect object
        try {
            m_isDetecting = true;

            // start detection on a separate thread
            QThread* thread = new QThread;
            ISMWorker* worker = new ISMWorker;
            worker->setImplicitShapeModel(m_ism);
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
    if (!m_isDetecting)
    {
        m_cloud->clear();
        pcl::copyPointCloud(*pointCloud, *m_cloud);
    }
}

void TrainingGUI::signalBoundingBox(const ism3d::Utils::BoundingBox& box)
{
    m_boundingBox = box;
    addBoundingBox(box, false, false);
}

void TrainingGUI::addBoundingBox(const ism3d::Utils::BoundingBox& box, const bool tp, const bool fp)
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
    if (fp && !tp)
        bboxActor->GetProperty()->SetColor(255, 0, 0);
    if (tp && !fp)
        bboxActor->GetProperty()->SetColor(0, 200, 0);
    if (tp == fp)
        bboxActor->GetProperty()->SetColor(0, 0, 255);
    bboxActor->GetProperty()->SetLineWidth(2);
    bboxActor->GetProperty()->SetRepresentationToWireframe();

    m_renderView->lock();
    m_renderView->getRendererScene()->AddActor(bboxActor);
    m_renderView->getRendererTop()->AddActor(bboxActor);
    m_renderView->getRendererSide()->AddActor(bboxActor);
    m_renderView->getRendererFront()->AddActor(bboxActor);
    m_renderView->unlock();
}

void TrainingGUI::updateRenderView(bool state)
{
    // TODO VS: if needed handle all checkboxes to update the view without pressing "train" or "detect"
    // NOTE: this already works for the m_pointsActor and the m_normalsActor
    // however, the boundig boxes are only stored inside the m_renderView and must be kept in a separate list
    // same for features etc.

//    m_renderView->reset();
//    m_renderView->addActorToAll(m_pointsActor);

//    if(m_chkShowNormals->isChecked())
//    {
//        m_renderView->getRendererScene(true)->AddActor(m_normalsActor);
//    }
}


void TrainingGUI::signalNormals(pcl::PointCloud<ism3d::PointT>::ConstPtr pointCloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
    vtkSmartPointer<vtkPoints> line = vtkSmartPointer<vtkPoints>::New();

    for (int i = 0; i < normals->size(); i++) {
        const ism3d::PointT& point = pointCloud->at(i);
        const pcl::Normal& normal = normals->at(i);

        Eigen::Vector3f nn = normal.getNormalVector3fMap();
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
    m_normals->clear();
    pcl::copyPointCloud(*normals, *m_normals);

    // Create a polydata to store everything in
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();

    polyData->SetPoints(line);
    polyData->SetLines(cells);

    // Setup actor and mapper
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polyData);

    m_normalsActor = vtkSmartPointer<vtkActor>::New();
    m_normalsActor->SetMapper(mapper);
    m_normalsActor->GetProperty()->SetColor(0, 0, 255);
    m_normalsActor->GetProperty()->SetLineWidth(2);
    if(m_chkShowNormals->isChecked())
    {
        m_renderView->getRendererScene(true)->AddActor(m_normalsActor);
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

        vtkSmartPointer<vtkAxesActor> featuresActor = vtkSmartPointer<vtkAxesActor>::New();
        featuresActor->SetUserTransform(transform);
        featuresActor->AxisLabelsOff();
        featuresActor->SetConeRadius(0);
        featuresActor->GetXAxisShaftProperty()->SetLineWidth(2);
        featuresActor->GetYAxisShaftProperty()->SetLineWidth(2);
        featuresActor->GetZAxisShaftProperty()->SetLineWidth(2);
        featuresActor->SetTotalLength(0.1, 0.1, 0.1);

        if(m_chkShowFeatures->isChecked())
        {
            m_renderView->getRendererScene(true)->AddActor(featuresActor);
        }

        // add a sphere for each keypoint
        vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New();
        sphere->SetCenter(feature.x, feature.y, feature.z);
        sphere->SetRadius(0.02);
        vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputConnection(sphere->GetOutputPort());
        vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        //actor->GetProperty()->SetRepresentationToWireframe();
        actor->GetProperty()->SetLighting(false);
        actor->GetProperty()->SetColor(0, 0.75, 0);

        if(m_chkShowKeypoints->isChecked())
        {
            m_renderView->getRendererScene()->AddActor(actor);
        }
    }

    m_renderView->unlock();
}

void TrainingGUI::signalMaxima(std::vector<ism3d::VotingMaximum> maxima)
{
    m_renderView->lock();

    // TODO VS: clean up debug output later
    std::cout << "--- debug   max size: " << maxima.size() << std::endl;
    std::cout << "--- debug   gt info: " << m_use_gt_info << std::endl;
    std::cout << "--- debug   gt file: " << m_gt_file << std::endl;

    std::cout << "--- debug   gt objects: " << m_gt_objects.size() << std::endl;
    for(auto x : m_gt_objects)
    {
        x.print();
    }

    std::vector<int> tp_list(maxima.size(), 0);
    std::vector<int> fp_list(maxima.size(), 0);
    if(m_use_gt_info)
    {
        // collect all detections
        std::vector<DetectionObject> detected_objects;
        for (int i = 0; i < (int)maxima.size(); i++)
        {
            DetectionObject detected_obj = convertMaxToObj(maxima[i], m_gt_file);
            detected_objects.push_back(std::move(detected_obj));
        }
        float dist_threshold = m_ism->getDetectionThreshold();
        std::map<unsigned, float> dist_thresholds = m_ism->getDetectionThreshold();
        std::tie(std::ignore, std::ignore, std::ignore, std::ignore, std::ignore,
                         tp_list, fp_list) = computeMetrics(m_gt_objects,
                                                            detected_objects,
                                                            dist_threshold);
        std::cout << "--- debug   det objects: " << detected_objects.size() << std::endl;
        for(auto x : detected_objects)
        {
            x.print();
        }
    }

    // TODO --- debug
    for(auto x : tp_list)
        std::cout << "tp: " << x << std::endl;
    std::cout << std::endl;
    for(auto x : fp_list)
        std::cout << "fp: " << x << std::endl;


    const std::map<unsigned, std::vector<ism3d::Vote> >& votes = m_ism->getVoting()->getVotes();
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

    // Setup colors
    vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(4);

    // enable or disable visualization of votes
    if(m_chkShowVotes->isChecked())
    {
        // save vote indices
        std::map<unsigned, std::vector<ism3d::Vote> > allVotes;
        for (int i = 0; i < (int)maxima.size(); i++)
        {
            const std::vector<ism3d::Vote> cur_max_votes = maxima[i].votes;
            std::vector<ism3d::Vote>& mapEntry = allVotes[maxima[i].classId];
            mapEntry.insert(mapEntry.end(), cur_max_votes.begin(), cur_max_votes.end());
        }

        for (std::map<unsigned, std::vector<ism3d::Vote> >::const_iterator it = votes.begin();
             it != votes.end(); it++)
        {
            unsigned classId = it->first;
            const std::vector<ism3d::Vote>& classVotes = it->second;

            for (int i = 0; i < (int)classVotes.size(); i++) {
                ism3d::Vote vote = classVotes[i];

                double classColor[3];
                getColor(classId, classColor[0], classColor[1], classColor[2]);

                //float alpha = 0.1f + (vote.weight * 0.9f);
                //float alpha = vote.weight;
                float alpha = 1.0f;

                bool isMaximumVote = true;
                const auto it = allVotes.find(classId);
                if (it == allVotes.end()) {
                    alpha = 0.3f;
                    isMaximumVote = false;
                }
                else {
                    const std::vector<ism3d::Vote>& cur_votes = it->second;
                    // TODO VS: if something is wrong with vote colors, check this
                    if (std::find(cur_votes.begin(), cur_votes.end(), vote) == cur_votes.end()) {
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
                colors->InsertNextTypedTuple(color);

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
                lineColors->InsertNextTypedTuple(color);
                lineColors->InsertNextTypedTuple(color);

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
    m_maxima_classes.resize(votes.size(), false);

    // enable / disable bounding box and center visualization
    if(m_chkShowBbAndCenters->isChecked())
    {
        //m_rendererScene->AddActor(actor);
        m_renderView->getRendererScene(true)->AddActor(actor);

        // add bounding boxes to represent object positions
        int min_votes = m_minMaxVotesToShowLine->text().toInt();
        LOG_INFO("Only maxima with at least " << min_votes << " votes will be displayed!");

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
            LOG_INFO("Only maxima of class IDs " << temp_text << " will be displayed!");
        }
        else
        {
            LOG_INFO("All class IDs will be displayed!");
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
                    continue; // one max with this class id has been displayed already
                }
                m_maxima_classes.at(max.classId) = true; // mark this class as shown
            }

            if(max.votes.size() < min_votes) // only report maxima with at least X votes
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
            bool tp = tp_list[i] == 1 ? true : false;
            bool fp = fp_list[i] == 1 ? true : false;
            addBoundingBox(max.boundingBox, tp, fp);
        }
    }

    // TEMP STUFF
//    // draw trajectories
//    const std::vector<std::vector<Eigen::Vector3f> >& trajectories = ((ism3d::VotingMeanShift*)m_ism->getVoting())->getTrajectories;
//    for (int i = 0; i < (int)trajectories.size(); i++) {
//        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
//        const std::vector<Eigen::Vector3f>& path = trajectories[i];
//        for (int j = 0; j < (int)path.size(); j++)
//            points->InsertNextPoint(&path[j][0]);

//        vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();
//        polyLine->GetPointIds()->SetNumberOfIds(path.size());
//        for(unsigned int i = 0; i < path.size(); i++)
//            polyLine->GetPointIds()->SetId(i,i);

//        // Create a cell array to store the lines in and add the lines to it
//        vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
//        cells->InsertNextCell(polyLine);

//        // Create a polydata to store everything in
//        vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();

//        // Add the points to the dataset
//        polyData->SetPoints(points);

//        // Add the lines to the dataset
//        polyData->SetLines(cells);

//        // Setup actor and mapper
//        vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
//        mapper->SetInputData(polyData);

//        vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
//        actor->SetMapper(mapper);
//        actor->GetProperty()->SetColor(255, 255, 0);
//        m_renderView->getRendererScene(true)->AddActor(actor);
//    }

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
