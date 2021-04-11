#include "ground_truth_gui.h"

#include "../vtk_utils/render_view.h"
#include "../implicit_shape_model/utils/utils.h"
#include "../implicit_shape_model/utils/normal_orientation.h"

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
#include <QFileInfo>

// PCL
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/features/normal_3d_omp.h>

// VTK
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkBox.h>

#include <jsoncpp/json/json.h>
#include <jsoncpp/json/writer.h>

#include <Eigen/Geometry>
#include <Eigen/Core>

GroundTruthGUI::GroundTruthGUI(QWidget* parent)
    : QWidget(parent),
      m_isLoaded(false)
{
    //srand(time(0));
    srand(0);

    // create pcl data
    m_cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
    m_displayCloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());

    setWindowTitle("ISM3D - Ground Truth GUI");

    // create vtk views
    m_renderView = new RenderView(this);
    connect(m_renderView, SIGNAL(moveXY(float, float)), SLOT(moveXY(float, float)));
    connect(m_renderView, SIGNAL(moveYZ(float, float)), SLOT(moveYZ(float, float)));
    connect(m_renderView, SIGNAL(moveXZ(float, float)), SLOT(moveXZ(float, float)));
    connect(m_renderView, SIGNAL(rotateX(float, Eigen::Vector3f)), SLOT(rotateX(float, Eigen::Vector3f)));
    connect(m_renderView, SIGNAL(rotateY(float, Eigen::Vector3f)), SLOT(rotateY(float, Eigen::Vector3f)));
    connect(m_renderView, SIGNAL(rotateZ(float, Eigen::Vector3f)), SLOT(rotateZ(float, Eigen::Vector3f)));

    // add points
    m_points = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPolyDataMapper> pointsMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    pointsMapper->SetInputData(m_points);
    m_pointsActor = vtkSmartPointer<vtkActor>::New();
    m_pointsActor->SetMapper(pointsMapper);
    //m_pointsActor->GetProperty()->SetPointSize(0.7);

    //m_renderView->addActorToScene(m_pointsActor);
    m_renderView->addActorToAll(m_pointsActor);

    // create navigator panes
    QVBoxLayout* navigatorLayout = new QVBoxLayout();
    navigatorLayout->addWidget(createNavigatorApplication());
    navigatorLayout->addWidget(createNavigatorGeneral());
    navigatorLayout->addItem(new QSpacerItem(150, 0, QSizePolicy::Minimum, QSizePolicy::Expanding));

    // put it all together
    QGridLayout* layout = new QGridLayout(this);    // NOTE: produces a warning
    layout->addWidget(m_renderView, 0,  0);
    layout->addLayout(navigatorLayout, 0, 1);
    this->setLayout(layout);

    resize(1024, 768);
}

GroundTruthGUI::~GroundTruthGUI()
{
}

QGroupBox* GroundTruthGUI::createNavigatorApplication()
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

QGroupBox* GroundTruthGUI::createNavigatorGeneral()
{
    QPushButton* btReset = new QPushButton(this);
    connect(btReset, SIGNAL(clicked()), this, SLOT(reset()));
    btReset->setText("Reset");

    QPushButton* btLoadScene = new QPushButton(this);
    connect(btLoadScene, SIGNAL(clicked()), this, SLOT(loadScene()));
    btLoadScene->setText("Load Scene");

    QPushButton* btAddModel = new QPushButton(this);
    connect(btAddModel, SIGNAL(clicked()), this, SLOT(addModel()));
    btAddModel->setText("Add Model");

    QPushButton* btRemoveModel = new QPushButton(this);
    connect(btRemoveModel, SIGNAL(clicked()), this, SLOT(removeModel()));
    btRemoveModel->setText("Remove Model");

    QPushButton* btExportGroundTruth = new QPushButton(this);
    connect(btExportGroundTruth, SIGNAL(clicked()), this, SLOT(exportGroundTruth()));
    btExportGroundTruth->setText("Export Ground-Truth");

    QPushButton* btComputeNormals = new QPushButton(this);
    connect(btComputeNormals, SIGNAL(clicked()), this, SLOT(computeNormals()));
    btComputeNormals->setText("Compute Normals");

    QLabel* lbNormalRadius = new QLabel(this);
    lbNormalRadius->setText("Normal Radius:");

    m_normalRadiusLineEdit = new QLineEdit(this);
    m_normalRadiusLineEdit->setToolTip(QString("Normals radius [m]"));
    m_normalRadiusLineEdit->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    m_normalRadiusLineEdit->setText("0.05");

    QLabel* lbNormalMethod = new QLabel(this);
    lbNormalMethod->setText("Normal Method:");

    m_normalMethodLineEdit = new QLineEdit(this);
    m_normalMethodLineEdit->setToolTip(QString("Normals method (0 or 1): 0 orient toward viewpoint, 1 orient away from centroid"));
    m_normalMethodLineEdit->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    m_normalMethodLineEdit->setText("0");

    QVBoxLayout* layout = new QVBoxLayout();
    layout->addWidget(btReset);
    layout->addWidget(btLoadScene);
    layout->addWidget(btAddModel);
    layout->addWidget(btRemoveModel);
    layout->addWidget(btExportGroundTruth);
    layout->addWidget(btComputeNormals);
    layout->addWidget(lbNormalRadius);
    layout->addWidget(m_normalRadiusLineEdit);
    layout->addWidget(lbNormalMethod);
    layout->addWidget(m_normalMethodLineEdit);

    QGroupBox* groupBox = new QGroupBox(this);
    groupBox->setTitle("General");
    groupBox->setLayout(layout);
    return groupBox;
}

void GroundTruthGUI::drawCloud()
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

                const PointT& point = m_cloud->at(i, j);
                points->InsertNextPoint(point.x, point.y, point.z);
                //colors->InsertNextTuple3(255, 255, 255);
                colors->InsertNextTuple3(0, 0, 0);
            }
        }
    }
    else
    {
        for (int i = 0; i < (int)m_cloud->size(); i += downsampling)
        {
            m_displayCloud->points.emplace_back(m_cloud->at(i));

            const PointT& point = m_cloud->at(i);
            points->InsertNextPoint(point.x, point.y, point.z);
            //colors->InsertNextTuple3(255, 255, 255);
            colors->InsertNextTuple3(0, 0, 0);
        }
    }


    vtkSmartPointer<vtkCellArray> conn = vtkSmartPointer<vtkCellArray>::New();
    for (vtkIdType i = 0; i < points->GetNumberOfPoints(); i++)
        conn->InsertNextCell(1, &i);

    m_points->SetPoints(points);
    m_points->GetPointData()->SetScalars(colors);
    m_points->SetVerts(conn);
    m_points->Modified();
}

void GroundTruthGUI::reset()
{
    if (m_isLoaded) {
        m_cloud->clear();
        drawCloud();
    }

    m_sceneFile.clear();
    m_models.clear();
    m_modelFiles.clear();
    m_transforms.clear();

    m_isLoaded = false;

    m_renderView->reset();
    m_renderView->update();
}

void GroundTruthGUI::loadScene()
{
    QString filename = QFileDialog::getOpenFileName(this, "Load Scene", QString(), tr("PCD-Files (*.pcd);;PLY-Files (*.ply);;All Files (*.*)"), 0, QFileDialog::DontUseNativeDialog);

    if (!filename.isEmpty()) {
        m_cloud->clear();

        if (filename.endsWith(".pcd", Qt::CaseInsensitive)) {
            if (pcl::io::loadPCDFile(filename.toStdString(), *m_cloud) < 0)
                QMessageBox::warning(this, "Error", "Could not load PCD file!");
            {
                m_isLoaded = true;
                m_sceneFile = filename.toStdString();
                drawCloud();
                m_renderView->update();
            }
        }
        else if (filename.endsWith(".ply", Qt::CaseInsensitive)) {
            if (pcl::io::loadPLYFile(filename.toStdString(), *m_cloud) < 0)
                QMessageBox::warning(this, "Error", "Could not load PLY file!");
            else {
                m_isLoaded = true;
                m_sceneFile = filename.toStdString();
                drawCloud();
                m_renderView->update();
            }
        }
    }
}

void GroundTruthGUI::computeNormals()
{
    pcl::PointCloud<pcl::Normal>::Ptr cloudNormals(new pcl::PointCloud<pcl::Normal>());
    pcl::search::KdTree<PointT>::Ptr search(new pcl::search::KdTree<PointT>());

    std::cout << "computing normals" << std::endl;

    float normal_radius = m_normalRadiusLineEdit->text().toFloat();

    // compute normals
    pcl::NormalEstimationOMP<PointT, pcl::Normal> normalEst;
    normalEst.setInputCloud(m_displayCloud);
    normalEst.setSearchMethod(search);
    normalEst.setRadiusSearch(normal_radius);
    normalEst.setNumberOfThreads(4);

    std::cout << "computing consistent orientation" << std::endl;

    int consistent_normals_method = m_normalMethodLineEdit->text().toInt();
    if(consistent_normals_method == 0)
    {
        // orient consistently towards the view point
        normalEst.setViewPoint(0,0,0);
        normalEst.compute(*cloudNormals);
    }
    else if(consistent_normals_method == 1)
    {
        // move model to origin, then point normals away from origin
        pcl::PointCloud<PointT>::Ptr model_no_centroid(new pcl::PointCloud<PointT>());
        pcl::copyPointCloud(*m_displayCloud, *model_no_centroid);

        // compute the object centroid
        Eigen::Vector4f centroid4f;
        pcl::compute3DCentroid(*model_no_centroid, centroid4f);
        Eigen::Vector3f centroid(centroid4f[0], centroid4f[1], centroid4f[2]);
        // remove centroid for normal computation
        for(PointT& point : model_no_centroid->points)
        {
            point.x -= centroid.x();
            point.y -= centroid.y();
            point.z -= centroid.z();
        }
        normalEst.setInputCloud(model_no_centroid);
        normalEst.setViewPoint(0,0,0);
        normalEst.compute(*cloudNormals);
        // invert normals
        for(pcl::Normal& norm : cloudNormals->points)
        {
            norm.normal_x *= -1;
            norm.normal_y *= -1;
            norm.normal_z *= -1;
        }
    }

    std::cout << "concatenating" << std::endl;

    pcl::PointCloud<PointNormalT>::Ptr newCloud(new pcl::PointCloud<PointNormalT>());
    pcl::concatenateFields(*m_displayCloud, *cloudNormals, *newCloud);

    QString filename = QFileDialog::getSaveFileName(this, "Save Scene", QString(), tr("PCD-Files (*.pcd);;All Files (*.*)"), 0, QFileDialog::DontUseNativeDialog);

    if (!filename.isEmpty())
    {
        if(!filename.contains(QString(".pcd")))
        {
            filename.append(".pcd");
        }
        pcl::io::savePCDFileBinary(filename.toStdString(), *newCloud);
    }
}

void GroundTruthGUI::addModel()
{
    QString filename = QFileDialog::getOpenFileName(this, "Add Model", QString(), tr("PCD-Files (*.pcd);;PLY-Files (*.ply);;All Files (*.*)"), 0, QFileDialog::DontUseNativeDialog);

    if (!filename.isEmpty()) {
        pcl::PointCloud<PointT>::Ptr model(new pcl::PointCloud<PointT>());

        bool loaded = false;
        if (filename.endsWith(".pcd", Qt::CaseInsensitive)) {
            if (pcl::io::loadPCDFile(filename.toStdString(), *model) < 0)
                QMessageBox::warning(this, "Error", "Could not load PCD file!");
            {
                loaded = true;
            }
        }
        else if (filename.endsWith(".ply", Qt::CaseInsensitive)) {
            if (pcl::io::loadPLYFile(filename.toStdString(), *model) < 0)
                QMessageBox::warning(this, "Error", "Could not load PLY file!");
            else {
                loaded = true;
            }
        }

        if (loaded) {
            m_models.push_back(model);
            m_modelFiles.push_back(filename.toStdString());

            vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
            vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
            colors->SetNumberOfComponents(3);

            for (size_t i = 0; i < model->size(); i++) {
                const PointT& point = model->at(i);
                points->InsertNextPoint(point.x, point.y, point.z);
                //colors->InsertNextTuple3(255, 255, 255);
                colors->InsertNextTuple3(0, 0, 0);
            }

            vtkSmartPointer<vtkCellArray> conn = vtkSmartPointer<vtkCellArray>::New();
            for (vtkIdType i = 0; i < points->GetNumberOfPoints(); i++)
                conn->InsertNextCell(1, &i);

            vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
            vtkSmartPointer<vtkPolyDataMapper> polyDataMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
            polyDataMapper->SetInputData(polyData);

            polyData->SetPoints(points);
            polyData->GetPointData()->SetScalars(colors);
            polyData->SetVerts(conn);
            polyData->Modified();

            vtkSmartPointer<vtkActor> modelActor = vtkSmartPointer<vtkActor>::New();
            modelActor->SetMapper(polyDataMapper);

            vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
            modelActor->SetUserTransform(transform);
            m_transforms.push_back(transform);

            m_modelActors.push_back(modelActor);
            //m_renderView->addActorToAll(modelActor);
            m_renderView->getRendererScene()->AddActor(modelActor);
            m_renderView->getRendererTop()->AddActor(modelActor);
            m_renderView->getRendererSide()->AddActor(modelActor);
            m_renderView->getRendererFront()->AddActor(modelActor);
        }
    }
}

void GroundTruthGUI::removeModel()
{
    if (m_models.size() > 0) {
        // remove from render view
        vtkSmartPointer<vtkActor> actor = m_modelActors.back();
        m_renderView->getRendererScene()->RemoveActor(actor);
        m_renderView->getRendererTop()->RemoveActor(actor);
        m_renderView->getRendererSide()->RemoveActor(actor);
        m_renderView->getRendererFront()->RemoveActor(actor);
        m_renderView->update();

        // erase
        m_modelActors.erase(m_modelActors.end());
        m_models.erase(m_models.end());
        m_modelFiles.erase(m_modelFiles.end());
        m_transforms.erase(m_transforms.end());
    }
}

void GroundTruthGUI::exportGroundTruth()
{
    if (m_sceneFile.empty()) {
        QMessageBox::warning(this, "Warning", "No scene file has been loaded, cannot export ground truth data.");
        return;
    }

    QString filename = QFileDialog::getSaveFileName(this, "Export Ground-Truth", QString(), tr("TXT-Files (*.txt);;All Files (*)"));

    if (!filename.isEmpty()) {
        std::ofstream file;
        if(!filename.contains(QString(".txt")))
        {
            filename.append(".txt");
        }
        file.open(filename.toStdString().c_str(), ios::out);
        QString scene = QString::fromStdString(m_sceneFile);
        file << "ISM3D ground truth data, scene: \"" << QFileInfo(scene).fileName().toStdString() << "\"\n";

        for (int i = 0; i < (int)m_models.size(); i++) {
            QString modelFilename = QString::fromStdString(m_modelFiles[i]);
            file << "\"" << QFileInfo(modelFilename).fileName().toStdString() << "\", ";

            // retrieve model
            const pcl::PointCloud<PointT>::ConstPtr model = m_models[i];
            vtkSmartPointer<vtkTransform> transform = m_transforms[i];
            pcl::PointCloud<PointNormalT>::Ptr newModel(new pcl::PointCloud<PointNormalT>());
            pcl::copyPointCloud(*model, *newModel);

            // compute bounding box
            ism3d::Utils::BoundingBox box = ism3d::Utils::computeMVBB<PointNormalT>(newModel);

            Eigen::Matrix3f rotMat;
            ism3d::Utils::quat2Matrix(box.rotQuat, rotMat);

            Eigen::Matrix4f localMat = Eigen::Matrix4f::Identity();
            localMat.block<3, 3>(0, 0) = rotMat;
            localMat.block<3, 1>(0, 3) = box.position;

            vtkSmartPointer<vtkMatrix4x4> tempMat = vtkSmartPointer<vtkMatrix4x4>::New();
            transform->GetMatrix(tempMat);
            Eigen::Matrix4f transformMat;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    transformMat(i, j) = (float)tempMat->Element[i][j];

            // compute new transformation
            Eigen::Matrix4f resultMat = transformMat * localMat;

            // store new position
            box.position[0] = resultMat(0, 3);
            box.position[1] = resultMat(1, 3);
            box.position[2] = resultMat(2, 3);

            // store new rotation quaternion
            Eigen::Matrix3f resultRotMat = resultMat.block<3, 3>(0, 0);
            boost::math::quaternion<float> quat;
            ism3d::Utils::matrix2Quat(resultRotMat, quat);
            box.rotQuat = quat;

            // save data
            file << box.position[0] << ", ";
            file << box.position[1] << ", ";
            file << box.position[2] << ", ";
            file << box.size[0] << ", ";
            file << box.size[1] << ", ";
            file << box.size[2] << ", ";
            file << box.rotQuat.R_component_1() << ", ";
            file << box.rotQuat.R_component_2() << ", ";
            file << box.rotQuat.R_component_3() << ", ";
            file << box.rotQuat.R_component_4() << "\n";

            {
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
                bboxActor->GetProperty()->SetColor(0, 0, 0);
                bboxActor->GetProperty()->SetLineWidth(2);
                bboxActor->GetProperty()->SetRepresentationToWireframe();

                m_renderView->lock();
                m_renderView->getRendererScene()->AddActor(bboxActor);
                m_renderView->unlock();
                m_renderView->update();
            }
        }
        file.close();
    }
}

void GroundTruthGUI::moveXY(float value1, float value2)
{
    if (m_transforms.size() > 0) {
        vtkSmartPointer<vtkTransform> lastTrans = m_transforms.back();

        lastTrans->PostMultiply();
        lastTrans->Translate(value1, value2, 0);
        m_renderView->update();
    }
}

void GroundTruthGUI::moveYZ(float value1, float value2)
{
    if (m_transforms.size() > 0) {
        vtkSmartPointer<vtkTransform> lastTrans = m_transforms.back();

        lastTrans->PostMultiply();
        lastTrans->Translate(0, value1, value2);
        m_renderView->update();
    }
}

void GroundTruthGUI::moveXZ(float value1, float value2)
{
    if (m_transforms.size() > 0) {
        vtkSmartPointer<vtkTransform> lastTrans = m_transforms.back();

        lastTrans->PostMultiply();
        lastTrans->Translate(value1, 0, value2);
        m_renderView->update();
    }
}

void GroundTruthGUI::rotateX(float value, Eigen::Vector3f)
{
    if (m_transforms.size() > 0) {
        vtkSmartPointer<vtkTransform> lastTrans = m_transforms.back();

        lastTrans->PostMultiply();
        double* pos = lastTrans->GetPosition();
        lastTrans->Translate(-pos[0], -pos[1], -pos[2]);
        lastTrans->RotateX(value);
        lastTrans->Translate(pos[0], pos[1], pos[2]);
        m_renderView->update();
    }
}

void GroundTruthGUI::rotateY(float value, Eigen::Vector3f)
{
    if (m_transforms.size() > 0) {
        vtkSmartPointer<vtkTransform> lastTrans = m_transforms.back();

        lastTrans->PostMultiply();
        double* pos = lastTrans->GetPosition();
        lastTrans->Translate(-pos[0], -pos[1], -pos[2]);
        lastTrans->RotateY(value);
        lastTrans->Translate(pos[0], pos[1], pos[2]);
        m_renderView->update();
    }
}

void GroundTruthGUI::rotateZ(float value, Eigen::Vector3f)
{
    if (m_transforms.size() > 0) {
        vtkSmartPointer<vtkTransform> lastTrans = m_transforms.back();

        lastTrans->PostMultiply();
        double* pos = lastTrans->GetPosition();
        lastTrans->Translate(-pos[0], -pos[1], -pos[2]);
        lastTrans->RotateZ(value);
        lastTrans->Translate(pos[0], pos[1], pos[2]);
        m_renderView->update();
    }
}
