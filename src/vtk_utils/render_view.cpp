#include "render_view.h"
#include "interactor_ortho.h"
#include "interactor_scene.h"

#include <vtkPolyDataMapper.h>
#include <vtkTextProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QComboBox>
#include <QLabel>
#include <QResizeEvent>

RenderView::RenderView(QWidget* parent)
    : QWidget(parent), m_update(true), m_resizing(false), m_stackedLayout(0)
{
    qRegisterMetaType<Eigen::Vector3f>("Eigen::Vector3f");

    m_mutex = new QMutex();

    m_resizeTimer.setSingleShot(true);
    connect(&m_resizeTimer, SIGNAL(timeout()), SLOT(resizeDone()));

    m_splitLayout = new QGridLayout();

    m_renderWidgetScene = new QVTKWidget();
    m_renderWidgetTop = new QVTKWidget();
    m_renderWidgetSide = new QVTKWidget();
    m_renderWidgetFront = new QVTKWidget();

    m_splitLayout->addWidget(m_renderWidgetScene, 0, 0);
    m_splitLayout->addWidget(m_renderWidgetTop, 0, 1);
    m_splitLayout->addWidget(m_renderWidgetSide, 1, 0);
    m_splitLayout->addWidget(m_renderWidgetFront, 1, 1);

    // front renderer
    m_cameraScene = vtkSmartPointer<vtkCamera>::New();
    //m_cameraScene->SetPosition(0, 0, -3);
    m_cameraScene->SetPosition(1, -1, -1);
    m_cameraScene->SetViewUp(0, -1, 0);
    m_cameraScene->SetFocalPoint(0, 0, 0);
    vtkSmartPointer<vtkRenderWindow> renderWindowScene =
            createRenderer(m_rendererScene,m_rendererSceneTop, m_cameraScene, m_textScene, "Scene View", m_renderWidgetScene);
    vtkSmartPointer<InteractorScene> interScene = vtkSmartPointer<InteractorScene>::New();
    interScene->setRenderView(this);
    renderWindowScene->GetInteractor()->SetInteractorStyle(interScene);

    // top renderer
    m_cameraTop = vtkSmartPointer<vtkCamera>::New();
    m_cameraTop->SetPosition(0, -20, 0);
    m_cameraTop->SetViewUp(0, 0, -1);
    m_cameraTop->SetFocalPoint(0, 0, 0);
    m_cameraTop->SetParallelProjection(true);
    m_cameraTop->Zoom(0.5);
    vtkSmartPointer<vtkRenderWindow> renderWindowTop =
            createRenderer(m_rendererTop, m_rendererTopTop, m_cameraTop, m_textTop, "Top View", m_renderWidgetTop);
    vtkSmartPointer<InteractorOrtho> interTop = vtkSmartPointer<InteractorOrtho>::New();
    interTop->setRenderView(this);
    renderWindowTop->GetInteractor()->SetInteractorStyle(interTop);

    // side renderer
    m_cameraSide = vtkSmartPointer<vtkCamera>::New();
    m_cameraSide->SetPosition(-20, 0, 0);
    m_cameraSide->SetViewUp(0, -1, 0);
    m_cameraSide->SetFocalPoint(0, 0, 0);
    m_cameraSide->SetParallelProjection(true);
    m_cameraSide->Zoom(0.5);
    vtkSmartPointer<vtkRenderWindow> renderWindowSide =
            createRenderer(m_rendererSide, m_rendererSideTop, m_cameraSide, m_textSide, "Side View", m_renderWidgetSide);
    vtkSmartPointer<InteractorOrtho> interSide = vtkSmartPointer<InteractorOrtho>::New();
    interSide->setRenderView(this);
    renderWindowSide->GetInteractor()->SetInteractorStyle(interSide);

    // front renderer
    m_cameraFront = vtkSmartPointer<vtkCamera>::New();
    m_cameraFront->SetPosition(0, 0, -20);
    m_cameraFront->SetViewUp(0, -1, 0);
    m_cameraFront->SetFocalPoint(0, 0, 0);
    m_cameraFront->SetParallelProjection(true);
    m_cameraFront->Zoom(0.5);
    vtkSmartPointer<vtkRenderWindow> renderWindowFront =
            createRenderer(m_rendererFront, m_rendererFrontTop, m_cameraFront, m_textFront, "Front View", m_renderWidgetFront);
    vtkSmartPointer<InteractorOrtho> interFront = vtkSmartPointer<InteractorOrtho>::New();
    interFront->setRenderView(this);
    renderWindowFront->GetInteractor()->SetInteractorStyle(interFront);

    // add world axes
    m_worldAxes = vtkSmartPointer<vtkAxesActor>::New();
    m_worldAxes->AxisLabelsOff();
    m_worldAxes->SetConeRadius(0.2);
    m_worldAxes->SetTotalLength(1.0, 1.0, 1.0);

    // text objects
    m_textStatus = vtkSmartPointer<vtkTextActor>::New();
    m_textStatus->GetTextProperty()->SetFontSize(16);
    m_textStatus->SetPosition(10, 10);
    m_textStatus->SetInput("Ready");
    m_textStatus->GetTextProperty()->SetColor(1, 1, 1);
    m_textStatus->GetTextProperty()->SetFontFamilyToCourier();
    m_textStatus->SetVisibility(true);

    m_renderWidgetScene->installEventFilter(this);
    m_renderWidgetTop->installEventFilter(this);
    m_renderWidgetSide->installEventFilter(this);
    m_renderWidgetFront->installEventFilter(this);

    reset();

    QVBoxLayout* layout = new QVBoxLayout();

    QHBoxLayout* top = new QHBoxLayout();
    top->addSpacerItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Minimum));
    QLabel* chooserLabel = new QLabel;
    chooserLabel->setText("Layout: ");
    top->addWidget(chooserLabel);
    QComboBox* chooser = new QComboBox();
    //chooser->addItem("Split View");
    chooser->addItem("Scene");
    chooser->addItem("Top");
    chooser->addItem("Side");
    chooser->addItem("Front");
    top->addWidget(chooser);

    layout->addLayout(top);

    m_stackedLayout = new QStackedLayout();
    layout->addLayout(m_stackedLayout);

    QWidget* splitWidget = new QWidget();
    splitWidget->setLayout(m_splitLayout);

    // UNDONE
    //m_stackedLayout->addWidget(splitWidget);
    m_stackedLayout->addWidget(m_renderWidgetScene);
    m_stackedLayout->addWidget(m_renderWidgetTop);
    m_stackedLayout->addWidget(m_renderWidgetSide);
    m_stackedLayout->addWidget(m_renderWidgetFront);

    //m_renderWidgetScene->setVisible(true);
    //m_renderWidgetScene->update();

    connect(chooser, SIGNAL(activated(int)), m_stackedLayout, SLOT(setCurrentIndex(int)));
    connect(chooser, SIGNAL(activated(int)), SLOT(changeView(int)));

    this->setLayout(layout);
}

RenderView::~RenderView()
{
}

vtkSmartPointer<vtkRenderWindow> RenderView::createRenderer(vtkSmartPointer<vtkRenderer>& renderer, vtkSmartPointer<vtkRenderer>& rendererTop,
                                                            vtkSmartPointer<vtkCamera>& camera, vtkSmartPointer<vtkTextActor>& textActor, QString text,
                                                            QVTKWidget* renderWidget)
{
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetNumberOfLayers(2);
    renderer = vtkSmartPointer<vtkRenderer>::New();
    rendererTop = vtkSmartPointer<vtkRenderer>::New();
    renderWindow->AddRenderer(rendererTop);
    rendererTop->SetLayer(1);
    renderWindow->AddRenderer(renderer);
    renderer->SetLayer(0);
    renderer->SetActiveCamera(camera);
    rendererTop->SetActiveCamera(camera);
    renderWidget->SetRenderWindow(renderWindow);

    textActor = vtkSmartPointer<vtkTextActor>::New();
    textActor->GetTextProperty()->SetFontSize(16);
    textActor->SetInput(text.toStdString().c_str());
    textActor->GetTextProperty()->SetColor(1, 1, 1);
    textActor->GetTextProperty()->SetFontFamilyToCourier();
    textActor->SetVisibility(true);
    renderer->AddActor(textActor);

    //  TEMP
    renderer->SetBackground(1.0, 1.0, 1.0);

    return renderWindow;
}

void RenderView::emitMove(float valueX, float valueY, float valueZ, InteractorOrtho* interactor)
{
    if (interactor == m_rendererFront->GetRenderWindow()->GetInteractor()->GetInteractorStyle())
        emit moveXY(valueX, valueY);
    else if (interactor == m_rendererTop->GetRenderWindow()->GetInteractor()->GetInteractorStyle())
        emit moveXZ(valueX, valueZ);
    else if (interactor == m_rendererSide->GetRenderWindow()->GetInteractor()->GetInteractorStyle())
        emit moveYZ(valueY, valueZ);
}

void RenderView::emitScale(float valueX, float valueY, float valueZ, InteractorOrtho* interactor)
{
    if (interactor == m_rendererFront->GetRenderWindow()->GetInteractor()->GetInteractorStyle())
        emit scaleXY(valueX, valueY);
    else if (interactor == m_rendererTop->GetRenderWindow()->GetInteractor()->GetInteractorStyle())
        emit scaleXZ(valueX, valueZ);
    else if (interactor == m_rendererSide->GetRenderWindow()->GetInteractor()->GetInteractorStyle())
        emit scaleYZ(valueY, valueZ);
}

void RenderView::emitRotate(float value, Eigen::Vector3f pickPoint, InteractorOrtho* interactor)
{
    if (interactor == m_rendererFront->GetRenderWindow()->GetInteractor()->GetInteractorStyle())
        emit rotateZ(value, pickPoint);
    else if (interactor == m_rendererTop->GetRenderWindow()->GetInteractor()->GetInteractorStyle())
        emit rotateY(value, pickPoint);
    else if (interactor == m_rendererSide->GetRenderWindow()->GetInteractor()->GetInteractorStyle())
        emit rotateX(value, pickPoint);
}

void RenderView::setStatus(std::string status)
{
    m_textStatus->SetInput(status.c_str());
}

void RenderView::resetStatus()
{
    m_textStatus->SetInput("Ready");
}

void RenderView::reset()
{
    // first remove all
    m_rendererSceneTop->RemoveAllViewProps();
    m_rendererScene->RemoveAllViewProps();
    m_rendererTopTop->RemoveAllViewProps();
    m_rendererTop->RemoveAllViewProps();
    m_rendererSideTop->RemoveAllViewProps();
    m_rendererSide->RemoveAllViewProps();
    m_rendererFrontTop->RemoveAllViewProps();
    m_rendererFront->RemoveAllViewProps();

    // add world axes
    //m_rendererScene->AddActor(m_worldAxes);
    m_rendererTop->AddActor(m_worldAxes);
    m_rendererSide->AddActor(m_worldAxes);
    m_rendererFront->AddActor(m_worldAxes);

    // add text
    m_rendererScene->AddActor(m_textScene);
    m_rendererTop->AddActor(m_textTop);
    m_rendererSide->AddActor(m_textSide);
    m_rendererFront->AddActor(m_textFront);
    m_rendererScene->AddActor(m_textStatus);

    // add all the other things
    for (int i = 0; i < (int)m_actorsScene.size(); i++)
        m_rendererScene->AddActor(m_actorsScene[i]);
    for (int i = 0; i < (int)m_actorsTop.size(); i++)
        m_rendererTop->AddActor(m_actorsTop[i]);
    for (int i = 0; i < (int)m_actorsSide.size(); i++)
        m_rendererSide->AddActor(m_actorsSide[i]);
    for (int i = 0; i < (int)m_actorsFront.size(); i++)
        m_rendererFront->AddActor(m_actorsFront[i]);
}

void RenderView::addActorToScene(vtkSmartPointer<vtkProp> actor)
{
    m_actorsScene.push_back(actor);
    m_rendererScene->AddActor(actor);
}

void RenderView::addActorToTop(vtkSmartPointer<vtkProp> actor)
{
    m_actorsTop.push_back(actor);
    m_rendererTop->AddActor(actor);
}

void RenderView::addActorToSide(vtkSmartPointer<vtkProp> actor)
{
    m_actorsSide.push_back(actor);
    m_rendererSide->AddActor(actor);
}

void RenderView::addActorToFront(vtkSmartPointer<vtkProp> actor)
{
    m_actorsFront.push_back(actor);
    m_rendererFront->AddActor(actor);
}

void RenderView::addActorToAll(vtkSmartPointer<vtkProp> actor)
{
    m_actorsScene.push_back(actor);
    m_rendererScene->AddActor(actor);
    m_actorsTop.push_back(actor);
    m_rendererTop->AddActor(actor);
    m_actorsSide.push_back(actor);
    m_rendererSide->AddActor(actor);
    m_actorsFront.push_back(actor);
    m_rendererFront->AddActor(actor);
}

void RenderView::update()
{
    m_renderWidgetScene->update();
    m_renderWidgetTop->update();
    m_renderWidgetSide->update();
    m_renderWidgetFront->update();
    QWidget::update();
}

bool RenderView::eventFilter(QObject* object, QEvent* event)
{
    if (event->type() == QEvent::Paint &&
            (object == m_renderWidgetScene ||
            object == m_renderWidgetTop ||
            object == m_renderWidgetSide ||
            object == m_renderWidgetFront)) {
        m_mutex->lock();
        if (m_resizing) {
            m_mutex->unlock();
            return true;
        }
        else if (!m_update) {
            m_mutex->unlock();
            return true;
        }
        m_mutex->unlock();
    }

    return false;
}

void RenderView::resizeEvent(QResizeEvent* event)
{
    if (event->type() == QResizeEvent::Resize) {
        m_resizing = true;
        m_resizeTimer.start(500);

        QSize size = m_renderWidgetScene->geometry().size();
        m_textScene->SetPosition(10, size.height() -
                m_textScene->GetTextProperty()->GetFontSize() - 10);

        size = m_renderWidgetScene->geometry().size();
        m_textTop->SetPosition(10, size.height() -
                m_textTop->GetTextProperty()->GetFontSize() - 10);

        size = m_renderWidgetScene->geometry().size();
        m_textSide->SetPosition(10, size.height() -
                m_textSide->GetTextProperty()->GetFontSize() - 10);

        size = m_renderWidgetScene->geometry().size();
        m_textFront->SetPosition(10, size.height() -
                m_textFront->GetTextProperty()->GetFontSize() - 10);
    }
}

void RenderView::resizeDone()
{
    m_resizing = false;
    update();
}

void RenderView::changeView(int index)
{
    switch (index)
    {
    case 0:
        //m_stackedLayout->removeWidget(m_renderWidgetScene);
        /*m_stackedLayout->removeWidget(m_renderWidgetTop);
        m_stackedLayout->removeWidget(m_renderWidgetSide);
        m_stackedLayout->removeWidget(m_renderWidgetFront);

        m_splitLayout->addWidget(m_renderWidgetScene, 0, 0);
        m_splitLayout->addWidget(m_renderWidgetTop, 0, 1);
        m_splitLayout->addWidget(m_renderWidgetSide, 1, 0);
        m_splitLayout->addWidget(m_renderWidgetFront, 1, 1);*/
        break;
    case 1:
        //m_stackedLayout->insertWidget(1, m_renderWidgetScene);
        //m_stackedLayout->addWidget(m_renderWidgetScene);
        //m_splitLayout->removeWidget(m_renderWidgetScene);
        break;
    case 2:
        break;
    case 4:
        break;
    }
}

vtkSmartPointer<vtkRenderer> RenderView::getRendererScene(bool top)
{
    if (top)
        return m_rendererSceneTop;
    else
        return m_rendererScene;
}

vtkSmartPointer<vtkRenderer> RenderView::getRendererTop(bool top)
{
    if (top)
        return m_rendererTopTop;
    else
        return m_rendererTop;
}

vtkSmartPointer<vtkRenderer> RenderView::getRendererSide(bool top)
{
    if (top)
        return m_rendererSideTop;
    else
        return m_rendererSide;
}

vtkSmartPointer<vtkRenderer> RenderView::getRendererFront(bool top)
{
    if (top)
        return m_rendererFrontTop;
    else
        return m_rendererFront;
}

void RenderView::lock()
{
    m_mutex->lock();
    m_update = false;
    m_mutex->unlock();
}

void RenderView::unlock()
{
    m_mutex->lock();
    m_update = true;
    m_mutex->unlock();
}

bool RenderView::isLocked()
{
    m_mutex->lock();
    bool result = !m_update;
    m_mutex->unlock();
    return result;
}
