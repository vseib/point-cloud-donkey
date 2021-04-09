#ifndef RENDERVIEW_H
#define RENDERVIEW_H

#include <QWidget>
#include <QVTKWidget.h>
#include <QStackedLayout>
#include <QGridLayout>

#include <vtkSmartPointer.h>
#include <vtkCamera.h>
#include <vtkTextActor.h>
#include <vtkProp.h>
#include <vtkCubeSource.h>
#include <vtkAxesActor.h>
#include <QMutex>
#include <QTimer>

#include <eigen3/Eigen/Core>

class InteractorOrtho;

class RenderView
        : public QWidget
{
    Q_OBJECT

public:
    RenderView(QWidget* parent = 0);
    ~RenderView();

    void setStatus(std::string);
    void resetStatus();
    void reset();
    void addActorToScene(vtkSmartPointer<vtkProp>);
    void addActorToTop(vtkSmartPointer<vtkProp>);
    void addActorToSide(vtkSmartPointer<vtkProp>);
    void addActorToFront(vtkSmartPointer<vtkProp>);
    void addActorToAll(vtkSmartPointer<vtkProp>);
    void update();

    void emitMove(float, float, float, InteractorOrtho*);
    void emitScale(float, float, float, InteractorOrtho*);
    void emitRotate(float, Eigen::Vector3f, InteractorOrtho*);

    vtkSmartPointer<vtkRenderer> getRendererScene(bool top = false);
    vtkSmartPointer<vtkRenderer> getRendererTop(bool top = false);
    vtkSmartPointer<vtkRenderer> getRendererSide(bool top = false);
    vtkSmartPointer<vtkRenderer> getRendererFront(bool top = false);

    void lock();
    void unlock();
    bool isLocked();

    bool eventFilter(QObject*, QEvent*);

public slots:
    void resizeDone();
    void changeView(int);

signals:
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
    void resizeEvent(QResizeEvent*);

private:
    vtkSmartPointer<vtkRenderWindow> createRenderer(vtkSmartPointer<vtkRenderer>&, vtkSmartPointer<vtkRenderer>&,
                                                    vtkSmartPointer<vtkCamera>&, vtkSmartPointer<vtkTextActor>&, QString,
                                                    QVTKWidget*);

    // 3d scene view
    QVTKWidget* m_renderWidgetScene;
    vtkSmartPointer<vtkCamera> m_cameraScene;
    vtkSmartPointer<vtkTextActor> m_textScene;
    vtkSmartPointer<vtkRenderer> m_rendererScene;
    vtkSmartPointer<vtkRenderer> m_rendererSceneTop;

    // top view
    QVTKWidget* m_renderWidgetTop;
    vtkSmartPointer<vtkCamera> m_cameraTop;
    vtkSmartPointer<vtkTextActor> m_textTop;
    vtkSmartPointer<vtkRenderer> m_rendererTop;
    vtkSmartPointer<vtkRenderer> m_rendererTopTop;

    // side view
    QVTKWidget* m_renderWidgetSide;
    vtkSmartPointer<vtkCamera> m_cameraSide;
    vtkSmartPointer<vtkTextActor> m_textSide;
    vtkSmartPointer<vtkRenderer> m_rendererSide;
    vtkSmartPointer<vtkRenderer> m_rendererSideTop;

    // front view
    QVTKWidget* m_renderWidgetFront;
    vtkSmartPointer<vtkCamera> m_cameraFront;
    vtkSmartPointer<vtkTextActor> m_textFront;
    vtkSmartPointer<vtkRenderer> m_rendererFront;
    vtkSmartPointer<vtkRenderer> m_rendererFrontTop;

    // drawing objects
    vtkSmartPointer<vtkAxesActor> m_worldAxes;
    vtkSmartPointer<vtkTextActor> m_textStatus;

    std::vector<vtkSmartPointer<vtkProp> > m_actorsScene;
    std::vector<vtkSmartPointer<vtkProp> > m_actorsTop;
    std::vector<vtkSmartPointer<vtkProp> > m_actorsSide;
    std::vector<vtkSmartPointer<vtkProp> > m_actorsFront;

    QGridLayout* m_splitLayout;
    QStackedLayout* m_stackedLayout;

    QMutex* m_mutex;
    bool m_update;
    bool m_resizing;
    QTimer m_resizeTimer;
};

#endif // RENDERVIEW_H
