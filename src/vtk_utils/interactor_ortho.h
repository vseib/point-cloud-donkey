#ifndef INTERACTORORTHO_H
#define INTERACTORORTHO_H

#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkObjectFactory.h>

#include <eigen3/Eigen/Core>

class RenderView;

class InteractorOrtho
        : public vtkInteractorStyleTrackballCamera
{

public:
    static InteractorOrtho* New();
    vtkTypeMacro(InteractorOrtho, vtkInteractorStyleTrackballCamera);

    InteractorOrtho();
    ~InteractorOrtho();

    void setRenderView(RenderView*);

    void OnLeftButtonDown();
    void OnRightButtonDown();
    void OnRightButtonUp();
    void Pan();
    void Rotate();

private:
    RenderView* m_renderView;
    double m_pickingPoint[4];
};

#endif // INTERACTORORTHO_H
