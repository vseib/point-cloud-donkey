#ifndef INTERACTORSCENE_H
#define INTERACTORSCENE_H

#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkObjectFactory.h>

class RenderView;

class InteractorScene
        : public vtkInteractorStyleTrackballCamera
{

public:
    static InteractorScene* New();
    vtkTypeMacro(InteractorScene, vtkInteractorStyleTrackballCamera);

    InteractorScene();
    ~InteractorScene();

    void setRenderView(RenderView*);

    void OnLeftButtonDown();
    void OnLeftButtonUp();
    void OnRightButtonDown();
    void OnRightButtonUp();
    void Rotate();
    void Spin();
    void Dolly();

private:
    RenderView* m_renderView;
};

#endif // INTERACTORSCENE_H
