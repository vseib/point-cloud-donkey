#include "interactor_ortho.h"
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkCamera.h>
#include "render_view.h"

vtkStandardNewMacro(InteractorOrtho);

InteractorOrtho::InteractorOrtho()
    : vtkInteractorStyleTrackballCamera()
{
}

InteractorOrtho::~InteractorOrtho()
{
}

void InteractorOrtho::setRenderView(RenderView* renderView)
{
    m_renderView = renderView;
}

void InteractorOrtho::OnLeftButtonDown()
{
    if (m_renderView && m_renderView->isLocked())
        return;

    this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
            this->Interactor->GetEventPosition()[1]);

    if (this->CurrentRenderer == NULL)
        return;

    this->StartPan();
}

void InteractorOrtho::OnRightButtonDown()
{
    if (m_renderView && m_renderView->isLocked())
        return;

    this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
            this->Interactor->GetEventPosition()[1]);

    if (this->CurrentRenderer == NULL)
        return;

    // get picking point
    vtkRenderWindowInteractor *rwi = this->Interactor;
    double viewFocus[4], focalDepth;
    vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();
    camera->GetFocalPoint(viewFocus);
    this->ComputeWorldToDisplay(viewFocus[0], viewFocus[1], viewFocus[2], viewFocus);
    focalDepth = viewFocus[2];
    this->ComputeDisplayToWorld(rwi->GetEventPosition()[0],
            rwi->GetEventPosition()[1], focalDepth, m_pickingPoint);

    this->StartRotate();
}

void InteractorOrtho::OnRightButtonUp()
{
    if (m_renderView && m_renderView->isLocked())
        return;

    switch (this->State) {
        case VTKIS_ROTATE:
        this->EndRotate();
        break;
    }

    if (this->Interactor)
        this->ReleaseFocus();
}

void InteractorOrtho::Pan()
{
    if (m_renderView && m_renderView->isLocked())
        return;

    if (this->CurrentRenderer == NULL)
        return;

    vtkRenderWindowInteractor *rwi = this->Interactor;

    int shift = rwi->GetShiftKey();
    int ctrl = rwi->GetControlKey();

    if (shift > 0 || ctrl > 0) {
        double viewFocus[4], focalDepth;
        double newPickPoint[4], oldPickPoint[4];

        vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();
        camera->GetFocalPoint(viewFocus);
        this->ComputeWorldToDisplay(viewFocus[0], viewFocus[1], viewFocus[2], viewFocus);
        focalDepth = viewFocus[2];

        this->ComputeDisplayToWorld(rwi->GetEventPosition()[0],
                rwi->GetEventPosition()[1], focalDepth, newPickPoint);

        this->ComputeDisplayToWorld(rwi->GetLastEventPosition()[0],
                rwi->GetLastEventPosition()[1], focalDepth, oldPickPoint);

        if (shift > 0) {
            m_renderView->emitMove(newPickPoint[0] - oldPickPoint[0],
                    newPickPoint[1] - oldPickPoint[1],
                    newPickPoint[2] - oldPickPoint[2],
                    this);
        }

        if (ctrl > 0) {
            m_renderView->emitScale(newPickPoint[0] - oldPickPoint[0],
                    newPickPoint[1] - oldPickPoint[1],
                    newPickPoint[2] - oldPickPoint[2],
                    this);
        }
    }
    else
        vtkInteractorStyleTrackballCamera::Pan();
}

void InteractorOrtho::Rotate()
{
    if (m_renderView && m_renderView->isLocked())
        return;

    if (this->CurrentRenderer == NULL)
        return;

    vtkRenderWindowInteractor *rwi = this->Interactor;

    int dx = rwi->GetEventPosition()[0] - rwi->GetLastEventPosition()[0];
    int dy = rwi->GetEventPosition()[1] - rwi->GetLastEventPosition()[1];

    int *size = this->CurrentRenderer->GetRenderWindow()->GetSize();

    double delta_elevation = -20.0 / size[1];
    double delta_azimuth = -20.0 / size[0];

    double rxf = dx * delta_azimuth * this->MotionFactor;
    double ryf = dy * delta_elevation * this->MotionFactor;

    // tell render view
    Eigen::Vector3f pickPoint(m_pickingPoint[0], m_pickingPoint[1], m_pickingPoint[2]);
    m_renderView->emitRotate((float)(rxf - ryf), pickPoint, this);
}
