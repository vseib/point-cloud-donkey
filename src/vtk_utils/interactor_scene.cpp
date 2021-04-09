#include "interactor_scene.h"
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkCamera.h>
#include "render_view.h"

vtkStandardNewMacro(InteractorScene);

InteractorScene::InteractorScene()
    : vtkInteractorStyleTrackballCamera(), m_renderView(0)
{
}

InteractorScene::~InteractorScene()
{
}

void InteractorScene::setRenderView(RenderView* renderView)
{
    m_renderView = renderView;
}

void InteractorScene::OnLeftButtonDown()
{
    if (m_renderView && m_renderView->isLocked())
        return;

    this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
            this->Interactor->GetEventPosition()[1]);

    if (this->CurrentRenderer == NULL)
        return;

    this->StartPan();
}

void InteractorScene::OnLeftButtonUp()
{
    if (m_renderView && m_renderView->isLocked())
        return;

    switch (this->State) {
        case VTKIS_PAN:
        this->EndPan();
        break;
    }

    if (this->Interactor)
        this->ReleaseFocus();
}

void InteractorScene::OnRightButtonDown()
{
    if (m_renderView && m_renderView->isLocked())
        return;

    this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
            this->Interactor->GetEventPosition()[1]);

    if (this->CurrentRenderer == NULL)
        return;

    this->StartRotate();
}

void InteractorScene::OnRightButtonUp()
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

void InteractorScene::Rotate()
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

    vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();
    camera->Azimuth(rxf);
    camera->Elevation(ryf);
    //camera->OrthogonalizeViewUp();

    if (this->AutoAdjustCameraClippingRange)
        this->CurrentRenderer->ResetCameraClippingRange();

    if (rwi->GetLightFollowCamera())
        this->CurrentRenderer->UpdateLightsGeometryToFollowCamera();

    rwi->Render();
}

void InteractorScene::Spin()
{
    // disabled
}

void InteractorScene::Dolly()
{
    // disabled
}
