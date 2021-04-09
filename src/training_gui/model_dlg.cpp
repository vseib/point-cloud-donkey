#include "model_dlg.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpacerItem>
#include <QIntValidator>
#include <QFileDialog>
#include <QMessageBox>

int ModelDlg::m_lastClassId = 0;

ModelDlg::ModelDlg()
    : m_classId(m_lastClassId)
{
    QLabel* lblClassId = new QLabel();
    lblClassId->setText("Class Id: ");

    QLineEdit* edClassId = new QLineEdit();
    edClassId->setMaximumWidth(40);

    QIntValidator* validator = new QIntValidator(0, 2147483647);
    edClassId->setValidator(validator);

    QPushButton* btLoad = new QPushButton();
    btLoad->setText("Load Model");
    connect(btLoad, SIGNAL(clicked()), this, SLOT(load()));

    m_lblFilename = new QLabel();
    m_lblFilename->setText("");

    m_btOK = new QPushButton();
    m_btOK->setText("OK");
    m_btOK->setEnabled(false);
    connect(m_btOK, SIGNAL(clicked()), this, SLOT(ok()));

    connect(edClassId, SIGNAL(textChanged(QString)), this, SLOT(classIdChanged(QString)));
    edClassId->setText(QString::number(m_classId));

    QPushButton* btCancel = new QPushButton();
    btCancel->setText("Cancel");
    connect(btCancel, SIGNAL(clicked(bool)), this, SLOT(cancel()));

    QHBoxLayout* layoutTop = new QHBoxLayout();

    layoutTop->addWidget(lblClassId);
    layoutTop->addWidget(edClassId);
    layoutTop->addWidget(btLoad);

    QHBoxLayout* layoutBottom = new QHBoxLayout();
    layoutBottom->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Minimum));
    layoutBottom->addWidget(m_btOK);
    layoutBottom->addWidget(btCancel);

    QVBoxLayout* layout = new QVBoxLayout();
    layout->addLayout(layoutTop);
    layout->addWidget(m_lblFilename);
    layout->addLayout(layoutBottom);
    layout->setSizeConstraint(QLayout::SetFixedSize);

    setLayout(layout);
    setWindowTitle("Add training model");
}

ModelDlg::~ModelDlg()
{
}

void ModelDlg::load()
{
    QString filename = QFileDialog::getOpenFileName(this, "Load Model", QString(), tr("PCD-Files (*.pcd);;PLY-Files (*.ply);;All Files (*.*)"), 0, QFileDialog::DontUseNativeDialog);

    if (!filename.isEmpty()) {
        if (filename.endsWith(".pcd", Qt::CaseInsensitive) || filename.endsWith(".ply", Qt::CaseInsensitive)) {
            m_filename = filename;
            m_lblFilename->setText(m_filename);
        }
        else
            QMessageBox::warning(this, "Error", "Invalid file extension");
    }

    m_btOK->setEnabled(m_classId >= 0 && !m_filename.isEmpty());
}

void ModelDlg::ok()
{
    m_lastClassId = m_classId;
    close();
}

void ModelDlg::cancel()
{
    m_filename.clear();
    m_classId = -1;
    close();
}

void ModelDlg::classIdChanged(const QString& text)
{
    m_btOK->setEnabled(!text.isEmpty() && !m_filename.isEmpty());
    m_classId = text.toInt();
}

bool ModelDlg::isValid() const
{
    return !m_filename.isEmpty() && m_classId >= 0;
}

const QString& ModelDlg::getFilename() const
{
    return m_filename;
}

unsigned ModelDlg::getClassId() const
{
    if (m_classId < 0)
        return 0;
    return (unsigned)m_classId;
}
