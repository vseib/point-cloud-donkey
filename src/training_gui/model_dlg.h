#ifndef MODELDLG_H
#define MODELDLG_H

#include <QDialog>
#include <QPushButton>
#include <QLabel>

class ModelDlg
        : public QDialog
{
    Q_OBJECT

public:
    ModelDlg();
    ~ModelDlg();

    bool isValid() const;
    const QString& getFilename() const;
    unsigned getClassId() const;

private slots:
    void load();
    void ok();
    void cancel();
    void classIdChanged(const QString&);

private:
    QLabel* m_lblFilename;
    QPushButton* m_btOK;
    QString m_filename;
    int m_classId;
    static int m_lastClassId;
};

#endif // MODELDLG_H
