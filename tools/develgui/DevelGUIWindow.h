#pragma once

#include <vector>
#include <memory>

#include <QMainWindow>
#include <QTimer>

#include <OCR.h>

#include <Timer.h>
#include "GUIAlgoElement.h"
#include "SequenceProvider.h"
#include "ContainerList.h"
#include "ContainerDisplay.h"

namespace Ui {
class DevelGUIWindow;
}


class AlgoBoxGeometryUpdate : public QObject
{
    Q_OBJECT

public:
    AlgoBoxGeometryUpdate(QObject* parent) : QObject(parent) {}

    bool eventFilter(QObject* ob, QEvent* ev);
};

class OpenDialog;

class DevelGUIWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit DevelGUIWindow(QWidget *parent = nullptr);
    ~DevelGUIWindow();

private slots:
    void onOpen();

    void openProject();

    void saveProject();

    void openLast();

    void next();
    void next(bool process);

    void processImage();

    void addAlgorithm();
    void addAlgorithm(QString name);

    void removeAlgo();
    void moveAlgoUp();
    void moveAlgoDown();

    void printAlgoRuntime(int index);

    void displayContainer();

    void saveImage();

    void liveUpdateSingleAlgo();

private:
    bool loadProjectFile(QString path);
    bool saveProjectFile(QString path);

    void onLoad();

    void startPlayback();
    void stopPlayback();

private:
    Ui::DevelGUIWindow *ui;

    ContainerDisplay*   m_view;

    SequenceProvider    m_sp;
    QTimer              m_timer;

    OpenDialog*         m_openDialog;


    std::vector<std::unique_ptr<GUIAlgoElement>> m_algoStack;

    std::vector<double> m_algoRuntimes;

    ContainerList m_containerList;

    std::map<std::string, std::shared_ptr<Container>> m_containerStack;

    OCR m_ocr;

    Timer m_frameTimer;
};

