#include "DevelGUIWindow.h"
#include "./ui_DevelGUIWindow.h"

#include <QMessageBox>
#include <QFileDialog>

#include <imageview.h>

#include "AlgoFactory.h"

#include "OpenDialog.h"

DevelGUIWindow::DevelGUIWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::DevelGUIWindow)
{
    ui->setupUi(this);

    QHBoxLayout* mainLayout = ui->mainLayout;

    m_view = new ContainerDisplay(this);
    m_view->setControlls(ui->frameContainerContent, ui->lContainerContent, ui->tbContainerContentLeft, ui->tbContainerContentRight);
    mainLayout->removeWidget(ui->view);
    delete ui->view;
    mainLayout->insertWidget(0, m_view);
    mainLayout->setStretch(0, 4);
    mainLayout->setStretch(1, 1);

    ui->sbFPS->setMaximum(1000);
    m_timer.setInterval(static_cast<int>(1000.0 / 30.0));
    ui->sbFPS->setValue(30);

    connect(ui->actionOpen, &QAction::triggered, this, [&](){
        m_openDialog = new OpenDialog(this);
        connect(m_openDialog, &QDialog::finished, this, &DevelGUIWindow::onOpen);
        m_openDialog->open();
    });

    connect(ui->tbPlayPause, &QToolButton::pressed, this, [&](){
        if (m_timer.isActive())
        {
            stopPlayback();
        }
        else
        {
            startPlayback();
        }
    });
    connect(ui->tbStop, &QToolButton::pressed, this, [&](){
        stopPlayback();

        if (m_sp.set(0))
        {
            QSignalBlocker b(ui->sbFrame), b3(ui->hsFrame);
            ui->sbFrame->setValue(m_sp.frameNumber());
            ui->hsFrame->setValue(m_sp.frameNumber());
            processImage();
        }
    });
    connect(ui->sbFPS, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, [&](int fps){
        m_timer.setInterval(static_cast<int>(1000.0 / fps));
    });
    connect(&m_timer, &QTimer::timeout, this, static_cast<void (DevelGUIWindow::*)(void)>(&DevelGUIWindow::next));
    connect(ui->tbRight, &QToolButton::pressed, this, static_cast<void (DevelGUIWindow::*)(void)>(&DevelGUIWindow::next));
    connect(ui->tbLeft, &QToolButton::pressed, this, [&](){
        if (m_sp.prev())
        {
            QSignalBlocker b(ui->sbFrame), b3(ui->hsFrame);
            ui->sbFrame->setValue(m_sp.frameNumber());
            ui->hsFrame->setValue(m_sp.frameNumber());
            processImage();
        }
    });
    connect(ui->sbFrame, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, [&](int frame){
        if (m_sp.set(frame))
        {
            QSignalBlocker b3(ui->hsFrame);
            ui->hsFrame->setValue(m_sp.frameNumber());
            processImage();
        }
    });
    connect(ui->hsFrame, static_cast<void (QSlider::*)(int)>(&QSlider::valueChanged), this, [&](int frame){
        if (m_sp.set(frame))
        {
            QSignalBlocker b(ui->sbFrame);
            ui->sbFrame->setValue(m_sp.frameNumber());
            processImage();
        }
    });
    connect(ui->tbRepeat, static_cast<void (QToolButton::*)(bool)>(&QToolButton::toggled), this, [&](bool repeat){
        m_sp.setRepeat(repeat);
    });
    connect(ui->tbRefresh, &QToolButton::pressed, this, &DevelGUIWindow::processImage);

    connect(ui->actionLoad_last_project, &QAction::triggered, this, &DevelGUIWindow::openLast);

    QStringList algoNames;
    for (const auto& name : allAlgoNames())
    {
        algoNames << QString::fromStdString(name);
    }
    ui->cobAlgorithms->addItems(algoNames);

    ui->lvContainers->setModel(&m_containerList);
    ui->lvContainers->setSelectionMode(QAbstractItemView::SingleSelection);
    m_containerList.setView(ui->lvContainers);

    connect(ui->lvContainers->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &DevelGUIWindow::displayContainer);

    connect(ui->pbAddContainer, &QPushButton::pressed, &m_containerList, static_cast<void (ContainerList::*) (void)>(&ContainerList::add));
    connect(ui->pbRemoveContainer, &QPushButton::pressed, &m_containerList, &ContainerList::remove);

    connect(ui->pbAddAlgo, &QPushButton::pressed, this, static_cast<void (DevelGUIWindow::*)(void)>(&DevelGUIWindow::addAlgorithm));

    connect(ui->toolBox, &QToolBox::currentChanged, this, &DevelGUIWindow::printAlgoRuntime);

    while (ui->toolBox->count() > 0)
    {
        ui->toolBox->removeItem(0);
    }

    connect(ui->actionLoad_project, &QAction::triggered, this, &DevelGUIWindow::openProject);
    connect(ui->actionSave_project, &QAction::triggered, this, &DevelGUIWindow::saveProject);

    connect(ui->tbContainerUp, &QToolButton::pressed, &m_containerList, &ContainerList::moveCurrentUp);
    connect(ui->tbContainerDown, &QToolButton::pressed, &m_containerList, &ContainerList::moveCurrentDown);

    m_ocr.loadOCR("param/ocr2.yml");

    connect(ui->actionSave_current_image, &QAction::triggered, this, &DevelGUIWindow::saveImage);


    connect(m_view, &ContainerDisplay::selected, this, [&](const QRectF& selectRect){
        ui->statusbar->showMessage(QString("(%1,%2) - (%3,%4) = (%5x%6)").arg(selectRect.x()).arg(selectRect.y())
                                   .arg(selectRect.x() + selectRect.width()).arg(selectRect.y() + selectRect.height())
                                   .arg(selectRect.width()).arg(selectRect.height()));
    });
}

DevelGUIWindow::~DevelGUIWindow()
{
    saveProjectFile("lastsession.yml");
    delete ui;
}

void DevelGUIWindow::processImage()
{
    cv::Mat src = m_sp.frame();
    auto sFrame = std::make_shared<cv::Mat>(src);

    Timer pipelineTimer;
    Timer algoTimer;

    m_algoRuntimes = std::vector<double>(m_algoStack.size());

    m_containerStack.clear();
    m_containerStack["Frame"] = std::make_shared<Container>(ContainerSpecification("Frame", ContainerSpecification::REFERENCE));
    m_containerStack["Frame"]->set(sFrame);

    for (const QString& s : m_containerList.list())
    {
        if (s == "Frame")
            continue;

        std::string str = s.toStdString();
        m_containerStack[str] = std::make_shared<Container>(ContainerSpecification(str, ContainerSpecification::REFERENCE));
        m_containerStack[str]->set(std::make_shared<cv::Mat>());
    }

    for (size_t i = 0; i < m_algoStack.size(); i++)
    {
        try
        {
            m_algoStack[i]->prepare(m_containerStack);
        }
        catch(std::exception& e)
        {
            QMessageBox::critical(this, tr("Pipeline Error"), tr("Error during preparation of Algorithm '%1':\n%2")
                                      .arg(QString::fromStdString(m_algoStack[i]->algo()->name()),
                                           QString::fromStdString(e.what())));
            stopPlayback();
            break;
        }

        try
        {
            algoTimer.restart();
            m_algoStack[i]->algo()->exec();
            m_algoRuntimes[i] = algoTimer.elapsed();
        }
        catch(std::exception& e)
        {
            QMessageBox::critical(this, tr("Alogrithm Error"), tr("Error during exection of Algorithm '%1':\n%2")
                                                               .arg(QString::fromStdString(m_algoStack[i]->algo()->name()),
                                                                    QString::fromStdString(e.what())));
            stopPlayback();
            break;
        }
    }

    double elapsedPipeline = pipelineTimer.elapsed();
    ui->lPipelineRuntime->setText(QString("%1 ms").arg(elapsedPipeline));

    printAlgoRuntime(ui->toolBox->currentIndex());

    displayContainer();
}

void DevelGUIWindow::liveUpdateSingleAlgo()
{
    GUIAlgoElement* sender = dynamic_cast<GUIAlgoElement*>(QObject::sender());
    if (sender == nullptr || m_containerStack.empty())
        return;

    sender->prepare(m_containerStack);
    sender->algo()->exec();
    displayContainer();
}

void DevelGUIWindow::displayContainer()
{
    QModelIndex idx = ui->lvContainers->selectionModel()->currentIndex();

    QString container = "Frame";
    if (idx.isValid())
    {
        container = m_containerList.data(idx, Qt::DisplayRole).toString();
    }

    auto it = m_containerStack.find(container.toStdString());

    if (it != m_containerStack.end())
    {
        m_view->display(it->second);
    }
}

void DevelGUIWindow::saveImage()
{
    QModelIndex idx = ui->lvContainers->selectionModel()->currentIndex();

    QString container = "Frame";
    if (idx.isValid())
    {
        container = m_containerList.data(idx, Qt::DisplayRole).toString();
    }

    container.append(".jpg");

    cv::imwrite(container.toStdString(), m_view->currentImage());
}

void DevelGUIWindow::addAlgorithm()
{
    addAlgorithm(ui->cobAlgorithms->currentText());
}
bool AlgoBoxGeometryUpdate::eventFilter(QObject* ob, QEvent* ev)
{
    if (ev->type() != QEvent::Show)
        return true;

    QObject* p = ob->parent();
    if (p)
    {
        QObject* pp = p->parent();
        if (pp)
        {
            auto* sa = static_cast<QAbstractScrollArea*>(pp);
            sa->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
            QWidget* w = static_cast<QWidget*>(ob);
            sa->setMinimumHeight(w->sizeHint().height());
        }
    }
    return true;
}

void DevelGUIWindow::addAlgorithm(QString name)
{
    std::unique_ptr<Algorithm> algo = createAlgorithm(name.toStdString(), m_ocr);
    if (algo == nullptr)
    {
        return;
    }

    m_algoStack.emplace_back(new GUIAlgoElement(name, std::move(algo), m_containerList));
    auto& gae = m_algoStack.back();
    connect(gae.get(), &GUIAlgoElement::remove, this, &DevelGUIWindow::removeAlgo);
    connect(gae.get(), &GUIAlgoElement::up, this, &DevelGUIWindow::moveAlgoUp);
    connect(gae.get(), &GUIAlgoElement::down, this, &DevelGUIWindow::moveAlgoDown);
    connect(gae.get(), &GUIAlgoElement::liveUpdate, this, &DevelGUIWindow::liveUpdateSingleAlgo);

    ui->toolBox->addItem(gae->widget(), name);
    gae->widget()->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    gae->widget()->installEventFilter(new AlgoBoxGeometryUpdate(gae->widget()));

    /*
    for (auto c : ui->toolBox->findChildren<QAbstractScrollArea*>())
    {
        auto* sa = static_cast<QAbstractScrollArea*>(c);
        sa->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        //sa->setMinimumHeight(600);
        sa->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    }
    */

}

void DevelGUIWindow::removeAlgo()
{
    for (auto it = m_algoStack.begin(); it != m_algoStack.end(); ++it)
    {
        if (it->get() == QObject::sender())
        {
            for (int i = 0; i < ui->toolBox->count(); ++i)
            {
                if (ui->toolBox->widget(i) == (*it)->widget())
                {
                    ui->toolBox->removeItem(i);
                    break;
                }
            }

            m_algoStack.erase(it);

            break;
        }
    }
}

void DevelGUIWindow::moveAlgoUp()
{
    int index = 0;
    for (auto it = m_algoStack.begin(); it != m_algoStack.end(); ++it, ++index)
    {
        if (it->get() == QObject::sender())
        {
            if (index == 0)
                return;

            QString name = (*it)->name();
            QWidget* w = ui->toolBox->widget(index);
            //w->setParent(nullptr);

            ui->toolBox->removeItem(index);
            ui->toolBox->insertItem(index - 1, w, name);

            ui->toolBox->setCurrentIndex(index - 1);

            std::iter_swap(it, it - 1);

            break;
        }
    }
}

void DevelGUIWindow::moveAlgoDown()
{
    int index = 0;
    for (auto it = m_algoStack.begin(); it != m_algoStack.end(); ++it, ++index)
    {
        if (it->get() == QObject::sender())
        {
            if (index == static_cast<int>(m_algoStack.size() - 1))
                return;

            QString name = (*it)->name();
            QWidget* w = ui->toolBox->widget(index);
            //w->setParent(nullptr);

            ui->toolBox->removeItem(index);
            ui->toolBox->insertItem(index + 1, w, name);

            ui->toolBox->setCurrentIndex(index + 1);

            std::iter_swap(it, it + 1);

            break;
        }
    }
}

void DevelGUIWindow::printAlgoRuntime(int index)
{
    if (index < m_algoRuntimes.size())
    {
        ui->lPluginRuntime->setText(QString("%1 ms").arg(m_algoRuntimes[index]));
    }
}

void DevelGUIWindow::next()
{
    next(true);
}

void DevelGUIWindow::next(bool process)
{
    Timer framegrabTimer;
    if (m_sp.next())
    {
        ui->lFramegrabRuntime->setText(QString("%1 ms").arg(framegrabTimer.elapsed()));

        QSignalBlocker b(ui->sbFrame), b3(ui->hsFrame);
        ui->sbFrame->setValue(m_sp.frameNumber());
        ui->hsFrame->setValue(m_sp.frameNumber());

        if (process)
            processImage();

        ui->lPipelineFPS->setText(QString::number(1000.0 / m_frameTimer.restart()));
    }
}

void DevelGUIWindow::onOpen()
{
    if (m_openDialog->result() == QDialog::Accepted)
    {
        if (m_openDialog->videoChoosen())
        {
            if (!m_sp.loadVideoSource(m_openDialog->videoFile()))
            {
                QMessageBox::critical(this, tr("error"), tr("Could not load video source"));
            }
        }
        else
        {
            if (!m_sp.loadFiles(m_openDialog->imageFileList()))
            {
                QMessageBox::critical(this, tr("error"), tr("Could not load images"));
            }
        }

        // Adjust controls
        QSignalBlocker b(ui->sbFrame), b3(ui->hsFrame);
        int maxFrame = m_sp.getNumFrames() - 1;
        if (maxFrame == -1)
        {
            ui->sbFrame->setMaximum(0);
            ui->hsFrame->setMaximum(0);
            ui->sbFrame->setSuffix(" / " + QString::number(maxFrame));
        }
        else
        {
            ui->sbFrame->setMaximum(maxFrame);
            ui->hsFrame->setMaximum(maxFrame);

            ui->sbFrame->setSuffix(" / " + QString::number(maxFrame));
        }

        ui->sbFrame->setValue(0);
        ui->hsFrame->setValue(0);

        if (m_openDialog->videoChoosen())
        {
            ui->sbFPS->setValue(30);
        }
        else
        {
            ui->sbFPS->setValue(1);
        }

        next();
    }
    m_openDialog->deleteLater();
}

void DevelGUIWindow::openLast()
{
    if (loadProjectFile("lastsession.yml"))
    {
        onLoad();
    }
}

void DevelGUIWindow::openProject()
{
    QString path = QFileDialog::getOpenFileName(this, tr("Open project file"));
    if (!path.isEmpty())
    {
        if (loadProjectFile(path))
        {
            onLoad();
        }
    }
}

void DevelGUIWindow::saveProject()
{
    QString path = QFileDialog::getSaveFileName(this, tr("Choose project file location"),
                                                QDir::currentPath() + "/projectexport.yml", tr("Project Files (*.yml *.yaml *.json *.xml)"));
    if (!path.isEmpty())
    {
        saveProjectFile(path);
    }
}

bool DevelGUIWindow::loadProjectFile(QString path)
{
    m_containerList.clear();
    while (ui->toolBox->count() > 0)
    {
        ui->toolBox->removeItem(0);
    }
    m_algoStack.clear();

    cv::FileStorage fs(path.toStdString(), cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;

    cv::FileNode node = fs["video"];
    if (node.isNone())
    {
        node = fs["images"];
        if (node.isNone())
            return false;

        QStringList images;
        for (cv::FileNodeIterator it = node.begin(); it != node.end(); it++)
        {
            images.push_back(QString::fromStdString(static_cast<std::string>(*it)));
        }

        if (!m_sp.loadFiles(images))
        {
            QMessageBox::critical(this, tr("error"), tr("Could not load images"));
        }
    }
    else
    {
        QString vidSource = QString::fromStdString(static_cast<std::string>(node));

        if (!m_sp.loadVideoSource(vidSource))
        {
            QMessageBox::critical(this, tr("error"), tr("Could not load video source: ") + vidSource);
        }
    }

    cv::FileNode containerStackNode = fs["container-stack"];
    if (!node.isNone())
    {
        for (cv::FileNodeIterator it = containerStackNode.begin(); it != containerStackNode.end(); ++it)
        {
            m_containerList.add(QString::fromStdString(static_cast<std::string>(*it)));
        }

        cv::FileNode algoStackNode = fs["algo-stack"];
        if (!algoStackNode.isNone())
        {
            for (cv::FileNodeIterator it = algoStackNode.begin(); it != algoStackNode.end(); ++it)
            {
                QString name = QString::fromStdString(static_cast<std::string>((*it)["name"]));
                addAlgorithm(name);
                auto& gae = m_algoStack.back();
                cv::FileNode n = *it;
                gae->loadFromFile(n);
            }
        }
    }

    return true;
}


bool DevelGUIWindow::saveProjectFile(QString path)
{
    cv::FileStorage fs(path.toStdString(), cv::FileStorage::WRITE);
    if (fs.isOpened())
    {
        if (m_sp.isVideo())
        {
            fs << "video" << m_sp.videoSource().toStdString();
        }
        else
        {
            fs << "images" << "[";
            for (const QString& s : m_sp.imageFiles())
            {
                fs << s.toStdString();
            }
            fs << "]";
        }
    }

    fs << "container-stack" << "[";

    for (const QString& s : m_containerList.list())
    {
        fs << s.toStdString();
    }

    fs << "]";

    fs << "algo-stack" << "[";

    for (auto& gae : m_algoStack)
    {
        fs << "{";

        fs << "name" << gae->algo()->name();

        gae->saveToFile(fs);

        fs << "}";
    }

    fs << "]";

    return true;
}

void DevelGUIWindow::onLoad()
{
    // Adjust controls
    QSignalBlocker b(ui->sbFrame), b3(ui->hsFrame);
    int maxFrame = m_sp.getNumFrames() - 1;
    if (maxFrame == -1)
    {
        ui->sbFrame->setMaximum(0);
        ui->hsFrame->setMaximum(0);
        ui->sbFrame->setSuffix(" / " + QString::number(maxFrame));
    }
    else
    {
        ui->sbFrame->setMaximum(maxFrame);
        ui->hsFrame->setMaximum(maxFrame);

        ui->sbFrame->setSuffix(" / " + QString::number(maxFrame));
    }

    ui->sbFrame->setValue(0);
    ui->hsFrame->setValue(0);

    if (m_sp.isVideo())
    {
        ui->sbFPS->setValue(30);
    }
    else
    {
        ui->sbFPS->setValue(1);
    }

    next(false);
}

void DevelGUIWindow::startPlayback()
{
    if (!m_timer.isActive())
    {
        m_timer.start();
        ui->tbPlayPause->setIcon(QIcon(":/icons/res/pause.svg"));
    }
}

void DevelGUIWindow::stopPlayback()
{
    if (m_timer.isActive())
    {
        m_timer.stop();
        ui->tbPlayPause->setIcon(QIcon(":/icons/res/play.svg"));
    }
}
