#include "OpenDialog.h"

#include <QListWidgetItem>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>

OpenDialog::OpenDialog(QWidget* parent)
 : QDialog(parent)
{

    QVBoxLayout* mainLayout = new QVBoxLayout;

    {
        QGroupBox* videoGB = new QGroupBox(tr("Load Video Source"));
        QVBoxLayout* videoGBLayout = new QVBoxLayout;

        videoGBLayout->addWidget(new QLabel(tr("Path to video source:")));

        QHBoxLayout* leLayout = new QHBoxLayout;
        m_leVideoFile = new QLineEdit;
        QPushButton* button = new QPushButton("...");
        connect(button, &QPushButton::clicked, this, &OpenDialog::openVideoFileDialog);
        leLayout->addWidget(m_leVideoFile);
        leLayout->addWidget(button);

        videoGBLayout->addItem(leLayout);

        QHBoxLayout* bLayout = new QHBoxLayout;
        bLayout->addStretch();
        QPushButton* loadButton = new QPushButton(tr("Load"));
        connect(loadButton, &QPushButton::clicked, this, &OpenDialog::loadVideoFile);
        bLayout->addWidget(loadButton);

        videoGBLayout->addItem(bLayout);

        videoGB->setLayout(videoGBLayout);
        mainLayout->addWidget(videoGB);
    }

    {
        QGroupBox* filesGB = new QGroupBox(tr("Load image files"));
        QVBoxLayout* filesGBLayout = new QVBoxLayout;

        QHBoxLayout* dirLayout = new QHBoxLayout;
        m_leDirectory = new QLineEdit;
        QPushButton* dirButton = new QPushButton("...");
        connect(dirButton, &QPushButton::clicked, this, &OpenDialog::openFolderDialog);
        QPushButton* addDirButton = new QPushButton(tr("Add"));
        connect(addDirButton, &QPushButton::clicked, this, &OpenDialog::addFolder);
        dirLayout->addWidget(m_leDirectory);
        dirLayout->addWidget(dirButton);
        dirLayout->addWidget(addDirButton);

        QHBoxLayout* fileLayout = new QHBoxLayout;
        m_leFiles = new QLineEdit;
        QPushButton* filesButton = new QPushButton("...");
        connect(filesButton, &QPushButton::clicked, this, &OpenDialog::openImagesFileDialog);
        QPushButton* addFilesButton = new QPushButton(tr("Add"));
        connect(addFilesButton, &QPushButton::clicked, this, &OpenDialog::addFiles);
        fileLayout->addWidget(m_leFiles);
        fileLayout->addWidget(filesButton);
        fileLayout->addWidget(addFilesButton);

        filesGBLayout->addItem(dirLayout);
        filesGBLayout->addItem(fileLayout);

        m_lwFiles = new QListWidget;
        filesGBLayout->addWidget(m_lwFiles);


        QHBoxLayout* bLayout = new QHBoxLayout;
        bLayout->addStretch();
        QPushButton* loadButton = new QPushButton(tr("Load"));
        connect(loadButton, &QPushButton::clicked, this, &OpenDialog::loadImageFiles);
        bLayout->addWidget(loadButton);
        filesGBLayout->addItem(bLayout);
        filesGB->setLayout(filesGBLayout);
        mainLayout->addWidget(filesGB);
    }

    {
        QHBoxLayout* l = new QHBoxLayout;
        l->addStretch();
        QPushButton* b = new QPushButton(tr("Cancel"));
        connect(b, &QPushButton::clicked, this, &QDialog::reject);
        l->addWidget(b);
        mainLayout->addItem(l);
    }

    setLayout(mainLayout);
}

QString OpenDialog::videoFile() const
{
    return m_leVideoFile->text();
}

QStringList OpenDialog::imageFileList() const
{
    QStringList rtn;
    for (int i = 0; i < m_lwFiles->count(); i++)
    {
        rtn.append(m_lwFiles->item(i)->data(Qt::DisplayRole).toString());
    }
    return rtn;
}

void OpenDialog::openVideoFileDialog()
{
    QUrl path = QFileDialog::getOpenFileUrl(this, tr("Open Video Source"));
    if (path.isValid())
    {
        m_leVideoFile->setText(path.toString());
    }
}

void OpenDialog::openFolderDialog()
{
    QString path = QFileDialog::getExistingDirectory(this, tr("Open image directory"));
    if (!path.isEmpty())
    {
        m_leDirectory->setText(path);
    }
}

void OpenDialog::openImagesFileDialog()
{
    QStringList files = QFileDialog::getOpenFileNames(this, tr("Open Image Files"));
    if (!files.empty())
    {
        m_leFiles->clear();
        QString str;
        for (QString s : files)
        {
            str += s + " ";
        }
        str.chop(1);
        m_leFiles->setText(str);
    }
}

void OpenDialog::addFolder()
{
    QString str = m_leDirectory->text();
    m_leDirectory->clear();
    QDir d(str);
    if (d.exists())
    {
        for (QFileInfo inf : d.entryInfoList(QDir::Files))
        {
            new QListWidgetItem(inf.filePath(), m_lwFiles);
        }
    }
}

void OpenDialog::addFiles()
{
    QString str = m_leFiles->text();
    m_leFiles->clear();

    for (QString s : str.split(" "))
    {
        QFileInfo inf(s);
        if (inf.exists())
        {
            new QListWidgetItem(s, m_lwFiles);
        }
    }
}

void OpenDialog::loadVideoFile()
{
    m_videoFile = true;
    accept();
}

void OpenDialog::loadImageFiles()
{
    m_videoFile = false;
    accept();
}
