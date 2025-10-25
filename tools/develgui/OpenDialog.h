#pragma once

#include <QDialog>

#include <QLineEdit>
#include <QListView>
#include <QListWidget>

class OpenDialog : public QDialog
{
public:
    OpenDialog(QWidget* parent = nullptr);

    bool videoChoosen() const { return m_videoFile; }

    QString videoFile() const;
    QStringList imageFileList() const;

private slots:
    void openVideoFileDialog();

    void openImagesFileDialog();
    void openFolderDialog();

    void addFolder();
    void addFiles();

    void loadVideoFile();
    void loadImageFiles();

private:
    bool m_videoFile = false;

    QLineEdit*  m_leVideoFile;

    QLineEdit*  m_leDirectory;
    QLineEdit*  m_leFiles;

    QListWidget* m_lwFiles;
};
