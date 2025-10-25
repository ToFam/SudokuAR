#pragma once

#include <QString>
#include <QList>

#include <opencv2/opencv.hpp>

class SequenceProvider
{
public:
    bool loadFiles(QStringList files);
    bool loadVideoSource(QString path);

    cv::Mat frame();

    bool next();
    bool prev();
    bool set(int frameNumber);

    int frameNumber() const;
    int getNumFrames() const;

    bool isVideo() const;

    QString videoSource() const;
    QStringList imageFiles() const;

public:
    void setRepeat(bool repeat) { m_repeat = repeat; }
    bool repeat() const { return m_repeat; }

private:
    bool m_validInput = false;
    bool m_videoMode = false;
    bool m_videoSeekable = false;

    bool m_repeat = false;

    int m_frameNumber = -1;
    cv::Mat m_frame;

    QStringList m_files;
    QString m_vidSource;
    cv::VideoCapture m_vc;
};
