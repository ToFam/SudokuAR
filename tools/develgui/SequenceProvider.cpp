#include "SequenceProvider.h"
#include "PylonDriver.h"

#include <QFileInfo>
#include <thread>
#include <chrono>

bool SequenceProvider::loadFiles(QStringList files)
{
    m_files.clear();
    m_videoMode = false;
    m_frameNumber = -1;

    for (QString f : files)
    {
        QFileInfo info(f);
        if (!info.exists())
        {
            m_validInput = false;
            return false;
        }
    }

    m_files = files;
    m_validInput = true;
    return true;
}

bool SequenceProvider::loadVideoSource(QString path)
{
    m_videoMode = true;
    m_vidSource = path;
    m_frameNumber = -1;

    if (path == "pylon")
    {
        try
        {
            m_pylon = std::make_unique<PylonDriver>();
            m_validInput = true;
            return true;
        }
        catch(...)
        {
            m_validInput = false;
            return false;
        }
    }

    if (!m_vc.open(path.toInt()))//path.toStdString()))
    {
        std::cout << "Open Video Capture " << path.toStdString() << " failed" << std::endl;
        m_validInput = false;
        return false;
    }

    double w = m_vc.get(cv::CAP_PROP_FRAME_WIDTH);
    double h = m_vc.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Video Capture Resolution: " << w << " x " << h << std::endl;

    bool sucW = m_vc.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    if (sucW)
    {
        std::cout << "Set width" << std::endl;
    }
    else
    {
        std::cout << "Could not change width" << std::endl;
    }
    bool sucH = m_vc.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    if (sucH)
    {
        std::cout << "Set to height" << std::endl;
    }
    else
    {
        std::cout << "Could not change height" << std::endl;
    }

    w = m_vc.get(cv::CAP_PROP_FRAME_WIDTH);
    h = m_vc.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Video Capture Resolution: " << w << " x " << h << std::endl;


    m_validInput = true;
    return true;
}

cv::Mat SequenceProvider::frame()
{
    return m_frame;
}

int SequenceProvider::frameNumber() const
{
    return m_frameNumber;
}

bool SequenceProvider::next()
{
    m_frameNumber++;

    if (!m_validInput)
           return false;

    if (m_videoMode)
    {
        if (m_pylon)
        {
            m_frame = m_pylon->grab();
            return !m_frame.empty();
        }

        if (!m_vc.read(m_frame))
        {
            if (m_repeat && m_vc.open(m_vidSource.toInt())) //m_vidSource.toStdString()))
            {
                m_frameNumber = 0;
                if (!m_vc.read(m_frame))
                    return false;
            }
            else
            {
                return false;
            }
        }
    }
    else
    {
        if (m_frameNumber >= m_files.size())
        {
            if (m_repeat)
            {
                m_frameNumber = 0;
            }
            else
            {
                m_frameNumber--;
                return false;
            }
        }

        m_frame = cv::imread(m_files[m_frameNumber].toStdString());
    }

    return true;
}

bool SequenceProvider::prev()
{
    m_frameNumber--;

    if (!m_validInput)
    {
        m_frameNumber = -1;
        return false;
    }

    if (m_frameNumber < 0)
    {
        m_frameNumber = 0;
        return false;
    }

    if (m_videoMode)
    {
        m_frameNumber++;
        return false;
    }
    else
    {
        m_frame = cv::imread(m_files[m_frameNumber].toStdString());
        return true;
    }
}

bool SequenceProvider::set(int frameNumber)
{
    if (!m_validInput)
           return false;

    if (m_videoMode)
    {
        if (m_pylon)
        {
            return false;
        }

        if (frameNumber == 0)
        {
            std::cout << "Close Video Capture " << m_vidSource.toStdString() << std::endl;
            m_vc.release();

            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(1000));
            std::cout << "Re-Open Video Capture " << m_vidSource.toStdString() << std::endl;
            if (m_vc.open(m_vc.open(m_vidSource.toInt()))) //m_vidSource.toStdString()))
            {
                std::cout << "Try to read frame " << std::endl;
                if (m_vc.read(m_frame))
                {
                    m_frameNumber = 0;
                    std::cout << "Success " << std::endl;
                    return  true;
                }
                std::cout << "Could not read frame" << std::endl;
            }
            else
            {
                std::cout << "Re-Open Video Capture " << m_vidSource.toStdString() << " failed" << std::endl;
            }
        }
        return false;
    }
    else
    {
        if (m_frameNumber < 0 || m_frameNumber >= getNumFrames())
            return false;


        m_frameNumber = frameNumber;
        m_frame = cv::imread(m_files[m_frameNumber].toStdString());
        return true;
    }
}

int SequenceProvider::getNumFrames() const
{
    if (m_videoMode)
    {
        return -1;
    }
    else
    {
       return m_files.size();
    }
}

bool SequenceProvider::isVideo() const
{
    return m_videoMode;
}

QString SequenceProvider::videoSource() const
{
    return m_vidSource;
}

QStringList SequenceProvider::imageFiles() const
{
    return m_files;
}

SequenceProvider::SequenceProvider() = default;
SequenceProvider::~SequenceProvider() = default;
