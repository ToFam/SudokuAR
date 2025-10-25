#include "OCR.h"

#include <opencv2/opencv.hpp>

const std::vector<Digit> &OCR::digits() const
{
    return m_digits;
}

double OCR::cellResolution() const
{
    return m_cellResolution;
}

double OCR::scoreThreshold() const
{
    return m_scoreThreshold;
}

void OCR::loadOCR(std::string path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);

    m_cellResolution = static_cast<double>(fs["cellResolution"]);
    m_scoreThreshold = static_cast<double>(fs["scoreThreshold"]);

    cv::FileNode n = fs["digits"];

    for (auto it = n.begin(); it != n.end(); ++it)
    {
        Digit d;
        d.value = static_cast<int>((*it)["value"]);
        cv::Mat mat;
        ((*it)["matrix"]) >> mat;
        mat.copyTo(d.templ);
        m_digits.push_back(d);
    }
}

void OCR::scaled(double targetCellResolution, std::vector<Digit> &outScaledDigits) const
{
    double scaleFactor = targetCellResolution / m_cellResolution;
    for (int i = 0; i < m_digits.size(); ++i)
    {
        Digit d;
        d.value = m_digits[i].value;

        cv::resize(m_digits[i].templ, d.templ, cv::Size(), scaleFactor, scaleFactor);

        outScaledDigits.push_back(d);
    }
}

void OCR::rotated(cv::RotateFlags rotation, std::vector<Digit> &outRotatedDigits) const
{
    for (int i = 0; i < m_digits.size(); ++i)
    {
        Digit d;
        d.value = m_digits[i].value;

        cv::rotate(m_digits[i].templ, d.templ, rotation);

        outRotatedDigits.push_back(d);
    }
}

void OCR::scaledRotated(cv::RotateFlags rotation, double targetCellResolution, std::vector<Digit> &outDigits) const
{
    double scaleFactor = targetCellResolution / m_cellResolution;
    for (int i = 0; i < m_digits.size(); ++i)
    {
        Digit d;
        d.value = m_digits[i].value;

        cv::Mat tmpl;
        cv::rotate(m_digits[i].templ, tmpl, rotation);

        cv::resize(tmpl, d.templ, cv::Size(), scaleFactor, scaleFactor);

        outDigits.push_back(d);
    }
}
