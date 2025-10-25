#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

struct Digit
{
    cv::Mat templ;
    int value;
};

enum class Rotation : int
{
    Zero = 0,
    Rot_90 = 1,
    Rot_180 = 2,
    Rot_270 = 3
};

class OCR
{
public:
    void loadOCR(std::string file);

    const std::vector<Digit> &digits() const;
    void scaled(double targetCellResolution, std::vector<Digit>& outScaledDigits) const;
    void rotated(cv::RotateFlags rotation, std::vector<Digit>& outRotatedDigits) const;
    void scaledRotated(cv::RotateFlags rotation, double targetCellResolution, std::vector<Digit> &outDigits) const;

    double cellResolution() const;
    double scoreThreshold() const;

private:
    std::vector<Digit> m_digits;

    double m_cellResolution;
    double m_scoreThreshold;
};

