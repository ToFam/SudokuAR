#pragma once

#include "../Algorithm.h"
#include "../OCR.h"

class SolutionDisplay : public Algorithm
{
public:
    SolutionDisplay(const OCR& ocr);

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;

private:
    void render(int frameNum, cv::Mat inImage, cv::Mat frames, std::vector<int>& numbers, std::vector<int>& solved, cv::Mat& outImage);

private:
    const OCR& m_ocr;
};
