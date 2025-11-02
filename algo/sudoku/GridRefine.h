#pragma once

#include "../Algorithm.h"

class GridRefine : public Algorithm
{
public:
    GridRefine();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;

private:
    bool processFrame(std::vector<cv::Point2f>& frame, std::vector<cv::Point2f>& framePoints,
                      float pointMargin, int pointDistAvgWindow, float templateDiscardFactor,
                      cv::Mat& outImg);

private:
    std::vector<cv::Point2f> m_intersectPoints;
};
