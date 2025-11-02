#pragma once

#include <pylon/PylonIncludes.h>

#include <opencv2/opencv.hpp>

class PylonDriver
{
public:
    PylonDriver();

    cv::Mat grab();

private:
    Pylon::PylonAutoInitTerm m_plyonContext;
    Pylon::CInstantCamera m_cam;
};
