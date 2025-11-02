#include "PylonDriver.h"

#include <iostream>

PylonDriver::PylonDriver() : m_cam(Pylon::CTlFactory::GetInstance().CreateFirstDevice())
{
    std::cout << "Using Basler device " << m_cam.GetDeviceInfo().GetModelName() << std::endl;
    m_cam.StartGrabbing(Pylon::EGrabStrategy::GrabStrategy_LatestImageOnly);
}

cv::Mat PylonDriver::grab()
{
    Pylon::CGrabResultPtr res;
    Pylon::CPylonImage pylonImage;
    Pylon::CImageFormatConverter formatConverter;
    formatConverter.OutputPixelFormat = Pylon::PixelType_BGR8packed;
    formatConverter.OutputBitAlignment=Pylon::OutputBitAlignment_MsbAligned;


    if (m_cam.RetrieveResult(5000, res))
    {
        size_t w = res->GetWidth();
        size_t h = res->GetHeight();
        //std::cout << "SizeX: " << w << std::endl;
        //std::cout << "SizeY: " << h << std::endl;

        cv::Mat frame = cv::Mat::zeros(h, w, CV_8UC3);

        //const uint8_t* pImageBuffer = (uint8_t*) res->GetBuffer();
        formatConverter.Convert(pylonImage, res);
        memcpy(frame.ptr(), pylonImage.GetBuffer(), w * h * 3);

        //std::cout << "Gray value of first pixel: " << (uint32_t) pImageBuffer[0] << std::endl << std::endl;
        return frame;
    }
    else
    {
         std::cout << "Error: " << std::hex << res->GetErrorCode()
                  << std::dec << " " << res->GetErrorDescription() << std::endl;
    }

    return cv::Mat();
}
