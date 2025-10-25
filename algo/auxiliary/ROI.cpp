#include "ROI.h"

#include <Timer.h>
#include <opencv2/opencv.hpp>

ROI::ROI() : Algorithm("ROI")
{
    ContainerSpecification input("input", ContainerSpecification::READ_ONLY);
    ContainerSpecification output("output", ContainerSpecification::REFERENCE);
    m_argumentsSpecification.push_back(input);
    m_argumentsSpecification.push_back(output);

    m_settings.add(Option("x", OptionValue<int>(0, 0, 0, 99999)));
    m_settings.add(Option("y", OptionValue<int>(0, 0, 0, 99999)));
    m_settings.add(Option("width", OptionValue<int>(0, 0, 0, 99999)));
    m_settings.add(Option("height", OptionValue<int>(0, 0, 0, 99999)));
}

std::vector<Algorithm::ImplementationType> ROI::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(OPENCV_CPU);
    return v;
}

bool ROI::exec()
{
    std::shared_ptr<cv::Mat> input = m_arguments[0]->get();
    std::shared_ptr<cv::Mat> output = m_arguments[1]->get();

    int x = m_settings.get("x").valueInt().value();
    int y = m_settings.get("y").valueInt().value();
    int width = m_settings.get("width").valueInt().value();
    int height = m_settings.get("height").valueInt().value();

    cv::Mat& in = *input;

    if (x > in.cols - 1 || y > in.rows - 1
       || x + width > in.cols || y + height > in.rows)
    {
        return false;
    }

    in.rowRange(y, y + height).colRange(x, y + width).copyTo(*output);

    return true;
}
