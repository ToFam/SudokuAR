#include "Close.h"

#include <opencv2/opencv.hpp>

Close::Close() : Algorithm("Close")
{
    ContainerSpecification input("input", ContainerSpecification::READ_ONLY);
    ContainerSpecification output("output", ContainerSpecification::REFERENCE);
    m_argumentsSpecification.push_back(input);
    m_argumentsSpecification.push_back(output);

    m_settings.add(Option("iterations", OptionValue<int>(1, 1, 0, 10)));
    m_settings.add(Option("kernel-radius", OptionValue<int>(1, 1, 1, 10)));
    m_settings.add(Option("erode", OptionValue<bool>(true, true)));
}

std::vector<Algorithm::ImplementationType> Close::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(OPENCV_GPU);
    return v;
}

bool Close::exec()
{
    int iterations = m_settings.get("iterations").valueInt().value();

    if (iterations > 0)
    {
        std::shared_ptr<cv::Mat> input = m_arguments[0]->get();
        std::shared_ptr<cv::Mat> output = m_arguments[1]->get();

        int kernelSize = m_settings.get("kernel-radius").valueInt().value() * 2 + 1;
        bool erode = m_settings.get("erode").valueBool().value();

        cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
        cv::morphologyEx(*input, *output, erode ? cv::MORPH_CLOSE : cv::MORPH_DILATE, kernel, cv::Point(-1, -1), iterations);
    }

    return true;
}
