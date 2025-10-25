#include "Gray.h"

#include <opencv2/opencv.hpp>

Gray::Gray() : Algorithm("Gray")
{
    ContainerSpecification input("input", ContainerSpecification::READ_ONLY);
    ContainerSpecification output("output", ContainerSpecification::REFERENCE);
    m_argumentsSpecification.push_back(input);
    m_argumentsSpecification.push_back(output);
}

std::vector<Algorithm::ImplementationType> Gray::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(OPENCV_GPU);
    return v;
}

bool Gray::exec()
{
    std::shared_ptr<cv::Mat> input = m_arguments[0]->get();
    std::shared_ptr<cv::Mat> output = m_arguments[1]->get();

    cv::cvtColor( *input.get(), *output.get(), cv::COLOR_RGB2GRAY );

    return true;
}
