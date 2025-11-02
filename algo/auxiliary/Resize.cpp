#include "Resize.h"

#include <opencv2/opencv.hpp>

Resize::Resize() : Algorithm("Resize")
{
    ContainerSpecification input("input", ContainerSpecification::READ_ONLY);
    ContainerSpecification output("output", ContainerSpecification::REFERENCE);
    m_argumentsSpecification.push_back(input);
    m_argumentsSpecification.push_back(output);

    m_settings.add(Option("max_width", OptionValue<int>(1920, 1920, 1, 99999)));
    m_settings.add(Option("max_height", OptionValue<int>(1080, 1080, 1, 99999)));
}

std::vector<Algorithm::ImplementationType> Resize::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(OPENCV_CPU);
    return v;
}

bool Resize::exec()
{
    int maxWidth = m_settings.get("max_width").valueInt().value();
    int maxHeight = m_settings.get("max_height").valueInt().value();

    std::shared_ptr<cv::Mat> input = m_arguments[0]->get();
    std::shared_ptr<cv::Mat> output = m_arguments[1]->get();

    int inWidth = input->size().width;
    int inHeight = input->size().height;

    if (inWidth <= maxWidth && inHeight <= maxHeight)
    {
        input->copyTo(*output);
        return true;
    }

    double sw = maxWidth / static_cast<double>(inWidth);
    double sh = maxHeight / static_cast<double>(inHeight);
    double s = std::min(sw, sh);

    cv::Size newSize(static_cast<int>(s * inWidth), static_cast<int>(s * inHeight));

    //std::cout << newSize.width << " x " << newSize.height << std::endl;

    cv::resize(*input, *output, newSize);

    return true;
}
