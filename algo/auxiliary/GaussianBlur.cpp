#include "GaussianBlur.h"

#include <opencv2/opencv.hpp>

GaussianBlur::GaussianBlur() : Algorithm("GaussianBlur")
{
    ContainerSpecification input("input", ContainerSpecification::READ_ONLY);
    ContainerSpecification output("output", ContainerSpecification::REFERENCE);
    m_argumentsSpecification.push_back(input);
    m_argumentsSpecification.push_back(output);

    m_settings.add(Option("iterations", OptionValue<int>(1, 1, 0, 10)));
    m_settings.add(Option("kernel-radius", OptionValue<int>(1, 1, 1, 10)));
    m_settings.add(Option("sigma", OptionValue<double>(1.0, 1.0, 0.0, 100.0)));
}

std::vector<Algorithm::ImplementationType> GaussianBlur::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(OPENCV_GPU);
    return v;
}

bool GaussianBlur::exec()
{
    std::shared_ptr<cv::Mat> input = m_arguments[0]->get();
    std::shared_ptr<cv::Mat> output = m_arguments[1]->get();

    int iterations = m_settings.get("iterations").valueInt().value();
    int kernelSize = m_settings.get("kernel-radius").valueInt().value() * 2 + 1;
    double sigma = m_settings.get("sigma").valueDouble().value();

    if (iterations > 0)
    {
        cv::Size k(kernelSize, kernelSize);

        cv::GaussianBlur(*input, *output, k, sigma, sigma);

        for (int i = 1; i < iterations; ++i)
        {
            cv::GaussianBlur(*output, *output, k, sigma, sigma);
        }
    }

    return true;
}
