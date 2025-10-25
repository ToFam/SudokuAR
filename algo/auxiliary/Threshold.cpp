#include "Threshold.h"

#include <Timer.h>

#include <opencv2/opencv.hpp>

Threshold::Threshold() : Algorithm("Threshold")
{
    ContainerSpecification input("input", ContainerSpecification::READ_ONLY);
    ContainerSpecification output("output", ContainerSpecification::REFERENCE);
    m_argumentsSpecification.push_back(input);
    m_argumentsSpecification.push_back(output);

    m_settings.add(Option("Gauss", OptionValue<bool>(true, true)));
    m_settings.add(Option("blockRadius", OptionValue<int>(5, 5, 1, 255)));
    m_settings.add(Option("C", OptionValue<double>(4.0, 4.0, -255.0, 255.0)));
}

std::vector<Algorithm::ImplementationType> Threshold::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(OPENCV_CPU);
    v.push_back(GPU);
    return v;
}

bool Threshold::exec()
{
    std::shared_ptr<cv::Mat> out_lines = m_arguments[1]->get();

    std::shared_ptr<cv::Mat> input = m_arguments[0]->get();
    std::shared_ptr<cv::Mat> output = m_arguments[1]->get();

    bool gauss = m_settings.get("Gauss").valueBool().value();
    double C = m_settings.get("C").valueDouble().value();
    int radius = m_settings.get("blockRadius").valueInt().value();

    Timer t;
    cv::adaptiveThreshold(*input, *output, 255.0, gauss ? cv::ADAPTIVE_THRESH_GAUSSIAN_C : cv::ADAPTIVE_THRESH_MEAN_C,
                          cv::THRESH_BINARY_INV, radius * 2 + 1, C);


    std::cout << "thresh: " << t.restart() << std::endl;


    return true;
}
