#include "Canny.h"

#include <Timer.h>

#include <opencv2/opencv.hpp>

Canny::Canny() : Algorithm("Canny")
{
    ContainerSpecification input("input", ContainerSpecification::READ_ONLY);
    ContainerSpecification output("output", ContainerSpecification::REFERENCE);
    m_argumentsSpecification.push_back(input);
    m_argumentsSpecification.push_back(output);

    m_settings.add(Option("thresh1", OptionValue<double>(100.0, 100.0, 0.0, 99999.9)));
    m_settings.add(Option("thresh2", OptionValue<double>(200.0, 200.0, 0.0, 99999.9)));
    m_settings.add(Option("blockRadius", OptionValue<int>(1, 1, 1, 3)));
    m_settings.add(Option("L2Grad", OptionValue<bool>(false)));
}

std::vector<Algorithm::ImplementationType> Canny::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(OPENCV_CPU);
    return v;
}

bool Canny::exec()
{
    std::shared_ptr<cv::Mat> input = m_arguments[0]->get();
    std::shared_ptr<cv::Mat> output = m_arguments[1]->get();

    double thresh1 = m_settings.get("thresh1").valueDouble().value();
    double thresh2 = m_settings.get("thresh2").valueDouble().value();
    int radius = m_settings.get("blockRadius").valueInt().value();
    bool l2Grad = m_settings.get("L2Grad").valueBool().value();

    cv::Canny(*input, *output, thresh1, thresh2, radius * 2 + 1, l2Grad);

    return true;
}
