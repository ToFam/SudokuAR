#include "HoughTransform.h"

#include <opencv2/opencv.hpp>
#include <vector>

#include <opencv2/core/ocl.hpp>

#include <Timer.h>

HoughTransform::HoughTransform() : Algorithm("HoughTransform")
{
    ContainerSpecification input("input", ContainerSpecification::READ_ONLY);
    ContainerSpecification output("out_lines", ContainerSpecification::REFERENCE);
    m_argumentsSpecification.push_back(input);
    m_argumentsSpecification.push_back(output);

    m_settings.add(Option("threshold", OptionValue<int>(100, 100, 0, 10000)));

    m_settings.add(Option("theta-resolution", OptionValue<double>(1.0, 1.0, 0.01, 90.0)));
    m_settings.add(Option("rho-resolution", OptionValue<double>(1.0, 1.0, 0.01, 10.0)));
}

std::vector<Algorithm::ImplementationType> HoughTransform::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(OPENCV_GPU);
    return v;
}

bool HoughTransform::exec()
{
    std::shared_ptr<cv::Mat> input = m_arguments[0]->get();
    std::shared_ptr<cv::Mat> out_lines = m_arguments[1]->get();

    int threshold = m_settings.get("threshold").valueInt().value();

    const double rhoRes = m_settings.get("rho-resolution").valueDouble().value();
    const double thetaRes = m_settings.get("theta-resolution").valueDouble().value() * CV_PI / 180.0;

    cv::UMat im, li;
    input->copyTo(im);

    cv::HoughLines( im, li, rhoRes, thetaRes, threshold);

    if (!li.empty())
    {
        li.copyTo(*out_lines);
    }

    return true;
}
