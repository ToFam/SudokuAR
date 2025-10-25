#include "FindPerpLines.h"

#include <opencv2/opencv.hpp>
#include <vector>

#include "LineAlgorithms.h"
#include "util.h"

#include <iostream>

FindPerpLines::FindPerpLines() : Algorithm("FindPerpLines")
{
    ContainerSpecification input("input", ContainerSpecification::READ_ONLY);
    ContainerSpecification inlines("in_lines", ContainerSpecification::READ_ONLY);
    ContainerSpecification output("out_lines", ContainerSpecification::REFERENCE);
    m_argumentsSpecification.push_back(input);
    m_argumentsSpecification.push_back(inlines);
    m_argumentsSpecification.push_back(output);

    m_settings.add(Option("threshold", OptionValue<int>(100, 100, 0, 10000)));
    m_settings.add(Option("angle-tolerance", OptionValue<double>(5.0, 5.0, 0, 180.0)));

    m_settings.add(Option("theta-resolution", OptionValue<double>(1.0, 1.0, 0.01, 90.0)));
    m_settings.add(Option("rho-resolution", OptionValue<double>(0.5, 0.5, 0.01, 10.0)));
}

std::vector<Algorithm::ImplementationType> FindPerpLines::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(OPENCV_GPU);
    return v;
}

bool FindPerpLines::exec()
{
    std::shared_ptr<cv::Mat> input = m_arguments[0]->get();
    std::shared_ptr<cv::Mat> in_lines = m_arguments[1]->get();
    std::shared_ptr<cv::Mat> out_lines = m_arguments[2]->get();

    int threshold = m_settings.get("threshold").valueInt().value();
    double angleTolerance = m_settings.get("angle-tolerance").valueDouble().value() * CV_PI / 180.0;

    if (in_lines->rows == 0)
    {
        return false;
    }

    // Find best angle
    cv::Mat hist;
    angleHistogram(*in_lines, hist);

    int maxPeak, secondPeak;
    dualAnglePeak(hist, 1, maxPeak, secondPeak);

    std::cout << "theta0: " << maxPeak << ", theta1: " << secondPeak << std::endl;

    if (maxPeak < 90)
    {
        maxPeak += 90;
    }
    else
    {
        maxPeak -= 90;
    }

    const double rhoRes = m_settings.get("rho-resolution").valueDouble().value();
    const double thetaRes = m_settings.get("theta-resolution").valueDouble().value() * CV_PI / 180.0;

    double targetTheta = maxPeak * CV_PI / 180.0;

    std::vector<cv::Vec3f> li;
    houghLines(*input, li, rhoRes, thetaRes, threshold, targetTheta, angleTolerance);

    //filterHoughLinesNonMax(li, 100, 3.14);
    //filterHoughLinesAngle(li, targetTheta, 10, 0.1);

    std::vector<cv::Vec2f> oli;
    for (auto& l : li)
    {
        oli.emplace_back(l[0], l[1]);
    }

    if (!oli.empty())
    {
        cv::Mat(oli).copyTo(*out_lines);
    }

    return true;
}
