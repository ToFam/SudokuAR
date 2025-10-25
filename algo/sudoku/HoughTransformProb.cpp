#include "HoughTransformProb.h"

#include <opencv2/opencv.hpp>
#include <vector>

#include <Timer.h>

HoughTransformProb::HoughTransformProb() : Algorithm("HoughTransformProb")
{
    ContainerSpecification input("input", ContainerSpecification::READ_ONLY);
    ContainerSpecification output("out_lines", ContainerSpecification::REFERENCE);
    m_argumentsSpecification.push_back(input);
    m_argumentsSpecification.push_back(output);

    m_settings.add(Option("threshold", OptionValue<int>(100, 100, 0, 10000)));
    m_settings.add(Option("minlength", OptionValue<int>(50, 50, 0, 10000)));
    m_settings.add(Option("maxgap", OptionValue<int>(10, 10, 0, 1000)));
}

std::vector<Algorithm::ImplementationType> HoughTransformProb::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(OPENCV_CPU);
    v.push_back(OPENCV_GPU);
    return v;
}

bool HoughTransformProb::exec()
{
    if (!m_implSet)
    {
        setImplementation(OPENCV_CPU);
    }

    std::shared_ptr<cv::Mat> input = m_arguments[0]->get();
    std::shared_ptr<cv::Mat> out_lines = m_arguments[1]->get();

    int threshold = m_settings.get("threshold").valueInt().value();
    int minlength = m_settings.get("minlength").valueInt().value();
    int maxgap = m_settings.get("maxgap").valueInt().value();

    if (m_activeImpl == OPENCV_CPU)
    {
        std::vector<cv::Vec4i> lines;

        Timer t;
        cv::HoughLinesP(*input, lines, 1.0, CV_PI/180.0, threshold, minlength, maxgap);
        std::cout << "houghlinesP (CPU): " << t.elapsed() << "ms " << std::endl;

        if (lines.size() > 0)
        {
            cv::Mat(lines).copyTo(*out_lines);
            return true;
        }
    }
    else if (m_activeImpl == OPENCV_GPU)
    {
        cv::UMat im;// = input->getUMat(cv::ACCESS_READ);
        input->copyTo(im);
        cv::UMat ou;
        Timer t;
        cv::HoughLinesP(im, ou, 1.0, CV_PI/180.0, threshold, minlength, maxgap);
        std::cout << "houghlinesP (GPU): " << t.elapsed() << "ms " << std::endl;

        if (ou.rows > 0)
        {
            ou.copyTo(*out_lines);
            return true;
        }
    }

    return false;
}
