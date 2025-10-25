#include "HoughLinesDisplay.h"

#include <opencv2/opencv.hpp>

HoughLinesDisplay::HoughLinesDisplay() : Algorithm("HoughLinesDisplay")
{
    ContainerSpecification in_image("in_binary", ContainerSpecification::READ_ONLY);
    ContainerSpecification in_lines("in_houghlines", ContainerSpecification::READ_ONLY);
    ContainerSpecification output("out_image", ContainerSpecification::REFERENCE);
    m_argumentsSpecification.push_back(in_image);
    m_argumentsSpecification.push_back(in_lines);
    m_argumentsSpecification.push_back(output);

    m_settings.add(Option("colorindex", OptionValue<int>(0, 0, 0, 5)));
}

std::vector<Algorithm::ImplementationType> HoughLinesDisplay::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(CPU);
    return v;
}

bool HoughLinesDisplay::exec()
{
    std::shared_ptr<cv::Mat> in_image = m_arguments[0]->get();
    std::shared_ptr<cv::Mat> in_lines = m_arguments[1]->get();
    std::shared_ptr<cv::Mat> out_image = m_arguments[2]->get();

    if (in_image.get() != out_image.get())
    {
        if (in_image->channels() == 1)
        {
            cv::cvtColor(*in_image, *out_image, cv::COLOR_GRAY2BGR);
        }
        else
        {
            in_image->copyTo(*out_image);
        }
    }

    int colorIndex = m_settings.get("colorindex").valueInt().value();
    std::vector<cv::Scalar> colors{ cv::Scalar(0, 0, 255),
                                    cv::Scalar(0, 255, 255),
                                    cv::Scalar(255, 0, 255),
                                    cv::Scalar(255, 0, 0),
                                    cv::Scalar(255, 255, 0),
                                    cv::Scalar(0, 255, 0)
    };

    if (colorIndex == 1)
    {
        for (size_t i = 0; i < in_lines->rows; ++i)
        {
            std::cout << (*in_lines).at<cv::Vec2f>(i)[0] << std::endl;
        }
    }

    cv::Mat& lineMat = *in_lines;

    int channels = lineMat.channels();

    int bignum = in_image->rows + in_image->cols;

    if (channels == 2)
    {
        std::vector<cv::Vec2f> lines = *in_lines;

        for( size_t i = 0; i < lines.size(); i++ )
        {
            float rho = lines[i][0], theta = lines[i][1];
            cv::Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + bignum*(-b));
            pt1.y = cvRound(y0 + bignum*(a));
            pt2.x = cvRound(x0 - bignum*(-b));
            pt2.y = cvRound(y0 - bignum*(a));
            cv::line( *out_image, pt1, pt2, colors[colorIndex], 2, cv::LINE_AA);
        }
    }
    else if (channels == 4)
    {
        std::vector<cv::Vec4i> lines = *in_lines;

        for (size_t i = 0; i < lines.size(); ++i)
        {
            cv::Point pt1, pt2;
            pt1.x = lines[i][0];
            pt1.y = lines[i][1];
            pt2.x = lines[i][2];
            pt2.y = lines[i][3];
            cv::line( *out_image, pt1, pt2, colors[colorIndex], 2, cv::LINE_AA);
        }
    }

    return true;
}
