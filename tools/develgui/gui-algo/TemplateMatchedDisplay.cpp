#include "TemplateMatchedDisplay.h"

#include <opencv2/opencv.hpp>

TemplateMatchedDisplay::TemplateMatchedDisplay(const OCR& ocr) : Algorithm("TemplateMatchedDisplay"), m_ocr(ocr)
{
    ContainerSpecification in_image("in_frame", ContainerSpecification::READ_ONLY);
    ContainerSpecification in_numbers("in_numbers", ContainerSpecification::READ_ONLY);
    ContainerSpecification in_rot("in_rotations", ContainerSpecification::READ_ONLY);
    ContainerSpecification out_image("out_image", ContainerSpecification::REFERENCE);


    m_argumentsSpecification.push_back(in_image);
    m_argumentsSpecification.push_back(in_numbers);
    m_argumentsSpecification.push_back(in_rot);
    m_argumentsSpecification.push_back(out_image);
}

std::vector<Algorithm::ImplementationType> TemplateMatchedDisplay::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(CPU);
    return v;
}

void drawOutput(cv::Mat& img, cv::Mat& outImg, const std::vector<int>& numbers,
                const std::vector<Digit>& digits, double dist, double size, double margin)
{
    for (int row = 0; row < 9; row++)
    {
        for (int col = 0; col < 9; col++)
        {
            auto& guess = numbers[row * 9 + col];

            int left = col * dist + margin;
            int top =  row * dist + margin;
            int right = left + size;
            int bottom = top + size;

            cv::Point2i p1(left, top);
            cv::Point2i p2(right, bottom);
            cv::rectangle(outImg, p1, p2, cv::Scalar(255, 0, 0));

            cv::Mat target;
            img.rowRange(top, bottom).colRange(left, right).copyTo(target);

            for (const Digit& d : digits)
            {
                if (d.value == guess)
                {
                    cv::Mat detected;
                    cv::Size s(target.cols, target.rows);
                    cv::resize(d.templ, detected, s);

                    cv::Mat bla;
                    addWeighted(detected, 0.5, target, 0.5, 0, bla);
                    cvtColor(bla, outImg.rowRange(top, bottom).colRange(left, right), cv::COLOR_GRAY2BGR);

                    break;
                }
            }
        }
    }
}

bool TemplateMatchedDisplay::exec()
{
    if (m_ocr.digits().size() == 0)
    {
        throw std::runtime_error("No ocr loaded");
    }

    auto in = m_arguments[0];
    auto inNumbers = m_arguments[1];

    std::vector<int> rotations(*m_arguments[2]->get());

    if (inNumbers->size() < in->size() || rotations.size() < in->size())
    {
        throw std::runtime_error(std::format("Mismatched input count: {} frames, {} numbers, {} rotations",
                                             in->size(), inNumbers->size(), rotations.size()));
    }

    for (size_t i = 0; i < in->size(); ++i)
    {
        cv::Mat inImage = *in->get(i);

        std::vector<int> numbers(*inNumbers->get(i));

        if (inImage.empty() || numbers.size() == 0)
        {
            std::cout << "Template Display skipping input " << i << std::endl;
            continue;
        }

        auto outImage = std::make_shared<cv::Mat>();

        cvtColor( inImage, *outImage, cv::COLOR_GRAY2BGR );

        double dist = inImage.rows / 9.0;
        double size = dist * 0.9;
        double margin = dist * 0.05;

        std::vector<Digit> digits;

        if (rotations[i] != static_cast<int>(Rotation::Zero))
        {
            m_ocr.scaledRotated(static_cast<cv::RotateFlags>(rotations[i] - 1), dist, digits);
        }
        else
        {
            m_ocr.scaled(dist, digits);
        }

        drawOutput(inImage, *outImage, numbers, digits, dist, size, margin);

        m_arguments[3]->add(outImage);
    }

    return true;
}
