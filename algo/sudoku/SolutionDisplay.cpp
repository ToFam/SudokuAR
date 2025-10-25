#include "SolutionDisplay.h"

#include <opencv2/opencv.hpp>
#include <vector>

#include "../util.h"

SolutionDisplay::SolutionDisplay(const OCR& ocr) : Algorithm("SolutionDisplay"), m_ocr(ocr)
{
    ContainerSpecification inputImage("in_image", ContainerSpecification::READ_ONLY);
    ContainerSpecification inputFrames("in_frames", ContainerSpecification::READ_ONLY);
    ContainerSpecification inputNumbers("in_numbers", ContainerSpecification::READ_ONLY);
    ContainerSpecification inputSolved("in_solved", ContainerSpecification::READ_ONLY);
    ContainerSpecification outputImage("out_image", ContainerSpecification::READ_ONLY);

    m_argumentsSpecification.push_back(inputImage);
    m_argumentsSpecification.push_back(inputFrames);
    m_argumentsSpecification.push_back(inputNumbers);
    m_argumentsSpecification.push_back(inputSolved);
    m_argumentsSpecification.push_back(outputImage);

    m_settings.add(Option("frame_index", OptionValue<int>(-1, -1, -1, 100)));
}

std::vector<Algorithm::ImplementationType> SolutionDisplay::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(OPENCV_GPU);
    return v;
}

bool SolutionDisplay::exec()
{
    std::shared_ptr<cv::Mat> in_image = m_arguments[0]->get();
    std::shared_ptr<cv::Mat> in_frames = m_arguments[1]->get();
    auto in_numbers = m_arguments[2];
    auto in_solved = m_arguments[3];
    std::shared_ptr<cv::Mat> out_image = m_arguments[4]->get();

    if (in_numbers->size() != in_solved->size())
        return false;

    int frameIndex = m_settings.get("frame_index").valueInt().value();

    *out_image = in_image->clone();

    int n = 0;
    for (size_t i = 0; i < in_frames->rows && n < in_numbers->size(); ++i)
    {
        if (frameIndex > -1 && frameIndex != i)
            continue;

        if (in_frames->at<cv::Vec2f>(i, 0)[0] < 0.f)
            continue;

        std::vector<int> numbers(*in_numbers->get(n)), solved(*in_solved->get(n));
        ++n;

        if (numbers.size() == 0 || solved.size() == 0)
            continue;

        render(i, *in_image, *in_frames, numbers, solved, *out_image);
    }

    return true;
}

void SolutionDisplay::render(int frameNum, cv::Mat inImage, cv::Mat frames, std::vector<int> &numbers, std::vector<int> &solved, cv::Mat &outImage)
{
    double dist = m_ocr.cellResolution();
    double size = dist * 0.80;
    double margin = dist * 0.10;

    double tmpSize = dist * 9.0;
    cv::Mat tmpImage(tmpSize, tmpSize, CV_8U, cv::Scalar(0));

    for (int row = 0; row < 9; ++row)
    {
        for (int col = 0; col < 9; ++col)
        {
            int left = col * dist + margin;
            int top =  row * dist + margin;
            int right = left + size;
            int bottom = top + size;

            cv::Point2i p1(left, top);
            cv::Point2i p2(right, bottom);


            cv::Mat target;

            int number = numbers[row * 9 + col];

            if (number > -1)
                continue;

            number = solved[row * 9 + col];
            const Digit* digit = nullptr;
            for (const Digit& d : m_ocr.digits())
            {
                if (d.value == number)
                {
                    digit = &d;
                    break;
                }
            }
            if (digit == nullptr)
                continue;


            cv::Mat detected;
            cv::Size s(size, size);
            cv::resize(digit->templ, detected, s);

            detected.copyTo(tmpImage.rowRange(top, bottom).colRange(left, right));
        }
    }

    cv::Vec2f v0 = frames.at<cv::Vec2f>(frameNum, 0);
    cv::Vec2f v1 = frames.at<cv::Vec2f>(frameNum, 1);
    cv::Vec2f v2 = frames.at<cv::Vec2f>(frameNum, 2);
    cv::Vec2f v3 = frames.at<cv::Vec2f>(frameNum, 3);

    cv::Point2f src[4];
    cv::Point2f dst[4];

    dst[0] = cv::Point2f(v0[0], v0[1]);
    dst[1] = cv::Point2f(v1[0], v1[1]);
    dst[2] = cv::Point2f(v2[0], v2[1]);
    dst[3] = cv::Point2f(v3[0], v3[1]);

    arrangePoints(dst);;

    src[0] = cv::Point2f(tmpSize, tmpSize);
    src[1] = cv::Point2f(  0.f,    tmpSize);
    src[2] = cv::Point2f(  0.f,    0.f);
    src[3] = cv::Point2f(tmpSize, 0.f);

    cv::Mat t = getPerspectiveTransform(src, dst);
    cv::Mat warpedImage;
    warpPerspective(tmpImage, warpedImage, t, cv::Size(inImage.cols, inImage.rows));

    cv::Mat color(warpedImage.size(), CV_8UC3, cv::Scalar(0, 0, 255));


    color.copyTo(outImage, warpedImage);

    //cv::Mat empty = cv::Mat::zeros(warpedImage.size(), warpedImage.depth());
    //std::vector<cv::Mat> channels(3, empty);
    //channels.at(0) = warpedImage;
    //
    //cv::Mat warpedColor, output;
    //cv::merge(channels, warpedColor);
    //
    //cv::addWeighted(*in_image, 0.5, warpedColor, 0.5, 0.0, output);
}
