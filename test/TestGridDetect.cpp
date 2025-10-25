#include "TestGridDetect.h"

#include <opencv2/opencv.hpp>

#include <util.h>

#include <iostream>

using namespace std;

double minAngleDeg = 80.0;

void outputLocalNeighborhood(cv::Mat hist, float index)
{
    cv::Mat mIndx(1, 5, CV_8U);
    cv::Mat mVal(1, 5, CV_8U);
    for (int i = 0; i < 5; ++i)
    {
        int indx = std::min(std::max(0, (int)index - 2 + i), hist.cols - 1);
        mIndx.at<unsigned char>(0, i) = indx;
        mVal.at<unsigned char>(0, i) = (int)hist.at<unsigned char>(0, indx);
    }
    cout << index << endl << mIndx << endl << mVal << endl;
}

void testLocalMax(cv::Mat hist)
{
    int histSize = 180;

    for (int i = 0; i < histSize; ++i)
    {
        hist.at<unsigned char>(0, i + histSize) = hist.at<unsigned char>(0, i);
    }

    cv::Mat angleHistogramSmoothed = cv::Mat(1, histSize * 2, CV_8U);
    cv::GaussianBlur(hist, angleHistogramSmoothed, cv::Size(5, 5), 0, 0, cv::BORDER_REPLICATE);

    cv::Mat data = angleHistogramSmoothed.colRange(histSize / 2, histSize / 2 + histSize);
    std::cout << data << std::endl;


    auto maxima = localMax(data);
    for (float f : maxima)
    {
        outputLocalNeighborhood(data, f);
    }
}

void testDualAnglePeak(cv::Mat hist)
{
    int m1, m2;
    dualAnglePeak(hist, minAngleDeg, m1, m2);

    cout << "algo: m1: " << endl;
    outputLocalNeighborhood(hist, m1);
    cout << "m2:" << endl;
    outputLocalNeighborhood(hist, m2);
}

void TestGridDetect::execTest()
{
    cout << "test grid detect" << endl;

    for (int i = 0; i < 5; ++i)
    {
        std::string file("eval/grid_detect/");
        file += std::to_string(i);
        file += ".yml";

        cv::FileStorage fs(file, cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            cout << "Cannot open file " << file << endl;
            continue;
        }

        cv::Mat hist;
        fs["hist"] >> hist;

        int expectedM1, expectedM2;
        fs["m1"] >> expectedM1;
        fs["m2"] >> expectedM2;

        testLocalMax(hist);
        cout << "----------...------------" << endl;

        testDualAnglePeak(hist);

        cout << "expected: m1: " << endl;
        outputLocalNeighborhood(hist, expectedM1);
        cout << "m2:" << endl;
        outputLocalNeighborhood(hist, expectedM2);
        cout << "-------------------------" << endl;
        cout << "-------------------------" << endl;
    }
}
