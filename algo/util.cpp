#include "util.h"
using namespace std;
using namespace cv;

std::vector<float> localMax(cv::Mat data)
{
    unsigned char v0, v1, v2;

    std::vector<float> rtn;

    int size = data.cols;

    int peakSize = 1;
    for (int i = 0; i < size; i++)
    {
        unsigned char v = data.at<unsigned char>(0, i);
        if (i == 0)
        {
            v0 = v - 1;
            peakSize = 1;
        }
        else if (v == v1)
        {
            peakSize++;
        }
        else
        {
            v0 = v1;
            peakSize = 1;
        }
        v1 = v;

        if (i < size - 1)
        {
            v2 = data.at<unsigned char>(0, i + 1);
        }
        else
        {
            v2 = v1 - 1;
        }

        if ((v1 > v0) && (v1 > v2))
        {
            double d = (peakSize - 1.0) / 2.0;
            rtn.push_back(i - d);
        }
    }

    return rtn;
}

void dualAnglePeak(cv::Mat histogram, int minDist, int& maxPeak, int& secondPeak)
{
    int histSize = 180;

    // replicate entries in second histogram half
    for (int i = 0; i < histSize; ++i)
    {
        histogram.at<unsigned char>(0, i + histSize) = histogram.at<unsigned char>(0, i);
    }
    //std::cout << histogram << std::endl;

    // Smooth histogram and shift to middle
    cv::Mat angleHistogramSmoothed = cv::Mat(1, histSize * 2, CV_8U);
    cv::GaussianBlur(histogram, angleHistogramSmoothed, cv::Size(5, 5), 0, 0, cv::BORDER_REPLICATE);
    //std::cout << angleHistogramSmoothed << std::endl;

    cv::Mat data = angleHistogramSmoothed.colRange(histSize / 2, histSize / 2 + histSize);


    std::vector<float> peaks = localMax(data);
    std::sort(peaks.begin(), peaks.end(), [&](float f1, float f2){
       return data.at<unsigned char>((int)f1) > data.at<unsigned char>((int)f2);
    });

    if (peaks.empty())
    {
        maxPeak = -1;
        secondPeak = -1;
        return;
    }

    /*
    for (int i = 0; i < peaks.size(); ++i)
    {
        std::cout << peaks[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << "minDist " << minDist << std::endl;
    */

    int mX0 = peaks[0];
    int mX1 = -1;
    for (int p = 1; p < peaks.size(); ++p)
    {
        int i = peaks[p];
        int d1, d2;
        if (i < mX0)
        {
            d1 = mX0 - i;
            d2 = i - (mX0 - histSize);
        }
        else
        {
            d1 = i - mX0;
            d2 = mX0 - (i - histSize);
        }

        //std::cout << i << ": " << d1 << ", " << d2 << std::endl;

        if (d1 > minDist && d2 > minDist)
        {
            mX1 = i;
            break;
        }
    }

    // Shift back to input histogram indices
    maxPeak = (mX0 + histSize / 2) % histSize;
    if (mX1 < 0) secondPeak = -1;
    else secondPeak = (mX1 + histSize / 2) % histSize;

    // Find true max value in unsmoothed histogram
    cv::Point loc;
    int win_l = std::max(0, maxPeak - 2);
    int win_r = std::min(histSize, maxPeak + 3);
    cv::Mat window = histogram.colRange(win_l, win_r);
    cv::minMaxLoc(window, 0, 0, 0, &loc);
    maxPeak = win_l + loc.x;

    if (secondPeak >= 0)
    {
        win_l = std::max(0, secondPeak - 2);
        win_r = std::min(histSize, secondPeak + 3);
        cv::Mat window = histogram.colRange(win_l, win_r);
        cv::minMaxLoc(window, 0, 0, 0, &loc);
        secondPeak = win_l + loc.x;
    }
}
