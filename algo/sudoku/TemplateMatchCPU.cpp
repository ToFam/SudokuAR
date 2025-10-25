#include "TemplateMatch.h"

#include <opencv2/opencv.hpp>
#include <vector>

#include "util.h"
#include "Timer.h"

void matchTemplateCPU(cv::Mat& img, cv::Mat& templt, cv::Mat& outResult, cv::TemplateMatchModes mode)
{
    int w = templt.cols;
    int h = templt.rows;
    int W = img.cols;
    int H = img.rows;
    int Rw = W - w + 1;
    int Rh = H - h + 1;
    outResult = cv::Mat::zeros(Rh, Rw, CV_32F);

    double Tnorm = 0.0;
    for (int y_ = 0; y_ < h; ++y_)
    for (int x_ = 0; x_ < w; ++x_)
    {
        double v = templt.at<uchar>(y_, x_);
        Tnorm += v*v;
    }
    double TnormR = sqrt(Tnorm);

    double cvtnorm = cv::norm(templt, cv::NORM_L2SQR);
    std::cout << "norm diff " << abs(cvtnorm - Tnorm) << std::endl;
    //Tnorm = sqrt(Tnorm);

    if (mode == cv::TM_SQDIFF)
    {
        for (int y = 0; y < Rh; ++y)
        for (int x = 0; x < Rw; ++x)
        {
            float sum = 0;

            for (int y_ = 0; y_ < h; ++y_)
            for (int x_ = 0; x_ < w; ++x_)
            {
                float d = templt.at<uchar>(y_, x_) - img.at<uchar>(y + y_, x + x_);
                sum += d*d;
            }

            outResult.at<float>(y, x) = sum;
        }
    }
    else if (mode == cv::TM_SQDIFF_NORMED)
    {
        double t;
        for (int y = 0; y < Rh; ++y)
        for (int x = 0; x < Rw; ++x)
        {
            double sum = 0;
            double Inorm = 0;

            for (int y_ = 0; y_ < h; ++y_)
            for (int x_ = 0; x_ < w; ++x_)
            {
                double iv = img.at<uchar>(y + y_, x + x_);
                double d = templt.at<uchar>(y_, x_) - iv;
                sum += d*d;
                Inorm += iv*iv;
            }
            sum = std::max(sum, 0.0);

            double diff2 = MAX(Inorm, 0);
            if (diff2 <= std::min(0.5, 10 * FLT_EPSILON * Inorm))
                t = 0; // avoid rounding errors
            else
                t = std::sqrt(diff2)*TnormR;

            if( fabs(sum) < t )
                sum /= t;
            else if( fabs(sum) < t*1.125 )
                sum = sum > 0 ? 1 : -1;
            else
                sum = 1;

            outResult.at<float>(y, x) = (float)sum;
        }
    }
    else if (mode == cv::TM_CCORR)
    {
        for (int y = 0; y < Rh; ++y)
        for (int x = 0; x < Rw; ++x)
        {
            float sum = 0;

            for (int y_ = 0; y_ < h; ++y_)
            for (int x_ = 0; x_ < w; ++x_)
            {
                float d = templt.at<uchar>(y_, x_) * img.at<uchar>(y + y_, x + x_);
                sum += d;
            }

            outResult.at<float>(y, x) = sum;
        }
    }
    else if (mode == cv::TM_CCORR_NORMED)
    {
        double t;
        for (int y = 0; y < Rh; ++y)
        for (int x = 0; x < Rw; ++x)
        {
            double sum = 0;
            double Inorm = 0;

            for (int y_ = 0; y_ < h; ++y_)
            for (int x_ = 0; x_ < w; ++x_)
            {
                double iv = img.at<uchar>(y + y_, x + x_);
                double d = templt.at<uchar>(y_, x_) * iv;
                sum += d;
                Inorm += iv*iv;
            }
            sum = std::max(sum, 0.0);

            double diff2 = MAX(Inorm, 0);
            if (diff2 <= std::min(0.5, 10 * FLT_EPSILON * Inorm))
                t = 0; // avoid rounding errors
            else
                t = std::sqrt(diff2)*TnormR;

            if( fabs(sum) < t )
                sum /= t;
            else if( fabs(sum) < t*1.125 )
                sum = sum > 0 ? 1 : -1;
            else
                sum = 0;

            outResult.at<float>(y, x) = (float)sum;
        }
    }
}

void TemplateMatch::matchCPU(cv::Mat image, cv::Mat &outNumbers, Rotation &outRot, cv::Mat &outDebug)
{
    double cm = m_settings.get("cellMargin").valueDouble().value();

    double df = m_settings.get("templateDiscardFactor").valueDouble().value();

    double dist = image.rows / 9.0;
    double size = dist * (1.0 - 2 * cm);
    double margin = dist * cm;

    Timer t;
    std::vector<Digit> digits[4];
    m_ocr.scaled(dist, digits[0]);
    m_ocr.scaledRotated(cv::ROTATE_90_CLOCKWISE, dist, digits[1]);
    m_ocr.scaledRotated(cv::ROTATE_180, dist, digits[2]);
    m_ocr.scaledRotated(cv::ROTATE_90_COUNTERCLOCKWISE, dist, digits[3]);

    std::vector<std::vector<std::pair<int, double>>> directionGuesses;

    std::vector<double> avgScore(4);
    //directionGuesses.push_back(matchDirection(image, dist, size, margin, digits[0], avgScore[0]));

    std::cout << "ocr scale " << t.elapsed() << " ms" << std::endl;

    t.restart();
    cv::Mat compressed;
    cellImage(image, compressed, dist, size, margin);
    cv::UMat ucomp;
    compressed.copyTo(ucomp);
    std::cout << "resize " << t.elapsed() << " ms" << std::endl;

    t.restart();
    std::vector<cv::UMat> res(digits[0].size());
    for (int i = 0; i < digits[0].size(); ++i)
    {
        cv::UMat templ;
        digits[0][i].templ.copyTo(templ);
        cv::matchTemplate(ucomp, templ, res[i], cv::TM_CCORR_NORMED);
    }

    std::cout << "match " << t.elapsed() << " ms" << std::endl;

    t.restart();
    cv::Mat guesses = cv::Mat::zeros(9, 9, CV_8U);
    cv::Mat guessScores = cv::Mat::zeros(9, 9, CV_64F);
    for (int d = 0; d < digits[0].size(); ++d)
    {

        int rSizeX = res[d].cols / 9;
        int rSizeY = res[d].rows / 9;
    for (int i = 0; i < 9; ++i)
    for (int j = 0; j < 9; ++j)
    {
        double max;
        cv::minMaxLoc(res[d].rowRange(i*rSizeY, (i+1)*rSizeY).colRange(j*rSizeX, (j+1)*rSizeX),
                      nullptr, &max, nullptr, nullptr);
        double m = guessScores.at<double>(i, j);
        if (max > m)
        {
            guessScores.at<double>(i, j) = max;
            guesses.at<uchar>(i, j) = d;
        }
    }
    }

    std::cout << "maxloc " << t.elapsed() << " ms"
              << std::endl << guesses << std::endl << guessScores << std::endl;



    cv::Mat r = res[0].getMat(cv::ACCESS_READ);
    outDebug = cv::Mat::zeros(r.rows, r.cols, CV_8U);
    for (int i = 0; i < r.rows; ++i)
    for (int j = 0; j < r.cols; ++j)
    {
        //outDebug.at<uchar>(i, j) = static_cast<uchar>(255 * r.at<float>(i, j));
        float v = r.at<float>(i, j);
        if (v > df)
        {
            outDebug.at<uchar>(i, j) = static_cast<uchar>(255 * v);
        }
    }
}
