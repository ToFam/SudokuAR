#include "TemplateMatch.h"

#include <opencv2/opencv.hpp>
#include <vector>

#include "util.h"
#include "Timer.h"

bool matchSingleDigit(cv::Mat& img, int row, int col,
                      double dist, double size, double margin,
                      const std::vector<Digit>& digits, double ratioDiscardFactor,
                      int& outDigit, double& outScoreDigit)
{
    int left = col * dist + margin;
    int top =  row * dist + margin;
    int right = left + size;
    int bottom = top + size;

    //cv::UMat target = img.rowRange(top, bottom).colRange(left, right);
    cv::Mat target;
    img.rowRange(top, bottom).colRange(left, right).copyTo(target);

    std::vector<double> scores;
    double maxScore = 0.f;
    int maxScoreDigit = -1;

    for (const Digit& d : digits)
    {
        cv::Mat res;

        if (target.rows < d.templ.rows ||
            target.cols < d.templ.cols )
        {
            maxScoreDigit = -1;
            break;
        }

        cv::matchTemplate(target, d.templ, res, cv::TM_CCORR_NORMED);
        double max;
        cv::minMaxLoc(res, nullptr, &max, nullptr, nullptr);

        scores.push_back(max);
        if (max > maxScore)
        {
            maxScore = max;
            maxScoreDigit = d.value;
        }
    }

    if (scores.size() > 1)
    {
        std::sort(scores.begin(), scores.end());

        if (*(scores.end() - 2) > ratioDiscardFactor * maxScore)
        {
            std::cout << "discard. (" << maxScoreDigit << ", score: " << maxScore << ", next best: " << *(scores.end() - 2) << ")" << std::endl;
            maxScoreDigit = -1;
        }
    }

    outScoreDigit = maxScore;
    outDigit = maxScoreDigit;

    return maxScoreDigit > -1;
}

std::vector<std::pair<int, double>> matchDirection(cv::Mat& img, double dist, double size, double margin, const std::vector<Digit>& digits,
                                                   double templateDiscardFactor, double ratioDiscardFactor, double& outAvgScore)
{
    std::vector<std::pair<int, double>> guesses;

    double maxOverallScore = 0.;
    outAvgScore = 0.;
    int guessSize = 0;
    for (int row = 0; row < 9; row++)
    {
        for (int col = 0; col < 9; col++)
        {
            double scoreDigit;
            int digit;
            bool ok = matchSingleDigit(img, row, col,
                                       dist, size, margin,
                                       digits, ratioDiscardFactor, digit, scoreDigit);

            /*
            double avg = 0., variance = 0.;
            for (double s : scores)
                avg += s;
            avg /= scores.size();

            for (double s : scores)
                variance += pow(avg - s, 2);
            variance /= scores.size();

            double stdDeviation = sqrt(variance);
            */

            if (ok && scoreDigit > templateDiscardFactor)// maxScore > m_ocr.scoreThreshold() // && maxScore > avg + stdDeviation)
            {
                if (scoreDigit > maxOverallScore)
                {
                    maxOverallScore = scoreDigit;
                }

                guesses.push_back(std::pair<int, double>(digit, scoreDigit));
                outAvgScore += scoreDigit;
                guessSize++;
            }
            else
            {
                guesses.push_back(std::pair<int, double>(-1, 0.0));
            }
        }
    }

    outAvgScore /= guessSize;

    return guesses;
}

void TemplateMatch::matchOCVGPU(cv::Mat image, cv::Mat &outNumbers, Rotation &outRot)
{
    double cm = m_settings.get("cellMargin").valueDouble().value();
    double dist = image.rows / 9.0;
    double size = dist * (1.0 - 2 * cm);
    double margin = dist * cm;

    double templateDF = m_settings.get("templateDiscardFactor").valueDouble().value();
    double ratioDF = m_settings.get("ratioDiscardFactor").valueDouble().value();

    std::vector<Digit> digits[4];
    m_ocr.scaled(dist, digits[0]);
    m_ocr.scaledRotated(cv::ROTATE_90_CLOCKWISE, dist, digits[1]);
    m_ocr.scaledRotated(cv::ROTATE_180, dist, digits[2]);
    m_ocr.scaledRotated(cv::ROTATE_90_COUNTERCLOCKWISE, dist, digits[3]);

    std::vector<std::vector<std::pair<int, double>>> directionGuesses;

    std::vector<double> avgScore(4);
    directionGuesses.push_back(matchDirection(image, dist, size, margin, digits[0], templateDF, ratioDF, avgScore[0]));
    directionGuesses.push_back(matchDirection(image, dist, size, margin, digits[1], templateDF, ratioDF, avgScore[1]));
    directionGuesses.push_back(matchDirection(image, dist, size, margin, digits[2], templateDF, ratioDF, avgScore[2]));
    directionGuesses.push_back(matchDirection(image, dist, size, margin, digits[3], templateDF, ratioDF, avgScore[3]));

    for (int i = 0; i < 4; ++i)
    {
        std::cout << avgScore[i] << ", ";
    }

    int idx[2];
    cv::minMaxIdx(avgScore, nullptr, nullptr, nullptr, idx);
    int bestDirection = idx[1];

    outRot = static_cast<Rotation>(bestDirection);

    std::cout << " best: " << bestDirection << std::endl;

    std::vector<int> numbers;
    for (int row = 0; row < 9; row++)
    {
        for (int col = 0; col < 9; col++)
        {
            auto& guess = directionGuesses[bestDirection][row * 9 + col];
            numbers.push_back(guess.first);
        }
    }

    cv::Mat(numbers).copyTo(outNumbers);
}
