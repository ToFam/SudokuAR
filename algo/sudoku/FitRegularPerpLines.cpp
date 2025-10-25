#include "FitRegularPerpLines.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <random>

#include "LineAlgorithms.h"
#include "util.h"

#include <iostream>

FitRegularPerpLines::FitRegularPerpLines() : Algorithm("FitRegularPerpLines")
{
    ContainerSpecification input("input", ContainerSpecification::READ_ONLY);
    ContainerSpecification inlines("in_lines", ContainerSpecification::READ_ONLY);
    ContainerSpecification output("out_lines", ContainerSpecification::REFERENCE);
    ContainerSpecification output2("out_lines_inliers", ContainerSpecification::REFERENCE);
    ContainerSpecification output3("out_lines_bestfit", ContainerSpecification::REFERENCE);
    m_argumentsSpecification.push_back(input);
    m_argumentsSpecification.push_back(inlines);
    m_argumentsSpecification.push_back(output);
    m_argumentsSpecification.push_back(output2);
    m_argumentsSpecification.push_back(output3);

    m_settings.add(Option("threshold", OptionValue<int>(100, 100, 0, 10000)));
    m_settings.add(Option("offset-angle", OptionValue<double>(90.0, 90.0, 0, 180.0)));
    m_settings.add(Option("angle-tolerance", OptionValue<double>(5.0, 5.0, 0, 180.0)));

    m_settings.add(Option("theta-resolution", OptionValue<double>(1.0, 1.0, 0.01, 90.0)));

    m_settings.add(Option("rho-resolution", OptionValue<double>(0.5, 0.5, 0.01, 10.0)));
    m_settings.add(Option("rho-resolution-fit", OptionValue<double>(1.0, 1.0, 0.01, 10.0)));

    m_settings.add(Option("gridlength", OptionValue<int>(10, 10, 2, 10000)));
}

std::vector<Algorithm::ImplementationType> FitRegularPerpLines::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(CPU);
    return v;
}

double lineDist(std::vector<cv::Vec3f>& li, size_t index1, size_t index2)
{
    double rho1 = li[index1][0];
    if (rho1 < 0.0) rho1 = -rho1;
    double rho2 = li[index2][0];
    if (rho2 < 0.0) rho2 = -rho2;
    return std::abs(rho1 - rho2);
}

double findRegularDistanceByNeighbors(std::vector<cv::Vec3f>& li, const double distMin, const double distMax)
{
    // Sort rho values
    std::vector<double> rhoSorted;
    for (auto& l : li)
    {
        double rho = l[0];

        if (rho < 0.0)
        {
            rho = -rho;
        }

        bool found = false;
        for (auto it = rhoSorted.begin(); it != rhoSorted.end(); ++it)
        {
            if (*it > rho)
            {
                rhoSorted.insert(it, rho);
                found = true;
                break;
            }
        }

        if (!found)
        {
            rhoSorted.push_back(rho);
        }
    }

    // Find distances between lines
    // ignore lines that are too close to each other
    //  -> todo: if there are, take the mean value as "real" line
    std::vector<double> sortedDistances;
    for (size_t i = 1; i < rhoSorted.size(); ++i)
    {
        double d = rhoSorted[i] - rhoSorted[i - 1];

        if (d < distMin || d > distMax)
        {
            continue;
        }

        bool found = false;
        for (auto it = sortedDistances.begin(); it != sortedDistances.end(); ++it)
        {
            if (*it > d)
            {
                sortedDistances.insert(it, d);
                found = true;
                break;
            }
        }

        if (!found)
        {
            sortedDistances.push_back(d);
        }
    }

    // Output mean dist
    double r = 0.0;
    for (double d : sortedDistances)
    {
        r += d;
    }
    r /= sortedDistances.size();

    return r;
}

double findRegularDistanceRANSAC(std::vector<cv::Vec3f>& li, double minDist, double maxDist, std::vector<double>& inlierRhoValues)
{
    //assert(li.size() > 1);

    const double inlierMaxError = 2.0;
    const int iterations = 100;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, li.size() - 1);

    double maxInlierRho = 0.0;
    int maxInlierCount = 0;

    // Sort rho values
    std::vector<double> rhoSorted;
    for (auto& l : li)
    {
        double rho = l[0];

        if (rho < 0.0)
        {
            rho = -rho;
        }

        bool found = false;
        for (auto it = rhoSorted.begin(); it != rhoSorted.end(); ++it)
        {
            if (*it > rho)
            {
                rhoSorted.insert(it, rho);
                found = true;
                break;
            }
        }

        if (!found)
        {
            rhoSorted.push_back(rho);
        }
    }

    // RANSAC
    for (int i = 0; i < iterations; ++i)
    {
        // Pick two random lines until a distance in [minDist,maxDist] is found
        double r = 0.0;
        int rndTries = 0;
        while (r < minDist || r > maxDist)
        {
            r = lineDist(li, uni(rng), uni(rng));

            rndTries++;
            if (rndTries > iterations)
            {
                // give up
                i = iterations;
                r = 1.0;
                break;
            }
        }

        // Check for inliers
        int currentInlierCount = 0;
        for (size_t x = 0; x < rhoSorted.size(); ++x)
        {
            for (size_t y = x + 1; y < rhoSorted.size(); ++y)
            {
                double d = rhoSorted[y] - rhoSorted[x];

                if (d < minDist)
                {
                    continue;
                }

                if (d > r + inlierMaxError)
                {
                    break;
                }

                if (std::abs(d - r) < inlierMaxError)
                {
                    currentInlierCount++;
                    break;
                }
            }
        }

        if (currentInlierCount > maxInlierCount)
        {
            maxInlierCount = currentInlierCount;
            maxInlierRho = r;
        }
    }

    // Output inliers
    int currentInlierCount = 0;
    for (size_t x = 0; x < rhoSorted.size(); ++x)
    {
        for (size_t y = x + 1; y < rhoSorted.size(); ++y)
        {
            double d = rhoSorted[y] - rhoSorted[x];

            if (d < minDist)
            {
                continue;
            }

            if (d > maxInlierRho + inlierMaxError)
            {
                break;
            }

            if (std::abs(d - maxInlierRho) < inlierMaxError)
            {
                inlierRhoValues.push_back(rhoSorted[x]);
                break;
            }
        }
    }

    return maxInlierRho;
}

double findMaxGridAgreement(cv::Mat img, double theta, double rhoRes, double rhoX0, double rhoDist, int gridlength)
{
    double irho = 1.0 / rhoRes;

    const uchar* image = img.ptr();
    int step = (int)img.step;
    int width = img.cols;
    int height = img.rows;

    int max_rho = width + height;
    int min_rho = -max_rho;

    int numrho = cvRound(((max_rho - min_rho) + 1) * irho);
    int halfNumRho = (numrho - 1) * 0.5;
    double cos_ = cos(theta) * irho;
    double sin_ = sin(theta) * irho;

    cv::Mat _accum = cv::Mat::zeros(1, numrho + 2, CV_32SC1);
    int *accum = _accum.ptr<int>();


    int i, j;
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            if (image[i * step + j] != 0)
            {
                int r = cvRound(j * cos_ + i * sin_);
                r += halfNumRho;
                accum[r + 1]++;
            }
        }
    }

    //double rho = (i - 1 - halfNumRho) * rhoRes;

    // Start at rhoX0

    int rhoDist_i = rhoDist * irho; // Distance between lines in "integer rho frame"
    int rhoX0_i = (rhoX0 * irho) + 1 + halfNumRho; // Starting rho value in "integer rho frame"

    std::vector<std::pair<int, int>> scores;

    // go right
    for (int i = 0;; ++i)
    {
        int idx = rhoX0_i + i * rhoDist_i;
        if (idx > numrho - 1)
        {
            break;
        }

        int score = accum[idx];

        scores.push_back({ i, score });
    }

    // go left
    for (int i = -1;; --i)
    {
        int idx = rhoX0_i + i * rhoDist_i;
        if (idx < 0)
        {
            break;
        }

        int score = accum[idx];

        scores.insert(scores.begin(), { i, score });
    }

    // Shift grid and find max agreement
    double maxScore = 0.0;
    int maxScoreIndex = 0;
    for (int i = 0; i < scores.size() - gridlength; ++i)
    {
        double s = 0.0;
        for (int j = 0; j < gridlength; ++j)
        {
            s += scores[i + j].second;
        }

        if (s > maxScore)
        {
            maxScore = s;
            maxScoreIndex = i;
        }
    }

    int finalI = rhoX0_i + scores[maxScoreIndex].first * rhoDist_i;
    double finalRho = (finalI - 1 - halfNumRho) * rhoRes;
    return finalRho;
}

bool FitRegularPerpLines::exec()
{
    std::shared_ptr<cv::Mat> input = m_arguments[0]->get();
    std::shared_ptr<cv::Mat> in_lines = m_arguments[1]->get();
    std::shared_ptr<cv::Mat> out_lines = m_arguments[2]->get();
    std::shared_ptr<cv::Mat> out_lines_inlier = m_arguments[3]->get();
    std::shared_ptr<cv::Mat> out_lines_bestfit = m_arguments[4]->get();

    const int threshold = m_settings.get("threshold").valueInt().value();
    const double offsetAngleDegree = m_settings.get("offset-angle").valueDouble().value();
    double angleTolerance = m_settings.get("angle-tolerance").valueDouble().value() * CV_PI / 180.0;

    const int gridlength = m_settings.get("gridlength").valueInt().value();

    const double rhoRes = m_settings.get("rho-resolution").valueDouble().value();
    const double rhoResFit = m_settings.get("rho-resolution-fit").valueDouble().value();

    const double thetaRes = m_settings.get("theta-resolution").valueDouble().value() * CV_PI / 180.0;

    if (in_lines->rows == 0)
    {
        return false;
    }

    // Find best angle
    int maxPeak;
    {
        cv::Mat hist;
        angleHistogram(*in_lines, hist);

        int secondPeak;
        dualAnglePeak(hist, 1, maxPeak, secondPeak);

        std::cout << "theta0: " << maxPeak << ", theta1: " << secondPeak << std::endl;

        std::vector<cv::Vec2f> inlines(*in_lines);
        for (size_t i = 0; i < in_lines->rows && i < 10; ++i)
        {
            std::cout << " t: " << inlines[i][0] << " r: " << inlines[i][1] << std::endl;
        }
    }

    if (maxPeak < 90)
    {
        maxPeak += 90;
    }
    else
    {
        maxPeak -= 90;
    }

    double targetTheta = maxPeak * CV_PI / 180.0;

    std::vector<cv::Vec3f> li;
    houghLines(*input, li, rhoRes, thetaRes, threshold, targetTheta, angleTolerance);

    if (li.size() < 2)
    {
        return false;
    }

    const double thresholdDistMin = 20.0;
    const double thresholdDistMax = sqrt(input->rows * input->rows + input->cols * input->cols) * 0.5;

    // Find regular distance for parallel lines
    //double r = findRegularDistanceByNeighbors(li, thresholdDistMin, thresholdDistMax);

    std::vector<double> inlierRhoValues;
    double gridRho = findRegularDistanceRANSAC(li, thresholdDistMin, thresholdDistMax, inlierRhoValues);

    // Find best alignment
    double bestRho = findMaxGridAgreement(*input, targetTheta, rhoResFit, inlierRhoValues[0], gridRho, gridlength);

    // Output
    std::vector<cv::Vec2f> oli;
    for (auto& l : li)
    {
        oli.emplace_back(l[0], l[1]);
    }

    if (!oli.empty())
    {
        cv::Mat(oli).copyTo(*out_lines);
    }

    std::vector<cv::Vec2f> outputInliers;
    for (double r : inlierRhoValues)
    {
        cv::Vec2f v;
        v[0] = r;
        v[1] = targetTheta;
        outputInliers.push_back(v);
    }

    if (!outputInliers.empty())
    {
        cv::Mat(outputInliers).copyTo(*out_lines_inlier);
    }

    std::vector<cv::Vec2f> outputFit;
    for (int i = 0; i < gridlength; ++i)
    {
        cv::Vec2f v;
        v[0] = bestRho + i * gridRho;
        v[1] = targetTheta;
        outputFit.push_back(v);
    }

    if (!outputFit.empty())
    {
        cv::Mat(outputFit).copyTo(*out_lines_bestfit);
    }


    return true;
}
