#include "LineAlgorithms.h"

#include <opencv2/opencv.hpp>
#include <vector>

#include "util.h"

void getPolar(cv::Vec4i line, cv::Point2f origin, double& rho, double& theta)
{
    cv::Point2f start1 = cv::Point2f(line[0], line[1]);
    cv::Point2f end1 = cv::Point2f(line[2], line[3]);
    cv::Point2f d1 = (end1 - start1);
    d1 /= cv::norm(d1);
    cv::Point2f n1(-d1.y, d1.x);

    double t;
    cv::Point2f r;
    intersect2D(start1, d1, origin, n1, r, t);
    cartesianToPolar(r.x - origin.x, r.y - origin.y, rho, theta);

    /*
    if (origin.x > 0.f && origin.y > 0.0f)
    {
    std::cout << rho << " - " << theta << std::endl;

    origin = cv::Point2f(0.f, 0.f);
    intersect2D(start1, d1, origin, n1, r, t);
    cartesianToPolar(r.x - origin.x, r.y - origin.y, rho, theta);

    std::cout << rho << " - " << theta << std::endl;
    }
    */
}

bool angleHistogram(cv::Mat lines, cv::Mat& outHistogram, float angleResolution)
{
    if (lines.channels() == 2)
    {
        outHistogram = cv::Mat::zeros(1, 360, CV_8U); // double size needed in dualAnglePeak()
        std::vector<cv::Vec2f> vl(lines);

        for (cv::Vec2f l : vl)
        {
            float theta = l[1];
            int degr = theta / CV_PI * 180.0;
            degr = std::min(180, degr);
            outHistogram.at<unsigned char>(degr)++;
        }

        return true;
    }
    else if (lines.channels() == 4)
    {
        // todo
        return false;
    }

    return false;
}

void houghLines(cv::Mat inputImage, std::vector<cv::Vec3f>& outLines, double rhoResolution, double thetaResolution, double threshold,
                double targetTheta, double angleTolerance)
{
    if (targetTheta < 0.0)
    {
        cv::HoughLines(inputImage, outLines, rhoResolution, thetaResolution, threshold);
    }
    else
    {
        if (targetTheta >= CV_PI)
        {
            targetTheta -= CV_PI; // Cap to [0, pi)
        }

        double minTheta = targetTheta - angleTolerance;
        double maxTheta = targetTheta + angleTolerance;

        if (maxTheta > CV_PI || minTheta < 0.0)
        {
            std::cout << "Extra pass for theta near zero" << std::endl;

            // [D1, PI] -> li1
            double mint = maxTheta > CV_PI ? minTheta : CV_PI + minTheta;
            double maxt = CV_PI;

            std::vector<cv::Vec3f> li1;
            std::cout << "theta_min: " << mint << ", theta_max: " << maxt << std::endl;
            cv::HoughLines(inputImage, li1, rhoResolution, thetaResolution, threshold, 0.0, 0.0, mint, maxt);

            // [0, D2]
            mint = 0.0;
            maxt = maxTheta > CV_PI ? maxTheta - CV_PI : maxTheta;

            std::vector<cv::Vec3f> li2;
            std::cout << "theta_min: " << mint << ", theta_max: " << maxt << std::endl;
            cv::HoughLines(inputImage, li2, rhoResolution, thetaResolution, threshold, 0.0, 0.0, mint, maxt);

            // Merge sorted lists
            int n = li1.size() + li2.size();
            outLines.resize(n);
            for (int i1 = 0, i2 = 0; i1 + i2 < n;)
            {
                auto pick1 = [&]()
                {
                    outLines[i1 + i2] = li1[i1];
                    ++i1;
                };
                auto pick2 = [&]()
                {
                    outLines[i1 + i2] = li2[i2];
                    ++i2;
                };

                if (i1 >= li1.size())
                {
                    pick2();
                }
                else if (i2 >= li2.size())
                {
                    pick1();
                }
                else
                {
                    int l1a = li1.at(i1)[2];
                    int l2a = li2.at(i2)[2];

                    if (l1a > l2a)
                    {
                        pick1();
                    }
                    else
                    {
                        pick2();
                    }
                }
            }
        }
        else
        {
            cv::HoughLines(inputImage, outLines, rhoResolution, thetaResolution, threshold, 0.0, 0.0, minTheta, maxTheta);
        }
    }
}

void houghLines(cv::Mat inputImage, std::vector<cv::Vec2f>& outLines, double rhoResolution, double thetaResolution, double threshold,
                double targetTheta, double angleTolerance)
{
    std::vector<cv::Vec3f> lines;
    houghLines(inputImage, lines, rhoResolution, thetaResolution, threshold, targetTheta, angleTolerance);

    outLines.resize(lines.size());
    for (size_t i = 0; i < outLines.size(); ++i)
    {
        cv::Vec3f l = lines[i];
        outLines[i] = cv::Vec2f(l[0], l[1]);
    }
}

struct LineCluster
{
    std::vector<size_t> indices;
    double meanRho;
    double meanTheta;

    double deltaRho_;
    double deltaTheta_;

    LineCluster(size_t i, double rho, double theta, double deltaRho, double deltaTheta)
        : indices{i}, meanRho(rho), meanTheta(theta), deltaRho_(deltaRho), deltaTheta_(deltaTheta)
    {

    }

    template <typename T>
    bool tryPut(const std::vector<T>& lines, size_t index)
    {
        double rho = lines[index][0];
        double theta = lines[index][1];

        if (std::abs(rho - meanRho) < deltaRho_ && std::abs(theta - meanTheta) < deltaTheta_)
        {
            indices.push_back(index);

            meanTheta = 0.0;
            meanRho = 0.0;
            for (auto& i : indices)
            {
                meanRho += lines[i][0];
                meanTheta += lines[i][1];
            }
            meanTheta /= indices.size();
            meanRho /= indices.size();

            return true;
        }
        return false;
    }
};

template <typename T>
void buildClusters(const std::vector<T>& lines, std::vector<LineCluster>& clusters, double deltaRho, double deltaTheta)
{
    for (size_t i = 0; i < lines.size(); ++i)
    {
        bool found = false;
        for (auto& c : clusters)
        {
            if (c.tryPut(lines, i))
            {
                found =true;
                break;
            }
        }

        if (!found)
        {
            clusters.emplace_back(i, lines[i][0], lines[i][1], deltaRho, deltaTheta);
        }
    }
}


void filterHoughLinesNonMax(std::vector<cv::Vec3f>& lines, double deltaRho, double deltaTheta)
{
    std::vector<LineCluster> clusters;
    buildClusters(lines, clusters, deltaRho, deltaTheta);

    std::vector<cv::Vec3f> rtn;
    for (auto& c : clusters)
    {
        size_t maxIndex = c.indices[0];
        double maxAcc = lines[maxIndex][2];

        for (auto& i : c.indices)
        {
            if (lines[i][2] > maxAcc)
            {
                maxIndex = i;
                maxAcc = lines[i][2];
            }
        }

        rtn.push_back(lines[maxIndex]);
    }

    lines = rtn;
}

template <typename T>
void filterHoughLinesAngleT(std::vector<T>& lines, double targetAngle, double deltaRho, double deltaTheta)
{
    std::vector<LineCluster> clusters;
    buildClusters(lines, clusters, deltaRho, deltaTheta);

    std::vector<T> rtn;
    for (auto& c : clusters)
    {
        size_t maxIndex = c.indices[0];
        double minDiff = std::abs(lines[maxIndex][1] - targetAngle);

        for (auto& i : c.indices)
        {
            double diff = abs(lines[i][1] - targetAngle);
            if (diff < minDiff)
            {
                maxIndex = i;
                minDiff = diff;
            }
        }

        rtn.push_back(lines[maxIndex]);
    }

    lines = rtn;
}

void filterHoughLinesAngle(std::vector<cv::Vec3f>& lines, double targetAngle, double deltaRho, double deltaTheta)
{
    filterHoughLinesAngleT(lines, targetAngle, deltaRho, deltaTheta);
}

void filterHoughLinesAngle(std::vector<cv::Vec2f>& lines, double targetAngle, double deltaRho, double deltaTheta)
{
    filterHoughLinesAngleT(lines, targetAngle, deltaRho, deltaTheta);
}
