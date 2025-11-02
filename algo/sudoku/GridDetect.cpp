#include "GridDetect.h"

#include <opencv2/opencv.hpp>
#include <vector>

#include "LineAlgorithms.h"
#include "util.h"

GridDetect::GridDetect() : Algorithm("GridDetect")
{
    ContainerSpecification in_image("in_binary", ContainerSpecification::READ_ONLY);
    ContainerSpecification in_lines("in_lines", ContainerSpecification::READ_ONLY);
    ContainerSpecification in_seglines("in_linegroups", ContainerSpecification::REFERENCE);

    ContainerSpecification out_frames("out_frames", ContainerSpecification::REFERENCE);

    m_argumentsSpecification.push_back(in_image);
    m_argumentsSpecification.push_back(in_lines);
    m_argumentsSpecification.push_back(in_seglines);
    m_argumentsSpecification.push_back(out_frames);

    m_settings.add(Option("min-intersect-angle", OptionValue<int>(80, 80, 0, 359)));

    m_settings.add(Option("min-max-angle-deviation", OptionValue<float>(5.f, 5.f, 0.f, 100.f)));

    m_settings.add(Option("min-frameedge-dist", OptionValue<int>(90, 90, 0, 999999)));
}

std::vector<Algorithm::ImplementationType> GridDetect::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(CPU);
    return v;
}

bool getBB(size_t groupID, std::vector<cv::Vec2i>& groups, std::vector<cv::Vec4i>& lines, int theta1Deg, int theta2Deg, float angleTolerance, std::vector<cv::Point2f>& outBB)
{
    angleTolerance = angleTolerance / 180.0 * CV_PI;

    double theta1 = theta1Deg / 180.0 * CV_PI;
    double theta2 = theta2Deg / 180.0 * CV_PI;

    double t11 = theta1;
    double t12 = fmod(theta1 + CV_PI, 2*CV_PI);
    double t21 = theta2;
    double t22 = fmod(theta2 + CV_PI, 2*CV_PI);

    cv::Vec4d tTarget(t11, t21, t12, t22);
    cv::Vec4d rMax(0.f, 0.f, 0.f, 0.f), trMax;
    cv::Vec4i lrMax;

    // Find group center
    cv::Point2f origin(0.f, 0.f);
    int n = 0;
    for (auto it = groups.begin(); it != groups.end(); ++it)
    {
        if ((*it)[1] != groupID)
            continue;

        int lineIndex = (*it)[0];
        origin.x += lines[lineIndex][0] + lines[lineIndex][2];
        origin.y += lines[lineIndex][1] + lines[lineIndex][3];
        n += 2;
    }
    origin.x /= n;
    origin.y /= n;

    // Find max distant line in all 4 directions from group center
    for (auto it = groups.begin(); it != groups.end(); ++it)
    {
        if ((*it)[1] != groupID)
            continue;

        int lineIndex = (*it)[0];
        double t, r;
        getPolar(lines[lineIndex], origin, r, t);

        for (int i = 0; i < 4; ++i)
        {
            if (isDirection(t, tTarget[i], angleTolerance))
            {
                if (r > rMax[i])
                {
                    rMax[i] = r;
                    lrMax[i] = lineIndex;
                }
            }
        }
    }

    // Convert rho theta of max lines back to image coord system
    for (int i = 0; i < 4; ++i)
    {
        int lineIndex = lrMax[i];
        getPolar(lines[lineIndex], cv::Point2f(0.f, 0.f), rMax[i], trMax[i]);
    }

    // Output four intersect points of max lines as quasi BoundingBox
    cv::Point2f intersect;
    intersect2D(rMax[0], trMax[0], rMax[1], trMax[1], intersect);
    outBB.push_back(intersect);
    intersect2D(rMax[1], trMax[1], rMax[2], trMax[2], intersect);
    outBB.push_back(intersect);
    intersect2D(rMax[2], trMax[2], rMax[3], trMax[3], intersect);
    outBB.push_back(intersect);
    intersect2D(rMax[3], trMax[3], rMax[0], trMax[0], intersect);
    outBB.push_back(intersect);

    return true;
}

bool GridDetect::exec()
{
    std::shared_ptr<cv::Mat> in_image = m_arguments[0]->get();
    std::shared_ptr<cv::Mat> in_lines = m_arguments[1]->get();
    std::shared_ptr<cv::Mat> in_seglines = m_arguments[2]->get();

    std::shared_ptr<cv::Mat> out_frames = m_arguments[3]->get();

    int minAngleDeg = m_settings.get("min-intersect-angle").valueInt().value();

    float angleTolerance = m_settings.get("min-max-angle-deviation").valueFloat().value();

    int minFrameEdgeDist = m_settings.get("min-frameedge-dist").valueInt().value();

    if (in_lines->empty() || in_seglines->empty())
        return true;

    int width = in_image->cols;
    int height = in_image->rows;

    cv::Mat& lineMat = *in_lines;

    int channels = lineMat.channels();

    if (channels != 4)
    {
        return false;
    }

    std::vector<cv::Vec2i> lineGroups(*in_seglines);
    std::vector<cv::Vec4i> lines(*in_lines);

    std::set<size_t> groupReps;
    for (size_t g = 0; g < lineGroups.size(); ++g)
    {
        int groupRep = lineGroups[g][1];
        groupReps.insert(groupRep);
    }


    // Find Frames
    // ===========================
    std::vector<std::vector<cv::Point2f>> frames;

    {
        // Per Group:
        // Create map of Theta angles
        // Find two most common ones
        //  (if no clear distinction, drop group)
        // Find outermost lines with these angles and create bounding rect
        for (size_t groupNumber : groupReps)
        {
            cv::Mat angleHistogram = cv::Mat::zeros(1, 360, CV_8U);
            for (auto it = lineGroups.begin(); it != lineGroups.end(); ++it)
            {
                if ((*it)[1] == groupNumber)
                {
                    size_t i = (*it)[0];
                    double theta, rho;
                    getPolar(lines[i], cv::Point2f(0.f, 0.f), rho, theta);
                    theta = fmod(theta, CV_PI);

                    int degr = theta / CV_PI * 180.0;
                    angleHistogram.at<unsigned char>(degr)++;
                }
            }

            int maxPeak, secondPeak;
            dualAnglePeak(angleHistogram, minAngleDeg, maxPeak, secondPeak);

            std::vector<cv::Point2f> frame;

            bool ok = false;
            if (secondPeak > -1 && getBB(groupNumber, lineGroups, lines, maxPeak, secondPeak, angleTolerance, frame))
            {
                ok = true;

                // Sanity check
                for (int i = 0; ok && i < 4; ++i)
                {
                    if (frame[i].x < 0.f || frame[i].x > width
                     || frame[i].y < 0.f || frame[i].y > height)
                    {
                        ok = false;
                        break;
                    }

                    for (int j = 0; j < 4; ++j)
                    {
                        if (j == i) continue;

                        if (cv::norm(frame[i] - frame[j]) < minFrameEdgeDist)
                        {
                            ok = false;
                            break;
                        }
                    }
                }
            }

            if (ok)
            {
                frames.push_back(frame);
            }
            else
            {
                frame.clear();
                frame.push_back(cv::Point2f(-1.f, -1.f));
                frame.push_back(cv::Point2f(-1.f, -1.f));
                frame.push_back(cv::Point2f(-1.f, -1.f));
                frame.push_back(cv::Point2f(-1.f, -1.f));
                frames.push_back(frame);
            }
        }
    }

    // Output
    // =======================================
    if (frames.size() > 0)
    {
        cv::Mat outFrames = cv::Mat(frames.size(), 4, CV_32FC2);

        for (int i = 0; i < frames.size(); ++i)
        {
            outFrames.at<cv::Vec2f>(i, 0) = cv::Vec2f(frames[i][0].x, frames[i][0].y);
            outFrames.at<cv::Vec2f>(i, 1) = cv::Vec2f(frames[i][1].x, frames[i][1].y);
            outFrames.at<cv::Vec2f>(i, 2) = cv::Vec2f(frames[i][2].x, frames[i][2].y);
            outFrames.at<cv::Vec2f>(i, 3) = cv::Vec2f(frames[i][3].x, frames[i][3].y);
        }

        outFrames.copyTo(*out_frames);
    }

    return true;
}
