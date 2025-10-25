#include "LineGrouping.h"

#include <opencv2/opencv.hpp>

#include "../util.h"

LineGrouping::LineGrouping() : Algorithm("LineGrouping")
{
    ContainerSpecification in_lines("in_lines", ContainerSpecification::READ_ONLY);

    ContainerSpecification out_points("out_points", ContainerSpecification::REFERENCE);
    ContainerSpecification out_seglines("out_linegroups", ContainerSpecification::REFERENCE);

    m_argumentsSpecification.push_back(in_lines);
    m_argumentsSpecification.push_back(out_points);
    m_argumentsSpecification.push_back(out_seglines);

    m_settings.add(Option("min-intersect-angle", OptionValue<int>(80, 80, 0, 359)));

    m_settings.add(Option("intersect-tolerance", OptionValue<float>(5.f, 5.f, 0.f, 100.f)));
}

std::vector<Algorithm::ImplementationType> LineGrouping::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(CPU);
    return v;
}

bool LineGrouping::exec()
{
    std::shared_ptr<cv::Mat> in_lines = m_arguments[0]->get();

    std::shared_ptr<cv::Mat> out_points = m_arguments[1]->get();
    std::shared_ptr<cv::Mat> out_seglines = m_arguments[2]->get();

    int minAngleDeg = m_settings.get("min-intersect-angle").valueInt().value();
    double minAngle = CV_PI * static_cast<double>(minAngleDeg) / 180.0;

    float intersectTolerance = m_settings.get("intersect-tolerance").valueFloat().value();

    if (in_lines->empty())
        return true;

    cv::Mat& lineMat = *in_lines;

    int channels = lineMat.channels();

    std::vector<cv::Point2f> intersectPoints;

    if (channels != 4)
    {
        return false;
    }

    // Group lines and calculate intersect points
    // ==========================================
    std::vector<cv::Vec4i> lines = *in_lines;

    std::map<size_t, size_t> lineGroups;

    lineGroups[0] = 0;

    for (size_t i = 0; i < lines.size(); ++i)
    {
        cv::Point2f start1 = cv::Point2f(lines[i][0], lines[i][1]);
        cv::Point2f end1 = cv::Point2f(lines[i][2], lines[i][3]);
        cv::Point2f d1 = (end1 - start1);
        d1 /= cv::norm(d1);

        cv::Point2f n1(-d1.y, d1.x);

        double t, r1, t1;
        cv::Point2f r;
        intersect2D(start1, d1, cv::Point2f(0, 0), n1, r, t);
        cartesianToPolar(r.x, r.y, r1, t1);
        double t1f = fmod(t1, CV_PI);

        double t1Max;
        if (d1.x > EPS)
        {
            t1Max = (end1.x - start1.x) / d1.x;
        }
        else
        {
            t1Max = (end1.y - start1.y) / d1.y;
        }

        for (size_t j = i + 1; j < lines.size(); ++j)
        {
            cv::Point2f start2 = cv::Point2f(lines[j][0], lines[j][1]);
            cv::Point2f end2 = cv::Point2f(lines[j][2], lines[j][3]);
            cv::Point2f d2 = end2 - start2;
            d2 /= cv::norm(d2);

            cv::Point2f n2(-d2.y, d2.x);

            double t, r2, t2;
            cv::Point2f r;
            intersect2D(start2, d2, cv::Point2f(0, 0), n2, r, t);
            cartesianToPolar(r.x, r.y, r2, t2);
            double t2f = fmod(t2, CV_PI);

            double t2Max;
            if (d2.x > EPS)
            {
                t2Max = (end2.x - start2.x) / d2.x;
            }
            else
            {
                t2Max = (end2.y - start2.y) / d2.y;
            }

            double angle = abs(t2f - t1f);

            if (angle > minAngle && CV_PI - angle > minAngle)
            {
                cv::Point2f intersect;
                if (intersect2D(start1, d1, start2, d2, intersect, t))
                {
                    double t2;
                    if (d2.x > EPS)
                    {
                        t2 = (intersect.x - start2.x) / d2.x;
                    }
                    else
                    {
                        t2 = (intersect.y - start2.y) / d2.y;
                    }


                    if (t < -intersectTolerance || t > t1Max + intersectTolerance || t2 < -intersectTolerance || t2 > t2Max + intersectTolerance)
                    {
                        continue;
                    }

                    // Put into group
                    auto it = lineGroups.find(i);
                    if (it == lineGroups.end())
                    {
                        // i has no group
                        it = lineGroups.find(j);
                        if (it == lineGroups.end())
                        {
                            // j has no group either,
                            //　create group for i and add j
                            lineGroups[i] = i;
                            lineGroups[j] = lineGroups[i];
                        }
                        else
                        {
                            // j has group but i does not,
                            // add i to group of j
                            lineGroups[i] = lineGroups[j];
                        }
                    }
                    else
                    {
                        // i has group
                        auto it = lineGroups.find(j);
                        if (it == lineGroups.end())
                        {
                            // j has no group, add j to group of i
                            lineGroups[j] = lineGroups[i];
                        }
                        else
                        {
                            // bot have groups, merge them if not the same
                            // put all in group of j into group of i

                            size_t jGroup = lineGroups[j];
                            size_t iGroup = lineGroups[i];
                            if (iGroup != jGroup)
                            {
                                for (it = lineGroups.begin(); it != lineGroups.end(); ++it)
                                {
                                    if (it->second == jGroup)
                                    {
                                        it->second = iGroup;
                                    }
                                }
                            }
                        }
                    }

                    // Add intersection point
                    intersectPoints.push_back(intersect);
                }
            }
            else
            {
                continue;
            }
        }
    }

    std::vector<cv::Vec2i> outLineGroups;

    for (auto it = lineGroups.begin(); it != lineGroups.end(); ++it)
    {
        outLineGroups.push_back(cv::Vec2i(it->first, it->second));
    }

    cv::Mat(outLineGroups).copyTo(*out_seglines);

    if (intersectPoints.size() > 0)
    {
        cv::Mat(intersectPoints).copyTo(*out_points);
    }

    return true;
}
