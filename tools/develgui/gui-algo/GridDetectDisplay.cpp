#include "GridDetectDisplay.h"

#include <opencv2/opencv.hpp>

GridDetectDisplay::GridDetectDisplay() : Algorithm("GridDetectDisplay", true)
{
    ContainerSpecification in_binary("in_binary", ContainerSpecification::READ_ONLY);
    ContainerSpecification in_points("in_points", ContainerSpecification::READ_ONLY);
    ContainerSpecification in_lines("in_lines", ContainerSpecification::READ_ONLY);
    ContainerSpecification in_linegroups("in_line-groups", ContainerSpecification::READ_ONLY);
    ContainerSpecification in_frames("in_frames", ContainerSpecification::READ_ONLY);

    ContainerSpecification in_framePoints("in_framePoints", ContainerSpecification::READ_ONLY);
    ContainerSpecification in_frameImages("inout_frameImages", ContainerSpecification::REFERENCE);

    ContainerSpecification out_img("out_image", ContainerSpecification::REFERENCE);

    m_argumentsSpecification.push_back(in_binary);
    m_argumentsSpecification.push_back(in_points);
    m_argumentsSpecification.push_back(in_lines);
    m_argumentsSpecification.push_back(in_linegroups);
    m_argumentsSpecification.push_back(in_frames);
    m_argumentsSpecification.push_back(in_framePoints);
    m_argumentsSpecification.push_back(in_frameImages);
    m_argumentsSpecification.push_back(out_img);


    m_settings.add(Option("pointsize", OptionValue<int>(5, 5, 1, 100)));

    m_settings.add(Option("draw_segment", OptionValue<int>(-1, -1, -1, 10000)));
}

std::vector<Algorithm::ImplementationType> GridDetectDisplay::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(CPU);
    return v;
}

bool GridDetectDisplay::exec()
{
    std::shared_ptr<cv::Mat> in_binary = m_arguments[0]->get();
    std::shared_ptr<cv::Mat> in_points = m_arguments[1]->get();
    std::shared_ptr<cv::Mat> in_lines = m_arguments[2]->get();
    std::shared_ptr<cv::Mat> in_linegroups = m_arguments[3]->get();
    std::shared_ptr<cv::Mat> in_frames = m_arguments[4]->get();
    std::shared_ptr<cv::Mat> out_img = m_arguments[7]->get();

    std::shared_ptr<Container>& in_framePoints = m_arguments[5];
    std::shared_ptr<Container>& in_frameImages = m_arguments[6];

    int pointSize = m_settings.get("pointsize").valueInt().value();

    std::vector<cv::Scalar> colors;
    colors.push_back(cv::Scalar(0, 0, 255));
    colors.push_back(cv::Scalar(0, 255, 255));
    colors.push_back(cv::Scalar(255, 0, 255));
    colors.push_back(cv::Scalar(0, 255, 0));
    colors.push_back(cv::Scalar(255, 255, 0));
    colors.push_back(cv::Scalar(255, 0, 0));

    cv::cvtColor(*in_binary, *out_img, cv::COLOR_GRAY2BGR);

    if (!in_points->empty())
    {
        std::vector<cv::Point2f> intersectPoints(*in_points);

        for (cv::Point2f p : intersectPoints)
        {
            cv::circle(*out_img, p, pointSize, cv::Scalar(0, 255, 0), -1);
        }
    }

    int drawSegment = m_settings.get("draw_segment").valueInt().value();

    if (!in_linegroups->empty() && !in_lines->empty())
    {

        std::vector<cv::Vec2i> lineGroups(*in_linegroups);

        std::vector<cv::Vec4i> lines(*in_lines);


        std::set<size_t> groupReps;
        for (size_t g = 0; g < lineGroups.size(); ++g)
        {
            int groupRep = lineGroups[g][1];
            groupReps.insert(groupRep);
        }

        for (size_t g = 0; g < lineGroups.size(); ++g)
        {
            int lineIndex = lineGroups[g][0];
            int groupRep = lineGroups[g][1];
            auto& line = lines[lineIndex];

            int groupIndex = -1;

            int i = 0;
            for (auto it = groupReps.begin(); it != groupReps.end(); ++it, ++i)
            {
                if (*it == groupRep)
                {
                    groupIndex = i;
                    break;
                }
            }


            cv::Point pt1, pt2;
            pt1.x = line[0];
            pt1.y = line[1];
            pt2.x = line[2];
            pt2.y = line[3];

            if (drawSegment < 0 || drawSegment == groupIndex)
            {
                cv::line( *out_img, pt1, pt2, colors[groupIndex % colors.size()], 2, cv::LINE_AA);
            }
        }

    }

    if (!in_frames->empty())
    {
        int frameNum = in_frames->rows;
        for (int i = 0; i < frameNum; ++i)
        {

            if (drawSegment < 0 || drawSegment == i)
            {


                cv::Vec2f v0 = in_frames->at<cv::Vec2f>(i, 0);
                cv::Vec2f v1 = in_frames->at<cv::Vec2f>(i, 1);
                cv::Vec2f v2 = in_frames->at<cv::Vec2f>(i, 2);
                cv::Vec2f v3 = in_frames->at<cv::Vec2f>(i, 3);

                cv::Point p0(v0[0], v0[1]);
                cv::Point p1(v1[0], v1[1]);
                cv::Point p2(v2[0], v2[1]);
                cv::Point p3(v3[0], v3[1]);

                cv::line( *out_img, p0, p1, colors[(i + 1) % colors.size()], 4, cv::LINE_AA);
                cv::line( *out_img, p1, p2, colors[(i + 1) % colors.size()], 4, cv::LINE_AA);
                cv::line( *out_img, p2, p3, colors[(i + 1) % colors.size()], 4, cv::LINE_AA);
                cv::line( *out_img, p3, p0, colors[(i + 1) % colors.size()], 4, cv::LINE_AA);

            }

            /*
            if (in_frameImages->size() > i && in_framePoints->size() > i)
            {
                cv::Mat frameImage = *in_frameImages->get(i);

                if (frameImage.empty()) continue;
                if (in_framePoints->get(i)->empty()) continue;

                cv::Mat img;
                if (frameImage.channels() == 1)
                {
                    cv::cvtColor(frameImage, img, cv::COLOR_GRAY2BGR);
                }

                std::vector<cv::Point2f> framePoints(*in_framePoints->get(i));

                size_t size = framePoints.size();
                float sizeF = size;
                for (size_t i = 0; i < size; ++i)
                {
                    cv::circle(img, framePoints[i], pointSize, cv::Scalar(0, 255, 255.0f * (i / sizeF)), -1);
                }

                img.copyTo(*in_frameImages->get(i));
            }
            */
        }

    }
    return true;
}
