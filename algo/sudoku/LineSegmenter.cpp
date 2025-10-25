#include "LineSegmenter.h"

#include <opencv2/opencv.hpp>
#include <vector>

#include "util.h"

LineSegmenter::LineSegmenter() : Algorithm("LineSegmenter")
{
    ContainerSpecification in_image("in_binary", ContainerSpecification::READ_ONLY);
    ContainerSpecification in_lines("in_lines", ContainerSpecification::READ_ONLY);
    ContainerSpecification out_lines("out_lines", ContainerSpecification::REFERENCE);

    ContainerSpecification out_lineImage("out_lineImg", ContainerSpecification::REFERENCE);

    m_argumentsSpecification.push_back(in_image);
    m_argumentsSpecification.push_back(in_lines);
    m_argumentsSpecification.push_back(out_lines);

    m_argumentsSpecification.push_back(out_lineImage);

    m_settings.add(Option("minlength", OptionValue<int>(50, 50, 0, 10000)));
    m_settings.add(Option("maxgap", OptionValue<int>(10, 10, 0, 1000)));
}

std::vector<Algorithm::ImplementationType> LineSegmenter::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(CPU);
    return v;
}

bool find(float x, float y, cv::Mat& binary)
{
    bool found = false;
    for (int i = -1; i < 1; ++i)
    {
        if (y + i < 0 || y + i > binary.rows)
            continue;

        for (int j = -1; j < 1; ++j)
        {
            if (x + j < 0 || x + j > binary.cols)
                continue;

            found |= binary.at<unsigned char>(y + i, x + j) == 255;
        }
    }
    return found;
}

bool LineSegmenter::exec()
{
    std::shared_ptr<cv::Mat> in_image = m_arguments[0]->get();
    std::shared_ptr<cv::Mat> in_lines = m_arguments[1]->get();

    std::shared_ptr<cv::Mat> out_lines = m_arguments[2]->get();

    int minlength = m_settings.get("minlength").valueInt().value();
    int maxgap = m_settings.get("maxgap").valueInt().value();

    cv::Mat img = *in_image;

    cv::Mat& lineMat = *in_lines;
    int channels = lineMat.channels();

    std::vector<cv::Vec2f> lines = *in_lines;
    std::vector<cv::Vec4i> allSegments;


    std::shared_ptr<cv::Mat> out_img = m_arguments[3]->get();

    cv::cvtColor(*in_image, *out_img, cv::COLOR_GRAY2BGR);

    if (channels == 2)
    {
        float width = static_cast<float>(img.cols);
        float height = static_cast<float>(img.rows);

        for (size_t i = 0; i < lines.size(); ++i)
        {

            float r1 = lines[i][0], t1 = lines[i][1];

            cv::Point2f start, end;
            intersectRect(r1, t1, width, height, start, end);

            allSegments.push_back(cv::Vec4i(start.x, start.y, end.x, end.y));


            // Rasterize

            float xs = start.x, ys = start.y, xe = end.x, ye = end.y;

            float deltaX = abs(xe - xs), deltaY = abs(ye - ys);

            if (deltaX < EPS)
                continue;

            float m = deltaY / deltaX;

            if (m > 1.f)
                continue;


            for (int p = 0; p < deltaX; ++p)
            {
                int x = p;
                int y = ceil(ys + m * static_cast<float>(p));

                if (x < 0 || x > width - 1 || y < 0 || y > height - 1)
                    continue;

                out_img->at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0);
            }

            /*
            int signX = 1, signY = 1;
            if (xe < xs)
                signX = -1;
            if (ye < ys)
                signY = -1;

            int maxp = (m <= 1) ? deltaX : deltaY;
            for (int p = 0; p <= maxp; ++p)
            {
                int x, y;
                if (m <= 1)
                {
                    x = ceil(xs + p * signX);
                    y = ceil(ys + p * m * signY);
                }
                else
                {
                    x = ceil(xs + p / m * signX);
                    y = ceil(ys + p * signY);
                }

                out_img->at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
            }
            */
        }

        cv::Mat(allSegments).copyTo(*out_lines);
    }


    return true;
}
