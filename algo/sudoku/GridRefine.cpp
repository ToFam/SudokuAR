#include "GridRefine.h"

#include <opencv2/opencv.hpp>
#include <vector>

#include "util.h"

GridRefine::GridRefine() : Algorithm("GridRefine")
{
    ContainerSpecification in_image("in_binary", ContainerSpecification::READ_ONLY);
    ContainerSpecification in_points("in_points", ContainerSpecification::READ_ONLY);
    ContainerSpecification in_frames("in_frames", ContainerSpecification::READ_ONLY);

    ContainerSpecification out_frames("out_frames", ContainerSpecification::REFERENCE);
    ContainerSpecification out_framesImage("out_framesImage", ContainerSpecification::REFERENCE);

    ContainerSpecification out_framePoints("out_framePoints", ContainerSpecification::REFERENCE);

    m_argumentsSpecification.push_back(in_image);
    m_argumentsSpecification.push_back(in_points);
    m_argumentsSpecification.push_back(in_frames);
    m_argumentsSpecification.push_back(out_frames);
    m_argumentsSpecification.push_back(out_framesImage);
    m_argumentsSpecification.push_back(out_framePoints);

    m_settings.add(Option("point_margin", OptionValue<float>(5.0f, 5.f, 0.f, 99999.f)));
    m_settings.add(Option("pointdist_avg_window", OptionValue<int>(2, 2, 0, 100)));
    m_settings.add(Option("discardFactor", OptionValue<float>(0.5f, 0.5f, 0.0f, 1.0f)));
}

std::vector<Algorithm::ImplementationType> GridRefine::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(CPU);
    return v;
}

cv::Mat perspectiveTF(std::vector<cv::Point2f> frame, int& dstWidth, int& dstHeight)
{
    cv::Point2f src[4];
    cv::Point2f dst[4];

    for (int i = 0; i < 4; i++)
    {
        src[i] = frame[i];
    }

    arrangePoints(src);

    dstWidth = cvRound(cv::norm(src[0] - src[1]));
    dstHeight = cvRound(dstWidth);

    dst[0] = cv::Point2f(dstWidth, dstHeight);
    dst[1] = cv::Point2f(  0.f,    dstHeight);
    dst[2] = cv::Point2f(  0.f,    0.f);
    dst[3] = cv::Point2f(dstWidth, 0.f);

    return getPerspectiveTransform(src, dst);
}

bool GridRefine::processFrame(std::vector<cv::Point2f> &frame, std::vector<cv::Point2f>& outFramePoints,
                              float margin, int pointDistAvgWindow, float templateDiscardFactor, cv::Mat& outImg)
{
    int width, height;
    cv::Mat t = perspectiveTF(frame, width, height);

    // Find all intersection points in frame
    for (cv::Point2f p : m_intersectPoints)
    {
        cv::Mat_<double> mp = cv::Mat_<double>(3, 1);
        mp << p.x, p.y, 1.f;

        mp = t * mp;

        double x = mp.at<double>(0, 0) / mp.at<double>(2, 0);
        double y = mp.at<double>(1, 0) / mp.at<double>(2, 0);

        if (x >= 0.f && x <= width && y >= 0.f && y <= height)
        {
            outFramePoints.push_back(cv::Point2f(x, y));
        }
    }

    // Cluster points, keep only representatives
    std::sort(outFramePoints.begin(), outFramePoints.end(), std::bind(sortGridYX<cv::Point2f, float>, std::placeholders::_1, std::placeholders::_2, margin));

    for (auto firstIT = outFramePoints.begin();firstIT != outFramePoints.end();++firstIT)
    {
        auto secondIT = firstIT + 1;
        for(;secondIT != outFramePoints.end() && cv::norm(*firstIT - *secondIT) < margin;++secondIT);
        --secondIT;
        if (secondIT > firstIT)
        {
            cv::Point2f mean = *firstIT;
            int clusterSize = secondIT - firstIT + 1;
            auto it = firstIT + 1;
            for (int i = 1; i < clusterSize; ++i)
            {
                mean += *it;
                outFramePoints.erase(it);
            }
            mean /= clusterSize;
            *firstIT = mean;
        }
    }

    // Find an estimate for the width and height of a single grid cell
    std::vector<float> distancesX, distancesY;
    for (auto it = outFramePoints.begin(); it < outFramePoints.end() - 1; ++it)
    {
        cv::Point2f d = *it - *(it + 1);
        if (abs(d.y) < margin)
        {
            distancesX.push_back(abs(d.x));
        }
    }

    std::sort(outFramePoints.begin(), outFramePoints.end(), std::bind(sortGridXY<cv::Point2f, float>, std::placeholders::_1, std::placeholders::_2, margin));
    for (auto it = outFramePoints.begin(); it < outFramePoints.end() - 1; ++it)
    {
        cv::Point2f d = *it - *(it + 1);
        if (abs(d.x) < margin)
        {
            distancesY.push_back(abs(d.y));
        }
    }

    if (distancesY.size() == 0 || distancesX.size() == 0)
        return false;

    float avgCellWidth = avgMean<float>(distancesX.begin(), distancesX.end(), pointDistAvgWindow);
    float avgCellHeight = avgMean<float>(distancesY.begin(), distancesY.end(), pointDistAvgWindow);

    std::cout << "points " << outFramePoints.size()
              << " frame (" << width << "x" << height
              << ") cell: (" << avgCellWidth << "x" << avgCellHeight << ")"
              << " 9xcell (" << avgCellWidth * 9 << "x" << avgCellHeight*9 << ")" << std::endl;
    if (width < avgCellWidth * 9)
    {
        avgCellWidth = width / 9.f;
    }
    if (height < avgCellHeight * 9)
    {
        avgCellHeight = height / 9.f;
    }

    // Create binary pictures (one compelete with all intersections and one template with 10x10)
    cv::Mat intersectPic = cv::Mat::zeros(height, width, CV_8U);
    int radius = 5;
    int tmpltSizeX = avgCellWidth * 9;
    int tmpltSizeY = avgCellHeight * 9;
    cv::Mat templt = cv::Mat::zeros(tmpltSizeY, tmpltSizeX, CV_8U);

    for (int x = 0; x < 10; ++x)
    for (int y = 0; y < 10; ++y)
    {
        cv::circle(templt, cv::Point2f(x * avgCellWidth, y * avgCellHeight), radius, cv::Scalar(255), -1);
    }

    for (cv::Point2f& p : outFramePoints)
    {
        cv::circle(intersectPic, p, radius, cv::Scalar(255), -1);
    }

    // Match template to intersection pic
    cv::Mat R;
    double min, max;
    cv::Point minLoc, maxLoc;
    cv::matchTemplate(intersectPic, templt, R, cv::TM_CCORR_NORMED);
    cv::minMaxLoc(R, &min, &max, &minLoc, &maxLoc);

    cv::Point loc = maxLoc;
    double val = max;

    if (val < templateDiscardFactor)
    {
        return false;
    }

    std::cout << R.cols << " " << R.rows << " == " << loc.x << " " << loc.y << ", " << val  << std::endl;

   // cv::rectangle(intersectPic, loc, loc + cv::Point(9 * avgCellWidth, 9 * avgCellHeight), cv::Scalar(200), 3);

    //intersectPic.copyTo(outImg);

    std::vector<cv::Point2f> newFramePointsLocal, newFramePointsGlobal;
    newFramePointsLocal.push_back(cv::Point2f(loc.x, loc.y));
    newFramePointsLocal.push_back(cv::Point2f(loc.x + tmpltSizeX, loc.y));
    newFramePointsLocal.push_back(cv::Point2f(loc.x + tmpltSizeX, loc.y + tmpltSizeY));
    newFramePointsLocal.push_back(cv::Point2f(loc.x, loc.y + tmpltSizeY));

    cv::perspectiveTransform(newFramePointsLocal, newFramePointsGlobal, t.inv());

    frame = newFramePointsGlobal;

    return true;
}


bool GridRefine::exec()
{
    std::shared_ptr<cv::Mat> in_image = m_arguments[0]->get();
    std::shared_ptr<cv::Mat> in_points = m_arguments[1]->get();
    std::shared_ptr<cv::Mat> in_frames = m_arguments[2]->get();
    std::shared_ptr<cv::Mat> out_frames = m_arguments[3]->get();

    std::shared_ptr<Container>& out_frameImages = m_arguments[4];

    std::shared_ptr<Container>& out_framePoints = m_arguments[5];

    if (in_frames->empty())
        return true;

    float intersectPointMargin = m_settings.get("point_margin").valueFloat().value();
    int pointDistAvgWindow = m_settings.get("pointdist_avg_window").valueInt().value();
    float templateDiscardFactor = m_settings.get("discardFactor").valueFloat().value();

    m_intersectPoints = std::vector<cv::Point2f>(*in_points);

    size_t frameNum = in_frames->rows;
    std::vector<std::vector<cv::Point2f>> frames;

    for (size_t i = 0; i < frameNum; ++i)
    {
        cv::Vec2f v0 = in_frames->at<cv::Vec2f>(i, 0);
        cv::Vec2f v1 = in_frames->at<cv::Vec2f>(i, 1);
        cv::Vec2f v2 = in_frames->at<cv::Vec2f>(i, 2);
        cv::Vec2f v3 = in_frames->at<cv::Vec2f>(i, 3);

        frames.emplace_back();
        frames[i].push_back(cv::Point2f(v0[0], v0[1]));
        frames[i].push_back(cv::Point2f(v1[0], v1[1]));
        frames[i].push_back(cv::Point2f(v2[0], v2[1]));
        frames[i].push_back(cv::Point2f(v3[0], v3[1]));
    }

    // Refine Frames
    // =======================================
    for (auto it = frames.begin(); it != frames.end(); ++it)
    {
        if ((*it)[0].x < 0.f)
            continue;

        cv::Mat outImg;

        std::vector<cv::Point2f> framePoints;
        if (processFrame(*it, framePoints, intersectPointMargin, pointDistAvgWindow, templateDiscardFactor, outImg))
        {
            auto m = std::make_shared<cv::Mat>();
            if (framePoints.size() > 0)
            {
               cv::Mat(framePoints).copyTo(*m);
            }
            out_framePoints->add(m);
        }
        else
        {
            (*it)[0].x = -1.f;
        }
    }

    // Create Frame output images
    // =======================================
    {
        for (size_t f = 0; f < frames.size(); ++f)
        {
            auto& frame = frames[f];

            if (frame[0].x < 0.f)
                continue;

            int dstWidth, dstHeight;
            cv::Mat t =  perspectiveTF(frame, dstWidth, dstHeight);
            cv::Mat warpedImage;
            warpPerspective(*in_image, warpedImage, t, cv::Size(dstWidth, dstHeight));

            warpedImage = warpedImage.rowRange(dstHeight - dstWidth, dstHeight);

            auto outFramesImage = std::make_shared<cv::Mat>();
            warpedImage.copyTo(*outFramesImage);
            out_frameImages->add(outFramesImage);
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
