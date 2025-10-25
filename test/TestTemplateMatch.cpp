#include "TestTemplateMatch.h"

#include <CLUtil.h>
#include <OCR.h>
#include <sudoku/TemplateMatch.h>
#include <Timer.h>

#include <opencv2/opencv.hpp>

#include <iostream>

TestTemplateMatch::TestTemplateMatch(const CLUtil::CLHandler& handler) : m_context(handler)
{

}

double comp(cv::Mat& mat, cv::Mat& gt)
{
    double rtn = 0;
    for (int i = 0; i < mat.rows; ++i)
    for (int j = 0; j < mat.cols; ++j)
    {
        if (mat.type() == CV_8U)
        {
            rtn += abs((int)mat.at<uchar>(i, j) - gt.at<uchar>(i, j));
        }
        else if (mat.type() == CV_32S)
        {
            //rtn += abs((long long)mat.at<uint>(i, j) - gt.at<uint>(i, j));
        }
        else if (mat.type() == CV_32F)
        {
            rtn += abs(mat.at<float>(i, j) - gt.at<float>(i, j));
        }
        else if (mat.type() == CV_64F)
        {
            rtn += abs(mat.at<double>(i, j) - gt.at<double>(i, j));
        }
    }
    return rtn;
}

void writeR(cv::Mat& r, std::string filename, bool norm)
{
    double maxVal, minVal;
    cv::minMaxLoc(r, &minVal, &maxVal, 0, 0);

    cv::Mat outDebug = cv::Mat::zeros(r.rows, r.cols, CV_8U);
    for (int i = 0; i < r.rows; ++i)
    for (int j = 0; j < r.cols; ++j)
    {
        if (r.type() == CV_32F)
        {
            float v = r.at<float>(i, j);
            //if (v > df)
            {
                outDebug.at<uchar>(i, j) = static_cast<uchar>(255 * (norm ? (v / maxVal) : v));
            }
        }
        if (r.type() == CV_64F)
        {
            double v = r.at<double>(i, j);
            {
                outDebug.at<uchar>(i, j) = static_cast<uchar>(255 * (norm ? (v / maxVal) : v));
            }
        }
        else if (r.type() == CV_8U)
        {
            uchar v = r.at<uchar>(i, j);
            outDebug.at<uchar>(i, j) = v;
        }
        /*
        else if (r.type() == CV_32S)
        {
            uint v = r.at<uint>(i, j);
            outDebug.at<uchar>(i, j) = static_cast<uchar>(255 * (norm ? (v / maxVal) : MIN(v, 255)));
        }
        */
    }

    std::string f = "output/" + filename + ".png";
    cv::imwrite(f, outDebug);

    std::cout << f << "(" << minVal << "-" << maxVal << ")" << std::endl;
}

bool testIntegralFunc(TemplateMatch& tm, cv::Mat& src, bool saveResults)
{
    cv::UMat srcU, R_ocvU;

    cv::Mat R_ocv, R_gpu, b, R_ocvUM;

    int iterations = 10;

    std::cout << "  Integral OCV ..." << std::flush;

    Timer t;
    cv::integral(src, R_ocv);
    std::cout << t.elapsed() << "ms" << std::endl;


    t.restart();
    src.copyTo(srcU);
    //srcU = src.getUMat(cv::ACCESS_READ);
    double t_c1 = t.restart();

    std::cout << "  Integral OCV GPU ..." << std::flush;
    t.restart();
    cv::integral(srcU, R_ocvU);
    double t_3 = t.restart();
    for (int i = 0; i < iterations; ++i)
    {
        cv::integral(srcU, R_ocvU);
    }
    std::cout << t.elapsed() / iterations << "ms ";

    t.restart();
    R_ocvU.copyTo(R_ocvUM);
    double t_c2 = t.restart();

    std::cout << t_c1 << "ms (c1) " << t_c2 << "ms (c2)" << t_3 << "ms (inital kernel)" << std::endl;

    std::cout << "  Integral GPU ..." << std::flush;

    t.restart();
    tm.integral(src, R_gpu, b);
    std::cout << t.elapsed() << "ms" << std::endl;

    std::cout << "ocv gpu-cpu: " << comp(R_ocv, R_ocvUM) << std::endl;
    std::cout << "gpu-ocv(cpu): " << comp(R_gpu, R_ocv) << std::endl;

    if (saveResults)
    {
        writeR(R_ocv, std::string("integral_ocv"), true);
        writeR(R_ocvUM, std::string("integral_ocv_gpu"), true);
        writeR(R_gpu, std::string("integral_gpu"), true);

        cv::Mat diffOcv, diffOcvGpu;
        cv::absdiff(R_ocv, R_ocvUM, diffOcv);
        cv::absdiff(R_gpu, R_ocv, diffOcvGpu);
        writeR(diffOcv, "integral_diff_ocv", true);
        writeR(diffOcvGpu, "integral_diff_gpu_ocv", true);
    }

    return true;
}

bool testBaseFunc(TemplateMatch& tm, OCR& ocr, cv::Mat& src, bool saveResults, double cellSize, bool compare, cv::TemplateMatchModes mode)
{
    bool normed = mode == cv::TM_CCOEFF_NORMED || mode == cv::TM_CCORR_NORMED || mode == cv::TM_SQDIFF_NORMED;

    bool ocv_cpu = true;
    bool ocv_gpu = false;
    bool cpu = false;
    bool gpu = true;

    cv::Mat R_ocv[10], R_cpu[10], R_gpu[10];
    cv::Mat templt[10];
    cv::UMat templtU[10];
    cv::UMat srcU, R_ocvU[10];
    src.copyTo(srcU);

    std::vector<Digit> digits;
    ocr.scaled(cellSize, digits);

    for (int i = 0; i < 9; ++i)
    {
        templt[i] = digits[i].templ;
        templt[i].copyTo(templtU[i]);
    }

    Timer t;

    if (ocv_cpu)
    {
        std::cout << "  OpenCV CPU ..." << std::flush;
        t.restart();
        for (int i = 0; i < 9; ++i)
        {
            cv::matchTemplate(src, templt[i], R_ocv[i], mode); // TM_CCORR_NORMED
        }
        std::cout << t.elapsed() << "ms" << std::endl;

        if (saveResults)
        {
            for (int i = 0; i < 9; ++i)
            {
                writeR(R_ocv[i], std::string("tmbase_ocv_") + std::to_string(i + 1), !normed);
            }
        }
    }

    if (ocv_gpu)
    {
        std::cout << "  OpenCV GPU ..." << std::flush;
        cv::matchTemplate(srcU, templtU[0], R_ocvU[0], mode);
        t.restart();
        int n = 20;
        for (int di = 0; di < 9; ++di)
        {
            for (int i = 0; i < n; ++i)
            {
                cv::matchTemplate(srcU, templtU[di], R_ocvU[di], mode);
            }
        }

        std::cout << t.elapsed() / n << "ms" << std::endl;

        if (saveResults)
        {
            for (int i = 0; i < 9; ++i)
            {
                cv::Mat m = R_ocvU[i].getMat(cv::AccessFlag::ACCESS_READ);
                writeR(m, std::string("tmbase_ocv_gpu_") + std::to_string(i + 1), !normed);
            }
        }
    }

    if (cpu)
    {
        std::cout << "  CPU        ..." << std::flush;
        t.restart();
        for (int di = 0; di < 9; ++di)
        {
            matchTemplateCPU(src, templt[di], R_cpu[di], mode);
        }

        std::cout << t.elapsed() << "ms" << std::endl;

        if (saveResults)
        {
            for (int i = 0; i < 9; ++i)
            {
                writeR(R_cpu[i], std::string("tmbase_cpu_") + std::to_string(i + 1), !normed);
            }
        }
    }

    if (gpu)
    {
        std::cout << "  GPU        ..." << std::flush;
        t.restart();
        int numIts = 1;
        //tm.setImplementation(Algorithm::ImplementationType::GPU);
        tm.setIterations(numIts);
        for (int di = 0; di < 9; ++di)
        {
            tm.matchTemplateGPU(src, templt[di], R_gpu[di], mode);
        }

        std::cout << t.elapsed() / numIts << "ms" << std::endl;

        if (saveResults)
        {
            for (int i = 0; i < 9; ++i)
            {
                writeR(R_gpu[i], std::string("tmbase_gpu_") + std::to_string(i + 1), !normed);
            }
        }
    }

    if (compare)
    {
        int diffCV = 0, diffCVMY = 0, diffCpuGpu = 0;

        if (ocv_cpu && ocv_gpu)
        {
            for (int i = 0; i < 9; ++i)
            {
                cv::Mat t = R_ocvU[i].getMat(cv::ACCESS_READ);
                diffCV += comp(R_ocv[i], t);

                cv::Mat diffImg;
                cv::absdiff(R_ocv[i], t, diffImg);
                writeR(diffImg, std::string("tmbase_diff_ocv_") + std::to_string(i + 1), true);
            }
            std::cout << "Diff OCV - OCV GPU: " << diffCV << std::endl;
        }

        if (ocv_cpu && cpu)
        {
            for (int i = 0; i < 9; ++i)
            {
                diffCVMY += comp(R_ocv[i], R_cpu[i]);

                cv::Mat diffImg;
                cv::absdiff(R_ocv[i], R_cpu[i], diffImg);
                writeR(diffImg, std::string("tmbase_diff_ocv_cpu_") + std::to_string(i + 1), true);
            }
            std::cout << "Diff OCV - CPU: " << diffCVMY << std::endl;
        }

        if (cpu && gpu)
        {
            for (int i = 0; i < 9; ++i)
            {
                diffCpuGpu += comp(R_gpu[i], R_cpu[i]);

                cv::Mat diffImg;
                cv::absdiff(R_cpu[i], R_gpu[i], diffImg);
                writeR(diffImg, std::string("tmbase_diff_gpu_cpu_") + std::to_string(i + 1), true);
            }
            std::cout << "Diff GPU - CPU: " << diffCpuGpu << std::endl;
        }
        else if (ocv_cpu && gpu)
        {
            for (int i = 0; i < 9; ++i)
            {
                diffCpuGpu += comp(R_gpu[i], R_ocv[i]);

                cv::Mat diffImg;
                cv::absdiff(R_ocv[i], R_gpu[i], diffImg);
                writeR(diffImg, std::string("tmbase_diff_gpu_ocvcpu_") + std::to_string(i + 1), true);
            }
            std::cout << "Diff GPU - CPU(ocv): " << diffCpuGpu << std::endl;
        }
    }

    return true;
}

bool TestTemplateMatch::DoCompute()
{
    std::cout << "Testing TemplateMatching" << std::endl;


    Timer t;

    OCR ocr;
    ocr.loadOCR("param/ocr2.yml");

    TemplateMatch tm(ocr);
    m_context.initTask(tm);

    cv::Mat img = cv::imread("eval/template_match/template_test.png", cv::ImreadModes::IMREAD_GRAYSCALE);
    cv::Mat imgBinary;

    int kernelSize = 3;
    cv::Size k(kernelSize, kernelSize);
    cv::blur(img, img, k);

    bool gauss = true;
    double C = 20.0;
    int radius = 100;

    cv::adaptiveThreshold(img, imgBinary, 255.0, gauss ? cv::ADAPTIVE_THRESH_GAUSSIAN_C : cv::ADAPTIVE_THRESH_MEAN_C,
                          cv::THRESH_BINARY_INV, radius * 2 + 1, C);

    cv::imwrite("output/tm_imgBinary.png", imgBinary);

    double margin = 0.05;
    double d = imgBinary.cols / 9.0;
    double cellRes = d * (1.0 - margin * 2);
    margin = d * margin;

    cv::Mat imgCells;
    Timer t2;
    cellImage(imgBinary, imgCells, d, cellRes, margin);
    std::cout << "  cellimg..." << t2.elapsed() << "ms" << std::endl;

    cv::imwrite("output/tm_imgCells.png", imgCells);

    std::cout << "  prepare..." << t.elapsed() << "ms" << std::endl;

    bool success = true;
    //success &= testIntegralFunc(tm, imgCells, true);
    success &= testBaseFunc(tm, ocr, imgCells, true, d, true, cv::TM_CCORR);

    return success;
}
