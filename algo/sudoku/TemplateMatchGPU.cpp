#include "TemplateMatch.h"

#include <opencv2/opencv.hpp>

#include <CLUtil.h>
#include <Timer.h>
#include <Utils.h>


size_t GetGlobalWorkSize(size_t DataElemCount, size_t LocalWorkSize)
{
    size_t r = DataElemCount % LocalWorkSize;
    if(r == 0)
        return DataElemCount;
    else
        return DataElemCount + LocalWorkSize - r;
}

bool TemplateMatch::InitResources(cl_device_id Device, cl_context Context, cl_command_queue CommandQueue)
{
    Algorithm::InitResources(Device, Context, CommandQueue);

    cl_int clError;

    m_program = CLUtil::compileProgram(Device, Context, Utils::loadFile("algo/sudoku/TemplateMatch.cl"));
    if(m_program == nullptr) return false;

    //create kernels
    m_tmKernel = clCreateKernel(m_program, "MatchTemplate", &clError);
    CLUtil::handleCLErrors(clError, "Failed to create kernel: MatchTemplate.");
    m_tmKernelTiled = clCreateKernel(m_program, "MatchTemplateTiled", &clError);
    CLUtil::handleCLErrors(clError, "Failed to create kernel: MatchTemplate.");

    m_reduceBasicKernel = clCreateKernel(m_program, "ReduceBasic", &clError);
    CLUtil::handleCLErrors(clError, "Failed to create kernel: ReduceBasic.");

    m_reduceDecompKernel = clCreateKernel(m_program, "ReduceDecomp", &clError);
    CLUtil::handleCLErrors(clError, "Failed to create kernel: ReduceDecomp.");

    m_reduceEABasicKernel = clCreateKernel(m_program, "ReduceEABasic", &clError);
    CLUtil::handleCLErrors(clError, "Failed to create kernel: ReduceEABasic.");

    m_reduceEADecompKernel = clCreateKernel(m_program, "ReduceEADecomp", &clError);
    CLUtil::handleCLErrors(clError, "Failed to create kernel: ReduceEADecomp.");


    m_prefixSumBasicKernel = clCreateKernel(m_program, "PrefixSumBasic", &clError);
    CLUtil::handleCLErrors(clError, "Failed to create kernel: PrefixSumBasic.");
    m_prefixSumAddKernel = clCreateKernel(m_program, "PrefixSumAdd", &clError);
    CLUtil::handleCLErrors(clError, "Failed to create kernel: IntPrefixSumAddegral.");

    m_integralKernel = clCreateKernel(m_program, "Integral", &clError);
    CLUtil::handleCLErrors(clError, "Failed to create kernel: Integral.");
    m_integralAddKernel = clCreateKernel(m_program, "IntegralAdd", &clError);
    CLUtil::handleCLErrors(clError, "Failed to create kernel: IntegralAdd.");
    m_integralGSKernel = clCreateKernel(m_program, "IntegralGroupSums", &clError);
    CLUtil::handleCLErrors(clError, "Failed to create kernel: IntegralGroupSums.");
    m_integralConvInputKernel = clCreateKernel(m_program, "IntegralConvInput", &clError);
    CLUtil::handleCLErrors(clError, "Failed to create kernel: IntegralConvInput.");
    m_integralConvOutputKernel = clCreateKernel(m_program, "IntegralConvOutput", &clError);
    CLUtil::handleCLErrors(clError, "Failed to create kernel: IntegralConvOutput.");
    m_rotCWKernel = clCreateKernel(m_program, "RotateCW", &clError);
    CLUtil::handleCLErrors(clError, "Failed to create kernel: RotateCW.");
    m_rotCCWKernel = clCreateKernel(m_program, "RotateCCW", &clError);
    CLUtil::handleCLErrors(clError, "Failed to create kernel: RotateCCW.");

    return true;
}

void TemplateMatch::ReleaseResources()
{
    clReleaseKernel(m_tmKernel);
    clReleaseKernel(m_tmKernelTiled);
    clReleaseKernel(m_reduceBasicKernel);
    clReleaseKernel(m_reduceDecompKernel);
    clReleaseKernel(m_reduceEABasicKernel);
    clReleaseKernel(m_reduceEADecompKernel);
    clReleaseKernel(m_prefixSumBasicKernel);
    clReleaseKernel(m_prefixSumAddKernel);
    clReleaseKernel(m_integralKernel);
    clReleaseKernel(m_integralAddKernel);
    clReleaseKernel(m_integralGSKernel);
    clReleaseKernel(m_integralConvInputKernel);
    clReleaseKernel(m_integralConvOutputKernel);
    clReleaseKernel(m_rotCWKernel);
    clReleaseKernel(m_rotCCWKernel);
    clReleaseProgram(m_program);
}

bool TemplateMatch::_integralGS(cl_mem dA, cl_mem dB, cl_mem dP, uint N, uint groupsize)
{
    cl_int err;
    err = clSetKernelArg(m_integralGSKernel, 0, sizeof(cl_mem), static_cast<void*>(&dA));
    err |= clSetKernelArg(m_integralGSKernel, 1, sizeof(cl_mem), static_cast<void*>(&dB));
    err |= clSetKernelArg(m_integralGSKernel, 2, sizeof(cl_mem), static_cast<void*>(&dP));
    err |= clSetKernelArg(m_integralGSKernel, 3, sizeof(cl_uint), static_cast<void*>(&groupsize));
    err |= clSetKernelArg(m_integralGSKernel, 4, sizeof(cl_uint), static_cast<void*>(&N));
    CLUtil::handleCLErrors(err, "Error setting kernel args for m_integralGSKernel");

    uint groupCount = N / groupsize;

    size_t lws[] = {16 /* groupsize / 2 */, 1};
    size_t gws[] = {GetGlobalWorkSize(groupCount, lws[0]), N};

    /*double t =*/ CLUtil::profileKernel(m_CommandQueue, m_integralGSKernel, 2, gws, lws, 1);
    //std::cout << "N " << N << " GS " << groupsize << " GC " << groupCount << " lws " << lws[0] << " gws " << gws[0] << " time: " << t << std::endl;
    return true;
}

bool TemplateMatch::_integral(cl_mem dA, cl_mem dB, cl_mem dO, cl_mem dP, uint N, uint groupsize)
{
    size_t globalWorkSize[2] = {N/2*N, 1};
    size_t localWorkSize[2] = {groupsize/2, 1};

    cl_int clError;
    clError = clSetKernelArg(m_integralKernel, 0, sizeof(cl_mem), static_cast<void*>(&dA));
    clError |= clSetKernelArg(m_integralKernel, 1, sizeof(cl_mem), static_cast<void*>(&dB));
    clError |= clSetKernelArg(m_integralKernel, 3, sizeof(cl_uint), static_cast<void*>(&N));
    CLUtil::handleCLErrors(clError, "Error setting kernel args for m_integralKernel");

    CLUtil::profileKernel(m_CommandQueue, m_integralKernel, 1, globalWorkSize, localWorkSize, 1);

    if (N > groupsize)
    {
        //_integralGS_cpu(dA, dB, N, groupsize, src, dst);
        _integralGS(dA, dB, dP, N, groupsize);

        clError = clSetKernelArg(m_integralKernel, 0, sizeof(cl_mem), static_cast<void*>(&dP));
        clError |= clSetKernelArg(m_integralKernel, 1, sizeof(cl_mem), static_cast<void*>(&dO));
        clError |= clSetKernelArg(m_integralKernel, 3, sizeof(cl_uint), static_cast<void*>(&groupsize));
        CLUtil::handleCLErrors(clError, "Error setting kernel args for m_integralKernel");

        globalWorkSize[0] = groupsize/2 * N;

        CLUtil::profileKernel(m_CommandQueue, m_integralKernel, 1, globalWorkSize, localWorkSize, 1);

        // Add groupsums
        clError = clSetKernelArg(m_integralAddKernel, 0, sizeof(cl_mem), static_cast<void*>(&dB));
        clError |= clSetKernelArg(m_integralAddKernel, 1, sizeof(cl_mem), static_cast<void*>(&dO));
        clError |= clSetKernelArg(m_integralAddKernel, 2, sizeof(cl_uint), static_cast<void*>(&groupsize));
        clError |= clSetKernelArg(m_integralAddKernel, 3, sizeof(cl_uint), static_cast<void*>(&N));
        CLUtil::handleCLErrors(clError, "Error setting kernel args for m_integralAddKernel");

        globalWorkSize[0] = N*N;

        CLUtil::profileKernel(m_CommandQueue, m_integralAddKernel, 1, globalWorkSize, localWorkSize, 1);
    }
    return true;
}

bool TemplateMatch::_rot(cl_mem dN, cl_mem dM, uint lw, uint N, bool cw)
{
    size_t globalWorkSizeRot[] = {N, N};
    size_t localWorkSizeRot[] = {lw, lw};

    cl_kernel kern = cw ? m_rotCWKernel : m_rotCCWKernel;

    cl_int clError;

    clError = clSetKernelArg(kern, 2, sizeof(cl_uint) * lw * lw, nullptr);

    clError |= clSetKernelArg(kern, 0, sizeof(cl_mem), static_cast<void*>(&dN));
    clError |= clSetKernelArg(kern, 1, sizeof(cl_mem), static_cast<void*>(&dM));
    clError |= clSetKernelArg(kern, 3, sizeof(cl_uint), static_cast<void*>(&N));

    clError |= clEnqueueNDRangeKernel(m_CommandQueue, kern, 2, nullptr, globalWorkSizeRot, localWorkSizeRot, 0, nullptr, nullptr);

    CLUtil::handleCLErrors(clError, "Error in rotation");
    return true;
}

bool TemplateMatch::_integralConvInput(cl_mem dConvInput, cl_mem M, uchar* data, uint width, uint height, uint N)
{
    CLUtil::handleCLErrors(clEnqueueWriteBuffer(m_CommandQueue, dConvInput, CL_FALSE, 0, sizeof(cl_uchar) * width*height, data, 0, NULL, NULL), "Error copying data");

    size_t gws[] = {N, N};
    size_t lws[] = {16, 16};

    cl_int err;
    err = clSetKernelArg(m_integralConvInputKernel, 0, sizeof(cl_mem), static_cast<void*>(&dConvInput));
    err |= clSetKernelArg(m_integralConvInputKernel, 1, sizeof(cl_mem), static_cast<void*>(&M));
    err |= clSetKernelArg(m_integralConvInputKernel, 2, sizeof(cl_uint), static_cast<void*>(&width));
    err |= clSetKernelArg(m_integralConvInputKernel, 3, sizeof(cl_uint), static_cast<void*>(&N));
    CLUtil::handleCLErrors(err, "Error setting kernel args for m_integralConvInputKernel");

    CLUtil::profileKernel(m_CommandQueue, m_integralConvInputKernel, 2, gws, lws, 1);

    return true;
}

bool TemplateMatch::_integralConvOutput(cl_mem M, cl_mem N, uint* dest, uint widthDest, uint heightDest, uint N_m)
{
    size_t gws[] = {N_m, N_m};
    size_t lws[] = {16, 16};

    cl_int err;
    err = clSetKernelArg(m_integralConvOutputKernel, 0, sizeof(cl_mem), static_cast<void*>(&M));
    err |= clSetKernelArg(m_integralConvOutputKernel, 1, sizeof(cl_mem), static_cast<void*>(&N));
    err |= clSetKernelArg(m_integralConvOutputKernel, 2, sizeof(cl_uint), static_cast<void*>(&widthDest));
    err |= clSetKernelArg(m_integralConvOutputKernel, 3, sizeof(cl_uint), static_cast<void*>(&heightDest));
    err |= clSetKernelArg(m_integralConvOutputKernel, 4, sizeof(cl_uint), static_cast<void*>(&N_m));
    CLUtil::handleCLErrors(err, "Error setting kernel args for m_integralConvOutputKernel");

    CLUtil::profileKernel(m_CommandQueue, m_integralConvOutputKernel, 2, gws, lws, 1);

    CLUtil::handleCLErrors(clEnqueueReadBuffer(m_CommandQueue, N, CL_TRUE, 0, sizeof(uint)*widthDest*heightDest, dest, 0, NULL, NULL), "Error reading data from device!");

    return true;
}

void TemplateMatch::integral(cv::Mat& img, cv::Mat& outSum, cv::Mat& outSqSum)
{
    uint rotationLocalSize = 16;
    uint targetGroupSize = 512;
    uint minN = std::max(img.rows, img.cols) + 1;
    uint N;
    uint groupsize;

    if (minN <= targetGroupSize)
    {
        N = minN;
        groupsize = minN;
    }
    else
    {
        N = std::ceil(minN / static_cast<double>(targetGroupSize)) * targetGroupSize;
        groupsize = targetGroupSize;
    }

    /*
    std::cout << "targetGroupSize: " << targetGroupSize << std::endl;
    std::cout << "minN: " << minN << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "groupsize: " << groupsize << std::endl;
    */

    cl_int clError = CL_SUCCESS, clError2;
    cl_mem dConv = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, sizeof(cl_uchar) * img.rows*img.cols, nullptr, &clError2);
    clError |= clError2;
    cl_mem dM = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * N*N, nullptr, &clError2);
    clError |= clError2;
    cl_mem dN = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * N*N, nullptr, &clError2);
    clError |= clError2;
    cl_mem dO = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * groupsize*N, nullptr, &clError2);
    clError |= clError2;
    cl_mem dP = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * groupsize*N, nullptr, &clError2);
    clError |= clError2;
    CLUtil::handleCLErrors(clError, "Error allocating device arrays");

    clError |= clSetKernelArg(m_integralKernel, 2, sizeof(uint) * groupsize, NULL);
    CLUtil::handleCLErrors(clError, "Error allocating local memory");

    _integralConvInput(dConv, dN, img.data, img.cols, img.rows, N);

    _integral(dN, dM, dO, dP, N, groupsize);

    _rot(dM, dN, rotationLocalSize, N, false);

    _integral(dN, dM, dO, dP, N, groupsize);

    _rot(dM, dN, rotationLocalSize, N, true);

    outSum = cv::Mat::zeros(img.rows + 1, img.cols + 1, CV_32S);
    //outSqSum = cv::Mat::zeros(img.rows + 1, img.cols + 1, CV_64F);

    _integralConvOutput(dN, dM, reinterpret_cast<uint*>(outSum.data), img.cols + 1, img.rows + 1, N);

    clReleaseMemObject(dConv);
    clReleaseMemObject(dM);
    clReleaseMemObject(dN);
    clReleaseMemObject(dO);
    clReleaseMemObject(dP);
}

void TemplateMatch::_matchTemplateGPU_Tiled(cv::Mat& img, cv::Mat& templt, cv::Mat& outResult)
{
    uint W = img.cols;
    uint H = img.rows;
    uint w = templt.cols;
    uint h = templt.rows;
    uint Rw = W - w + 1;
    uint Rh = H - h + 1;

    cl_int clError = CL_SUCCESS, clError2;
    cl_mem dImg = clCreateBuffer(m_Context, CL_MEM_READ_ONLY, sizeof(cl_uchar) * W * H, nullptr, &clError2);
    clError |= clError2;
    cl_mem dTmplt = clCreateBuffer(m_Context, CL_MEM_READ_ONLY, sizeof(cl_uchar) * w * h, nullptr, &clError2);
    clError |= clError2;
    cl_mem dR = clCreateBuffer(m_Context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * Rw * Rh, nullptr, &clError2);
    clError |= clError2;
    CLUtil::handleCLErrors(clError, "Error allocating device arrays");


    //write input data to the GPU
    CLUtil::handleCLErrors(clEnqueueWriteBuffer(m_CommandQueue, dImg, CL_FALSE, 0, sizeof(cl_uchar) * W * H, img.data, 0, NULL, NULL), "Error copying data from host to device!");
    CLUtil::handleCLErrors(clEnqueueWriteBuffer(m_CommandQueue, dTmplt, CL_FALSE, 0,  sizeof(cl_uchar) * w * h, templt.data, 0, NULL, NULL), "Error copying data from host to device!");

    size_t localWorkSize[2] = {16, 16};
    uint tileSizeX = localWorkSize[0] + w;
    uint tileSizeY = localWorkSize[1] + h;

    size_t globalWorkSize[2] = {GetGlobalWorkSize(Rw, localWorkSize[0]), GetGlobalWorkSize(Rh, localWorkSize[1])};

    int tmpltLoadCyclesX = (int)ceil((float)w / localWorkSize[0]);
    int tmpltLoadCyclesY = (int)ceil((float)h / localWorkSize[1]);
    std::cout << "Template (" << w << "x" << h << ") -> load cycles: " << tmpltLoadCyclesX << "x" << tmpltLoadCyclesY << ", tile: (" << tileSizeX << "x" << tileSizeY << ")" << std::endl;

    clError  = clSetKernelArg(m_tmKernelTiled, 0, sizeof(cl_mem), static_cast<void*>(&dImg));
    clError |= clSetKernelArg(m_tmKernelTiled, 1, sizeof(cl_mem), static_cast<void*>(&dTmplt));
    clError |= clSetKernelArg(m_tmKernelTiled, 2, sizeof(cl_mem), static_cast<void*>(&dR));
    clError |= clSetKernelArg(m_tmKernelTiled, 4, sizeof(cl_uint), static_cast<void*>(&W));
    clError |= clSetKernelArg(m_tmKernelTiled, 5, sizeof(cl_uint), static_cast<void*>(&H));
    clError |= clSetKernelArg(m_tmKernelTiled, 6, sizeof(cl_uint), static_cast<void*>(&w));
    clError |= clSetKernelArg(m_tmKernelTiled, 7, sizeof(cl_uint), static_cast<void*>(&h));
    clError |= clSetKernelArg(m_tmKernelTiled, 8, sizeof(cl_uint), static_cast<void*>(&tileSizeX));
    clError |= clSetKernelArg(m_tmKernelTiled, 9, sizeof(cl_uint), static_cast<void*>(&tileSizeY));
    clError |= clSetKernelArg(m_tmKernelTiled, 10, sizeof(cl_uint), static_cast<void*>(&Rw));
    clError |= clSetKernelArg(m_tmKernelTiled, 11, sizeof(cl_uint), static_cast<void*>(&Rh));
    CLUtil::handleCLErrors(clError, "Error setting kernel args for TemplateMatchKernel");

    clError = clSetKernelArg(m_tmKernelTiled, 3, sizeof(cl_uchar) * (w * h + tileSizeX * tileSizeY), NULL);
    CLUtil::handleCLErrors(clError, "Error allocating local memory");

    double kernelTime = CLUtil::profileKernel(m_CommandQueue, m_tmKernelTiled, 2, globalWorkSize, localWorkSize, 1);
    std::cout << "Kernel: " << kernelTime << "ms" << std::endl;

    outResult = cv::Mat(Rh, Rw, CV_32F);
    CLUtil::handleCLErrors(clEnqueueReadBuffer(m_CommandQueue, dR, CL_FALSE, 0, sizeof(cl_float) * Rw * Rh, outResult.data, 0, NULL, NULL), "Error reading data from device!");

    clReleaseMemObject(dImg);
    clReleaseMemObject(dTmplt);
    clReleaseMemObject(dR);
}

void TemplateMatch::matchTemplateGPU(cv::Mat& img, cv::Mat& templt, cv::Mat& outResult, cv::TemplateMatchModes mode)
{
    _matchTemplateGPU_Tiled(img, templt, outResult);
}

std::vector<std::pair<int, double>>
    matchDirection(cv::Mat& img, double dist, double size, double margin,
               const std::vector<Digit>& digits, double& outAvgScore)
{
    std::vector<std::pair<int, double>> guesses;



    return guesses;
}

void TemplateMatch::matchGPU(cv::Mat image, cv::Mat &outNumbers, Rotation& outRot, cv::Mat &outDebug)
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
    directionGuesses.push_back(matchDirection(image, dist, size, margin, digits[0], avgScore[0]));

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
