#pragma once

#include "../Algorithm.h"
#include "../OCR.h"

#include <opencv2/imgproc.hpp>

void cellImage(cv::Mat& in, cv::Mat& out, double dist, double size, double margin);

void matchTemplateCPU(cv::Mat& img, cv::Mat& templt, cv::Mat& outResult, cv::TemplateMatchModes mode);

class TemplateMatch : public Algorithm
{
public:
    TemplateMatch(const OCR& ocr);
    ~TemplateMatch();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;

    void matchTemplateGPU(cv::Mat& img, cv::Mat& templt, cv::Mat& outResult, cv::TemplateMatchModes mode);
    void integral(cv::Mat& img, cv::Mat& outSum, cv::Mat& outSumSq);

private:
    bool _integralConvOutput(cl_mem M, cl_mem N, uint* dest, uint widthDest, uint heightDest, uint N_m);
    bool _integralConvInput(cl_mem dConvInput, cl_mem M, uchar* data, uint width, uint height, uint N);
    bool _integralGS(cl_mem dA, cl_mem dB, cl_mem dP, uint N, uint groupsize);
    bool _integral(cl_mem dIn, cl_mem dOut, cl_mem dO, cl_mem dP, uint N, uint groupsize);
    bool _rot(cl_mem dIn, cl_mem dOut, uint lw, uint N, bool cw);

    void _matchTemplateGPU_Tiled(cv::Mat& img, cv::Mat& templt, cv::Mat& outResult);
    void _matchTemplateGPU();

private:
    void matchOCVGPU(cv::Mat inSingle, cv::Mat& outNumbers, Rotation& outRot);
    void matchCPU(cv::Mat inSingle, cv::Mat& outNumbers, Rotation& outRot, cv::Mat& outDebug);
    void matchGPU(cv::Mat inSingle, cv::Mat& outNumbers, Rotation& outRot, cv::Mat& outDebug);

public: // IComputeTask
    bool InitResources(cl_device_id Device, cl_context Context, cl_command_queue CommandQueue) override;
    void ReleaseResources();

    // Side tasks
private:
    uint reduceExtendedArray(uint* src, uint N);
    uint reducePingPong(uint* src, uint N);
    uint reduceBasic(uint* src, uint N);
    void prefixSumBasic(uint* src, uint* dst, uint N);

private:
    const OCR& m_ocr;

    cl_program m_program;
    cl_kernel  m_tmKernel;
    cl_kernel  m_tmKernelTiled;
    cl_kernel  m_reduceBasicKernel;
    cl_kernel  m_reduceDecompKernel;
    cl_kernel  m_reduceEABasicKernel;
    cl_kernel  m_reduceEADecompKernel;
    cl_kernel  m_prefixSumBasicKernel;
    cl_kernel  m_prefixSumAddKernel;
    cl_kernel  m_integralKernel;
    cl_kernel  m_integralAddKernel;
    cl_kernel  m_integralGSKernel;
    cl_kernel  m_integralConvInputKernel;
    cl_kernel  m_integralConvOutputKernel;
    cl_kernel  m_rotCCWKernel;
    cl_kernel  m_rotCWKernel;
};
