#include "TemplateMatch.h"

#include <opencv2/opencv.hpp>

#include <CLUtil.h>
#include "Timer.h"

uint TemplateMatch::reduceBasic(uint* src, uint N)
{
    cl_int clError = CL_SUCCESS, clError2;
    cl_mem dM = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * N, nullptr, &clError2);
    clError |= clError2;
    CLUtil::handleCLErrors(clError, "Error allocating device arrays");

    CLUtil::handleCLErrors(clEnqueueWriteBuffer(m_CommandQueue, dM, CL_FALSE, 0, sizeof(cl_uint) * N, src, 0, NULL, NULL), "Error copying data");


    size_t globalWorkSize[2] = {N/2, 1};
    size_t localWorkSize[2] = {16, 1};
    if (localWorkSize[0] > globalWorkSize[0])
        localWorkSize[0] = globalWorkSize[0];

    clError = clSetKernelArg(m_reduceBasicKernel, 0, sizeof(cl_mem), static_cast<void*>(&dM));

    int iterations = 1;
    double time = 0.0;
    uint stride = globalWorkSize[0];
    while (stride >= 1)
    {
        clError |= clSetKernelArg(m_reduceBasicKernel, 1, sizeof(uint), static_cast<void*>(&stride));
        CLUtil::handleCLErrors(clError, "Error setting kernel args for m_reduceBasicKernel");


        clError = clFinish(m_CommandQueue);
        Timer t;
        for (int i = 0; i < iterations; ++i)
        {
        clError |= clEnqueueNDRangeKernel(m_CommandQueue, m_reduceBasicKernel, 1, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
        }
        clError = clFinish(m_CommandQueue);
        time += t.elapsed() / iterations;
        CLUtil::handleCLErrors(clError, "Error executing m_reduceBasicKernel!");


        globalWorkSize[0] /= 2;
        stride /= 2;
        if (localWorkSize[0] > globalWorkSize[0])
            localWorkSize[0] = globalWorkSize[0];
    }

    std::cout << "elapsed: " << time << "ms" << std::endl;


    Timer t;
    uint res;
    CLUtil::handleCLErrors(clEnqueueReadBuffer(m_CommandQueue, dM, CL_TRUE, 0, sizeof(uint), &res, 0, NULL, NULL), "Error reading data from device!");
    std::cout << "read: " << t.elapsed() << std::endl;

    return res;
}
uint TemplateMatch::reducePingPong(uint* src, uint N)
{
    cl_int clError = CL_SUCCESS, clError2;
    cl_mem dM = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * N, nullptr, &clError2);
    clError |= clError2;
    cl_mem dN = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * N, nullptr, &clError2);
    clError |= clError2;
    CLUtil::handleCLErrors(clError, "Error allocating device arrays");

    CLUtil::handleCLErrors(clEnqueueWriteBuffer(m_CommandQueue, dM, CL_FALSE, 0, sizeof(cl_uint) * N, src, 0, NULL, NULL), "Error copying data");


    size_t globalWorkSize[2] = {N/2, 1};
    size_t localWorkSize[2] = {16, 1};
    if (localWorkSize[0] > globalWorkSize[0])
        localWorkSize[0] = globalWorkSize[0];

    CLUtil::handleCLErrors(clSetKernelArg(m_reduceDecompKernel, 2, sizeof(uint) * localWorkSize[0], NULL), "Error creating local mem");

    int iterations = 100;
    double time = 0.0;
    cl_mem Ping = dM;
    cl_mem Pong = dN;
    while (true)
    {
        clError = clSetKernelArg(m_reduceDecompKernel, 0, sizeof(cl_mem), static_cast<void*>(&Ping));
        clError |= clSetKernelArg(m_reduceDecompKernel, 1, sizeof(cl_mem), static_cast<void*>(&Pong));
        CLUtil::handleCLErrors(clError, "Error setting kernel args for m_reduceBasicKernel");


        clError = clFinish(m_CommandQueue);
        Timer t;
        for (int i = 0; i < iterations; ++i)
        {
        clError |= clEnqueueNDRangeKernel(m_CommandQueue, m_reduceDecompKernel, 1, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
        }
        clError = clFinish(m_CommandQueue);
        time += t.elapsed() / iterations;
        CLUtil::handleCLErrors(clError, "Error executing m_reduceBasicKernel!");

        cl_mem tmp = Ping;
        Ping = Pong;
        Pong = tmp;

        if (globalWorkSize[0] <= localWorkSize[0] * 2)
        {
            break;
        }

        globalWorkSize[0] /= localWorkSize[0] * 2;
        if (localWorkSize[0] > globalWorkSize[0])
            localWorkSize[0] = globalWorkSize[0];
    }

    std::cout << "elapsed: " << time << "ms" << std::endl;


    Timer t;
    uint res;
    CLUtil::handleCLErrors(clEnqueueReadBuffer(m_CommandQueue, Ping, CL_TRUE, 0, sizeof(uint), &res, 0, NULL, NULL), "Error reading data from device!");
    std::cout << "read: " << t.elapsed() << std::endl;

    return res;
}

uint TemplateMatch::reduceExtendedArray(uint* src, uint N)
{
#if false
    cl_int clError = CL_SUCCESS, clError2;
    cl_mem dM = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * (N*2), nullptr, &clError2);
    clError |= clError2;
    CLUtil::handleCLErrors(clError, "Error allocating device arrays");

    uint hM[N*2];
    for (int i = 0; i < N; ++i)
    {
        hM[i] = (uint)src[i];
    }
    for (int i = N; i < N*2;++i)
    {
        hM[i] = 0;
    }

    CLUtil::handleCLErrors(clEnqueueWriteBuffer(m_CommandQueue, dM, CL_FALSE, 0, sizeof(cl_uint) * (N*2), hM, 0, NULL, NULL), "Error copying data");

    clError = clSetKernelArg(m_reduceEABasicKernel, 0, sizeof(cl_mem), static_cast<void*>(&dM));
    CLUtil::handleCLErrors(clError, "Error setting kernel args for m_reduceBasicKernel");

    int iterations = 100;
    double time = 0.0;

    size_t globalWorkSize[2] = {N/2, 1};
    size_t localWorkSize[2] = {32, 1};
    if (localWorkSize[0] > globalWorkSize[0])
        localWorkSize[0] = globalWorkSize[0];

    uint step = 0;
    uint offsetRead = 0;
    uint offsetWrite = N;
    uint stride = globalWorkSize[0];
    while (globalWorkSize[0] > 0)
    {
        //std::cout << offsetRead << " " << offsetWrite << " " << stride << std::endl;

        clError |= clSetKernelArg(m_reduceEABasicKernel, 1, sizeof(cl_uint), static_cast<void*>(&offsetRead));
        clError |= clSetKernelArg(m_reduceEABasicKernel, 2, sizeof(cl_uint), static_cast<void*>(&offsetWrite));
        clError |= clSetKernelArg(m_reduceEABasicKernel, 3, sizeof(cl_uint), static_cast<void*>(&stride));
        CLUtil::handleCLErrors(clError, "Error setting kernel args for m_reduceBasicKernel");

        clError = clFinish(m_CommandQueue);
        Timer t;
        for (int i = 0; i < iterations; ++i)
        {
        clError |= clEnqueueNDRangeKernel(m_CommandQueue, m_reduceEABasicKernel, 1, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
        }
        clError = clFinish(m_CommandQueue);
        time += t.elapsed() / iterations;
        CLUtil::handleCLErrors(clError, "Error executing m_reduceBasicKernel!");

        /*
        CLUtil::handleCLErrors(clEnqueueReadBuffer(m_CommandQueue, dM, CL_TRUE, 0, sizeof(cl_uint) * (N*2), dst, 0, NULL, NULL), "Error reading data from device!");

        std::cout << "s" << step << ":";
        for (int i = 0; i < N*2; ++i)
        {
            std::cout << dst[i] << " ";
        }
        std::cout << std::endl;
        */

        offsetRead += globalWorkSize[0]*2;
        offsetWrite += globalWorkSize[0];

        globalWorkSize[0] /= 2;
        if (localWorkSize[0] > globalWorkSize[0])
            localWorkSize[0] = globalWorkSize[0];

        stride /= 2;

        step++;
    }

    std::cout << "elapsed: " << time << "ms" << std::endl;


    Timer t;
    uint res;
    CLUtil::handleCLErrors(clEnqueueReadBuffer(m_CommandQueue, dM, CL_TRUE, (N*2-2)*sizeof(uint), sizeof(uint), &res, 0, NULL, NULL), "Error reading data from device!");
    std::cout << "read: " << t.elapsed() << std::endl;
    return res;
#endif
    return 0;
}

void TemplateMatch::prefixSumBasic(uint* src, uint* dst, uint N)
{
    size_t globalWorkSize[2] = {N/2, 1};
    size_t localWorkSize[2] = {256, 1};
    if (localWorkSize[0] > globalWorkSize[0])
        localWorkSize[0] = globalWorkSize[0];


    uint groupsize = localWorkSize[0]*2;
    std::cout << "groupsize: " << groupsize << std::endl;

    cl_int clError = CL_SUCCESS, clError2;
    cl_mem dM = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * N, nullptr, &clError2);
    clError |= clError2;
    cl_mem dN = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * N, nullptr, &clError2);
    clError |= clError2;
    cl_mem dO = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * groupsize, nullptr, &clError2);
    clError |= clError2;
    CLUtil::handleCLErrors(clError, "Error allocating device arrays");

    CLUtil::handleCLErrors(clEnqueueWriteBuffer(m_CommandQueue, dM, CL_FALSE, 0, sizeof(cl_uint) * N, src, 0, NULL, NULL), "Error copying data");

    assert(globalWorkSize[0] % localWorkSize[0] == 0);

    CLUtil::handleCLErrors(clSetKernelArg(m_prefixSumBasicKernel, 2, sizeof(uint) * localWorkSize[0] * 2, NULL), "Error creating local mem");

    int iterations = 100;
    double time = 0.0;
    //while (true)
    {
        clError = clSetKernelArg(m_prefixSumBasicKernel, 0, sizeof(cl_mem), static_cast<void*>(&dM));
        clError |= clSetKernelArg(m_prefixSumBasicKernel, 1, sizeof(cl_mem), static_cast<void*>(&dN));
        CLUtil::handleCLErrors(clError, "Error setting kernel args for m_reduceBasicKernel");


        clError = clFinish(m_CommandQueue);
        Timer t;
        for (int i = 0; i < iterations; ++i)
        {
        clError |= clEnqueueNDRangeKernel(m_CommandQueue, m_prefixSumBasicKernel, 1, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
        }
        clError = clFinish(m_CommandQueue);
        time += t.elapsed() / iterations;
        CLUtil::handleCLErrors(clError, "Error executing m_reduceBasicKernel!");
    }

    std::cout << "elapsed: " << time << "ms" << std::endl;


    Timer t;
    CLUtil::handleCLErrors(clEnqueueReadBuffer(m_CommandQueue, dN, CL_TRUE, 0, sizeof(uint)*N, dst, 0, NULL, NULL), "Error reading data from device!");
    std::cout << "read: " << t.elapsed() << std::endl;

    if (globalWorkSize[0] > localWorkSize[0])
    {
        // Fill groupsums array with last elements of local groups
        uint groupCount = globalWorkSize[0] / localWorkSize[0];

        std::vector<uint> groupSums(groupsize, 0);

        for (uint i = 0; i < groupCount; ++i)
        {
            uint idx = i * groupsize + groupsize - 1;
            groupSums[i] = dst[idx] + src[idx];
        }

        for (uint i = 0; i < groupSums.size(); ++i)
        {
            std::cout << groupSums[i] << " ";
        }
        std::cout << std::endl;

        // Start prefix sum of local prefix sums
        CLUtil::handleCLErrors(clEnqueueWriteBuffer(m_CommandQueue, dM, CL_FALSE, 0, sizeof(cl_uint) * groupsize, &groupSums[0], 0, NULL, NULL), "Error copying data");

        clError = clSetKernelArg(m_prefixSumBasicKernel, 0, sizeof(cl_mem), static_cast<void*>(&dM));
        clError |= clSetKernelArg(m_prefixSumBasicKernel, 1, sizeof(cl_mem), static_cast<void*>(&dO));
        CLUtil::handleCLErrors(clError, "Error setting kernel args for m_prefixSumBasicKernel");

        globalWorkSize[0] = localWorkSize[0];

        clError |= clEnqueueNDRangeKernel(m_CommandQueue, m_prefixSumBasicKernel, 1, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);

        CLUtil::handleCLErrors(clEnqueueReadBuffer(m_CommandQueue, dO, CL_TRUE, 0, sizeof(uint)*groupsize, &groupSums[0], 0, NULL, NULL), "Error reading data from device!");

        for (uint i = 0; i < groupSums.size(); ++i)
        {
            std::cout << groupSums[i] << " ";
        }
        std::cout << std::endl;

        // Add groupsums
        clError = clSetKernelArg(m_prefixSumAddKernel, 0, sizeof(cl_mem), static_cast<void*>(&dN));
        clError |= clSetKernelArg(m_prefixSumAddKernel, 1, sizeof(cl_mem), static_cast<void*>(&dO));
        clError |= clSetKernelArg(m_prefixSumAddKernel, 2, sizeof(cl_uint), static_cast<void*>(&groupsize));
        CLUtil::handleCLErrors(clError, "Error setting kernel args for m_prefixSumAddKernel");

        globalWorkSize[0] = N;

        clError |= clEnqueueNDRangeKernel(m_CommandQueue, m_prefixSumAddKernel, 1, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);

        CLUtil::handleCLErrors(clEnqueueReadBuffer(m_CommandQueue, dN, CL_TRUE, 0, sizeof(uint)*N, dst, 0, NULL, NULL), "Error reading data from device!");
    }
}
