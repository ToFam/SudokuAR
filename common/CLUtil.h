#pragma once

#include <stdexcept>
#if defined(WIN32)
    #include <CL/opencl.h>
#elif defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

#include <string>

class IComputeTask;

namespace CLUtil
{
cl_program compileProgram(cl_device_id device, cl_context context,
                          const std::string& sourceCode, const std::string& options = "");

void printBuildLog(cl_program program, cl_device_id device);

/**
 * @brief profileKernel Execute \a kernel \a NIterations times and measure runtime
 * @return average runtime in milliseconds
 */
double profileKernel(cl_command_queue commandQueue, cl_kernel kernel, cl_uint dimensions,
    const size_t* pGlobalWorkSize, const size_t* pLocalWorkSize, int NIterations);

/**
 * @return printable name of opencl error code
 */
const char* errorToString(cl_int CLErrorCode);

class CLHandler
{
public:
    CLHandler();
    virtual ~CLHandler();

    void initTask(IComputeTask &task) const;

private:
    void initContext();

private:
    cl_platform_id		m_CLPlatform;
    cl_device_id		m_CLDevice;
    cl_context			m_CLContext;
    cl_command_queue	m_CLCommandQueue;
};

class CLException : public std::runtime_error
{
public:
    CLException(cl_int errorCode, std::string_view msg = "OpenCL error: ");
};

void handleCLErrors(cl_int errorCode, std::string_view msg = "OpenCL error: ");

}
