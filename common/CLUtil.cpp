#include "CLUtil.h"
#include "Timer.h"
#include "IComputeTask.h"

#include <iostream>
#include <vector>

namespace CLUtil
{

CLException::CLException(cl_int errorCode, std::string_view msg)
    : std::runtime_error(std::format("{} {}", msg, errorToString(errorCode))) {}

void handleCLErrors(cl_int errorCode, std::string_view msg)
{
    if (errorCode != CL_SUCCESS)
    {
        throw CLException(errorCode, msg);
    }
}

cl_program compileProgram(cl_device_id device, cl_context context,
                          const std::string& sourceCode, const std::string& options)
{
    cl_program prog = nullptr;

    const char* src = sourceCode.c_str();
    size_t length = sourceCode.size();

	cl_int clError;
    prog = clCreateProgramWithSource(context, 1, &src, &length, &clError);
    handleCLErrors(clError, "Could not create program");

	// program created, now build it:
    const char* pCompileOptions = options.size() > 0 ? options.c_str() : nullptr;
    clError = clBuildProgram(prog, 1, &device, pCompileOptions, NULL, NULL);
    printBuildLog(prog, device);
    if(clError != CL_SUCCESS)
	{
        clReleaseProgram(prog);
        throw CLException(clError, "Failed to build CL program");
	}

	return prog;
}

void printBuildLog(cl_program program, cl_device_id device)
{
	cl_build_status buildStatus;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS,
                          sizeof(cl_build_status), &buildStatus, NULL);

	size_t logSize;
    cl_int clError = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    handleCLErrors(clError, "Could not get build log");
    std::string buildLog(logSize, '\0');

    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], NULL);
    handleCLErrors(clError, "Could not get build log");

	if(buildStatus != CL_SUCCESS)
    {
        std::cout << "Errors during OpenCL program build!" << std::endl;
    }
    std::cout << "OpenCL program build log:" << std::endl
              << buildLog << std::endl
              << "=================================" << std::endl;
}

double profileKernel(cl_command_queue q, cl_kernel kernel, cl_uint Dimensions,
        const size_t* pGlobalWorkSize, const size_t* pLocalWorkSize, int NIterations)
{
	cl_int clErr;

    // sync
    clErr = clFinish(q);
    Timer timer;
    for(int i = 0; i < NIterations; ++i)
	{
        clErr |= clEnqueueNDRangeKernel(q, kernel, Dimensions, NULL,
                                        pGlobalWorkSize, pLocalWorkSize, 0, NULL, NULL);
    }
    clErr |= clFinish(q);
    double elapsed = timer.elapsed();
    handleCLErrors(clErr);

    return elapsed / double(NIterations);
}

#define CL_ERROR(x) case (x): return #x;

const char* errorToString(cl_int CLErrorCode)
{
	switch(CLErrorCode)
	{
        CL_ERROR(CL_SUCCESS);
        CL_ERROR(CL_DEVICE_NOT_FOUND);
        CL_ERROR(CL_DEVICE_NOT_AVAILABLE);
        CL_ERROR(CL_COMPILER_NOT_AVAILABLE);
        CL_ERROR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        CL_ERROR(CL_OUT_OF_RESOURCES);
        CL_ERROR(CL_OUT_OF_HOST_MEMORY);
        CL_ERROR(CL_PROFILING_INFO_NOT_AVAILABLE);
        CL_ERROR(CL_MEM_COPY_OVERLAP);
        CL_ERROR(CL_IMAGE_FORMAT_MISMATCH);
        CL_ERROR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
        CL_ERROR(CL_BUILD_PROGRAM_FAILURE);
        CL_ERROR(CL_MAP_FAILURE);
        CL_ERROR(CL_INVALID_VALUE);
        CL_ERROR(CL_INVALID_DEVICE_TYPE);
        CL_ERROR(CL_INVALID_PLATFORM);
        CL_ERROR(CL_INVALID_DEVICE);
        CL_ERROR(CL_INVALID_CONTEXT);
        CL_ERROR(CL_INVALID_QUEUE_PROPERTIES);
        CL_ERROR(CL_INVALID_COMMAND_QUEUE);
        CL_ERROR(CL_INVALID_HOST_PTR);
        CL_ERROR(CL_INVALID_MEM_OBJECT);
        CL_ERROR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
        CL_ERROR(CL_INVALID_IMAGE_SIZE);
        CL_ERROR(CL_INVALID_SAMPLER);
        CL_ERROR(CL_INVALID_BINARY);
        CL_ERROR(CL_INVALID_BUILD_OPTIONS);
        CL_ERROR(CL_INVALID_PROGRAM);
        CL_ERROR(CL_INVALID_PROGRAM_EXECUTABLE);
        CL_ERROR(CL_INVALID_KERNEL_NAME);
        CL_ERROR(CL_INVALID_KERNEL_DEFINITION);
        CL_ERROR(CL_INVALID_KERNEL);
        CL_ERROR(CL_INVALID_ARG_INDEX);
        CL_ERROR(CL_INVALID_ARG_VALUE);
        CL_ERROR(CL_INVALID_ARG_SIZE);
        CL_ERROR(CL_INVALID_KERNEL_ARGS);
        CL_ERROR(CL_INVALID_WORK_DIMENSION);
        CL_ERROR(CL_INVALID_WORK_GROUP_SIZE);
        CL_ERROR(CL_INVALID_WORK_ITEM_SIZE);
        CL_ERROR(CL_INVALID_GLOBAL_OFFSET);
        CL_ERROR(CL_INVALID_EVENT_WAIT_LIST);
        CL_ERROR(CL_INVALID_EVENT);
        CL_ERROR(CL_INVALID_OPERATION);
        CL_ERROR(CL_INVALID_GL_OBJECT);
        CL_ERROR(CL_INVALID_BUFFER_SIZE);
        CL_ERROR(CL_INVALID_MIP_LEVEL);
        default:
			return "Unknown error code";
	}
}

CLHandler::CLHandler()
{
    initContext();
}

CLHandler::~CLHandler()
{
    if (m_CLCommandQueue != nullptr)
    {
        clReleaseCommandQueue(m_CLCommandQueue);
        m_CLCommandQueue = nullptr;
    }

    if (m_CLContext != nullptr)
    {
        clReleaseContext(m_CLContext);
        m_CLContext = nullptr;
    }
}

void CLHandler::initTask(IComputeTask& task) const
{
    if (!task.InitResources(m_CLDevice, m_CLContext, m_CLCommandQueue))
    {
        throw std::runtime_error("Error during resource allocation of task");
    }
}

#define PRINT_INFO(title, buffer, bufferSize, maxBufferSize, expr) { expr; buffer[bufferSize] = '\0'; std::cout << title << ": " << buffer << std::endl; }

void CLHandler::initContext()
{
    std::vector<cl_platform_id> platformIds;
    const cl_uint c_MaxPlatforms = 16;
    platformIds.resize(c_MaxPlatforms);

    cl_uint countPlatforms;
    handleCLErrors(clGetPlatformIDs(c_MaxPlatforms, &platformIds[0], &countPlatforms), "Failed to get CL platform ID");
    platformIds.resize(countPlatforms);

    std::vector<cl_device_id> deviceIds;
    const int maxDevices = 16;
    deviceIds.resize(maxDevices);
    int countAllDevices = 0;

    // Searching for the graphics device with the most dedicated video memory.
    cl_device_type deviceType = CL_DEVICE_TYPE_GPU;

    cl_ulong maxGlobalMemorySize = 0;
    cl_device_id bestDeviceId = NULL;

    for (size_t i = 0; i < platformIds.size(); i++)
    {
        // Getting the available devices.
        cl_uint countDevices;
        auto res = clGetDeviceIDs(platformIds[i], deviceType, 1, &deviceIds[countAllDevices], &countDevices);
        if(res != CL_SUCCESS) // Maybe there are no GPU devices and some poor implementation
                              // doesn't set count devices to zero and return CL_DEVICE_NOT_FOUND.
        {
            char buffer[1024];
            clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, 1024, buffer, nullptr);
            std::cout << std::format("[WARNING]: clGetDeviceIDs() failed. Error type: {}, Platform name: {}!",
                   CLUtil::errorToString(res), buffer) << std::endl;
            continue;
        }
        for (size_t j = 0; j < countDevices; j++)
        {
            cl_device_id currentDeviceId = deviceIds[countAllDevices + j];
            cl_ulong globalMemorySize;
            cl_bool isUsingUnifiedMemory;
            clGetDeviceInfo(currentDeviceId, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalMemorySize, NULL);
            clGetDeviceInfo(currentDeviceId, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &isUsingUnifiedMemory, NULL);

            if (!isUsingUnifiedMemory && globalMemorySize > maxGlobalMemorySize)
            {
                bestDeviceId = currentDeviceId;
                maxGlobalMemorySize = globalMemorySize;
            }
        }

        countAllDevices += countDevices;
    }
    deviceIds.resize(countAllDevices);

    if (countAllDevices == 0)
    {
        throw std::runtime_error("No device with OpenCL support was found.");
    }

    // No discrete graphics device was found: falling back to the first found device.
    if (bestDeviceId == NULL)
    {
        bestDeviceId = deviceIds[0];
    }

    // Choosing the first available device.
    m_CLDevice = bestDeviceId;
    clGetDeviceInfo(m_CLDevice, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &m_CLPlatform, NULL);

    // Printing platform and device data.
    const int maxBufferSize = 1024;
    char buffer[maxBufferSize];
    size_t bufferSize;
    std::cout << "OpenCL platform:" << std::endl << std::endl;
    PRINT_INFO("Name", buffer, bufferSize, maxBufferSize, clGetPlatformInfo(m_CLPlatform, CL_PLATFORM_NAME, maxBufferSize, (void*)buffer, &bufferSize));
    PRINT_INFO("Vendor", buffer, bufferSize, maxBufferSize, clGetPlatformInfo(m_CLPlatform, CL_PLATFORM_VENDOR, maxBufferSize, (void*)buffer, &bufferSize));
    PRINT_INFO("Version", buffer, bufferSize, maxBufferSize, clGetPlatformInfo(m_CLPlatform, CL_PLATFORM_VERSION, maxBufferSize, (void*)buffer, &bufferSize));
    PRINT_INFO("Profile", buffer, bufferSize, maxBufferSize, clGetPlatformInfo(m_CLPlatform, CL_PLATFORM_PROFILE, maxBufferSize, (void*)buffer, &bufferSize));
    std::cout << std::endl << "Device:" << std::endl << std::endl;
    PRINT_INFO("Name", buffer, bufferSize, maxBufferSize, clGetDeviceInfo(m_CLDevice, CL_DEVICE_NAME, maxBufferSize, (void*)buffer, &bufferSize));
    PRINT_INFO("Vendor", buffer, bufferSize, maxBufferSize, clGetDeviceInfo(m_CLDevice, CL_DEVICE_VENDOR, maxBufferSize, (void*)buffer, &bufferSize));
    PRINT_INFO("Driver version", buffer, bufferSize, maxBufferSize, clGetDeviceInfo(m_CLDevice, CL_DRIVER_VERSION, maxBufferSize, (void*)buffer, &bufferSize));
    cl_ulong localMemorySize;
    clGetDeviceInfo(m_CLDevice, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemorySize, &bufferSize);
    std::cout << "Local memory size: " << localMemorySize << " Byte" << std::endl;
    std::cout << std::endl << "******************************" << std::endl << std::endl;


    cl_int clError;
    m_CLContext = clCreateContext(NULL, 1, &m_CLDevice, NULL, NULL, &clError);
    handleCLErrors(clError, "Failed to create OpenCL context.");

    m_CLCommandQueue = clCreateCommandQueue(m_CLContext, m_CLDevice, 0, &clError);
    handleCLErrors(clError, "Failed to create the command queue in the context");
}

} // namespace CLUtil
