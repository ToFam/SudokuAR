#pragma once

#if defined(WIN32)
    #include <CL/opencl.h>
#elif defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif 

class IComputeTask
{
public:
    virtual ~IComputeTask() = default;

	//! Init any resources specific to the current task
    virtual bool InitResources(cl_device_id Device, cl_context Context, cl_command_queue CommandQueue) = 0;
};
