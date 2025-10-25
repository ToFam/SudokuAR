#pragma once

#include <IComputeTask.h>

#include "AlgorithmSettings.h"
#include "Container.h"

class Algorithm : public IComputeTask
{
public:
    enum ImplementationType
    {
        CPU,
        GPU,
        OPENCV_CPU,
        OPENCV_GPU
    };

public:
    Algorithm(std::string_view name, bool canDoLiveUpdate = false);
    virtual ~Algorithm();

public:
    virtual std::vector<ImplementationType> supportedImplementations() const = 0;
    virtual bool exec() = 0;

    std::string name() const { return m_name; }

    bool canDoLiveUpdate() const { return m_canDoLiveUpdate; }

    std::vector<ContainerSpecification> specification() const;
    AlgorithmSettings settings() const;
    AlgorithmSettings& settings();

    double runtime() const;
    void setIterations(int iterations);

    void clearContainerStack();
    void addContainer(std::shared_ptr<Container> c);

    bool setImplementation(ImplementationType impl);

public: // IComputeTask
    bool InitResources(cl_device_id Device, cl_context Context, cl_command_queue CommandQueue) override;

private:
    std::string m_name;
    bool m_canDoLiveUpdate;

protected:
    std::vector<ContainerSpecification> m_argumentsSpecification;
    AlgorithmSettings m_settings;

    std::vector<std::shared_ptr<Container>> m_arguments;

    bool    m_implSet;
    ImplementationType m_activeImpl;

    cl_device_id m_Device;
    cl_context m_Context;
    cl_command_queue m_CommandQueue;

    double m_runtime;
    int m_iterations;
};
