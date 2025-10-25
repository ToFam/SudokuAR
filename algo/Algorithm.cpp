#include "Algorithm.h"

Algorithm::Algorithm(std::string_view name, bool canDoLiveUpdate)
    : m_name(name.begin(), name.end()), m_canDoLiveUpdate(canDoLiveUpdate),
    m_implSet(false), m_runtime(0.0), m_iterations(1)
{

}

Algorithm::~Algorithm()
{

}

std::vector<ContainerSpecification> Algorithm::specification() const
{
    return m_argumentsSpecification;
}

AlgorithmSettings Algorithm::settings() const
{
    return m_settings;
}

AlgorithmSettings& Algorithm::settings()
{
    return m_settings;
}

void Algorithm::clearContainerStack()
{
    m_arguments.clear();
}

void Algorithm::addContainer(std::shared_ptr<Container> c)
{
    m_arguments.push_back(c);
}

bool Algorithm::setImplementation(Algorithm::ImplementationType impl)
{
    for (auto si : supportedImplementations())
    {
        if (si == impl)
        {
            m_implSet = true;
            m_activeImpl = impl;
            return true;
        }
    }

    return false;
}

double Algorithm::runtime() const
{
    return m_runtime;
}

void Algorithm::setIterations(int iterations)
{
    m_iterations = iterations;
}

bool Algorithm::InitResources(cl_device_id Device, cl_context Context, cl_command_queue CommandQueue)
{
    m_Device = Device;
    m_Context = Context;
    m_CommandQueue = CommandQueue;

    return true;
}
