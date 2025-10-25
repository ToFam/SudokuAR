#pragma once

#include <string>

class Container;
class Algorithm;

namespace CLUtil
{
class CLHandler;
}

class TestSolver
{
public:
    TestSolver(const CLUtil::CLHandler& handler) : m_context(handler) {}
    virtual ~TestSolver() {};

	virtual bool DoCompute();

    virtual bool RunAlgorithm(Algorithm* algo, int iterationsCPU, int iterationsGPU,
                              Container& outputContainer, const std::string& outputFilePrefix,
                              const std::string& outputFileElement, bool outputImage);

private:
    const CLUtil::CLHandler& m_context;
};

