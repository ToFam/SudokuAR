#pragma once

#include "../Algorithm.h"
class Field;

class Sudoku : public Algorithm
{
public:
    Sudoku(size_t N);
    ~Sudoku() override;

private:
    bool solveTrial(Field &f, size_t &outRow, size_t &outCol, int &outValue, int recursionDepth, Field& outResult);
    bool solveStep(Field& f, int recursionDepth, Field& outResult);

    void print(int* field, int recursionDepth);

    void DoCompute();

public: // IComputeTask
    bool InitResources(cl_device_id Device, cl_context Context, cl_command_queue CommandQueue) override;

    void ReleaseResources();
    bool ValidateResults();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;

public:
    void setLogLevel(int level);

private:
    unsigned int m_N;
    unsigned int m_possArrayCellSize;

    std::vector<int> m_hArray;

    std::vector<int> m_hResultCPU;

    std::vector<int> m_hResultGPU;
    int m_hResultGPUFlags[3];
    std::vector<unsigned char> m_hDebugArray;

    cl_mem m_dArray;
    cl_mem m_dFlags;
    cl_mem m_dDebugArray;

    cl_program m_Program;
    cl_kernel  m_SolverKernel;

    int m_logLevel;
};
