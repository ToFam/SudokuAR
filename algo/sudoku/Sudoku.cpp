#include "Sudoku.h"

#include <cmath>
#include <assert.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include <CLUtil.h>
#include <Timer.h>
#include <Utils.h>

#define EPS 1e-8

class Cell
{
public:
    Cell(size_t N);

    /**
     * @return -1 if invalid (no solution possible);
     *          0 if no value set yet (multiple possibilities left)
     *          1-N if value set
     */
    int value() const;

    void setValue(int value);

    void set(int value);

    void disable(int value);
    //void setPossibilities(bool* possibilities);
    const std::vector<bool>& possibilities() const;

    /**
     * @brief solve write value if only one possibility left
     * @return true if value was set, false if more than one possibility remains
     */
    bool solve();

private:
    size_t m_N;
    int m_value;
    std::vector<bool> m_possible;
};

class Field
{
public:
    Field(size_t N);

    void setValue(size_t row, size_t col, int value);
    int value(size_t row, size_t col) const;
    const std::vector<bool>& possibilities(size_t row, size_t col) const;

    /**
     * @brief solveStep solve cells with only one possibility left
     * @return count of solved cells in this step
     */
    int solveStep();

    void print(int recursionDepth) const;

    bool valid() const;
    bool solved() const;

    std::vector<Cell>& cells() { return m_cells; }

private:
    size_t m_N;
    std::vector<Cell> m_cells;
};


Sudoku::Sudoku(size_t N) : Algorithm("Sudoku"), m_N(N),
    m_dArray(nullptr), m_Program(nullptr), m_SolverKernel(nullptr)
{
    ContainerSpecification input("in_grid", ContainerSpecification::READ_ONLY);
    ContainerSpecification output("out_grid", ContainerSpecification::REFERENCE);
    m_argumentsSpecification.push_back(input);
    m_argumentsSpecification.push_back(output);

    // only square numbers are allowed for N, otherwise we cant have blocks
    assert(sqrt(m_N) == std::trunc(sqrt(m_N)));

    // How many bytes are needed to store flags for each possible number
    m_possArrayCellSize = static_cast<unsigned int>(ceil(double(m_N) / 8.0));
}

Sudoku::~Sudoku()
{
    ReleaseResources();
}

void Sudoku::setLogLevel(int level)
{
    m_logLevel = level;
}

std::vector<Algorithm::ImplementationType> Sudoku::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(CPU);
    v.push_back(GPU);
    return v;
}

bool Sudoku::exec()
{
    if (!m_implSet)
    {
        m_activeImpl = CPU;
    }

    std::vector<double> runtimes;

    auto input = m_arguments[0];
    for (size_t s = 0; s < input->size(); ++s)
    {
        auto sudoku = input->get(s);

        if (sudoku->empty())
            continue;

        m_hArray = std::vector<int>(*sudoku);
        if (m_hArray.size() != m_N * m_N)
            return false;

        Timer timer;

        for (int i = 0; i < m_iterations; ++i)
        {
            DoCompute();
        }

        runtimes.push_back(timer.elapsed() / double(m_iterations));


        auto output = std::make_shared<cv::Mat>();
        if (m_activeImpl == ImplementationType::CPU)
            cv::Mat(m_hResultCPU).copyTo(*output);
        else if (m_activeImpl == ImplementationType::GPU)
            cv::Mat(m_hResultGPU).copyTo(*output);

        m_arguments[1]->add(output);
    }

    m_runtime = 0.0;
    for (double r : runtimes)
    {
        m_runtime += r;
    }
    m_runtime /= runtimes.size();

    return true;
}

bool Sudoku::InitResources(cl_device_id Device, cl_context Context, cl_command_queue CommandQueue)
{
    Algorithm::InitResources(Device, Context, CommandQueue);

    //device resources
    cl_int clError, clError2;
    m_dArray = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_int) * m_N * m_N, nullptr, &clError2);
    clError = clError2;
    m_dDebugArray = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_N * m_N, nullptr, &clError2);
    clError = clError2;
    m_dFlags = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_int) * 3, nullptr, &clError2);
    clError = clError2;
    CLUtil::handleCLErrors(clError, "Error allocating device arrays");

    m_Program = CLUtil::compileProgram(Device, Context, Utils::loadFile("algo/sudoku/Sudoku.cl"));
    if(m_Program == nullptr) return false;

    //create kernels
    m_SolverKernel = clCreateKernel(m_Program, "Sudoku", &clError);
    CLUtil::handleCLErrors(clError, "Failed to create kernel: Sudoku.");

    return true;
}

void Sudoku::ReleaseResources()
{
    if (m_dArray) clReleaseMemObject(m_dArray);
    if (m_SolverKernel) clReleaseKernel(m_SolverKernel);
    if (m_Program) clReleaseProgram(m_Program);
}

bool Sudoku::ValidateResults()
{
    for (size_t i = 0; i < m_N * m_N; ++i)
    {
        if (m_hResultCPU[i] != m_hResultGPU[i])
            return false;
    }

    return true;
}

void Sudoku::DoCompute()
{
    Field f(m_N), fResult(m_N);
    for (size_t row = 0; row < m_N; ++row)
    {
        for (size_t col = 0; col < m_N; ++col)
        {
            int val = m_hArray[row * m_N + col];
            if (val > 0)
            {
                f.setValue(row, col, val);
            }
            else
            {
                f.cells()[row * m_N + col].set(0);
            }
        }
    }



    if (m_logLevel > 0)
        f.print(0);

    if (!solveStep(f, 0, fResult))
    {
        std::cout << "Could not solve" << std::endl;
    }

    if (m_logLevel > 0)
        fResult.print(0);

    if (m_activeImpl == ImplementationType::CPU)
        m_hResultCPU.clear();
    else
        m_hResultGPU.clear();

    for (size_t row = 0; row < m_N; ++row)
    {
        for (size_t col = 0; col < m_N; ++col)
        {
            if (m_activeImpl == ImplementationType::CPU)
                m_hResultCPU.push_back(fResult.value(row, col));
            else
                m_hResultGPU.push_back(fResult.value(row, col));
        }
    }
}

bool Sudoku::solveStep(Field& f, int recursionDepth, Field& outResult)
{
    if (m_activeImpl == ImplementationType::CPU)
    {
        int singleStepSolved, acc = 0;
        do
        {
            singleStepSolved = f.solveStep();
            acc += singleStepSolved;
        } while (singleStepSolved > 0);

        if (m_logLevel > 0)
        {
            for (int tab = 0; tab < recursionDepth; tab++)
            {
                std::cout << " | ";
            }

            std::cout << "single step solve: " << acc << std::endl;
        }

        if (m_logLevel > 1)
            f.print(recursionDepth);

        if (f.solved())
        {
            outResult = f;
            return true;
        }

        if (!f.valid())
            return false;

        size_t row, col; int val;
        m_logLevel--;
        bool success = solveTrial(f, row, col, val, recursionDepth + 1, outResult);
        m_logLevel++;

        return success;
    }
    else if (m_activeImpl == ImplementationType::GPU)
    {
        // Write Field f to array
        for (size_t row = 0; row < m_N; ++row)
        {
            for (size_t col = 0; col < m_N; ++col)
            {
                m_hArray[row * m_N + col] = f.value(row, col);
            }
        }

        //write input data to the GPU
        CLUtil::handleCLErrors(clEnqueueWriteBuffer(m_CommandQueue, m_dArray, CL_FALSE, 0, m_N * m_N * sizeof(cl_int), m_hArray.data(), 0, NULL, NULL), "Error copying data from host to device!");

        cl_int clErr;

        size_t globalWorkSize[2] = {m_N, m_N};
        size_t localWorkSize[2] = {m_N, m_N};

        clErr = clSetKernelArg(m_SolverKernel, 0, sizeof(cl_mem), static_cast<void*>(&m_dArray));
        clErr |= clSetKernelArg(m_SolverKernel, 1, sizeof(cl_mem), static_cast<void*>(&m_dDebugArray));
        clErr |= clSetKernelArg(m_SolverKernel, 2, sizeof(cl_mem), static_cast<void*>(&m_dFlags));
        clErr |= clSetKernelArg(m_SolverKernel, 6, sizeof(cl_uint), static_cast<void*>(&m_N));
        clErr |= clSetKernelArg(m_SolverKernel, 7, sizeof(cl_uint), static_cast<void*>(&m_possArrayCellSize));
        CLUtil::handleCLErrors(clErr, "Error setting kernel args for SolverKernel");

        clErr = clSetKernelArg(m_SolverKernel, 3, sizeof(cl_int) * 4, NULL);
        clErr |= clSetKernelArg(m_SolverKernel, 4, sizeof(cl_int) * m_N * m_N, NULL);
        clErr |= clSetKernelArg(m_SolverKernel, 5, sizeof(cl_char) * m_N * m_N * m_possArrayCellSize, NULL);
        CLUtil::handleCLErrors(clErr, "Error allocating shared memory!");

        clErr = clEnqueueNDRangeKernel(m_CommandQueue, m_SolverKernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
        CLUtil::handleCLErrors(clErr, "Error executing SolverKernel!");


        //m_hDebugArray.resize(m_N * m_N * m_possArrayCellSize);

        m_hResultGPU.resize(m_N * m_N);
        CLUtil::handleCLErrors(clEnqueueReadBuffer(m_CommandQueue, m_dArray, CL_FALSE, 0, m_N * m_N * sizeof(cl_int), m_hResultGPU.data(), 0, NULL, NULL), "Error reading data from device!");
        CLUtil::handleCLErrors(clEnqueueReadBuffer(m_CommandQueue, m_dFlags, CL_TRUE, 0, 3 * sizeof(cl_int), m_hResultGPUFlags, 0, NULL, NULL), "Error reading data from device!");


        // CLUtil::handleCLErrors(clEnqueueReadBuffer(m_CommandQueue, m_dDebugArray, CL_FALSE, 0, m_N * m_N * m_possArrayCellSize * sizeof(cl_uchar),
        //                               m_hDebugArray.data(), 0, NULL, NULL), "Error reading data from device!");

        //cv::Mat debugResult(m_N * m_N * m_possArrayCellSize, 1, CV_8U, m_hDebugArray.data());
        //
        //cv::FileStorage fs("eval/debugArray.yml", cv::FileStorage::WRITE);
        //if (fs.isOpened())
        //{
        //    fs << "possiblities" << debugResult;
        //}

        // Write to F
        for (size_t row = 0; row < m_N; ++row)
        {
            for (size_t col = 0; col < m_N; ++col)
            {
                int value = m_hResultGPU[row * m_N + col];
                if (value > 0)
                {
                    f.setValue(row, col, value);
                }
            }
        }

        if (m_logLevel > 0)
        {
            for (int tab = 0; tab < recursionDepth; tab++)
            {
                std::cout << " | ";
            }

            std::cout << "single step solve: " << m_hResultGPUFlags[2] << std::endl;
        }

        if (m_logLevel > 1)
            print(m_hResultGPU.data(), recursionDepth);

        if (m_hResultGPUFlags[0] == 0) // solved
        {
            outResult = f;
            return true;
        }

        if (m_hResultGPUFlags[1] > 0) // invalid
        {
            return false;
        }

        size_t row, col; int val;
        m_logLevel--;
        bool success = solveTrial(f, row, col, val, recursionDepth + 1, outResult);
        m_logLevel++;

        return success;
    }
    else
    {
        return false;
    }
}

bool Sudoku::solveTrial(Field &f, size_t &outRow, size_t &outCol, int &outValue, int recursionDepth, Field& outResult)
{
    for (size_t i = 0; i < m_N * m_N; i++)
    {
        outRow = i / 9;
        outCol = i % 9;
        if (f.value(outRow, outCol) == 0)
        {
            const std::vector<bool>& ps = f.possibilities(outRow, outCol);

            for (int p = 0; p < m_N; p++)
            {
                if (ps[p] == true)
                {
                    outValue = p + 1;

                    if (m_logLevel > -1)
                    {
                        for (int tab = 0; tab < recursionDepth - 1; tab++)
                        {
                            std::cout << " | ";
                        }

                        std::cout << "trial solve: (" << outCol << "|" << outRow << ") = " << outValue << std::endl;
                    }

                    Field f2 = f;
                    f2.setValue(outRow, outCol, outValue);

                    if (m_logLevel > 1)
                        f2.print(recursionDepth);

                    if (solveStep(f2, recursionDepth, outResult))
                    {
                        return true;
                    }
                }
            }

            return false;
        }
    }

    assert(false);
    return f.solved();
}

Cell::Cell(size_t N) : m_N(N)
{
    m_value = -1;
    m_possible = std::vector<bool>(m_N, true);
}

void Cell::disable(int value)
{
    m_possible[value - 1] = false;
}

const std::vector<bool> &Cell::possibilities() const
{
    return m_possible;
}

void Cell::setValue(int value)
{
    for (int i = 0; i < m_N; i++)
        m_possible[i] = false;
    m_possible[value - 1] = true;
    m_value = value;
}

void Cell::set(int value)
{
    m_value = value;
}

int Cell::value() const
{
    return m_value;
}

bool Cell::solve()
{
    int maybeSolution = -1;
    bool invalid = true;
    for (int i = 0; i < m_N; i++)
    {
        if (m_possible[i] == true)
        {
            invalid = false;

            if (maybeSolution == -1)
            {
                maybeSolution = i;
            }
            else
            {
                maybeSolution = -1;
                break;
            }
        }
    }

    if (invalid)
    {
        // no possible solutions left
        m_value = -1;
    }
    else if (maybeSolution == -1)
    {
        // undecided
        m_value = 0;
    }
    else
    {
        // only value possible
        m_value = maybeSolution + 1;
    }

    return m_value > 0;
}

Field::Field(size_t N) : m_N(N)
{
    for (size_t i = 0; i < m_N * m_N; ++i)
        m_cells.emplace_back(m_N);
}

void Field::setValue(size_t row, size_t col, int value)
{
    assert(row >= 0 && row < m_N && col >= 0 && col < m_N);

    for (size_t i = 0; i < m_N; i++)
        m_cells[m_N * row + i].disable(value);

    for (size_t i = 0; i < m_N; i++)
        m_cells[m_N * i + col].disable(value);

    size_t r = sqrt(m_N);
    for (size_t i = r * (row / r); i < (r * (row / r) + r); i++)
    {
        for (size_t j = r * (col / r); j < (r * (col / r) + r); j++)
        {
            m_cells[m_N * i + j].disable(value);
        }
    }

    m_cells[m_N * row + col].setValue(value);
}

int Field::value(size_t row, size_t col) const
{
    assert(row >= 0 && row < m_N && col >= 0 && col < m_N);

    return m_cells[m_N * row + col].value();
}

const std::vector<bool>& Field::possibilities(size_t row, size_t col) const
{
    assert(row >= 0 && row < m_N && col >= 0 && col < m_N);

    return m_cells[m_N * row + col].possibilities();
}

int Field::solveStep()
{
    struct change{
        size_t row; size_t col; int value;
    };
    std::vector<change> changes;

    for (size_t i = 0; i < m_N * m_N; i++)
    {
        if (m_cells[i].value() > 0)
            continue;

        if (m_cells[i].solve())
        {
            changes.push_back({i / m_N, i % m_N, m_cells[i].value()});
        }
    }

    for (change c : changes)
    {
        if (m_cells[m_N * c.row + c.col].solve())
            setValue(c.row, c.col, c.value);
    }

    return changes.size();
}

bool Field::solved() const
{
    bool missing = false;
    for (size_t i = 0; i < m_N * m_N; i++)
    {
        if (m_cells[i].value() < 1)
            missing = true;
    }
    return !missing;
}

bool Field::valid() const
{
    for (size_t i = 0; i < m_N * m_N; i++)
    {
        if (m_cells[i].value() < 0)
            return false;
    }
    return true;
}

void Field::print(int recursionDepth) const
{
    size_t r = sqrt(m_N);
    for (size_t i = 0; i < m_N + r + 1; i++)
    {
        for (int tab = 0; tab < recursionDepth; tab++)
        {
            std::cout << " | ";
        }

        for (size_t j = 0; j < m_N + r + 1; j++)
        {
            if (i % (r + 1) == 0)
                std::cout << "--";
            else
            {
                if (j % (r + 1) == 0)
                    std::cout << "|";
                else
                {
                    size_t row = i - (i / (r + 1) + 1);
                    size_t col = j - (j / (r + 1) + 1);

                    int value = m_cells[m_N * row + col].value();
                    if (value == 0)
                        std::cout << " ";
                    else
                        std::cout << value;
                }

                std::cout << " ";
            }
        }

        std::cout << std::endl;
    }
}

void Sudoku::print(int *field, int recursionDepth)
{
    size_t r = sqrt(m_N);
    for (size_t i = 0; i < m_N + r + 1; i++)
    {
        for (int tab = 0; tab < recursionDepth; tab++)
        {
            std::cout << " | ";
        }

        for (size_t j = 0; j < m_N + r + 1; j++)
        {
            if (i % (r + 1) == 0)
                std::cout << "--";
            else
            {
                if (j % (r + 1) == 0)
                    std::cout << "|";
                else
                {
                    size_t row = i - (i / (r + 1) + 1);
                    size_t col = j - (j / (r + 1) + 1);

                    int value = field[m_N * row + col];
                    if (value == 0)
                        std::cout << " ";
                    else
                        std::cout << value;
                }

                std::cout << " ";
            }
        }

        std::cout << std::endl;
    }
}
