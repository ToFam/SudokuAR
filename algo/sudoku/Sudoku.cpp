#include "Sudoku.h"

#include <cmath>
#include <assert.h>
#include <iostream>
#include <thread>
#include <vector>
#include <numeric>

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

    bool valid() const { return m_value > -1; }
    bool solved() const { return m_value > 0; }

    /**
     * @brief setValue set cell to \a value if possible
     * @return false if \a value is not in list of possiblities
     */
    bool setValue(int value);
    void disable(int value);

    bool possible(int value) const;

    size_t possibilities() const;

    operator int() const { return value(); }

    /**
     * @brief solve write value if only one possibility left
     * @return true if value was set or marked as invalid, false if more than one possibility remains
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

    /**
     * @brief setValue set cell with coords (\a col, \a row) to \a value if possible
     * @returns false if \a value is not in list of possiblities for cell
     */
    bool setValue(size_t row, size_t col, int value);
    bool possible(size_t row, size_t col, int value) const;
    int value(size_t row, size_t col) const;

    /**
     * @brief solveStep solve cells with only one possibility left
     * @return if field can still be valid and number of changes in this step
     */
    std::tuple<bool, int> solveStep();

    void print(int recursionDepth) const;

    bool valid() const;
    bool solved() const;
    size_t numSolvedCells() const;

    /**
     * @return how many cells currently solved with \a value
     */
    size_t valueCount(int value) const;
    /**
     * @return number with the most already solved cells but not completely solved
     */
    size_t mostlySolvedNumber() const;

    /**
     * @brief mostSolvedCell finds yet unsolved cell with fewest remaining possiblities
     * @param outRow row coord of cell
     * @param outCol col coord of cell
     * @return how many remaining possibilties
     */
    size_t mostSolvedCell(size_t& outRow, size_t& outCol) const;

private:
    Cell& cell(size_t row, size_t col) const;

private:
    size_t m_N;
    std::vector<Cell> m_cells;

    std::vector<std::vector<int>> m_rows;
    std::vector<std::vector<int>> m_cols;
    std::vector<std::vector<int>> m_blocks;
};

void printIndent(int recursionDepth);

Sudoku::Sudoku(size_t N) : Algorithm("Sudoku"), m_N(N),
    m_dArray(nullptr), m_Program(nullptr), m_SolverKernel(nullptr), m_logLevel(1)
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

    std::cout << "Start solver" << std::endl;

    std::vector<double> runtimes;

    auto input = m_arguments[0];
    for (size_t s = 0; s < input->size(); ++s)
    {
        auto sudoku = input->get(s);

        if (sudoku->empty())
        {
            continue;
        }

        m_hArray = std::vector<int>(*sudoku);
        if (m_hArray.size() != m_N * m_N)
        {
            throw std::runtime_error("Invalid input size");
        }

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
    if (!runtimes.empty())
    {
        m_runtime /= runtimes.size();
    }

    std::cout << "solver finished" << std::endl;

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
    bool valid = false;
    for (size_t row = 0; row < m_N; ++row)
    {
        for (size_t col = 0; col < m_N; ++col)
        {
            int val = m_hArray[row * m_N + col];
            if (val > 0)
            {
                valid |= f.setValue(row, col, val);
            }
        }
    }

    if (!valid)
    {
        std::cout << "Invalid input" << std::endl;
        f.print(0);
        return;
    }

    if (f.numSolvedCells() < 17)
    {
        std::cout << "Skip grid with too few values (" << f.numSolvedCells() << ")" << std::endl;
        return;
    }

    if (m_logLevel > 0)
        f.print(0);

    if (!solveStep(f, 0, fResult))
    {
        std::cout << "Could not solve" << std::endl;
        return;
    }
    else
    {
        std::cout << "Solved" << std::endl;
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
        bool valid = true;
        int singleStepSolved, acc = 0, steps = 0;
        do
        {
            std::tie(valid, singleStepSolved) = f.solveStep();
            acc += singleStepSolved;
            ++steps;
        } while (valid && singleStepSolved > 0);

        if (m_logLevel > recursionDepth)
        {
            printIndent(recursionDepth);
            std::cout << steps << " forced step(s) solved " << acc << " cells" << std::endl;
            f.print(recursionDepth);
        }

        if (f.solved())
        {
            outResult = f;
            return true;
        }

        if (!valid)
            return false;

        size_t row, col; int val;
        bool success;
        success = solveTrial(f, row, col, val, recursionDepth + 1, outResult);

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

        if (m_logLevel > recursionDepth)
        {
            printIndent(recursionDepth);
            std::cout << "single step solve: " << m_hResultGPUFlags[2] << std::endl;
            print(m_hResultGPU.data(), recursionDepth);
        }

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
        bool success = solveTrial(f, row, col, val, recursionDepth + 1, outResult);
        return success;
    }
    else
    {
        return false;
    }
}

bool Sudoku::solveTrial(Field &f, size_t &outRow, size_t &outCol, int &outValue, int recursionDepth, Field& outResult)
{
    f.mostSolvedCell(outRow, outCol);

    for (size_t p = 1; p <= m_N; ++p)
    {
        if (f.possible(outRow, outCol, p))
        {
            outValue = p;

            if (m_logLevel >= recursionDepth)
            {
                printIndent(recursionDepth);
                std::cout << std::format("trial solve: ({}|{}) = {}", outCol, outRow, outValue) << std::endl;
            }

            Field f2 = f;
            f2.setValue(outRow, outCol, outValue);

            if (m_logLevel > recursionDepth)
            {
                f2.print(recursionDepth);
            }

            if (solveStep(f2, recursionDepth, outResult))
            {
                return true;
            }
        }
    }

    if (m_logLevel >= recursionDepth)
    {
        printIndent(recursionDepth);
        std::cout << std::format("Trial solve for ({}|{}) failed", outCol, outRow) << std::endl;
    }
    return false;
}

bool Sudoku::solveTrialSplit(Field &f, size_t &outRow, size_t &outCol, int &outValue, int recursionDepth, Field& outResult)
{
    for (size_t i = 0; i < m_N * m_N; i++)
    {
        size_t row = i / m_N;
        size_t col = i % m_N;
        if (f.value(row, col) == 0)
        {
            std::vector<Field> solves(m_N, Field(m_N));
            std::vector<std::thread> runners;
            for (size_t p = 1; p <= m_N; ++p)
            {
                if (f.possible(row, col, p) == true)
                {
                    if (m_logLevel >= recursionDepth)
                    {
                        for (int tab = 0; tab < recursionDepth - 1; tab++)
                        {
                            std::cout << " | ";
                        }

                        std::cout << "trial solve: (" << col << "|" << row << ") = " << p << std::endl;
                    }

                    runners.push_back(std::thread([=, this, &solves]{
                        Field f2 = f;
                        f2.setValue(row, col, p);

                        if (solveStep(f2, recursionDepth, solves[p-1]))
                        {
                            m_solved = true;
                        }
                    }));
                }
            }

            for (auto& r : runners)
            {
                r.join();
            }

            m_solved = false;

            outValue = 1;
            for (auto& s : solves)
            {
                if (s.solved())
                {
                    outRow = row;
                    outCol = col;
                    outResult = s;
                    return true;
                }
                ++outValue;
            }

            return false;
        }
    }

    assert(false);
    return f.solved();
}

Cell::Cell(size_t N) : m_N(N)
{
    m_value = 0;
    m_possible = std::vector<bool>(m_N, true);
}

void Cell::disable(int value)
{
    m_possible[value - 1] = false;
}

bool Cell::possible(int value) const
{
    assert(value > 0 && value <= m_N);

    if (!valid())
    {
        return false;
    }

    return m_possible[value - 1];
}

size_t Cell::possibilities() const
{
    return std::count(m_possible.begin(), m_possible.end(), true);
}

bool Cell::setValue(int value)
{
    if (!m_possible[value - 1])
    {
        return false;
    }

    for (size_t i = 0; i < m_N; i++)
        m_possible[i] = false;
    m_value = value;
    return true;
}

int Cell::value() const
{
    return m_value;
}

bool Cell::solve()
{
    int maybeSolution = -1;
    bool invalid = true;
    for (size_t i = 0; i < m_N; i++)
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
        return true;
    }
    else if (maybeSolution == -1)
    {
        // undecided
        m_value = 0;
        return false;
    }
    else
    {
        // only value possible
        m_value = maybeSolution + 1;
        return true;
    }
}

Field::Field(size_t N) : m_N(N), m_rows(m_N), m_cols(m_N), m_blocks(m_N)
{
    for (size_t i = 0; i < m_N * m_N; ++i)
    {
        m_cells.emplace_back(m_N);
    }

    for (size_t i = 0; i < m_N; ++i)
    {
        for (size_t j = 0; j < m_N; ++j)
        {
            m_rows[i].push_back(m_N * i + j);
            m_cols[i].push_back(m_N * j + i);
        }
    }

    size_t r = sqrt(m_N);
    for (size_t block = 0; block < m_N; ++block)
    {
        size_t brow = block / r;
        size_t bcol = block % r;
        for (size_t row = r * brow; row < r * (brow + 1); ++row)
        {
            for (size_t col = r * bcol; col < r * (bcol + 1); ++col)
            {
                m_blocks[block].push_back(m_N * row + col);
            }
        }
    }
}

bool Field::setValue(size_t row, size_t col, int value)
{
    assert(row >= 0 && row < m_N && col >= 0 && col < m_N);
    bool valid = m_cells[m_N * row + col].setValue(value);
    if (!valid)
    {
        return false;
    }

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

    return true;
}

Cell& Field::cell(size_t row, size_t col) const
{
    assert(row >= 0 && row < m_N && col >= 0 && col < m_N);

    return const_cast<Cell&>(m_cells[m_N * row + col]);
}

int Field::value(size_t row, size_t col) const
{
    return cell(row, col).value();
}

bool Field::possible(size_t row, size_t col, int value) const
{
    return cell(row, col).possible(value);
}

std::tuple<bool, int> Field::solveStep()
{
    int changes = 0;
    for (size_t i = 0; i < m_N * m_N; i++)
    {
        Cell& cell = m_cells[i];
        if (cell.solved())
            continue;

        if (cell.solve())
        {
            ++changes;
            if (!cell.valid())
            {
                return {false, changes};
            }

            setValue(i / m_N, i % m_N, cell.value());
        }
    }

    auto solveBlock = [&](int value, auto&& block)
    {
        size_t numCellsWherePossible = 0;
        int indexPossible = 0;
        for (const auto& i : block)
        {
            Cell& c = m_cells[i];
            if (!c.solved() && c.possible(value))
            {
                ++numCellsWherePossible;
                indexPossible = i;
            }
        }
        if (numCellsWherePossible == 1)
        {
            setValue(indexPossible / m_N, indexPossible % m_N, value);
            ++changes;
        }
    };

    for (const auto& row : m_rows)
    {
        for (size_t value = 1; value <= m_N; ++value)
        {
            solveBlock(value, row);
        }
    }
    for (const auto& col : m_cols)
    {
        for (size_t value = 1; value <= m_N; ++value)
        {
            solveBlock(value, col);
        }
    }
    for (const auto& block : m_blocks)
    {
        for (size_t value = 1; value <= m_N; ++value)
        {
            solveBlock(value, block);
        }
    }

    return {true, changes};
}

bool Field::solved() const
{
    return std::ranges::all_of(m_cells, [](const Cell& cell){ return cell.solved(); });
}

bool Field::valid() const
{
    return std::ranges::all_of(m_cells, [](const Cell& cell) { return cell.valid(); });
}

size_t Field::numSolvedCells() const
{
    return std::accumulate(m_cells.begin(), m_cells.end(),
                           0, [](size_t filled, const Cell& cell){ return filled + (cell.solved() ? 1 : 0); });
}

size_t Field::valueCount(int value) const
{
    return std::accumulate(m_cells.begin(), m_cells.end(),
                           0, [value](size_t count, const Cell& cell){ return count + (cell.value() == value ? 1 : 0); });
}

size_t Field::mostlySolvedNumber() const
{
    std::map<size_t, int, std::greater<size_t>> solveCountPerValueDesc;
    for (size_t p = 1; p <= m_N; ++p)
    {
        size_t count = valueCount(p);
        if (count < m_N)
        {
            solveCountPerValueDesc[count] = p;
        }
    }

    return solveCountPerValueDesc.empty() ? 0 : solveCountPerValueDesc.begin()->second;
}

size_t Field::mostSolvedCell(size_t& outRow, size_t& outCol) const
{
    size_t minIndex = 0, value = m_N;
    for (size_t i = 0; i < m_cells.size(); ++i)
    {
        auto p = m_cells[i].possibilities();
        if (p > 0 && p < value)
        {
            value = p;
            minIndex = i;
        }
    }

    outRow = minIndex / m_N;
    outCol = minIndex % m_N;
    return value;
}

void printIndent(int recursionDepth)
{
    for (int tab = 0; tab < recursionDepth; tab++)
    {
        std::cout << " | ";
    }
}

template<typename F>
void print(int N, F &field, int recursionDepth)
{
    size_t r = sqrt(N);
    for (size_t i = 0; i < N + r + 1; i++)
    {
        printIndent(recursionDepth);
        for (size_t j = 0; j < N + r + 1; j++)
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

                    int value = field[N * row + col];
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

void Field::print(int recursionDepth) const
{
    ::print(m_N, m_cells, recursionDepth);
}

void Sudoku::print(int *field, int recursionDepth)
{
    ::print(m_N, field, recursionDepth);
}
