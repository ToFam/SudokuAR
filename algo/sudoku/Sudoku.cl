
bool possible(uint row, uint col, __local uchar* possibilities, const int value, const uint N, const uint cellSize)
{
    uint byte = (value - 1) / 8;

    int index = (row * N + col) * cellSize + byte;
    uchar cell = possibilities[index];

    uchar res = cell & (0x1 << ((value - 1) - byte * 8));

    return res > 0;
}

void initializeCell(uint index, __local uchar* possibilities, const uint N, const uint cellSize)
{
    for (int byte = 0; byte < cellSize; byte++)
    {
        uint byteFlag = 0;
        for (int bit = 0; bit < 8; bit++)
        {
            if (bit + byte * 8 < N)
                byteFlag |= (0x1 << bit);
        }

        possibilities[index * cellSize + byte] = byteFlag;
    }
}

void disable(uint row, uint col, __local uchar* possibilities, const int value, const uint N, const uint cellSize)
{
    uint byte = (value - 1) / 8;

    int index = (row * N + col) * cellSize + byte;
    uchar cell = possibilities[index];

    cell = cell & ~(0x1 << ((value - 1) - byte * 8));

    possibilities[index] = cell;
}

bool solved(__local int* field, __local int* flags, const uint N)
{
    int col = get_global_id(0); // assume N
    int row = get_global_id(1); // assume N

    if (row == 0 && col == 0)
    {
        flags[0] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (row < N && col < N)
    {
        if (field[row * N + col] < 1)
        {
            atomic_inc(&flags[0]);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    return flags[0] == 0;
}

bool valid(__local int* field, __local int* flags, const uint N)
{
    int col = get_global_id(0); // assume N
    int row = get_global_id(1); // assume N

    if (row == 0 && col == 0)
    {
        flags[1] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (row < N && col < N)
    {
        if (field[row * N + col] < 0)
        {
            atomic_inc(&flags[1]);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    return flags[1] == 0;
}

void updatePossibilities(__local int* field, __local uchar* possibilities, const uint N, const uint cellSize, const uint rootN)
{
    int col = get_global_id(0); // assume N
    int row = get_global_id(1); // assume N

    // Disable Horizontal Sweep
    for (int c = 0; c < N; ++c)
    {
        int value = field[row * N + c]; // Get value in colum c
        if (value != -1 && col != c)
        {
            disable(row, col, possibilities, value, N, cellSize); // Disable in every cell in this row
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Disable Vertical Sweep
    for (int r = 0; r < N; ++r)
    {
        int value = field[r * N + col]; // Get value in row r
        if (value != -1 && row != r)
        {
            disable(row, col, possibilities, value, N, cellSize); // Disable in every cell in this row
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Disable Block sweep
    for (int numInBlock = 0; numInBlock < N; ++numInBlock)
    {
        int block = row;

        int blockY = block / rootN;
        int blockX = block - blockY * rootN;

        int rowInBlockSrc = numInBlock / rootN;
        int colInBlockSrc = numInBlock - rowInBlockSrc * rootN;

        int srcX = blockX * rootN + colInBlockSrc;
        int srcY = blockY * rootN + rowInBlockSrc;

        int src = field[srcY * N + srcX];

        int rowInBlockTarget = col / rootN;
        int colInBlockTarget = col - rowInBlockTarget * rootN;

        int dstX = blockX * rootN + colInBlockTarget;
        int dstY = blockY * rootN + rowInBlockTarget;

        if ((dstX != srcX) || (dstY != srcY))
        {
            disable(dstY, dstX, possibilities, src, N, cellSize);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void Sudoku(__global int* array, __global uchar* outDebugArray, __global int* outFlags, __local int* flags, __local int* field, __local uchar* possibilities, const uint N, const uint cellSize)
{
    // solved, valid, changesThisStep, overallChanges
    //__local int flags[4];

    uint rootN = sqrt((float)N);

    int col = get_global_id(0); // assume N
    int row = get_global_id(1); // assume N

    int groupSize = get_local_size(0) * get_local_size(1);

    uint posSize = N * N * cellSize;

    // init flags
    if (row == 0 && col < 4)
    {
        flags[col] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Init possibilities array
    int index = row * N + col;
    int passes = posSize / groupSize;
    if (posSize % groupSize > 0)
        passes++;

    for (int i = 0; i < passes; ++i)
    {
        int outIndex = index + groupSize * i;

        if (outIndex < posSize)
        {
            initializeCell(index, possibilities, N, cellSize);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Load Field from global mem
    if (row < N && col < N)
    {
        int value = array[row * N + col];
        if (value == -1)
            value = 0;
        field[row * N + col] = value;
    }

    // Solve loop
    while (true)
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        // Update possibilities array
        updatePossibilities(field, possibilities, N, cellSize, rootN);

        // reset flags (not overall changes counter)
        if (row == 0 && col < 3)
        {
            flags[col] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Single solve step
        if (row < N && col < N)
        {
            int maybeSolution = -2;

            int existingValue = field[row * N + col];

            bool invalid = true;
            for (int i = 0; i < N; i++)
            {
                if (possible(row, col, possibilities, i + 1, N, cellSize))
                {
                    invalid = false;

                    if (maybeSolution == -2)
                    {
                        maybeSolution = i + 1;
                    }
                    else
                    {
                        // more than one number possible
                        maybeSolution = -1;
                    }
                }
                else
                {
                    // i is not possible
                    if (existingValue == i + 1)
                    {
                        invalid = true;
                        break;
                    }
                }
            }


            if (invalid)
            {
                // no possible solutions left
                // or
                // value is set but not possible => error in previos step
                field[row * N + col] = -1;
                atomic_inc(&flags[1]); // set invalid flag
            }

            if (maybeSolution > 0)
            {
                // only value possible

                if (existingValue == 0)
                {
                    // new number found!
                    field[row * N + col] = maybeSolution;
                    atomic_inc(&flags[2]); // increase changes this step
                    atomic_inc(&flags[3]); // increase changes over all
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (solved(field, flags, N))
        {
            break;
        }

        if (flags[1] > 0) // invalid
        {
            break;
        }

        if (flags[2] == 0) // no changes
        {
            break;
        }

    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Write Field
    array[row * N + col] = field[row * N + col];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Write flags
    if (row == 0 && col < 2)
    {
        outFlags[col] = flags[col]; // solved, valid
    }
    if (row == 0 && col == 3)
    {
        outFlags[2] = flags[3]; // overall changes
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Write Debug (possibilities array)
    for (int i = 0; i < passes; ++i)
    {
        int outIndex = index + groupSize * i;

        if (outIndex < posSize)
        {
            outDebugArray[outIndex] = possibilities[outIndex];
        }
    }
}
