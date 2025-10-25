
__kernel void ReduceBasic(__global uint* M, uint stride)
{
    int x = get_global_id(0);
    uint s = M[x]+M[x+stride];
    M[x]=s;
}

__kernel void ReduceDecomp(__global const uint* M, __global uint* N, __local uint* L)
{
    int x = get_local_id(0);
    int group = get_global_id(0) / get_local_size(0);
    int offset = group * get_local_size(0) * 2;

    L[x] = M[offset + x] + M[offset + get_local_size(0) + x];

    uint stride = get_local_size(0) / 2;
    while (stride >= 1)
    {
        if (x < stride)
        {
            L[x] = L[x] + L[x+stride];
        }

        stride /= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x == 0)
    {
        N[group] = L[0];
    }
}

__kernel void ReduceEABasic(__global uint* M, uint offsetRead, uint offsetWrite, uint stride)
{
    int x = get_global_id(0);
    uint s = M[offsetRead+x] + M[offsetRead+x+stride];
    M[offsetWrite+x] = s;
}

__kernel void ReduceEADecomp(__global uint* M, uint offsetRead, uint offsetWrite)
{
    int x = get_global_id(0);
    uint s = M[offsetRead+x*2] + M[offsetRead+x*2+1];
    M[offsetWrite+x] = s;
}


__kernel void PrefixSumBasic(__global const uint* M, __global uint* N, __local uint* L)
{
    int x = get_local_id(0);
    int w = get_local_size(0);
    int group = get_global_id(0) / w;
    int offset = group * w * 2;

    int w2 = w / 2;

    L[x] = M[offset + x];
    L[x+w] = M[offset + w + x];

    barrier(CLK_LOCAL_MEM_FENCE);

    uint stride = 1;
    uint lw = w;
    while (lw >= 1)
    {
        if (x < lw)
        {
            uint dx = (x+1)*(stride*2) - 1;
            L[dx] = L[dx-stride] + L[dx];
        }

        stride *= 2;
        lw /= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    lw = 1;
    stride = w;

    if (x == w - 1)
    {
        L[2*w-1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    while (lw <= w)
    {
        if (x < lw)
        {
            uint dx = (x+1)*(stride*2) - 1;
            uint left = L[dx - stride];
            uint right = L[dx];
            L[dx] = left + right;
            L[dx - stride] = right;
        }

        stride /= 2;
        lw *= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    N[offset + x] = L[x];
    N[offset + w + x] = L[w+x];
}

__kernel void PrefixSumAdd(__global uint* M, __global uint* groupsums, uint groupsize)
{
    uint groupIndex = get_global_id(0) / groupsize;
    atomic_add(&M[get_global_id(0)], groupsums[groupIndex]);
}

__kernel void IntegralConvInput(__global const uchar* img, __global uint* M, uint widthOrig, uint N_m)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    uint c = 0;
    if (x < widthOrig)
    {
        c = img[y * widthOrig + x];
    }

    M[y * N_m + x] = c;
}

__kernel void IntegralConvOutput(__global const uint* M, __global uint* img, uint width, uint height, uint N_m)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height)
    {
        img[y * width + x] = M[y * N_m + x];
    }
}

__kernel void Integral(__global const uint* M, __global uint* N, __local uint* L, uint n)
{
    int Y = get_global_id(0) / (n / 2);
    int X = get_global_id(0) - Y * (n / 2);
    int x = get_local_id(0);
    int w = get_local_size(0);
    int group = X / w;
    int offset = Y * n + group * w * 2;

    int w2 = w / 2;

    L[x] = M[offset + x];
    L[x+w] = M[offset + w + x];

    barrier(CLK_LOCAL_MEM_FENCE);

    uint stride = 1;
    uint lw = w;
    while (lw >= 1)
    {
        if (x < lw)
        {
            uint dx = (x+1)*(stride*2) - 1;
            L[dx] = L[dx-stride] + L[dx];
        }

        stride *= 2;
        lw /= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    lw = 1;
    stride = w;

    if (x == w - 1)
    {
        L[2*w-1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    while (lw <= w)
    {
        if (x < lw)
        {
            uint dx = (x+1)*(stride*2) - 1;
            uint left = L[dx - stride];
            uint right = L[dx];
            L[dx] = left + right;
            L[dx - stride] = right;
        }

        stride /= 2;
        lw *= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    N[offset + x] = L[x];
    N[offset + w + x] = L[w+x];
}

__kernel void IntegralGroupSums(__global const uint* M, __global const uint* sums, __global uint* groupsums, uint groupsize, uint n)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int groupcount = n / groupsize;

    if (x < groupcount)
    {
        int idx = y * n + x * groupsize + groupsize - 1;

        int sum = M[idx] + sums[idx];
        groupsums[y * groupsize + x] = sum;
    }
}

__kernel void IntegralAdd(__global uint* M, __global uint* groupsums, uint groupsize, uint n)
{
    uint Y = get_global_id(0) / n;
    uint X = get_global_id(0) - Y * n;
    uint groupIndex = X / groupsize;
    atomic_add(&M[Y * n + X], groupsums[Y * groupsize + groupIndex]);
}

__kernel void RotateCCW(__global const uint* img, __global uint* out, __local uint* L, uint n)
{
    int c = get_local_size(0);
    int chunks = n / c;

    int x = get_local_id(0);
    int y = get_local_id(1);

    L[y * c + x] = img[get_global_id(1) * n + get_global_id(0)];

    barrier(CLK_LOCAL_MEM_FENCE);

    /*
    int X = get_global_id(1);
    int Y = n - get_global_id(0) - 1;
    out[Y * n + X] = L[y * c + x];
    */

    int cx = get_global_id(0) / c;
    int cy = get_global_id(1) / c;

    int CY = chunks - cx - 1;
    int CX = cy;
    out[(CY * c + y) * n + CX * c + x] = L[x * c + (c - y - 1)];
}

__kernel void RotateCW(__global const uint* img, __global uint* out, __local uint* L, uint n)
{
    int c = get_local_size(0);
    int chunks = n / c;

    int x = get_local_id(0);
    int y = get_local_id(1);

    L[y * c + x] = img[get_global_id(1) * n + get_global_id(0)];

    barrier(CLK_LOCAL_MEM_FENCE);

    /*
    int Y = get_global_id(0);
    int X = n - get_global_id(1) - 1;
    out[Y * n + X] = L[y * c + x];
    */

    int cx = get_global_id(0) / c;
    int cy = get_global_id(1) / c;

    int CY = cx;
    int CX = chunks - cy - 1;
    out[(CY * c + y) * n + CX * c + x] = L[(c - x - 1) * c + y];
}

__kernel void MatchTemplate()
{

}

__kernel void MatchTemplateTiled(__global const uchar* gimage, __global const uchar* gtmplt, __global float* result,
                            __local uchar* L,
                            uint W, uint H,
                            uint w, uint h,
                            uint tileW, uint tileH, // >= local_size + w/h
                            uint Rw, uint Rh)
{
    int Rx = get_global_id(0);
    int Ry = get_global_id(1);

    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int lw = get_local_size(0);
    int lh = get_local_size(1);

    int offsetTileX = (Rx / lw) * lw;
    int offsetTileY = (Ry / lh) * lh;

    local uchar* ltmplt = L;
    local uchar* tile = L + w*h;

    // Load template into local mem
    int tmpltLoadCyclesX = (int)ceil((float)w / lw);
    int tmpltLoadCyclesY = (int)ceil((float)h / lh);

    for (int i = 0; i < tmpltLoadCyclesY; ++i)
    {
        int dy = i * lh + ly;
        if (dy < h)
        {
            for (int j = 0; j < tmpltLoadCyclesX; ++j)
            {
                int dx = j * lw + lx;
                if (dx < w)
                {
                    ltmplt[dy * w + dx] = gtmplt[dy * w + dx];
                }
            }
        }
    }

    // Load image crop into local mem
    int tileCyclesX = (int)ceil((float)tileW / lw);
    int tileCyclesY = (int)ceil((float)tileH / lh);

    for (int i = 0; i < tileCyclesY; ++i)
    {
        int dy = (i * lh) + ly;
        int dgy = offsetTileY + dy;
        if (dgy < H && dy < tileH)
        {
            for (int j = 0; j < tileCyclesX; ++j)
            {
                int dx = (j * lw) + lx;
                int dgx = offsetTileX + dx;
                if (dgx < W && dx < tileW)
                {
                    tile[dy * tileW + dx] = gimage[dgy * W + dgx];
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (Rx >= Rw || Ry >= Rh)
        return;

    // Calc CCORR
    float sum = 0.0;
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            //sum += x*y;
            sum += ltmplt[y * w + x] * tile[(ly + y) * tileW + lx + x]; // (Ry + y) * W + Rx + x
        }
    }

    result[Ry * Rw + Rx] = sum;

    /*
    uchar v = tile[ly * tileW + lx];
    float res = (float)v / 255;
    if (lx == 0 || ly == 0)
    {
        res = 1.0;
    }

    result[Ry * Rw + Rx] = res;
    */
}
