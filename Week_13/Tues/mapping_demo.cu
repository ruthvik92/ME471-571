#include <stdio.h>
#include <sys/time.h>
#include <demo_util.h>
#include <cuda_util.h>

__global__ void addmat(int m, int n, int *A, int *B, int *C)
{
    /* Each thread processes one element */
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy * m + ix;
    if (ix < m && iy < n)
        C[idx] = A[idx] + B[idx];
}

void addmat_host(int m, int n, int *A, int *B, int *C)
{
    int ix, iy, idx;
    for (iy = 0; iy < n; iy++)
        for (ix = 0; ix < m; ix++) // try putting this loop outside (will result in strided reading from memory and the cache misses will be rampant)
        {
            idx = iy * m + ix;
            C[idx] = A[idx] + B[idx];
        }
}

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1e-6;
}

int main(int argc, char **argv)
{
    /* Host */
    int *A, *B, *C;

    /* Device */
    int *dev_A, *dev_B, *dev_C;

    /* scalars */
    size_t m, n, nbytes;
    double etime, start, total_device;
    total_device = 0.0;

    int run_host;
    int dimx, dimy;

    int err0;
    read_int(argc, argv, "--host", &run_host, &err0);
    if (err0 > 0)
        run_host = 0;

    int err1, err2;
    read_int(argc, argv, "--dimx", &dimx, &err1);
    read_int(argc, argv, "--dimy", &dimy, &err2);

    if (err1 > 0 || err2 > 0)
    {
        printf("Problem reading dimx or dimy\n");
        exit(0);
    }

    /* Matrix is m x n */
    m = 1 << 14;
    n = 1 << 14;

    nbytes = m * n * sizeof(int);

    printf("--------------- Memory -------------\n");
    printf("%20s %10d\n", "m", m);
    printf("%20s %10d\n", "n", n);
    printf("%20s %10.1f (mb)\n", "Memory", 3 * nbytes / (1024.0 * 1024.0));
    printf("%20s %10.1f (gb)\n", "Memory", 3 * nbytes / (1024.0 * 1024.0 * 1024.0));
    printf("\n");

    A = (int *)malloc(nbytes);
    B = (int *)malloc(nbytes);
    C = (int *)malloc(nbytes);

    int k;
    for (k = 0; k < m * n; k++)
    {
        A[k] = 1;
        B[k] = 2;
    }

    /* Allocate memory on the device */
    printf("----------- Initialization ---------\n");
    start = cpuSecond();
    cudaMalloc((void **)&dev_A, nbytes);
    cudaMalloc((void **)&dev_B, nbytes);
    cudaMalloc((void **)&dev_C, nbytes);
    etime = cpuSecond() - start;
    printf("%20s %10.3g (s)\n", "cudaMalloc", etime);
    total_device += etime;

    start = cpuSecond();
    cudaMemcpy(dev_A, A, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, nbytes, cudaMemcpyHostToDevice);
    etime = cpuSecond() - start;
    printf("%20s %10.3g (s)\n", "Copy (HtoD)", etime);
    total_device += etime;

    /* Launch nx blocks, each with ny threads */
    // Max threads per block :            1024
    // Max thread dimensions:  (1024, 1024, 64)
    // Max grid dimensions  :  2147483647, 65535, 65535

    dim3 block(dimx, dimy); /* Distribute each row to blocks */
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    start = cpuSecond();
    addmat<<<grid, block>>>(m, n, dev_A, dev_B, dev_C);
    cudaDeviceSynchronize();
    etime = cpuSecond() - start;
    CHECK(cudaPeekAtLastError());
    printf("%20s %10.3g (s)\n", "GPU Kernel", etime);
    total_device += etime;

    /* Copy contents from device back to host */
    start = cpuSecond();
    cudaMemcpy(C, dev_C, nbytes, cudaMemcpyDeviceToHost);
    etime = cpuSecond() - start;
    printf("%20s %10.3g (s)\n", "Copy (DtoH)", etime);
    total_device += etime;
    printf("%20s %10.3g (s)\n", "Total device time", total_device);
    printf("\n");

    // printf("%20s %d %d\n", "block(dimx,dimy)", dimx, dimy);
    // printf("%30s %d %d\n", "(m + block.x - 1) / block.x, (n + block.y - 1) / block.y", (m + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    printf("---------------- Host --------------\n");
    if (run_host)
    {
        start = cpuSecond();
        addmat_host(m, n, A, B, C);
        etime = cpuSecond() - start;
        printf("%20s %10.3g (s)\n", "Host", etime);
    }
    else
    {
        printf("%20s %10s\n", "Host", "N/A");
    }

    printf("\n");
    printf("----------- Grid/Block Info ---------\n");
    printf("%20s %6d %6d", "grid.x, block.x", grid.x, block.x);
    if (m <= block.x * grid.x)
    {
        printf("        m <= block.x*grid.x = %d\n", block.x * grid.x);
    }
    else
    {
        printf("        m > block.x*grid.x = %d\n", block.x * grid.x);
    }
    printf("%20s %6d %6d", "grid.y, block.y", grid.y, block.y);
    if (n <= block.y * grid.y)
    {
        printf("        m <= block.y*grid.y = %d\n", block.y * grid.y);
    }
    else
    {
        printf("        m > block.y*grid.y = %d\n", block.y * grid.y);
    }
    if (dimx * dimy > 1024)
    {
        printf("\n");
        printf("-----> WARNING : Threads per block exceed 1024.\n");
        printf("\n");
    }

    printf("\n");

#if 0
    printf("\n");


    printf("\n");
    int i,j;
    k = 0;
    for(j = 0; j < n; j++)
    {
        for(i = 0; i < m; i++)
        {
            printf("C[%2d,%2d] = %d\n",i,j,C[k]);
            k++;
        }
        printf("\n");
    }
#endif

    cudaFree(dev_A);
    cudaFree(dev_B);
    free(A);
    free(B);

    cudaDeviceSynchronize();
    cudaDeviceReset();
}

//  --- General Information for device 0 ---
// Name:  NVIDIA RTX A5500 Laptop GPU
//
// Compute capability    :             8.6
// Clock rate            :            1.50 (GHz)
//
//   --- Memory Information for device 0 ---
// Total global mem      :            15.7 (gb)
//
//   --- MP Information for device 0 ---
// Multiprocessor count :              58
// Shared mem per mp     :            48.0 (kb)
// Registers per mp      :            64.0 (kb)
// Threads in warp       :              32
// Max threads per block :            1024
// Max thread dimensions:  (1024, 1024, 64)
// Max grid dimensions  :  2147483647, 65535, 65535
