#include <stdio.h>
#include <sys/time.h>
// #include <demo_util.h>
// #include <cuda_util.h>


#define CLOCK_RATE 1076000     // Titan

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return (double) tp.tv_sec + (double)tp.tv_usec*1e-6;
}

__device__ void sleep(float t)
{
    clock_t t0 = clock64();
    clock_t t1 = t0;
    while ((t1 - t0)/(CLOCK_RATE*1000.0f) < t)
    {
        t1 = clock64();
    }
}

__global__ void worker() 
{
    sleep(1.0);
}

int main(int argc, char** argv) 
{
    cudaDeviceProp  prop;
    clock_t clock_rate;
    int mp;


    double etime, start;

    cudaGetDeviceProperties(&prop, 0); /* Only look at first processor */
    printf("Name:  %s\n", prop.name );

    mp = prop.multiProcessorCount;
    clock_rate = prop.clockRate;
    printf("Clock rate = %d\n",clock_rate);

    int threads_per_block = 16;
    int blocks_per_sm = 1;    

    dim3 block(threads_per_block);
    dim3 grid(mp*blocks_per_sm); 

    start = cpuSecond();
    worker<<<grid,block>>>();
    cudaDeviceSynchronize();
    etime = cpuSecond() - start;

    int total_threads = block.x*grid.x;
    printf("Device has %d SMs\n",mp);
    printf("%27s %12d\n", "Threads per block",block.x*block.y);
    printf("%27s %12d\n", "Total number of blocks",grid.x);
    printf("%27s %12d\n", "Total number of threads",total_threads);
    printf("%27s %12.3f (s)\n","GPU Kernel Time (scaled)", etime);
    printf("\n");

    cudaDeviceReset();

}


