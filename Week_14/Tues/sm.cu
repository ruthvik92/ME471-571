#include <stdio.h>
#include <sys/time.h>
#include <demo_util.h>
#include <cuda_util.h>


//#define CLOCK_RATE 1080000     // on Tesla
#define CLOCK_RATE 1124000     // On Kepler

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return (double) tp.tv_sec + (double)tp.tv_usec*1e-6;
}

__device__ uint get_smid(void) {

     uint ret;
     asm("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
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

__global__ void worker(float *t,uint *s) 
{
    int id = blockIdx.x;
    s[id] = get_smid();
    sleep(t[id]);
    // __syncthreads();
}

#define blocks_per_MP 32     /* for Kepler */

int main(int argc, char** argv) 
{
    cudaDeviceProp  prop;
    clock_t clock_rate;

    uint  *sm_id, *dev_sm_id;
    float *t, *dev_t; 
    int *blocks_per_SM;
    int i, mp, N, M;
    double etime, start;
    double scale_factor;

    cudaGetDeviceProperties(&prop, 0); /* Only look at first processor */

    mp = prop.multiProcessorCount;
    clock_rate = prop.clockRate;
    printf("clock rate = %d\n",clock_rate);

    /* Read in number of blocks to launch */
    int err; 
    read_int(argc, argv, "--grid", &N, &err);
    if (err > 0)
    {
        printf("Grid dimension (grid.x) set to default value 64\n");
        N = 64;
    }

    read_int(argc, argv, "--block", &M, &err);
    if (err > 0)
    {
        printf("Block dimension (block.x) set to default value 1\n");
        M = 1;
    }

    read_double(argc, argv, "--scale", &scale_factor, &err);
    if (err > 0)
    {
        scale_factor = 1.0;
        printf("Scale factor set to %g\n",scale_factor);
    }

    t = (float*) malloc(N*sizeof(float));
    sm_id = (uint*) malloc(N*sizeof(uint));

    /* Allocate memory on the device */
    cudaMalloc( (void**)&dev_t, N*sizeof(float));
    cudaMalloc( (void**)&dev_sm_id, N*sizeof(uint));

    printf("Memory requirement : %0.2f (kB)\n",N*(sizeof(float) + sizeof(uint))/(1024));

    // scale_factor = 100.0;

    /* thread work */
    for(i = 0; i < N; i++)
        t[i] = 1.0/scale_factor;

    cudaMemcpy(dev_t, t, N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(M);
    dim3 grid(N);  /* N blocks */

    start = cpuSecond();
    worker<<<grid,block>>>(dev_t, dev_sm_id);
    CHECK(cudaDeviceSynchronize());
    etime = cpuSecond() - start;

    CHECK(cudaPeekAtLastError());


    /* Copy contents of dev_t back to t */
    cudaMemcpy(sm_id, dev_sm_id, N*sizeof(uint), cudaMemcpyDeviceToHost);

    /* Post process data */
    blocks_per_SM = (int*) malloc(mp*sizeof(float));
    printf("Device has %d SMs\n",mp);

    for(i = 0; i < mp; i++)
        blocks_per_SM[i] = 0;

    printf("Distribution of blocks on SMs\n");
    printf("------------------------------------------------------------------------------\n");
    int prt = N <= (1 << 11);
    int j, k = 0;
    for(i = 0; i < (N+mp-1)/mp; i++)
    {
        if (prt && i % blocks_per_MP == 0)
        {
            printf("\n");
        }
        for(j = 0; j < mp; j++)
        {
            if (prt)
            {
                printf("(%3d,%2d)  ",k,sm_id[k]);                
            }
            blocks_per_SM[sm_id[k]] += 1;
            k++;            
            if (k == N)
                break;
        }       
        if (prt)
        {
            printf("\n");
        }
    }
    printf("------------------------------------------------------------------------------\n");
    printf("\n");
    printf("Blocks per SM\n");
    printf("---------------------\n");
    for(i = 0; i < mp; i++)
    {
        printf("SM[%2d] = %6d\n",i,blocks_per_SM[i]);
    }
    printf("---------------------\n");
    printf("\n");
    int total_threads = block.x*grid.x;
    printf("%27s %12d\n", "Threads per block",block.x*block.y);
    printf("%27s %12d\n", "Total number of blocks",grid.x);
    printf("%27s %12d\n", "Total number of threads",total_threads);
    printf("%27s %12g\n", "Time scaling factor",scale_factor);
    printf("%27s %12.3f (s)\n","GPU Kernel Time (scaled)", scale_factor*etime);
    printf("\n");

    cudaFree(dev_t);
    cudaFree(dev_sm_id);

    free(t);
    free(sm_id);
    free(blocks_per_SM);

    cudaDeviceReset();

}


