#include <stdio.h>
#include <sys/time.h>
#include <demo_util.h>
#include <cuda_util.h>


//#define CLOCK_RATE 1080000     // on Tesla
#define CLOCK_RATE 1124000     // On Kepler

#define PI 3.14159265358979323846264338327

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

__global__ void worker(int n, int m,float *img) 
{
    int ix,iy,idx;
    ix = threadIdx.x + blockIdx.x*blockDim.x;
    iy = threadIdx.y + blockIdx.y*blockDim.y;
    idx = ix + iy*n;
    float x = 3*((float)ix)/((float) blockDim.x*gridDim.x);
    float y = ((float)iy)/((float) blockDim.y*gridDim.y);
    //img[idx] = sin(2*PI*x)*cos(2*PI*y);
    img[idx] = idx;
}

#define blocks_per_MP 32     /* for Kepler */

int main(int argc, char** argv) 
{
    float *t, *dev_t; 
    int N, M;
    double scale_factor;
    double etime, start;

    /* Read in number of blocks to launch */
    int err; 
    read_int(argc, argv, "--N", &N, &err);
    if (err > 0)
    {
        printf("Grid dimension (grid.x) set to default value 64\n");
        N = 64;
    }

    read_int(argc, argv, "--M", &M, &err);
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

    /* Allocate memory on the device */
    cudaMalloc( (void**)&dev_t, M*N*sizeof(float));

    dim3 block(32,32);
    dim3 grid((N+block.x-1)/block.x,(M+block.y-1)/block.y);  /* N blocks */

    start = cpuSecond();
    worker<<<grid,block>>>(N,M,dev_t);
    CHECK(cudaDeviceSynchronize());
    etime = cpuSecond() - start;
    CHECK(cudaPeekAtLastError());

    printf("%20s %10.4e\n","GPU time",etime);

    t = (float*) malloc(N*M*sizeof(float));
    cudaMemcpy(t, dev_t, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    printf("Last idx = %d\n",(int) t[M*N-1]);


    cudaFree(dev_t);
    free(t);
    cudaDeviceReset();

}


