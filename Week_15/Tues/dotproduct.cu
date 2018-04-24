#include <stdio.h>
#include <sys/time.h>
#include <demo_util.h>
#include <cuda_util.h>

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return (double) tp.tv_sec + (double)tp.tv_usec*1e-6;
}


__global__ void cuda_dot(int N, double *a, double *b, double *c) 
{
    // __shared__ double localDot[threadsPerBlock];  /* Statically defined */
    extern __shared__ double localDot[];
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int localIndex = threadIdx.x;

    double localSum = 0;
    while (ix < N) {
        localSum += a[ix] * b[ix];
        ix += blockDim.x * gridDim.x;
    }
    
    /* Store sum computed by this thread */
    localDot[localIndex] = localSum;
    
    /* Wait for all threads to get to this point */
    __syncthreads();

    /* Every block should add up sum computed on  
       threads in the block */
    int i = blockDim.x/2;
    while (i != 0) 
    {
        if (localIndex < i)
            localDot[localIndex] += localDot[localIndex + i];
        __syncthreads();
        i /= 2;
    }

    /* Each block stores local dot product */
    if (localIndex == 0)
        c[blockIdx.x] = localDot[0];
}

double dot_gpu(int N, double *a, double *b)
{
    double   dot, *partial_c;
    double   *dev_a, *dev_b, *dev_partial_c;
    double start, etime;

    const int threadsPerBlock = 32;
    const int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;

    partial_c = (double*) malloc( blocksPerGrid*sizeof(double) );

    /* allocate the memory on the GPU */
    start = cpuSecond();
    CHECK(cudaMalloc((void**) &dev_a, N*sizeof(double)));
    CHECK(cudaMalloc((void**) &dev_b, N*sizeof(double)));
    CHECK(cudaMalloc((void**) &dev_partial_c, blocksPerGrid*sizeof(double) ) );
    etime = cpuSecond() - start;
    printf("%20s %12.4e\n","cudaMalloc",etime);

    /* copy the arrays 'a' and 'b' to the GPU */
    start = cpuSecond();
    CHECK(cudaMemcpy(dev_a, a, N*sizeof(double),
                              cudaMemcpyHostToDevice ) );
    CHECK(cudaMemcpy(dev_b, b, N*sizeof(double),
                              cudaMemcpyHostToDevice ) ); 
    etime = cpuSecond() - start;
    printf("%20s %12.4e\n","cudaMemcpy",etime);


    dim3 block(threadsPerBlock);  /* Values defined in macros */
    dim3 grid(blocksPerGrid);     /* defined in macros, above */
    start = cpuSecond();
    cuda_dot<<<grid,block,threadsPerBlock*sizeof(double)>>>(N, dev_a, dev_b, 
                                                            dev_partial_c );
    etime = cpuSecond() - start;
    printf("%20s %12.4e\n","cuda_dot",etime);


    /* copy the array 'c' back from the GPU to the CPU */
    CHECK( cudaMemcpy( partial_c, dev_partial_c,
                      blocksPerGrid*sizeof(double),
                      cudaMemcpyDeviceToHost ) );

    /* Sum of block sums */
    dot = 0;
    for (int i = 0; i < blocksPerGrid; i++) 
    {
        dot += partial_c[i];
    }

    /* free memory on the gpu side */
    CHECK(cudaFree(dev_a));
    CHECK(cudaFree(dev_b));
    CHECK(cudaFree(dev_partial_c));

    free(partial_c);

    return dot;
}

double dot_cpu(int n, double *a, double *b)
{
    double sum = 0;
    int i;

    for (i = 0; i < n; i++)
    {
        sum += a[i]*b[i];
    }
    return sum;
}

/* Compute a dot product */
int main( void ) 
{
    double   *a, *b;
    double c_gpu, c_cpu;
    int N;
    double etime, etime1, start;

    N = 1 << 14;

    /* CPU */
    printf("\n");
    printf("%20s\n","CPU");

    start = cpuSecond();
    a = (double*) malloc( N*sizeof(double) );
    b = (double*) malloc( N*sizeof(double) );
    etime1 = cpuSecond() - start;
    printf("%20s %12.4e\n","malloc", etime1);


    /* Define vectors a and b */
    for (int i = 0; i < N; i++) 
    {
        a[i] = i;
        b[i] = i;
    }

    start = cpuSecond();
    c_cpu = dot_cpu(N,a,b);
    etime = cpuSecond() - start;
    printf("%20s %12.4e\n","dot_cpu", etime);
    printf("%20s %12.4e\n","Total CPU (s)", etime + etime1);
    printf("\n");

    /* GPU */
    printf("%20s\n","GPU");
    start = cpuSecond();
    c_gpu = dot_gpu(N,a,b);
    etime = cpuSecond() - start;
    printf("%20s %12.4e\n","Total GPU (s)", etime);

    /* Check result */
    printf("\n");
    #define sum_squares(x)  (x*(x+1)*(2*x+1)/6)
    double s = sum_squares((double)(N-1));

    // double s = N;   /* Sum of 1s */
    printf("%20s %10f\n","Dot product (CPU)", c_cpu);
    printf("%20s %10f\n","Dot product (GPU)", c_gpu);
    printf("%20s %10f\n","True dot product", s);

    free(a);
    free(b);
}
