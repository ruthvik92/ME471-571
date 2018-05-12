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

const int N_def (1 << 14);
const int threadsPerBlock = 32;
const int blocksPerGrid = (N_def+threadsPerBlock-1) / threadsPerBlock;

int get_N()
{
    return N_def;
}

int get_blocksPerGrid()
{
    return blocksPerGrid;
}


__global__ void cuda_norm(int N, double *a, double *c) 
{
    __shared__ double localDot[threadsPerBlock];
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int localIndex = threadIdx.x;

    double localSum = 0;
    while (ix < N) 
    {
        localSum += a[ix] * a[ix];
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

double dot_norm(int N, double *a, double *dev_a,
               double *dev_partial_c)
{
    double   dot, *partial_c;

    partial_c = (double*) malloc( blocksPerGrid*sizeof(double) );

    /* copy the arrays 'a' and 'b' to the GPU */
    CHECK(cudaMemcpy(dev_a, a, N*sizeof(double),
                              cudaMemcpyHostToDevice ) );

    dim3 block(threadsPerBlock);  /* Values defined in macros */
    dim3 grid(blocksPerGrid);     /* defined in macros, above */
    cuda_norm<<<grid,block>>>(N, dev_a,                                                             dev_partial_c );
    // cudaDeviceSynchronize();
    CHECK(cudaPeekAtLastError());


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

    free(partial_c);

    return dot;
}


__global__ void cuda_dot(int N, double *a, double *b, double *c) 
{
    __shared__ double localDot[threadsPerBlock];
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

double dot_gpu(int N, double *a, double *b,
               double *dev_a, double *dev_b, 
               double *dev_partial_c)
{
    double   dot, *partial_c;

    partial_c = (double*) malloc( blocksPerGrid*sizeof(double) );

    /* copy the arrays 'a' and 'b' to the GPU */
    CHECK(cudaMemcpy(dev_a, a, N*sizeof(double),
                              cudaMemcpyHostToDevice ) );
    CHECK(cudaMemcpy(dev_b, b, N*sizeof(double),
                              cudaMemcpyHostToDevice ) ); 


    dim3 block(threadsPerBlock);  /* Values defined in macros */
    dim3 grid(blocksPerGrid);     /* defined in macros, above */
    cuda_dot<<<grid,block>>>(N, dev_a, dev_b, 
                                                            dev_partial_c );
    cudaDeviceSynchronize();
    CHECK(cudaPeekAtLastError());


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

    free(partial_c);

    return dot;
}

__global__ void cuda_cg_loop(int N, double alpha, 
                             double *pk, double *uk, 
                             double *rk, double *wk, 
                             double* c, double *d)
{
    extern __shared__ double localDot[];
    double *rk_norm = &localDot[0];
    double *zk_norm = &localDot[threadsPerBlock];
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int localIndex = threadIdx.x;

    double localSum = 0, localMax = 0;
    double zk;
    while (j < N) 
    {
        zk = alpha*pk[j];
        uk[j] = uk[j] + zk;
        rk[j] = rk[j] - alpha*wk[j];
        localSum += rk[j]*rk[j];
        localMax = fabs(zk) > localMax ? fabs(zk) : localMax;
        j += blockDim.x * gridDim.x;
    }
    /* Store sum computed by this thread */
    rk_norm[localIndex] = localSum;
    zk_norm[localIndex] = localMax;
    
    /* Wait for all threads to get to this point */
    __syncthreads();

    /* Every block should add up sum computed on  
       threads in the block */
    int i = blockDim.x/2;
    double z0, z1;
    while (i != 0) 
    {
        if (localIndex < i)
        {
            rk_norm[localIndex] += rk_norm[localIndex + i];
            z0 = zk_norm[localIndex];
            z1 = zk_norm[localIndex+i];
            zk_norm[localIndex] = z1 > z0 ? z1 : z0;
        }
        __syncthreads();
        i /= 2;
    }

    /* Each block stores local dot product */
    if (localIndex == 0)
    {
        c[blockIdx.x] = rk_norm[0];
        d[blockIdx.x] = zk_norm[0];
    }
}

double cg_loop(int N, double alpha, 
               double *pk, double *uk, 
               double *rk, double *wk,
               double *dev_pk, double *dev_uk,
               double *dev_rk, double *dev_wk,
               double *dev_partial_c, double* dev_partial_d,
               double *zk_norm)

{
    double   dot, *partial_c, *partial_d;

    partial_c = (double*) malloc( blocksPerGrid*sizeof(double) );
    partial_d = (double*) malloc( blocksPerGrid*sizeof(double) );

    CHECK(cudaMemcpy(dev_pk, pk, N*sizeof(double),
                              cudaMemcpyHostToDevice ) );
    CHECK(cudaMemcpy(dev_uk, uk, N*sizeof(double),
                              cudaMemcpyHostToDevice ) );
    CHECK(cudaMemcpy(dev_rk, rk, N*sizeof(double),
                              cudaMemcpyHostToDevice ) );
    CHECK(cudaMemcpy(dev_wk, wk, N*sizeof(double),
                              cudaMemcpyHostToDevice ) );


    dim3 block(threadsPerBlock);  /* Values defined in macros */
    dim3 grid(blocksPerGrid);     /* defined in macros, above */
    cuda_cg_loop<<<grid,block,2*threadsPerBlock*sizeof(double)>>>(N, alpha,
                                 dev_pk, dev_uk, 
                                 dev_rk, dev_wk, 
                                 dev_partial_c, dev_partial_d );
    // cudaDeviceSynchronize();
    CHECK(cudaPeekAtLastError());


    /* copy the array 'c' back from the GPU to the CPU */
    CHECK( cudaMemcpy( partial_c, dev_partial_c,
                      blocksPerGrid*sizeof(double),
                      cudaMemcpyDeviceToHost ) );

    CHECK( cudaMemcpy( partial_d, dev_partial_d,
                      blocksPerGrid*sizeof(double),
                      cudaMemcpyDeviceToHost ) );

    CHECK(cudaMemcpy(pk, dev_pk, N*sizeof(double),
                              cudaMemcpyDeviceToHost ) );
    CHECK(cudaMemcpy(uk, dev_uk, N*sizeof(double),
                              cudaMemcpyDeviceToHost ) );
    CHECK(cudaMemcpy(rk, dev_rk, N*sizeof(double),
                              cudaMemcpyDeviceToHost ) );


    /* Sum of block sums */
    dot = 0;
    *zk_norm = 0;
    for (int i = 0; i < blocksPerGrid; i++) 
    {
        dot += partial_c[i];
        *zk_norm = partial_d[i] > *zk_norm ? partial_d[i] : *zk_norm;
    }

    free(partial_c);
    free(partial_d);
    return dot;
}