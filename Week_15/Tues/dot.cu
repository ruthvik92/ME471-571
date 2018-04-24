#include <stdio.h>
#include <sys/time.h>
#include <demo_util.h>
#include <cuda_util.h>

#define imin(a,b) (a<b?a:b)

const int N = (1 << 14);
const int threadsPerBlock = 256;

__global__ void dot( double *a, double *b, double *c ) 
{
    __shared__ double localDot[threadsPerBlock];  /* Statically defined */
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

int main( void ) 
{
    double   *a, *b, c, *partial_c;
    double   *dev_a, *dev_b, *dev_partial_c;

    int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;

    /* Allocate memory on the CPU */
    a = (double*) malloc( N*sizeof(double) );
    b = (double*) malloc( N*sizeof(double) );
    partial_c = (double*) malloc( blocksPerGrid*sizeof(double) );

    /* allocate the memory on the GPU */
    CHECK(cudaMalloc((void**) &dev_a, N*sizeof(double)));
    CHECK(cudaMalloc((void**) &dev_b, N*sizeof(double)));
    CHECK(cudaMalloc((void**) &dev_partial_c, blocksPerGrid*sizeof(double) ) );

    /* Define vectors a and b */
    for (int i = 0; i < N; i++) 
    {
        a[i] = i;
        b[i] = i;
    }

    /* copy the arrays 'a' and 'b' to the GPU */
    CHECK(cudaMemcpy(dev_a, a, N*sizeof(double),
                              cudaMemcpyHostToDevice ) );
    CHECK(cudaMemcpy(dev_b, b, N*sizeof(double),
                              cudaMemcpyHostToDevice ) ); 

    dim3 block(threadsPerBlock);  /* Values defined in macros */
    dim3 grid(blocksPerGrid);     /* defined in macros, above */
    dot<<<grid,block>>>( dev_a, dev_b,dev_partial_c );

    /* copy the array 'c' back from the GPU to the CPU */
    CHECK( cudaMemcpy( partial_c, dev_partial_c,
                      blocksPerGrid*sizeof(double),
                      cudaMemcpyDeviceToHost ) );

    /* Sum of block sums */
    c = 0;
    for (int i = 0; i < blocksPerGrid; i++) 
    {
        c += partial_c[i];
    }

    /* Check result */
    #define sum_squares(x)  (x*(x+1)*(2*x+1)/6)
    double s = sum_squares((double)(N-1));
    // double s = N;   /* Sum of 1s */
    double diff = abs(c - s)/abs(s);
    printf("%20s %10f\n","Computed dot product", c);
    printf("%20s %10f\n","True dot product", s);

    /* free memory on the gpu side */
    CHECK(cudaFree(dev_a));
    CHECK(cudaFree(dev_b));
    CHECK(cudaFree(dev_partial_c));

    /* free memory on the cpu side */
    free(a );
    free(b );
    free(partial_c );
}
