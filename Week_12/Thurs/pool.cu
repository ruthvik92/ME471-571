#include <stdio.h>
#include <demo_util.h>

#define CLOCK_RATE 1500000 // in kHZ

// Launch a <<<48, 32>>> kernel and note for all 1536 threads the SM id
// And then accumulate how much "work" did each of the 58 SMs (RTXA5500) do

__device__ uint get_smid(void)
{

    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret)); // inline assembly to move 32bit int from %smid into operand %0
    return ret;                            // %0 must be copied into variable ret
}

__device__ void sleep(float t)
{
    clock_t t0 = clock64();
    clock_t t1 = t0;
    while ((t1 - t0) / (CLOCK_RATE * 1000.0f) < t)
    {
        t1 = clock64();
    }
}

__global__ void worker(float *t, int *s, int *d, int N)
{
    // int id = blockIdx.x;
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < N)

    {

        s[id] = get_smid();
        d[id] = id;
        sleep(t[id]);
    }
    //    __syncthreads();  // Optional: synchronize threads within block
}
#define N 1536

int main(void)
{
    float *dev_t;
    int *dev_s;
    int *dev_d;
    float t[N];
    int s[N];
    int d[N];
    float SM[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int i;

    /* Allocate memory on the device */
    cudaMalloc((void **)&dev_t, N * sizeof(float));
    cudaMalloc((void **)&dev_s, N * sizeof(int));
    cudaMalloc((void **)&dev_d, N * sizeof(int));

    random_seed();

    float maxt = 0;
    for (i = 0; i < N; i++)
    {
        // t[i] = 1; // 10*random_number();
        t[i] = 10 * random_number();
        maxt = t[i] > maxt ? t[i] : maxt;
    }

    for (i = 0; i < N; i++)
    {
        printf("t[%d] = %f\t", i, t[i]);
    }

    cudaMemcpy(dev_t, t, N * sizeof(float), cudaMemcpyHostToDevice);

    worker<<<48, 32>>>(dev_t, dev_s, dev_d, N);

    /* Copy contents of dev_t back to t */
    cudaMemcpy(s, dev_s, N * sizeof(int), cudaMemcpyDeviceToHost);
    /* Copy contents of dev_d back to d */
    cudaMemcpy(d, dev_d, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (i = 0; i < N; i++)
    {
        printf("Thread %2d worked for %8.4f seconds on SM %d\n", i, t[i], s[i]);
        SM[s[i]] += t[i];
    }
    printf("Max t = %8.3f\n", maxt);
    printf("\n");
    for (i = 7; i >= 0; i--)
    {
        printf("SM[%d] = %8.4f\n", i, SM[i]);
    }
    printf("\n");
    for (i = 0; i < N; i++)
    {
        printf("d[%d] = %d\t", i, d[i]);
    }
    cudaFree(dev_t);
    cudaFree(dev_s);
    cudaFree(dev_d);
}

//   --- General Information for device 0 ---
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
