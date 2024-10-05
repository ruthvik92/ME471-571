#include <stdio.h>

int main(void)
{
       cudaDeviceProp prop;

       int count;
       char str[4];
       cudaGetDeviceCount(&count);

       if (count == 0)
       {
              printf("No CUDA capable devices found.\n");
       }

       for (int i = 0; i < count; i++)
       {
              cudaGetDeviceProperties(&prop, i);

              printf("   --- General Information for device %d ---\n", i);
              printf("Name:  %s\n", prop.name);
              printf("\n");
              sprintf(str, "%d.%d", prop.major, prop.minor);
              printf("Compute capability    :  %14s\n", str);
              printf("Clock rate            :  %14.2f (GHz)\n", prop.clockRate / 1000000.0);
              printf("\n");

              printf("   --- Memory Information for device %d ---\n", i);
              // printf("Total global mem      :  %14.1f (bytes)\n", (double) prop.totalGlobalMem );
              // printf("Total global mem      :  %14.1f (kb)\n", prop.totalGlobalMem/1024.0);
              // printf("Total global mem      :  %14.1f (mb)\n", prop.totalGlobalMem/(1024.0*1024.0));
              printf("Total global mem      :  %14.1f (gb)\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
              printf("\n");

              printf("   --- MP Information for device %d ---\n", i);
              printf("Multiprocessor count :  %14d\n",
                     prop.multiProcessorCount);
              printf("Shared mem per mp     :  %14.1f (kb)\n", prop.sharedMemPerBlock / 1024.);
              printf("Registers per mp      :  %14.1f (kb)\n", prop.regsPerBlock / 1024.);
              printf("Threads in warp       :  %14d\n", prop.warpSize);
              printf("Max threads per block :  %14d\n",
                     prop.maxThreadsPerBlock);
              printf("Max thread dimensions:  (%d, %d, %d)\n",
                     prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                     prop.maxThreadsDim[2]);
              printf("Max grid dimensions  :  %d, %d, %d\n",
                     prop.maxGridSize[0], prop.maxGridSize[1],
                     prop.maxGridSize[2]);
              printf("\n");
       }
}

// ###############################################
// The threads in a warp execute the same instruction simultaneously, though they may operate on different data.
// Warps are important for performance optimization because GPUs are most efficient when all threads in a warp are
// doing the same thing. If some threads in the warp diverge (i.e., they take different branches in an if-else statement),
// the warp will execute the different branches serially, which can slow down execution.
// ###############################################

//// ###############################################
//    Max thread dimensions: (1024, 1024, 64): This means the maximum number of threads per block can be configured in a 3D grid with the following limits:
//        The x dimension can go up to 1024 threads.
//        The y dimension can also go up to 1024 threads.
//        The z dimension can go up to 64 threads.
//
//    However, the total number of threads in a block cannot exceed the max threads per block, which is 1024 in this case.
//      This means you can create a thread block with different dimensions, like:
//
//        1024 x 1 x 1 (a 1D block)
//        32 x 32 x 1 (a 2D block)
//        16 x 16 x 4 (a 3D block)
//
//    All these examples respect the total limit of 1024 threads per block.

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
