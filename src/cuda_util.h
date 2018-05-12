#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <stdio.h>

#define CHECK(call)                                                          \
{                                                                            \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess)                                                \
    {                                                                        \
        printf("Error : %s : %d, ", __FILE__, __LINE__);                     \
        printf("code : %d, reason : %s\n",error, cudaGetErrorString(error)); \
        exit(1);                                                             \
    }                                                                        \
}       

#endif
