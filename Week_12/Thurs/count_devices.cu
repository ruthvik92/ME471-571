#include <stdio.h>

int main( void ) 
{
    cudaDeviceProp  prop;

    int count;
    cudaGetDeviceCount( &count);

    for (int i = 0; i < count; i++) 
    {
        cudaGetDeviceProperties( &prop, i);

        printf( "   --- General Information for device %d ---\n", i );
        printf( "Name:  %s\n", prop.name );

        printf( "   --- Memory Information for device %d ---\n", i );
        printf( "Total global mem  :  %12d (bytes)\n", prop.totalGlobalMem );
        printf( "Total global mem  :  %12.2f (kb)\n", prop.totalGlobalMem/1024.0);
        printf( "Total global mem  :  %12.2f (mb)\n", prop.totalGlobalMem/(1024.0*1024.0));
        printf( "Total global mem  :  %12.2f (gb)\n", prop.totalGlobalMem/(1024.0*1024.0*1024.0));


        printf( "   --- MP Information for device %d ---\n", i );
        printf( "Multiprocessor count:  %d\n",
                    prop.multiProcessorCount );
        printf( "Shared mem per mp :  %12.1f (kb)\n", prop.sharedMemPerBlock/1024. );
        printf( "Registers per mp  :  %12.1f (kb)\n", prop.regsPerBlock/1024. );
        printf( "Threads in warp   :  %d\n", prop.warpSize );
        printf( "Max threads per block:  %d\n",
                    prop.maxThreadsPerBlock );
        printf( "Max thread dimensions:  (%d, %d, %d)\n",
                    prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                    prop.maxThreadsDim[2] );
        printf( "Max grid dimensions:  (%d, %d, %d)\n",
                    prop.maxGridSize[0], prop.maxGridSize[1],
                    prop.maxGridSize[2] );
        printf( "\n" );
    }
}
