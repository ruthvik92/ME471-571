#include "demo7.h"
#include <demo_util.h>

#include <stdio.h>
#include <string.h>
#include <mpi.h>


void main(int argc, char** argv)
{
    /* Data arrays */
    double *x;
    double s, total_sum;
    double mean;
    int n_global;

    /* MPI variables */
    int my_rank, nprocs;
    int tag = 0;
    int count;
    int source;
    MPI_Status status;

    /* Other local variables */
    int i, p;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (my_rank == 0)
    {        
        int p0, err;
        read_int(argc,argv, "-p",&p0,&err);
        n_global = pow2(p0);
        printf("p0 = %d; n = %d\n",p0,n_global);
        
        // random_array(n_global,&x);  /* Used for both sending and receiving */
    }

    MPI_Bcast((void*) &n_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

    printf("Processor %2d : n_global is %d\n",my_rank,n_global);

#if 0
    delete_array(&x);
#endif

    MPI_Finalize();

}