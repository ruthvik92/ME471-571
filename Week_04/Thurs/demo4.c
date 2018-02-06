#include "demo4.h"
#include "demo_util.h"

#include <stdio.h>
#include <mpi.h>

void main(int argc, char** argv)
{
    /* Data arrays */
    double *x;
    double s, total_sum;
    double mean;

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

    int n_global = pow2(10);
    int n_local = n_global/nprocs;

    if (my_rank == 0)
    {
        random_array(n_global,&x);  /* Used for both sending and receiving */
        int p;
        for(p = 1; p < nprocs; p++)
        {
            int dest = p;
            MPI_Send((void*) &x[p*n_local],n_local,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);
        }
    }
    else
    {
        source = 0;
        empty_array(n_local,&x);
        MPI_Recv((void*) x,n_local,MPI_DOUBLE,source,tag,MPI_COMM_WORLD,&status);
        MPI_Get_count(&status,MPI_DOUBLE,&count);
        if (count < n_local)
        {
            printf("Something went wrong\n");
            printf("n_local = %d\n",n_local);
            printf("count = %d\n",count);
            exit(0);
        }
    }


    source = 0;
    s = sum_array(n_local,x);   
    MPI_Reduce(&s,&total_sum,1,MPI_DOUBLE,MPI_SUM,source,MPI_COMM_WORLD);    
    mean = total_sum/n_global;

    printf("Processor %d : The mean is  %.16f\n",my_rank,mean);        

#if 0
    if (my_rank == 0)
    {
        double mean_true;
        mean_true = sum_array(n_global,x)/n_global;
        printf("Processor %d : True mean is %.16f\n",my_rank,mean_true);
    }

    if (my_rank == 0)
    {
        printf("\n");
        printf("Processor 0 : Broadcasting mean ...\n");
    }

    source = 0;
    MPI_Bcast(&mean,1,MPI_DOUBLE,source,MPI_COMM_WORLD);

    printf("Processor %d : The mean is %.16f\n",my_rank,mean);

    /* Fill in details for broadcasting to all processors, computing 
    sums needed for STD, and then reducing results */
#endif

    delete_array(&x);

    MPI_Finalize();

}