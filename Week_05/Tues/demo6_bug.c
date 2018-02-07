#include "demo6.h"
#include <demo_util.h>

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
    set_rank(my_rank);

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (nprocs < 2)
    {
        print_global("demo6 : Must have at least 2 processors.\n");
        exit(0);
    }

    if (my_rank == 0)
    {        
        int m;
        read_int(argc,argv, "-m",&m);
        n_global = pow2(m);

        print_debug("m = %d; n = 2^m = %d\n",m,n_global);

        random_array(n_global,&x);  /* global array */

        int tag = 0;
        int dest = 1;
        MPI_Send((void*) &n_global,1,MPI_INT,dest,tag,MPI_COMM_WORLD);       

        for(p = 1; p < nprocs; p++)
        {
            tag = 1;
            dest = p;
            print_debug("Sending %d doubles to %d\n",n_global,p);
            MPI_Send((void*) x,n_global,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);
        }
    }
    else
    {
        source = 0;
        tag = 0;
        MPI_Recv((void*) &n_global,1,MPI_INT,source,tag,MPI_COMM_WORLD,&status);
        
        empty_array(n_global,&x);  /* store global array here on processor 'P' */

        for(p = 1; p < nprocs; p++)
        {
            source = p;
            tag = 1;
            print_debug("Receiving ...\n",my_rank);
            MPI_Recv((void*) x,n_global,MPI_DOUBLE,source,tag,MPI_COMM_WORLD,&status);            

            MPI_Get_count(&status,MPI_DOUBLE,&count);
            if (count < n_global)
            {
                print_debug("Something went wrong\n");
                print_debug("count = %d\n",count);
                exit(0);
            }
            print_debug("Received %d doubles from %d\n",my_rank,count,source);
        }
    }

    delete_array(&x);

    MPI_Finalize();

}