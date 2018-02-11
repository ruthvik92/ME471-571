#include "demo7.h"
#include <demo_util.h>

#include <mpi.h>
#include <math.h>

#define PI 3.141592653589793116

/* Integrate this over [0,1] */
double f(double x)
{
    double fx;

    fx = exp(x)*pow(sin(2*PI*x),2);
    return fx;
}

/* Indefinite integral goes here */
double I_exact(double x)
{
    /* Use Wolframe Alpha to code indefinite integral */
    return 0;
}

void main(int argc, char** argv)
{
    /* Data arrays */
    int n_global;

    /* MPI variables */
    int my_rank, nprocs;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    set_rank(my_rank);  /* Used in printing */

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


    print_global("Greetings from rank 0!!\n");

    if (nprocs < 2)
    {
        print_global("demo7 : Must have at least 2 processors.\n");
        exit(0);
    }

    if (my_rank == 0)
    {        
        int p0,err;
        read_int(argc,argv, "-p",&p0,&err);
        if (err > 0)
        {
            print_global("Command line argument '-p' not found.\n");
            exit(0);
        }
        n_global = pow2(p0);     /* Number of sub-intervals used for integration */

        /* Hardwire values */
        double a,b;
        a = 0;
        b = 1;  

        /* Your Node P=0 work goes here */
        /* Send sub-interval to other processors */
        double yp = 0;
        double w = (b-a)/nprocs;

        int p;
        for(p = 1; p < nprocs; p++)
        {
            /* pass two values to processor p */
            double range[2];
            range[0] = p*w;
            range[1] = range[0] + w;

            int tag = 0;
            int dest = p;
            MPI_Send((void*) range,2,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

        }
    }
    else
    {
        MPI_Status status;
        int count;

        print_debug("Hello!\n");

        double range[2];

        /* Receive range values */
        int source = 0;
        int tag = 0;
        MPI_Recv((void*) range,2,MPI_DOUBLE,source,tag,MPI_COMM_WORLD,&status);   
        MPI_Get_count(&status,MPI_DOUBLE,&count);         
        print_debug("range is %f %f\n",range[0],range[1]);

    }

    /* Broadcast value of N to all processors;  compute number of panels in each 
    subinterval */

    /* Every processor now knows its range and number of panels */

    /* Apply trapezoidal rule (for loop) */

    /* Call MPI_Reduce to get final integral */

    /* Node 0 prints the results - print_global() */



    MPI_Finalize();

}