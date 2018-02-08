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

    if (nprocs < 2)
    {
        print_global("demo7 : Must have at least 2 processors.\n");
        exit(0);
    }

    if (my_rank == 0)
    {        
        int p0;
        read_int(argc,argv, "-p",&p0);
        n_global = pow2(p0);     /* Number of sub-intervals used for integration */

        print_debug("n_global = %d\n",n_global);

        /* Your Node P=0 work goes here */

    }
    else
    {
        print_debug("Hello!\n");
        
        /* Your node P>0 goes here */
    }

    MPI_Finalize();

}