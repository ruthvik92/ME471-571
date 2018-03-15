#include "demo.h"
#include <demo_util.h>

#include <mpi.h>
#include <math.h>

#define PI 3.14159265358979323846264338327

/* Integrate this over [0,1] */
double f(double x)
{
    double fx;

    fx = (x-1.0)*(x-1.0)*exp(-x*x);
    return fx;
}

/* Indefinite integral goes here */
double I_exact(double x)
{
    /* Use Wolframe Alpha to code indefinite integral */
    double I;
    I = -exp(x)*(4*PI*sin(4*PI*x) + cos(4*PI*x) - 16*pow(PI,2)- 1)/(2+32*pow(PI,2));
    return I;
}


void main(int argc, char** argv)
{
    /* Data arrays */
    double a,b;
    int n_global;
    double *x;
    double range[2];

    /* MPI variables */
    int my_rank, nprocs;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    set_rank(my_rank);  /* Used in printing */
    read_loglevel(argc,argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


    if (my_rank == 0)
    {        
        int m,err;
        read_int(argc,argv, "-m", &m, &err);
        if (err > 0)
        {
            print_global("Command line argument '-m' not found\n");
            exit(0);
        }        

        n_global = pow2(m);     /* Number of sub-intervals used for integration */

        /* Hardwire values */
        a = -1;
        b = 1;  
    }

    /* Broadcast value of N to all processors;  compute number of panels in each 
    subinterval */
    int root = 0;
    MPI_Bcast(&a,1,MPI_DOUBLE,root,MPI_COMM_WORLD);
    MPI_Bcast(&b,1,MPI_DOUBLE,root,MPI_COMM_WORLD);
    MPI_Bcast(&n_global,1,MPI_INT,root,MPI_COMM_WORLD);

    double w = (b-a)/nprocs;    
    int m = n_global/nprocs;   /* Number of panels in each section */

    range[0] = a + my_rank*w;
    range[1] = range[0] + w;
    double h = (range[1] - range[0])/m;


    /* Apply trapezoidal rule (for loop) */
    linspace_array(range[0],range[1],m+1,&x);

    /* Call MPI_Reduce to get final integral */
    double I_local=0;
    int i;
    for(i = 0; i < m+1; i++)
    {
        I_local += f(x[i]);
    }
    I_local -= 0.5*(f(range[0]) + f(range[1]));
    I_local *= h;

    double I;
    MPI_Reduce(&I_local,&I,1,MPI_DOUBLE,MPI_SUM,root,MPI_COMM_WORLD);    

    if (my_rank == 0)
    {
        double Ie = -1.0/exp(1.0) + 1.5*sqrt(PI)*erf(1.0);
        double Ie_wolf = 1.872592957265838754602878538; /* Wolfram */
        print_global("%10d %24.20f %24.20f %12.4e\n",n_global,I,Ie_wolf,fabs(I-Ie_wolf));
    }

    delete_array(&x);

    MPI_Finalize();

}