#include "demo.h"
#include <demo_util.h>

#include <mpi.h>
#include <math.h>
#include <stddef.h>

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

typedef struct 
{
    double a;
    double b; 
    int n_global;
} struct_domain_t;

void build_domain_type(MPI_Datatype *domain_t)
{
    int block_lengths[3] = {1,1,1};

    /* Set up types */
    MPI_Datatype typelist[3];
    typelist[0] = MPI_DOUBLE;
    typelist[1] = MPI_DOUBLE;
    typelist[2] = MPI_INT;

    /* Set up displacements */
    MPI_Aint disp[3];
    disp[0] = offsetof(struct_domain_t,a);
    disp[1] = offsetof(struct_domain_t,b);
    disp[2] = offsetof(struct_domain_t,n_global);

    MPI_Type_create_struct(3,block_lengths, disp, typelist,domain_t);
    MPI_Type_commit(domain_t);
}



void main(int argc, char** argv)
{
    /* Data arrays */
    struct_domain_t domain;
    double *x;
    double range[2];

    /* MPI variables */
    int my_rank, nprocs;

    /* Data type */
    MPI_Datatype domain_t;

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

        domain.n_global = pow2(m);     /* Number of sub-intervals used for integration */

        /* Hardwire values */
        domain.a = -1;
        domain.b = 1;  
    }

    build_domain_type(&domain_t);

    int root;
    MPI_Bcast(&domain,1, domain_t, root, MPI_COMM_WORLD);

    double w = (domain.b-domain.a)/nprocs;    
    int m = domain.n_global/nprocs;   /* Number of panels in each section */

    range[0] = domain.a + my_rank*w;
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
        print_global("%10d %24.20f %24.20f %12.4e\n",
                     domain.n_global,I,Ie_wolf,fabs(I-Ie_wolf));
    }

    delete_array((void*) &x);

    MPI_Finalize();

}