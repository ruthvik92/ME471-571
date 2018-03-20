#include "demo.h"
#include <demo_util.h>

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stddef.h>

#define PI 3.14159265358979323846264338327

double utrue(double x)
{
    double u;
    double pi2;
    pi2 = 2*PI;
    u = cos(pi2*x);
    return u;
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

    MPI_Type_create_struct(3,block_lengths, disp, typelist, domain_t);
    MPI_Type_commit(domain_t);
}



void main(int argc, char** argv)
{
    /* Data arrays */
    double *x, *u;
    double range[2];
    struct_domain_t domain;


    /* File I/O */
    // MPI_Offset offset;
    MPI_File   file;
    MPI_Status status;
    
    /* Data type */
    MPI_Datatype domain_t;
    MPI_Datatype localarray;

    int j;

    /* MPI variables */
    int my_rank, nprocs;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    set_rank(my_rank);  /* Used in printing */
    read_loglevel(argc,argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (my_rank == 0)
    {        
        int m,err,loglevel;
        read_int(argc,argv, "-m", &m, &err);
        if (err > 0)
        {
            print_global("Command line argument '-m' not found\n");
            exit(0);
        }        

        domain.n_global = pow2(m);     /* Number of sub-intervals used for integration */

        /* Hardwire values */
        domain.a = 0;
        domain.b = 1;  
    }

    build_domain_type(&domain_t);

    /* Broadcast global information */
    int root = 0;
    MPI_Bcast(&domain.a,1,domain_t,root,MPI_COMM_WORLD);

    double w = (domain.b-domain.a)/nprocs;    
    int m = domain.n_global/nprocs;   /* Number of panels in each section */

    range[0] = domain.a + my_rank*w;
    range[1] = range[0] + w;
    double h = (range[1] - range[0])/m;

    /* ---------------------------------------------------------------
       Set up array to write out
    --------------------------------------------------------------- */
    linspace_array(range[0],range[1],m+1,&x);
    int nsize = my_rank < nprocs-1 ? m : m+1;
    zeros_array(nsize,&u);

    for(j = 0; j < nsize; j++)
    {
        u[j] = utrue(x[j]);
    }

    /* ----------------------------------------------------------------
       Write out file
    ---------------------------------------------------------------- */

    int globalsize = domain.n_global+1;  /* Leave extra space */
    int localsize = nsize;
    int starts = m*my_rank;
    int order = MPI_ORDER_C;

    MPI_Type_create_subarray(1, &globalsize, &localsize, &starts, order, 
                             MPI_DOUBLE, &localarray);
    MPI_Type_commit(&localarray);

    /* Open file for real */
    MPI_File_open(MPI_COMM_WORLD, "bin.out", 
                  MPI_MODE_CREATE|MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &file);

    MPI_Offset offset = 0;
    MPI_File_set_view(file, offset,  MPI_DOUBLE, localarray, 
                           "native", MPI_INFO_NULL);

    MPI_File_write_all(file, u, localsize, MPI_DOUBLE,MPI_STATUS_IGNORE);
    MPI_File_close(&file);

    MPI_Type_free(&localarray);

    delete_array((void*) &x);
    delete_array((void*) &u);

    MPI_Finalize();

}