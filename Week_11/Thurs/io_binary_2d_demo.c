#include "demo.h"
#include <demo_util.h>

#include <mpi.h>
#include <math.h>
#include <stdio.h>  
#include <stddef.h>  /* offsetof */

#define PI 3.14159265358979323846264338327

/* Get indices assuming Morton ordering */
static
void get_index(int L, int p, int* iloc, int* jloc)
{
    unsigned int  M,i,w;

    M = 1;
    for(i = 0; i < L; i++) 
        M*=2;

    w = M;
    *iloc = 0;
    *jloc = 0;
    for (i = (1 << 2*L-1); i > 0; i /= 2)
    {
        w /= 2;

        *jloc += (i & p) ? w : 0;   /* x shift */

        i /= 2;
        *iloc += (i & p) ? w : 0;   /* y shift */
    }
}

double utrue(double x, double y)
{
    double u;
    double pi2;
    pi2 = 2*PI;
    // u = cos(2*pi2*x)*sin(2*pi2*y);
    u = cos(2*pi2*(x*x + y*y));
    return u;
}

typedef struct 
{
    double a[2];
    double b[2];
    int n_global[2];
    int L;  /* level */
    int m;
} struct_domain_t;

void build_domain_type(MPI_Datatype *domain_t)
{
    int block_lengths[5] = {2,2,2,1,1};

    /* Set up types */
    MPI_Datatype typelist[5];
    typelist[0] = MPI_DOUBLE;
    typelist[1] = MPI_DOUBLE;
    typelist[2] = MPI_INT;
    typelist[3] = MPI_INT;
    typelist[4] = MPI_INT;

    /* Set up displacements */
    MPI_Aint disp[5];
    disp[0] = offsetof(struct_domain_t,a);
    disp[1] = offsetof(struct_domain_t,b);
    disp[2] = offsetof(struct_domain_t,n_global);
    disp[3] = offsetof(struct_domain_t,L);
    disp[4] = offsetof(struct_domain_t,m);

    MPI_Type_create_struct(5,block_lengths, disp, typelist, domain_t);
    MPI_Type_commit(domain_t);
}


void main(int argc, char** argv)
{
    /* Data arrays */
    double *x, *y, **u;
    double range_x[2];
    double range_y[2];
    struct_domain_t domain;


    /* File I/O */
    MPI_File   file;
    MPI_Status status;
    
    /* Data type */
    MPI_Datatype domain_t;
    MPI_Datatype localarray;
    int rank, nprocs;

    int i,j;

    /* ---- MPI Initialization */
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    set_rank(rank);  
    read_loglevel(argc,argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* ---- Read data from command line */
    if (rank == 0)
    {        
        int L,err,m;
        read_int(argc,argv, "-L", &L, &err);
        if (err > 0)
        {
            print_global("Command line argument '-L' not found\n");
            exit(0);
        }        
        int i,n1 = 1;
        for(i=0; i < L; i++)
            n1 *= 4;
        if (n1 != nprocs)
        {
            print_essential("4^L != number of procs\n");
            exit(0);
        }

        /* Grid size on each processor */
        read_int(argc,argv, "-m", &m, &err);
        if (err > 0)
        {
            print_global("Command line argument '-m' not found\n");
            exit(0);
        }        


        domain.a[0] = -1;    /* xlower */
        domain.a[1] = -1;    /* ylower */
        domain.b[0] = 1;     /* xupper */
        domain.b[1] = 1;     /* yupper */

        domain.n_global[0] = pow2(L)*m;    
        domain.n_global[1] = pow2(L)*m;  
        domain.L = L;  
        domain.m = m;
    }

    /* ---- Communicate data and set up domain */
    build_domain_type(&domain_t);

    MPI_Bcast(&domain,1,domain_t,0,MPI_COMM_WORLD);

    int pdim = (int) sqrt(nprocs);  /* Procs in each direction */
    if (abs(pdim*pdim-nprocs) > 1e-8)
    {
        print_essential("pdim = %d; nprocs = %d\n",pdim,nprocs);
        print_essential("Proc count should be a square number.\n");
        exit(0);
    }

    double w[2];
    int m[2];
    for(i = 0; i < 2; i++)
    {
        w[i] = (domain.b[i]-domain.a[i])/pdim;    
        m[i] = domain.m;   
        if (m[i] != domain.n_global[i]/pdim)
        {
            /* We shouldn't ever end up here */
            print_essential("m = %d; n/pdim = %d\n",m[i],
                            domain.n_global[i]/pdim);
            print_essential("m != nglobal/pdim\n");
            exit(0);  
        }
    }

    /* ---- Get index of this grid in 2d array */
    int iloc,jloc;
    get_index(domain.L,rank,&iloc,&jloc);

    /* ---- Set up domain */
    range_x[0] = domain.a[0] + iloc*w[0];
    range_x[1] = range_x[0] + w[0];
    range_y[0] = domain.a[1] + jloc*w[1];
    range_y[1] = range_y[0] + w[1];
    double dx = (domain.b[0] - domain.a[0])/domain.n_global[0];
    double dy = (domain.b[1] - domain.a[1])/domain.n_global[1];


    /* ---- Get solution */
    linspace_array(range_x[0],range_x[1],m[0]+1,&x);
    linspace_array(range_y[0],range_y[1],m[1]+1,&y);
    int nsize[2] = {m[0],m[1]};
    empty_array2(m[0],m[1],&u);

    for(i = 0; i < m[0]; i++)
    {
        for(j = 0; j < m[1]; j++)
        {
            /* Get cell centered values */
            double xh = x[i] + dx/2;
            double yh = y[j] + dy/2;
            u[i][j] = utrue(xh,yh);            
        }
    }


    /* ---- Open file so we can write header and solution */
    MPI_File_open(MPI_COMM_WORLD, "bin2d.out", 
                  MPI_MODE_CREATE|MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &file);

    /* ---- Create header to store meta data */
    if (rank == 0)
    {
        MPI_File_write(file,&domain,1,domain_t, MPI_STATUS_IGNORE);  
    }

    /* ---- Create view for this processor into file */
    int globalsize[2] = {domain.n_global[0], domain.n_global[1]}; 
    int localsize[2] = {m[0],m[1]};

    int starts[2];
    starts[0] = m[0]*iloc;  
    starts[1] = m[1]*jloc;

    int order = MPI_ORDER_C;

    MPI_Type_create_subarray(2, globalsize, localsize, starts, order, 
                             MPI_DOUBLE, &localarray);
    MPI_Type_commit(&localarray);

    /* Skip header */
    MPI_Aint extent;
    MPI_Type_extent(domain_t,&extent); 
    MPI_Offset offset = extent;

    /* Set view (with offset for header) */
    MPI_File_set_view(file, offset,  MPI_DOUBLE, localarray, 
                           "native", MPI_INFO_NULL);

    /* ---- Write out file */
    MPI_File_write_all(file, *u, m[0]*m[1], MPI_DOUBLE, MPI_STATUS_IGNORE);

    /* ---- Clean up */
    MPI_File_close(&file);

    MPI_Type_free(&localarray);

    delete_array((void*) &x);    
    delete_array((void*) &y);
    delete_array2((void*) &u);

    MPI_Finalize();

}