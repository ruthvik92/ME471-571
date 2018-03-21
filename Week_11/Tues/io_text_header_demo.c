#include "demo.h"
#include <demo_util.h>

#include <mpi.h>
#include <math.h>
#include <stdio.h>  
#include <stddef.h>  /* offsetof */

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
    MPI_File   file;
    MPI_Status status;
    
    /* Data type */
    MPI_Datatype domain_t;
    MPI_Datatype row_t;
    MPI_Datatype localarray;
    int rank, nprocs;

    /* Length of each row in text file */
    char *text;
    const int chars_per_row = 24;

    /* Formatting parameters : 
        -- All formats   : Leave one space for '\n' (line feed character)
        -- Fixed point   : Leave at least 2 spaces for '.' and ('+' or '-')
        -- Sci. notation : Leave at least 6 spaces for 'e+NN', '.' and ('+' or '-')
    */
    const int width = chars_per_row-1;  
    const int precision = 16;           /* check : precision+6 <= width */

    int j;

    /* ---- MPI Initialization */
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    set_rank(rank);  
    read_loglevel(argc,argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* ---- Read data from command line */
    if (rank == 0)
    {        
        int m,err;
        read_int(argc,argv, "-m", &m, &err);
        if (err > 0)
        {
            print_global("Command line argument '-m' not found\n");
            exit(0);
        }        

        domain.n_global = pow2(m);    
        domain.a = 0;
        domain.b = 1;  
    }

    /* ---- Communicate data and set up domain */
    build_domain_type(&domain_t);

    MPI_Bcast(&domain,1,domain_t,0,MPI_COMM_WORLD);

    double w = (domain.b-domain.a)/nprocs;    
    int m = domain.n_global/nprocs;   

    range[0] = domain.a + rank*w;
    range[1] = range[0] + w;

    /* ---- Get solution */
    linspace_array(range[0],range[1],m+1,&x);
    int nsize = rank < nprocs-1 ? m : m+1;
    zeros_array(nsize,&u);

    for(j = 0; j < nsize; j++)
    {
        u[j] = utrue(x[j]);
    }


    /* ---- Open file so we can write header and solution */ 
    MPI_File_open(MPI_COMM_WORLD, "text_header.out", 
                  MPI_MODE_CREATE|MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &file);


    /* ---- Create header to store meta data */
    int hlen = chars_per_row;
    int header_size = 3*hlen;  /* 3 rows of meta data */
    if (rank == 0)
    {
        char *header;
        char_array(header_size+1,&header);
        sprintf(&header[0*hlen],"%*.*f\n",width,2,domain.a); /* 15 lines each */
        sprintf(&header[1*hlen],"%*.*f\n",width,2,domain.b);
        sprintf(&header[2*hlen],"%*d\n", width,domain.n_global);
        MPI_File_write(file,header,header_size,MPI_CHAR,MPI_STATUS_IGNORE);  
        delete_array((void*) &header);      
    }


    /* ---- Create text string to write out */
    int nlen = nsize*chars_per_row;    /* 1 byte per character */
    char_array(nlen+1,&text);  /* extra space for null termination character */

    for (j = 0; j < nsize; j++) 
    {
        sprintf(&text[j*chars_per_row],"%*.*e\n",width,precision,u[j]);      
    }

    /* ---- Create data type to store rows of each text file */
    MPI_Type_contiguous(chars_per_row, MPI_CHAR, &row_t); 
    MPI_Type_commit(&row_t); 

    int globalsize = domain.n_global+1; 
    int localsize = nsize;
    int starts = m*rank;
    int order = MPI_ORDER_C;

    /* ---- Create view for this processor into file */
    MPI_Type_create_subarray(1, &globalsize, &localsize, &starts, order, 
                             row_t, &localarray);
    MPI_Type_commit(&localarray);

    MPI_Offset offset = header_size;    /* in bytes;  skip header */
    MPI_File_set_view(file, offset,  MPI_CHAR, localarray, 
                           "native", MPI_INFO_NULL);

    /* ---- Write out file */
    MPI_File_write_all(file, text, localsize, row_t,MPI_STATUS_IGNORE);

    /* ---- Clean up */
    MPI_File_close(&file);

    MPI_Type_free(&localarray);
    MPI_Type_free(&row_t);    

    delete_array((void*) &text);
    delete_array((void*) &x);
    delete_array((void*) &u);

    MPI_Finalize();

}