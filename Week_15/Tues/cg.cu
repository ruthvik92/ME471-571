#include <demo_util.h>
#include <cuda_util.h>

#include <math.h>
#include <string.h>   /* For atoi */

#define PI 3.14159265358979323846264338327

#ifdef USE_GPU
/* From CG kernels */
int get_N();
int get_blocksPerGrid();
double dot_norm(int N, double *a, double *dev_a,
               double *dev_partial_c);

double dot_gpu(int N, double *a, double *b,
               double *dev_a, double *dev_b, 
               double *dev_partial_c);

double cg_loop(int N, double alpha, 
             double *pk, double *uk, 
             double *rk, double *wk,
             double *dev_pk, double *dev_uk,
             double *dev_rk, double *dev_wk,
             double *dev_partial_c, double *dev_partial_d,
             double *zk_norm);

#endif

double utrue(double x)
{
    double u;
    double pi2;
    pi2 = 2*PI;
    u = cos(pi2*x);
    return u;
}

double rhs(double x)
{
    double fx;
    double pi2;
    pi2 = 2*PI;
    fx = -(pi2)*(pi2)*cos(pi2*x);
    return fx;
}


int main(int argc, char** argv)
{
    /* Data arrays */
    double a,b;
    int n_global;
    double *x, *F, *B;

    /* Iterative variables */
    double tol;
    int kmax,j,k;
    double range[2];

    /* Stuff that was handled by MPI */
    int my_rank = 0;
    set_rank(my_rank);  /* Used in printing */
    read_loglevel(argc,argv);

#ifdef USE_GPU
    printf("Using GPU\n");
#endif    

    if (my_rank == 0)
    {
        /* Input */
        int err;
        int mp;
        read_int(argc,argv, "-m", &mp, &err);
        if (err > 0)
        {
            print_global("Command line argument '-m' not found\n");
            exit(0);
        }    
#ifdef USE_GPU        
        n_global = get_N();
        if (n_global != (1 << mp))
        {
            print_global("1<< mp != N (defined in cg_kernel.cu)\n");
            exit(0);
        }
#else
        n_global = 1 << mp;     /* Number of sub-intervals used for integration */
#endif        
        read_int(argc,argv, "--kmax", &kmax, &err);
        if (err > 0)
        {
            print_global("Command line argument '--kmax' not found\n");
            exit(0);
        }
        read_double(argc,argv, "--tol", &tol, &err);
        if (err > 0)
        {
            print_global("Command line argument '--tol' not found\n");
            exit(0);
        }

        /* Hardwire values */
        a = 0;
        b = 1;
    }  

    /* Setup mesh */
    range[0] = a;
    range[1] = b;
    int m = n_global;   /* Number of panels in each section */
    double h = (range[1] - range[0])/((double)m);
    double h2;
    h2 = h*h;


    /* ---------------------------------------------------------------
       Set up right hand side
    --------------------------------------------------------------- */
    zeros_array(m+2,&B);  /* Include ghost values */
    zeros_array(m+2,&F);
    linspace_array(range[0]-h/2.0,range[1]+h/2.0,m+2,&x);

    /* Left edge BC */
    double u0 = utrue(range[0]);
    B[0] = 2*u0;  

    /* Right edge BC */
    double u1 = utrue(range[1]);
    B[m+1] = 2*u1;       

    /* Compute right hand side */
    for(j = 1; j < m+1; j++)
    {
        F[j] = rhs(x[j]) - (B[j-1] - 2*B[j] + B[j+1])/h2;
    }

    /* ----------------------------------------------------------------
       Set up arrays and other vars needed for iterative method
    ---------------------------------------------------------------- */
    double *pk, *uk, *rk, *wk;

    zeros_array(m+2,&pk);
    zeros_array(m+2,&uk);  
    zeros_array(m+2,&rk);
    zeros_array(m+2,&wk);

    double alpha,beta;
    int it_cnt;
    double res;

    /* ----------------------------------------------------------------
       Start iterations
    ---------------------------------------------------------------- */
    for(j = 1; j < m+1; j++)
    {
        /* Compute residual rk - F - A*uk */
        rk[j] = F[j] - (uk[j-1] - 2*uk[j] + uk[j+1])/h2;    
        pk[j] = rk[j];        
    }        

    double bv[2] = {0,0};

#ifdef USE_GPU
    int bpg = get_blocksPerGrid();

    double *dev_pk, *dev_uk, *dev_rk, *dev_wk;
    CHECK(cudaMalloc((void**) &dev_pk, m*sizeof(double)));
    CHECK(cudaMalloc((void**) &dev_uk, m*sizeof(double)));
    CHECK(cudaMalloc((void**) &dev_rk, m*sizeof(double)));
    CHECK(cudaMalloc((void**) &dev_wk, m*sizeof(double)));

    double *dev_partial_c, *dev_partial_d;
    CHECK(cudaMalloc((void**) &dev_partial_c, bpg*sizeof(double) ) );
    CHECK(cudaMalloc((void**) &dev_partial_d, bpg*sizeof(double) ) );

    bv[0] = dot_norm(m, &rk[1], dev_rk,dev_partial_c);
#else
    for(j = 1; j < m+1; j++)
    {
        bv[0] += rk[j]*rk[j];
    }
#endif    
    
    for(k = 0; k < kmax; k++)
    {
        /* Left edge */
        pk[0] = -pk[1]; 

        /* Right edge */
        pk[m+1] = -pk[m];           

        for(j = 1; j < m+1; j++)
        {
            wk[j] = (pk[j-1] - 2*pk[j] + pk[j+1])/h2;
        }

        double av[2] = {0,0};
        av[0] = bv[0];
        for(j = 1; j < m+1; j++)
        {
            av[1] += pk[j]*wk[j];
        }

        alpha = av[0]/av[1];

        double norm_zk = 0;
        bv[1] = av[0];
#ifdef USE_GPU
        bv[0] = cg_loop(m, alpha, 
                       &pk[1], &uk[1], &rk[1], &wk[1],
                       dev_pk, dev_uk,dev_rk, dev_wk,
                       dev_partial_c, dev_partial_d,
                       &norm_zk);
#else        
        bv[0] = 0;
        double zk;
        for(j = 1; j < m+1; j++)
        {
            zk = alpha*pk[j];
            uk[j] = uk[j] + zk;
            rk[j] = rk[j] - alpha*wk[j];
            bv[0] += rk[j]*rk[j];
            norm_zk = fabs(zk) > norm_zk ? fabs(zk) : norm_zk;
        }
#endif        
        beta = bv[0]/bv[1];   /* (rkp1 dot rkp1)/(rk dot rk) */

        print_info("%8d %16.8e\n",k,norm_zk);

        /* save results for output */
        it_cnt = k+1;
        res = norm_zk;

        if (norm_zk < tol)
        {
            break;
        }
        for(j = 1; j < m+1; j++)
        {
            pk[j] = rk[j] + beta*pk[j];
        }
    }

    /* ----------------------------------------------------------------
       Calculate error and report results
    ---------------------------------------------------------------- */
    double err[3] = {0,0,0};
    for(j = 1; j < m+1; j++)
    {
        double udiff = uk[j] - utrue(x[j]);
        err[0] += fabs(udiff)*h;
        err[1] += fabs(udiff*udiff)*h;
        err[2] = fabs(udiff) > err[2] ? fabs(udiff) : err[2];
    }
    err[1] = sqrt(err[1]);    /* 2-norm */

    print_essential("%10d %10d %12.4e %12.4e %12.4e %12.4e\n",n_global,it_cnt, 
                 res, err[0],err[1],err[2]);

    delete_array((void**) &B);
    delete_array((void**) &F);
    delete_array((void**) &x);

    delete_array((void**) &pk);
    delete_array((void**) &uk);
    delete_array((void**) &rk);
    delete_array((void**) &wk);

#ifdef USE_GPU
    /* free memory on the gpu side */
    CHECK(cudaFree(dev_pk));
    CHECK(cudaFree(dev_uk));
    CHECK(cudaFree(dev_rk));
    CHECK(cudaFree(dev_wk));
    CHECK(cudaFree(dev_partial_c));
    CHECK(cudaFree(dev_partial_d));
#endif    

    return 0;

}

