#include <demo_util.h>
#include <cuda_util.h>

#include <math.h>
#include <string.h>   /* For atoi */

#define PI 3.14159265358979323846264338327

/* From CG kernels */
int get_N();
int get_blocksPerGrid();
double dot_norm(int N, double *a, double *dev_a,
               double *dev_partial_c);


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

    if (my_rank == 0)
    {
        /* Input */
        int err;
#if 0        
        int mp;
        read_int(argc,argv, "-m", &mp, &err);
        if (err > 0)
        {
            print_global("Command line argument '-m' not found\n");
            exit(0);
        }        
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

        // n_global = 1 << mp;     /* Number of sub-intervals used for integration */
        n_global = get_N();

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
    double *rk, *rkp1, *uk, *pk, *wk, *ukp1;

    zeros_array(m+2,&rk);
    zeros_array(m+2,&rkp1);
    zeros_array(m+2,&uk);  /* Initial conditions */
    zeros_array(m+2,&ukp1);  /* Initial conditions */
    zeros_array(m+2,&wk);
    zeros_array(m+2,&pk);

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

    double *dev_rk, *dev_partial_c;
    int bpg = get_blocksPerGrid();
    CHECK(cudaMalloc((void**) &dev_rk, m*sizeof(double)));
    CHECK(cudaMalloc((void**) &dev_partial_c, bpg*sizeof(double) ) );


    for(k = 0; k < kmax; k++)
    {

        /* Compute dot(rk,rk) */
        double a[2] = {0,0};
        a[0] = dot_norm(m, &rk[1], dev_rk,dev_partial_c);

        /* Left edge */
        pk[0] = -pk[1]; 

        /* Right edge */
        pk[m+1] = -pk[m];           

        for(j = 1; j < m+1; j++)
        {
            wk[j] = (pk[j-1] - 2*pk[j] + pk[j+1])/h2;
        }

        for(j = 1; j < m+1; j++)
        {
            // a[0] += rk[j]*rk[j];
            a[1] += pk[j]*wk[j];
        }

        alpha = a[0]/a[1];

        double b[2] = {0,0};
        double norm_zk = 0, zk;
        for(j = 1; j < m+1; j++)
        {
            zk = alpha*pk[j];
            ukp1[j] = uk[j] + zk;
            rkp1[j] = rk[j] - alpha*wk[j];
            b[0] += rkp1[j]*rkp1[j];
            norm_zk = fabs(zk) > norm_zk ? fabs(zk) : norm_zk;
        }
        b[1] = a[0];
        beta = b[0]/b[1];
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
            pk[j] = rkp1[j] + beta*pk[j];
            rk[j] = rkp1[j];
            uk[j] = ukp1[j];
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
    delete_array((void**) &uk);
    delete_array((void**) &ukp1);
    delete_array((void**) &wk);
    delete_array((void**) &pk);
    delete_array((void**) &rk);
    delete_array((void**) &rkp1);

    /* free memory on the gpu side */
    CHECK(cudaFree(dev_rk));
    CHECK(cudaFree(dev_partial_c));



    return 0;

}

