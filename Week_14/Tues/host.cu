#include <stdio.h>
#include <sys/time.h>

#define P (1 << 14)

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return (double) tp.tv_sec + (double)tp.tv_usec*1e-6;
}

void copymat_host_x(int m, int n, int* A, int *B)
{
    int ix,iy,idx;
    for(iy = 0; iy < n; iy++)
        for(ix = 0; ix < m; ix++)
        {
            idx = iy*m + ix;
            B[idx] = A[idx];
        }
}

void copymat_host_y(int m, int n, int* A, int *B)
{
    int ix,iy,idx;
    for(ix = 0; ix < m; ix++)
        for(iy = 0; iy < n; iy++)
        {
            idx = iy*m + ix;
            B[idx] = A[idx];
        }
}


int main(int argc, char** argv) 
{
    int *A, *B;
    size_t m, n, nbytes;
    double etime, start;

    m = 1 << 14;  
    n = 1 << 14;  
    nbytes = m*n*sizeof(int);

    printf("P = %d\n",P);

    A = (int*) malloc(nbytes);
    B = (int*) malloc(nbytes);

    start = cpuSecond();
#if 1
    copymat_host_x(m,n,A,B);
#else
    copymat_host_y(m,n,A,B);
#endif
    etime = cpuSecond() - start;
    printf("Host       %10.3g (s)\n",etime);


    free(A);
    free(B);

}


