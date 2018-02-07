#include "demo_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <stdarg.h>   /* for va_list, va_start, va_end, vprintf */
#include <string.h>   /* For atoi */

static int s_rank;

void set_rank(int  rank)
{
    s_rank = rank;
}

void sleep(double t_total)
{
    double t0, t1;
    t0 = clock();
    t1 = t0;
    while ((t1-t0)/CLOCKS_PER_SEC < t_total)
    {
        t1 = clock();
    }
}

double random_number()
{
  return (double) rand() / (double) RAND_MAX ;
}

void random_seed()
{
    srand(time(NULL));
}

int pow2(int p)
{
    /* Compute n = 2^p */
    int n,i;

    n = 1;
    for(i = 0; i < p; i++)
    {
        n *= 2;
    }
    return n;
}

/* Arrays */

void empty_array(int n,double **x)
{
    *x = malloc(n*sizeof(double));
}

void ones_array(int n,double **x)
{
    int i;
    *x = malloc(n*sizeof(double));
    for(i = 0; i < n; i++)
    {
        (*x)[i] = 1;
    }
}

void random_array(int n, double **array)
{
    *array = malloc(n*sizeof(double));    
    random_seed();   
    int i;

    for(i=0; i < n; i++)
    {
        (*array)[i] = random_number();
    }
}

void delete_array(double **x)
{
    free(*x);
}


double sum_array(int n, double *x)
{
    int i;
    double s;

    s = 0;
    for(i = 0; i < n; i++)
    {
        s += x[i];
    }
    return s;
}


void read_int(int argc, char** argv, char arg[], int* value)
{
    int arg_index = 1;     /* Skip first argument */
    while (arg_index < argc)
    {
        if (strcmp(argv[arg_index], arg) == 0)
        {
            arg_index++;
            *value = atoi(argv[arg_index++]);            
            return;
        }
        else
        {
            arg_index++;
        }
    }
}



/* Global print */

void print_global(const char* format, ... )
{
    /* Only print if on processor 0 */
    if (s_rank == 0)
    {
        va_list arglist;
        printf( "Processor [0] : " );
        va_start( arglist, format );
        vprintf( format, arglist );
        va_end( arglist );
    }
}

void print_debug(const char* format, ... )
{
    /* Include rank number in print statement */
    va_list arglist;
    printf( "Processor [%d] : ",s_rank);
    va_start( arglist, format );
    vprintf( format, arglist );
    va_end( arglist );
}
