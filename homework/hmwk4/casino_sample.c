#include "hmwk4.h"
#include <demo_util.h>

#include <mpi.h>
#include <math.h>
#include <stdio.h>  
#include <stddef.h>  /* offsetof */   

typedef struct 
{
    char suits[2][10];   /* "spades", "hearts", "diamonds", "clubs" */
    char values[2][3];      /* "K", "Q", "J", "10",...,"2", "A" */
    float bet;           /* Between $0.01 and $100.00 */
    int total;
} struct_player_t;

void build_player_type(MPI_Datatype *player_t)
{
    /* TODO : Build MPI Data type here */
}


void main(int argc, char** argv)
{
    /* Data arrays */
    int deck[52];
    int hand[2];
    struct_player_t player;


    /* File I/O */
    MPI_File   file;
    MPI_Status status;
    
    /* Data type */
    MPI_Datatype player_t;

    int rank, nprocs;

    int i,j;

    /* ---- MPI Initialization */
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    set_rank(rank);  
    read_loglevel(argc,argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* ---- Create and shuffle cards */
    if (rank == 0)
    {        
        random_seed();
        if (2*nprocs > 52)
        {
            print_essential("Too many players!\n");
            exit(0);
        }

        int deck_sorted[52];
        int i;
        for(i = 0; i < 52; i++)
        {
            deck_sorted[i] = i+1;
        }

        /* Shuffle the deck */

        /* TODO : Shuffle "deck_sorted" to get "deck" */
    }
    /* ---- Deal cards */

    /* TODO  : Call MPI_Scatter to deal the cards */


    /* ---- Build player type */
    build_player_type(&player_t);

    /* ---- Each processor needs to look at their hand, and save info */

    random_seed();
    const char suits[4][10] = {"spades","hearts","diamonds","clubs"};
    const char values[13][3] = {"A", "2", "3", "4", "5", "6", "7", "8", \
                                "9", "10", "J", "Q", "K"};

    player.total = 0;
    for(i = 0; i < 2; i++)
    {
        /* TODO : Set "suit", "value" fields in struct "player" for each card. 
           Compute  card total (sum of two card values) */
    }

    /* TODO : Make a bet and set "bet" field. */


    /* ---- Open file so we can write header and solution */
    MPI_File_open(MPI_COMM_WORLD, "blackjack.out", 
                  MPI_MODE_CREATE|MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &file);

    /* ---- TODO : Create view for this processor into file */

    /* ---- TODO : Set view (with offset for header) */

    /* ---- TODO : Write out file */

    /* ---- Clean up */
    MPI_File_close(&file);
    MPI_Type_free(&player_t);

    /* TODO : Clean up any other arrays */

    MPI_Finalize();

}