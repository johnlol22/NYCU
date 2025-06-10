#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Status status;
    int tag =  0;
    long long int inter_count = 0;
    long long int count = 0;

    const double scale = 2.0 / (RAND_MAX + 1.0);
    int round = tosses/world_size;
    unsigned int seed = time(NULL)*world_rank;

    MPI_Win win;
    MPI_Win_create(world_rank == 0 ? &count : NULL, 
        world_rank == 0 ? sizeof(long long int) : 0, 
        sizeof(int),
        MPI_INFO_NULL,
        MPI_COMM_WORLD,
        &win);
        
    double x, y;
    for (int i = 0;i < round;i++)
    {
        x = scale * rand_r(&seed) - 1.0;
        y = scale * rand_r(&seed) - 1.0;
        if (x*x+y*y <= 1.0)
            inter_count++;
    }
    MPI_Win_fence(0, win);
    if (world_rank == 0)
    {
        // Master
        count = inter_count;
    }
    else
    {
        // Workers
        MPI_Accumulate(&inter_count, 1, MPI_LONG_LONG, 0, 0, 1, MPI_LONG_LONG, MPI_SUM, win);
    }
    MPI_Win_fence(0, win);

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result = 4 * count / (double)tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}