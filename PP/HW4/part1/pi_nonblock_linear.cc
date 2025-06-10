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
    long long int inter_count[world_size] = {0};
    long long int count = 0;

    const double scale = 2.0 / (RAND_MAX + 1.0);
    int round = tosses/world_size;
    unsigned int seed = time(NULL)*world_rank;

    if (world_rank > 0)
    {
        // TODO: MPI workers
        double x, y;
        for (int i = 0;i < round;i++)
        {
            x = scale * rand_r(&seed) - 1.0;
            y = scale * rand_r(&seed) - 1.0;
            if (x*x+y*y <= 1.0)
                inter_count[world_rank]++;
        }
        MPI_Send(&inter_count[world_rank], 1, MPI_LONG_LONG, 0, tag, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request requests[world_size-1];
        for (int i=1;i<world_size;i++)
        {
            MPI_Irecv(&inter_count[i], 1, MPI_LONG_LONG, i, tag, MPI_COMM_WORLD, &requests[i-1]);
        }
        // world rank 0 do his job
        double x, y;
        for (int i = 0;i < round;i++)
        {
            x = scale * rand_r(&seed) - 1.0;
            y = scale * rand_r(&seed) - 1.0;
            if (x*x+y*y <= 1.0)
                inter_count[world_rank]++;
        }

        MPI_Waitall(world_size-1, requests, MPI_STATUSES_IGNORE);
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        for (int i=0;i<world_size;i++)
        {
            count += inter_count[i];
        }
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
