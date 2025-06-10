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

    // TODO: init MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Status status;
    long long int count;
    long long int inter_count = 0;
    int tag = 0;
    unsigned int seed = time(NULL)*world_rank;

    const double scale = 2.0 / (RAND_MAX + 1.0);
    int round = tosses/world_size;

    double x, y;
    for (int i = 0;i < round;i++)
    {
        x = scale * rand_r(&seed) - 1.0;
        y = scale * rand_r(&seed) - 1.0;
        if (x*x+y*y <= 1.0)
            inter_count++;
    }

    if (world_rank > 0)
    {
        // TODO: handle workers
        MPI_Send(&inter_count, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }

    else
    {
        // TODO: process PI result
        count = inter_count;
        for (int i = 1 ; i < world_size ; i++)
        {
            MPI_Recv(&inter_count, 1, MPI_LONG_LONG, i, tag, MPI_COMM_WORLD, &status);
            count += inter_count;
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
