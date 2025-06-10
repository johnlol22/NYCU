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
    int base = 2;
    long long int inter_count = 0;
    long long int count = 0;

    const double scale = 2.0 / (RAND_MAX + 1.0);
    int round = tosses/world_size;
    unsigned int seed = time(NULL)*world_rank;

    //computing inter_count
    double x, y;
    for (int i = 0;i < round;i++)
    {
        x = scale * rand_r(&seed) - 1.0;
        y = scale * rand_r(&seed) - 1.0;
        if (x*x+y*y <= 1.0)
            inter_count++;
    }
    count = inter_count;
    // TODO: binary tree redunction
    while (base <= world_size)
    {
        if (world_rank % base == 0)
        {
            MPI_Recv(&inter_count, 1, MPI_LONG_LONG, world_rank+base/2, tag, MPI_COMM_WORLD, &status);
            count += inter_count;
            inter_count = count;
        }
        else if (world_rank % base == base/2)
        {
            MPI_Send(&inter_count, 1, MPI_LONG_LONG, world_rank-base/2, tag, MPI_COMM_WORLD);
        }
        base *= 2;
    }
    if (world_rank == 0)
    {
        // TODO: PI result
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
