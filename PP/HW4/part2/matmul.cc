#include <mpi.h>
#include <fstream>
#include <iostream>

// *********************************************
// ** ATTENTION: YOU CANNOT MODIFY THIS FILE. **
// *********************************************

// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from in
//
// in:        input stream of the matrix file
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(std::ifstream &in, int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr)
{
    int tmp;
    in >> *n_ptr >> *m_ptr >> *l_ptr;
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    *a_mat_ptr = new int[(*n_ptr) * (*m_ptr)];
    *b_mat_ptr = new int[(*m_ptr) * (*l_ptr)];
    
    for (int i=0;i<(*n_ptr)*(*m_ptr);i++)
    {
        in >> tmp;
        (*a_mat_ptr)[i] = tmp;
    }
    for (int i=0;i<(*m_ptr) * (*l_ptr);i++)
    {
        in >> tmp;
        (*b_mat_ptr)[i] = tmp;
    }
}
// Just matrix multiplication (your should output the result in this function)
// 
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat)
{
    int world_size, world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    int rows_per_proc = n / world_size;
    int remaining_rows = n % world_size;
    int my_rows = (world_rank < remaining_rows) ? rows_per_proc + 1 : rows_per_proc;
    int start_row = (world_rank < remaining_rows) ? 
                      world_rank * (rows_per_proc + 1) : 
                      world_rank * rows_per_proc + remaining_rows;
    

    int *local_result = new int[my_rows * l];
    
    for (int i = 0; i < my_rows; i++) {
        for (int j = 0; j < l; j++) {
            int sum = 0;
            for (int k = 0; k < m; k++) {
                sum += a_mat[(start_row + i) * m + k] * b_mat[k * l + j];
            }
            local_result[i * l + j] = sum;
        }
    }
    
    if (world_rank == 0) {
        for (int i = 0; i < my_rows; i++) {
            for (int j = 0; j < l; j++) {
                printf("%d ", local_result[i*l+j]);
            }
            printf("\n");
        }
        
        for (int i = 1; i < world_size; i++) {
            int src_rows = (i < remaining_rows) ? rows_per_proc + 1 : rows_per_proc;
            int *temp_result = new int[src_rows * l];
            MPI_Recv(temp_result, src_rows * l, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int j = 0; j < src_rows; j++) {
                for (int k = 0; k < l; k++) {
                    printf("%d ", temp_result[j*l+k]);
                }
                printf("\n");
            }
            delete[] temp_result;
        }
    } else {
        MPI_Send(local_result, my_rows * l, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    
    delete[] local_result;
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    delete[] a_mat;
    delete[] b_mat;
}
