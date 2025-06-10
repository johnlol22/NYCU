#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ int mandel(float c_re, float c_im, int maxIterations) {
    float z_re = c_re;
    float z_im = c_im;
    
    for (int i = 0; i < maxIterations; ++i) {
        float z_re2 = z_re * z_re;
        float z_im2 = z_im * z_im;
        
        if (z_re2 + z_im2 > 4.0f) {
            return i;
        }
        
        z_im = 2.0f * z_re * z_im + c_im;
        z_re = z_re2 - z_im2 + c_re;
    }
    return maxIterations;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, 
                            int* d_img, int resX, int resY, int maxIterations) {
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thisX >= resX || thisY >= resY)
        return;

    float x = lowerX + thisX * stepX;
    float y = lowerY + thisY * stepY;
    
    int index = thisY * resX + thisX;
    
    d_img[index] = mandel(x, y, maxIterations);
}

void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations) {
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;


    int* h_img = (int*)malloc(resX * resY * sizeof(int));

    memset(h_img, 0, resX * resY * sizeof(int));

    int* d_img = nullptr;
    size_t size = resX * resY * sizeof(int);
    cudaMalloc(&d_img, size);
    /*
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        free(h_img);
        return;
    }
    */

    cudaMemset(d_img, 0, size);
    /*if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset error: %s\n", cudaGetErrorString(err));
        cudaFree(d_img);
        free(h_img);
        return;
    }*/

    dim3 block(16, 16);  // 256 threads per block
    dim3 grid((resX + block.x - 1) / block.x, 
              (resY + block.y - 1) / block.y);

    mandelKernel<<<grid, block>>>(lowerX, lowerY, stepX, stepY, 
                                 d_img, resX, resY, maxIterations);
    
    /*
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_img);
        free(h_img);
        return;
    }*/

    cudaDeviceSynchronize();
    /*if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err));
        cudaFree(d_img);
        free(h_img);
        return;
    }*/

    cudaMemcpy(h_img, d_img, size, cudaMemcpyDeviceToHost);
    /*if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy error: %s\n", cudaGetErrorString(err));
        cudaFree(d_img);
        free(h_img);
        return;
    }*/

    memcpy(img, h_img, size);

    free(h_img);
    cudaFree(d_img);
}