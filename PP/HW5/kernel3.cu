#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define PIXELS_PER_THREAD_X 4
#define PIXELS_PER_THREAD_Y 4

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
                            int* d_img, size_t pitch, int resX, int resY, int maxIterations) {
    int baseX = (blockIdx.x * blockDim.x + threadIdx.x) * PIXELS_PER_THREAD_X;
    int baseY = (blockIdx.y * blockDim.y + threadIdx.y) * PIXELS_PER_THREAD_Y;

    for (int dy = 0; dy < PIXELS_PER_THREAD_Y && (baseY + dy) < resY; dy++) {
        int thisY = baseY + dy;
        int* row = (int*)((char*)d_img + thisY * pitch);

        for (int dx = 0; dx < PIXELS_PER_THREAD_X && (baseX + dx) < resX; dx++) {
            int thisX = baseX + dx;
            
            float x = lowerX + thisX * stepX;
            float y = lowerY + thisY * stepY;
            
            row[thisX] = mandel(x, y, maxIterations);
        }
    }
}

void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations) {
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int* h_img = nullptr;
    cudaHostAlloc(&h_img, resX * resY * sizeof(int), cudaHostAllocDefault);
    /*if (err != cudaSuccess) {
        fprintf(stderr, "cudaHostAlloc error: %s\n", cudaGetErrorString(err));
        return;
    }*/

    memset(h_img, 0, resX * resY * sizeof(int));

    int* d_img = nullptr;
    size_t pitch;
    cudaMallocPitch(&d_img, &pitch, resX * sizeof(int), resY);
    /*if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch error: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_img);
        return;
    }*/

    
    cudaMemset2D(d_img, pitch, 0, resX * sizeof(int), resY);
    /*if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset2D error: %s\n", cudaGetErrorString(err));
        cudaFree(d_img);
        cudaFreeHost(h_img);
        return;
    }*/


    dim3 block(8, 8);
    dim3 grid((resX + (block.x * PIXELS_PER_THREAD_X) - 1) / (block.x * PIXELS_PER_THREAD_X),
              (resY + (block.y * PIXELS_PER_THREAD_Y) - 1) / (block.y * PIXELS_PER_THREAD_Y));

    mandelKernel<<<grid, block>>>(lowerX, lowerY, stepX, stepY, 
                                 d_img, pitch, resX, resY, maxIterations);
    

    cudaDeviceSynchronize();

    cudaMemcpy2D(h_img, resX * sizeof(int), d_img, pitch,
                       resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    
    memcpy(img, h_img, resX * resY * sizeof(int));

    cudaFree(d_img);
    cudaFreeHost(h_img);
}