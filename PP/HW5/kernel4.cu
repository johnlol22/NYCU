#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ int mandel(float c_re, float c_im, int maxIterations) {
    float z_re = c_re;
    float z_im = c_im;
    int i;
    for (i = 0; i < maxIterations; ++i) {
        if (z_re * z_re + z_im * z_im > 4.0f)
            break;
        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.0f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, 
                            int* d_img, size_t pitch, int resX, int resY, int maxIterations) {
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (thisX >= resX || thisY >= resY)
        return;

    float x = lowerX + thisX * stepX;
    float y = lowerY + thisY * stepY;
    
    int* row = (int*)((char*)d_img + thisY * pitch);
    row[thisX] = mandel(x, y, maxIterations);
}

void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations) {
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int* h_img = nullptr;
    cudaHostAlloc(&h_img, resX * resY * sizeof(int), cudaHostAllocDefault);// arguments (device pointer, allocated size, properties of allocated memory)
    

    int* d_img = nullptr;
    size_t pitch;   // the size in bytes of one line
    cudaMallocPitch(&d_img, &pitch, resX * sizeof(int), resY);

    cudaMemset2D(d_img, pitch, 0, resX * sizeof(int), resY);  // arguments (pointer to 2D device memory,
                                                                    //             bytes in one line, 
                                                                    //              value to set for each byte of specified memory, 
                                                                    //              width (in byte),
                                                                    //              height)

    dim3 block(16, 16);
    dim3 grid((resX + block.x - 1) / block.x, 
              (resY + block.y - 1) / block.y);


    mandelKernel<<<grid, block>>>(lowerX, lowerY, stepX, stepY, 
                                 d_img, pitch, resX, resY, maxIterations);
    

    cudaDeviceSynchronize();

    cudaMemcpy2D(h_img, resX * sizeof(int), d_img, pitch,
                       resX * sizeof(int), resY, cudaMemcpyDeviceToHost);


    memcpy(img, h_img, resX * resY * sizeof(int));

    cudaFree(d_img);
    cudaFreeHost(h_img);
}