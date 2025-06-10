#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    size_t imageSize = imageWidth * imageHeight * sizeof(float);

    // Create command queue
    cl_command_queue cmdQueue = clCreateCommandQueue(*context, *device, 0, &status);

    // Create memory buffers with appropriate flags
    cl_mem inputBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      imageSize, inputImage, &status);

    cl_mem outputBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY,
                                       imageSize, NULL, &status);

    cl_mem filterBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       filterSize * sizeof(float), filter, &status);


    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);

    // Set kernel arguments
    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &filterBuffer);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &filterWidth);
    status |= clSetKernelArg(kernel, 4, sizeof(int), &imageWidth);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &imageHeight);


    // Define work dimensions
    size_t globalWorkSize[2] = {imageWidth, imageHeight};
    // Choose work group size based on device capabilities
    size_t localWorkSize[2] = {16, 16};

    // Make sure local work size divides global work size
    globalWorkSize[0] = ((globalWorkSize[0] + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
    globalWorkSize[1] = ((globalWorkSize[1] + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1];

    // Execute the kernel
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, globalWorkSize,
                                   localWorkSize, 0, NULL, NULL);

    // Wait for kernel completion
    clFinish(cmdQueue);

    // Read the output buffer back to the host
    status = clEnqueueReadBuffer(cmdQueue, outputBuffer, CL_TRUE, 0,
                                imageSize, outputImage, 0, NULL, NULL);

    // Cleanup
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(filterBuffer);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(cmdQueue);
}