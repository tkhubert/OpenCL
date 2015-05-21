//
//  VectAdd_C.c
//  OpenCLTutorial
//
//  Created by Thomas Hubert on 01/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

//pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

extern double wtime();       // returns time since some fixed past point (wtime.c)

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c

//------------------------------------------------------------------------------
//
// kernel:  vadd
// Purpose: Compute the elementwise sum c = a+b
// input: a and b float vectors of length count
// output: c float vector of length count holding the sum a + b
//
const char* KernelSource =
"__kernel void vadd(__global float* a, __global float* b, __global float* c, unsigned int size)"
"{"
"    int i = get_global_id(0);"
""
"    if (i<size)"
"        c[i] = a[i]+b[i];"
"}";


int vecAddC(void)
{
    int          err;               // error code returned from OpenCL calls
    
    float*       h_a = (float*) calloc(LENGTH, sizeof(float));       // a vector
    float*       h_b = (float*) calloc(LENGTH, sizeof(float));       // b vector
    float*       h_c = (float*) calloc(LENGTH, sizeof(float));       // c vector (a+b) returned from the compute device
    
    // Fill vectors a and b with random float values
    int i = 0;
    int count = LENGTH;
    for(i = 0; i < count; i++)
    {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    //float* h_d = (float*) calloc(LENGTH, sizeof(float));
    //double rtimeCPU = wtime();
    //for (i=0; i<count; ++i)
    //    h_d[i] = h_a[i]+h_b[i];
    //rtimeCPU = wtime() - rtimeCPU;
    //printf("\nThe kernel ran in %lf seconds\n",rtimeCPU);
    //free(h_d);
    
    // OPEN CL -------------------------------------------
    
    // Setup Platform
    // Find NbPlatforms
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");
    if (numPlatforms == 0)
    {
        printf("Found 0 platforms!\n");
        return EXIT_FAILURE;
    }
    
    // Get All Platforms
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Getting platforms");
    
    // Secure a GPU
    cl_device_id device_id;
    for (i = 0; i < numPlatforms; i++)
    {
        err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
        if (err == CL_SUCCESS)
        {
            break;
        }
    }
    
    if (device_id == NULL)
        checkError(err, "Finding a device");
    
    // Create a Context
    cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    checkError(err, "Creating context");
    
    // Create a Command Queue
    cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Creating command queue");
    
    // Create a Program
    cl_program program = clCreateProgramWithSource(context, 1, (const char**) &KernelSource, NULL, &err);
    checkError(err, "Creating program");
    
    // Build the Program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        
        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }
    
    // Create Kernel for programm
    cl_kernel k_vadd = clCreateKernel(program, "vadd", &err);
    checkError(err, "Creating kernel");
    
    // Create the device args
    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(float)*count, NULL, &err); checkError(err, "Creating buffer d_a");
    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(float)*count, NULL, &err); checkError(err, "Creating buffer d_b");
    cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*count, NULL, &err); checkError(err, "Creating buffer d_v");
    
    // Write a and b vectors into compute device memory
    err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float)*count, h_a, 0, NULL, NULL); checkError(err, "Copying h_a to device at d_a");
    err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float)*count, h_b, 0, NULL, NULL); checkError(err, "Copying h_b to device at d_a");
    
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(k_vadd, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(k_vadd, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(k_vadd, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(k_vadd, 3, sizeof(unsigned int), &count);
    checkError(err, "Setting kernel arguments");
    
    double rtime = wtime();
    
    size_t global = count;
    err = clEnqueueNDRangeKernel(commands, k_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
    
    checkError(err, "Enqueueing kernel");
    
    // Wait for the commands to complete before stopping the timer
    err = clFinish(commands);
    checkError(err, "Waiting for kernel to finish");
    
    rtime = wtime() - rtime;
    printf("\nThe kernel ran in %lf seconds\n",rtime);
    
    // Read back the results from the compute device
    err = clEnqueueReadBuffer(commands, d_c, CL_TRUE, 0, sizeof(float) * count, h_c, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array!\n%s\n", err_code(err));
        exit(1);
    }
    
    // Test the results
    unsigned int correct = 0;
    float tmp;
    
    for(i = 0; i < count; i++)
    {
        tmp  = h_a[i] + h_b[i];     // assign element i of a+b to tmp
        tmp -= h_c[i];              // compute deviation of expected and output result
        if(tmp*tmp < TOL*TOL)       // correct if square deviation is less than tolerance squared
            correct++;
        else
            printf(" tmp %f h_a %f h_b %f h_c %f \n",tmp, h_a[i], h_b[i], h_c[i]);
    }
    
    // summarise results
    printf("C = A+B:  %d out of %d results were correct.\n", correct, count);

    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}