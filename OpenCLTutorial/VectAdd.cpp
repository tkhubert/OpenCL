//
//  VectAdd_CPP.cpp
//  OpenCLTutorial
//
//  Created by Thomas Hubert on 01/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include "util.hpp"
#include "err_code.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c

int main(void)
{
    std::vector<float> h_a(LENGTH);                // a vector
    std::vector<float> h_b(LENGTH);                // b vector
    std::vector<float> h_c(LENGTH);                // c = a + b, from compute device
    
    // Fill vectors a and b with random float values
    int count = LENGTH;
    for(int i = 0; i < count; i++)
    {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    try
    {
        // create a context
        cl::Context context(DEVICE);
        
        // Load in kernel source, creating a program object for the context
        cl::Program program(context, util::loadProgram("/Users/tkhubert/Documents/Etude/10.HeterogeneousParallelProgramming/OpenCL/OpenCLTutorial/OpenCLTutorial/vadd.cl"), true);
        
        // Get the command queue
        cl::CommandQueue queue(context);
        
        // Create the kernel functor
        auto vadd = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "vadd");

        cl::Buffer d_a(context, h_a.begin(), h_a.end(), true);
        cl::Buffer d_b(context, h_b.begin(), h_b.end(), true);
        cl::Buffer d_c(context, CL_MEM_WRITE_ONLY, sizeof(float)*count);
        
        util::Timer timer;
        vadd(cl::EnqueueArgs(queue, cl::NDRange(count)), d_a, d_b, d_c, count);
        queue.finish();
        double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
        printf("\nThe kernels ran in %lf seconds\n", rtime);
        
        cl::copy(queue, d_c, h_c.begin(), h_c.end());
        
        // Test the results
        int correct = 0;
        float tmp;
        for(int i = 0; i < count; i++)
        {
            tmp  = h_a[i] + h_b[i];
            tmp -= h_c[i];
            if(tmp*tmp < TOL*TOL)
                correct++;
            else
                printf(" tmp %f h_a %f h_b %f  h_c %f \n", tmp, h_a[i], h_b[i], h_c[i]);
        }
        
        // summarize results
        printf("vector add to find C = A+B:  %d out of %d results were correct.\n", correct, count);
    }
    catch (cl::Error err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")" << std::endl;
    }

    return 0;
    
}