//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix Multiplication driver
//
//  PURPOSE: This is a driver program to test various ways of computing
//           the product:
//
//                C  = A * B
//
//           A and B are set to constant matrices so we
//           can make a quick test of the multiplication.
//
//  USAGE:   The matrices are constant matrices, square and the order is
//           set as a constant, ORDER (see mult.h).
//
//  HISTORY: Written by Tim Mattson, August 2010
//           Modified by Simon McIntosh-Smith, September 2011
//           Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
//           Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//           Modified to assume square matrices by Simon McIntosh-Smith, Sep 2014
//
//------------------------------------------------------------------------------

#include "matmul.hpp"
#include "matrix_lib.hpp"
#include "util.hpp"
#include "err_code.h"
#include "device_picker.hpp"

int main(int argc, char *argv[])
{

    int N;                  // A[N][N], B[N][N], C[N][N]
    int size;               // Number of elements in each matrix


    double start_time;      // Starting time
    double run_time;        // Timing
    util::Timer timer;      // Timing

    N    = ORDER;
    size = N * N;

    std::vector<float> h_A(size); // Host memory for Matrix A
    std::vector<float> h_B(size); // Host memory for Matrix B
    std::vector<float> h_C(size); // Host memory for Matrix C

//--------------------------------------------------------------------------------
// Create a context and queue
//--------------------------------------------------------------------------------

    try
    {
        cl_uint deviceIndex = 1;
        parseArguments(argc, argv, &deviceIndex);

        // Get list of devices
        std::vector<cl::Device> devices;
        unsigned numDevices = getDeviceList(devices);

        // Check device index in range
        if (deviceIndex >= numDevices)
        {
          std::cout << "Invalid device index (try '--list')\n";
          return EXIT_FAILURE;
        }

        cl::Device device = devices[deviceIndex];

        std::string name;
        getDeviceName(device, name);
        std::cout << "\nUsing OpenCL device: " << name << "\n";

        std::vector<cl::Device> chosen_device;
        chosen_device.push_back(device);
        cl::Context      context(chosen_device);
        cl::CommandQueue queue  (context, device);

//--------------------------------------------------------------------------------
// Run sequential matmul
//--------------------------------------------------------------------------------

        initmat(N, h_A, h_B, h_C);

        timer.reset();

        printf("\n===== Sequential, matrix mult (dot prod), order %d on host CPU ======\n",N);
        for(int i = 0; i < COUNT; i++)
        {
            zero_mat(N, h_C);

            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
            seq_mat_mul_sdot(N, h_A, h_B, h_C);
            run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;
            results(N, h_C, run_time);
        }

//--------------------------------------------------------------------------------
// Setup the buffers, initialize matrices, and write them into global memory
//--------------------------------------------------------------------------------

        //  Reset A, B and C matrices (just to play it safe)
        initmat(N, h_A, h_B, h_C);

        cl::Buffer d_a(context, h_A.begin(), h_A.end(), true);
        cl::Buffer d_b(context, h_B.begin(), h_B.end(), true);
        cl::Buffer d_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * size);


        // Create the compute program from the source buffer
        cl::Program program0(context, util::loadProgram("/Users/tkhubert/Documents/Etude/10.HeterogeneousParallelProgramming/OpenCL/OpenCLTutorial/OpenCLTutorial/mmult0.cl"), true);
        cl::Program program1(context, util::loadProgram("/Users/tkhubert/Documents/Etude/10.HeterogeneousParallelProgramming/OpenCL/OpenCLTutorial/OpenCLTutorial/mmult1.cl"), true);
        cl::Program program2(context, util::loadProgram("/Users/tkhubert/Documents/Etude/10.HeterogeneousParallelProgramming/OpenCL/OpenCLTutorial/OpenCLTutorial/mmult2.cl"), true);
        cl::Program program3(context, util::loadProgram("/Users/tkhubert/Documents/Etude/10.HeterogeneousParallelProgramming/OpenCL/OpenCLTutorial/OpenCLTutorial/mmult3.cl"), true);
        cl::Program program4(context, util::loadProgram("/Users/tkhubert/Documents/Etude/10.HeterogeneousParallelProgramming/OpenCL/OpenCLTutorial/OpenCLTutorial/mmult4.cl"), true);

        // Create the compute kernel from the program
        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer>                    mmult0(program0, "mmult0");
        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer>                    mmult1(program1, "mmult1");
        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer>                    mmult2(program2, "mmult2");
        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg> mmult3(program3, "mmult3");
        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl::LocalSpaceArg> mmult4(program4, "mmult4");

//--------------------------------------------------------------------------------
// OpenCL matrix multiplication ... Naive
//--------------------------------------------------------------------------------
        printf("\n===== OpenCL, matrix mult, C(i,j) per work item, order %d ======\n",N);
        for (int i = 0; i < COUNT; i++)
        {
            zero_mat(N, h_C);

            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
            cl::NDRange global0(N, N);
            mmult0(cl::EnqueueArgs(queue, global0), N, d_a, d_b, d_c);
            queue.finish();
            run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;

            cl::copy(queue, d_c, h_C.begin(), h_C.end());
            results(N, h_C, run_time);
        }
        
        
//--------------------------------------------------------------------------------
// OpenCL matrix multiplication ... C row per work item
//--------------------------------------------------------------------------------
        printf("\n===== OpenCL, matrix mult, C row per work item, order %d ======\n",N);
        for (int i = 0; i < COUNT; i++)
        {
            zero_mat(N, h_C);
            
            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
            cl::NDRange global(N);
            mmult1(cl::EnqueueArgs(queue, global), N, d_a, d_b, d_c);
            queue.finish();
            run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;
            
            cl::copy(queue, d_c, h_C.begin(), h_C.end());
            results(N, h_C, run_time);
        }
        
        
//--------------------------------------------------------------------------------
// OpenCL matrix multiplication ... C row per work item, A row in pivate memory
//--------------------------------------------------------------------------------
        printf("\n===== OpenCL, matrix mult, C row, A row in priv mem, order %d ======\n",N);
        for (int i = 0; i < COUNT; i++)
        {
            zero_mat(N, h_C);
            
            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
            cl::NDRange global(N);
            cl::NDRange local (ORDER/16);
            mmult2(cl::EnqueueArgs(queue, global, local), N, d_a, d_b, d_c);
            queue.finish();
            run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;
            
            cl::copy(queue, d_c, h_C.begin(), h_C.end());
            results(N, h_C, run_time);
        } // end for loop
        
        
//--------------------------------------------------------------------------------
// OpenCL matrix multiplication ... C row per work item, A row pivate, B col local
//--------------------------------------------------------------------------------
        printf("\n===== OpenCL, mat mult, C row, priv A, B cols loc, order %d ======\n",N);
        for (int i = 0; i < COUNT; i++)
        {
            zero_mat(N, h_C);
            
            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
            cl::NDRange global(N);
            cl::NDRange local (ORDER/16);
            cl::LocalSpaceArg localMem = cl::Local(sizeof(float)*ORDER);
            mmult3(cl::EnqueueArgs(queue, global, local), N, d_a, d_b, d_c, localMem);
            queue.finish();
            run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;
            
            cl::copy(queue, d_c, h_C.begin(), h_C.end());
            results(N, h_C, run_time);
        }
        
//--------------------------------------------------------------------------------
// OpenCL matrix multiplication ... blocked
//--------------------------------------------------------------------------------
        printf("\n===== Parallel matrix mult (blocked), order %d on device ======\n",N);
        for (int i = 0; i < COUNT; i++)
        {
            zero_mat(N, h_C);
            
            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
            int TILE_WIDTH = 16;
            int GLOBAL_SIZE = ((N-1)/TILE_WIDTH+1)*TILE_WIDTH;
            cl::NDRange global(GLOBAL_SIZE, GLOBAL_SIZE);
            cl::NDRange local (TILE_WIDTH , TILE_WIDTH);
            cl::LocalSpaceArg Aloc = cl::Local(sizeof(float)*TILE_WIDTH*TILE_WIDTH);
            cl::LocalSpaceArg Bloc = cl::Local(sizeof(float)*TILE_WIDTH*TILE_WIDTH);
            mmult4(cl::EnqueueArgs(queue, global, local), N, d_a, d_b, d_c, Aloc, Bloc);
            queue.finish();
            run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;
            
            cl::copy(queue, d_c, h_C.begin(), h_C.end());
            results(N, h_C, run_time);
        }
    }
    catch (cl::Error err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: "
                  << err.what()
                  << "("
                  << err_code(err.err())
                  << ")"
                  << std::endl;
    }

    return EXIT_SUCCESS;
}
