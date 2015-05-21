__kernel void mmult3(int N, __global float* A, __global float* B, __global float* C, __local float* Bloc)
{
    int i    = get_global_id (0);
    int iloc = get_local_id  (0);
    int nloc = get_local_size(0);
    
    if (i<N)
    {
        float Apriv[1024];
        for (int k=0; k<N; ++k)
            Apriv[k] = A[i*N+k];
        
        for (int j=0; j<N; ++j)
        {
            for (int k=iloc; k<N; k+=nloc)
                Bloc[k] = B[k*N+j];
            
            barrier(CLK_LOCAL_MEM_FENCE);
            
            float tmp = 0;
            for (int k=0; k<N; ++k)
                tmp += Apriv[k]*Bloc[k];
            C[i*N+j] = tmp;
            
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}