__kernel void mmult2(int N, __global float* A, __global float* B, __global float* C)
{
    int i = get_global_id(0);
    
    if (i<N)
    {
        float Apriv[1024];
        for (int k=0; k<N; ++k)
            Apriv[k] = A[i*N+k];
        
        for (int j=0; j<N; ++j)
        {
            float tmp = 0;
            for (int k=0; k<N; ++k)
            {
                tmp += Apriv[k]*B[k*N+j];
            }
            C[i*N+j] = tmp;
        }
    }
}