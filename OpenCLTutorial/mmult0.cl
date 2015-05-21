
__kernel void mmult0(int N, __global float* A, __global float* B, __global float* C)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    float tmp;
    if (i<N && j<N)
    {
        tmp=0;
        for (int k=0; k<N; ++k)
            tmp+= A[i*N+k]*B[k*N+j];
    }
    C[i*N+j] = tmp;
}
