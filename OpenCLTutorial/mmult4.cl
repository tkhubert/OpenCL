__kernel void mmult4(
    const unsigned int N,
    __global const float* restrict A,
    __global const float* restrict B,
    __global       float* restrict C,
    __local        float* restrict Aloc,
    __local        float* restrict Bloc)
{
    int TILE_WIDTH = 16;//get_local_size(0);
    int col   = get_global_id(0);
    int row   = get_global_id(1);
    int colL  = get_local_id (0);
    int rowL  = get_local_id (1);
    
    float val=0.;
    for (int t=0; t<(N-1)/TILE_WIDTH+1; ++t)
    {
        int  colA   = t*TILE_WIDTH+colL;
        int  rowB   = t*TILE_WIDTH+rowL;
        bool checkA = row<N && colA<N;
        bool checkB = col<N && rowB<N;
        
        Aloc[rowL*TILE_WIDTH+colL] = checkA ? A[row*N+colA] : 0.;
        Bloc[rowL*TILE_WIDTH+colL] = checkB ? B[rowB*N+col] : 0.;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k=0; k<TILE_WIDTH; ++k)
            val+=Aloc[rowL*TILE_WIDTH+k]*Bloc[k*TILE_WIDTH+colL];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (row<N && col<N)
        C[row*N+col] = val;
}