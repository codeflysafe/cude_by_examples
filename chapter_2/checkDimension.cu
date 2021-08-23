#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkDimension(){
    printf("threadIdx:(%d,%d,%d), blockIdx:(%d,%d,%d),blockDim:(%d,%d,%d),gridDim:(%d,%d,%d)\n",
    threadIdx.x,threadIdx.y,threadIdx.z,blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,blockDim.z,
    gridDim.x,gridDim.y,gridDim.z);
}



int main(){
    // 数据总量
    int nElem = 6;

    dim3 block(3);
    dim3 grid((nElem + block.x - 1)/block.x);

    printf("grid:(%d,%d,%d)\n",grid.x,grid.y,grid.z);
    printf("block:(%d,%d,%d)\n",block.x,block.y,block.z);

    checkDimension<<<grid,block>>>();
    
    // reset device before exit;
    cudaDeviceReset();
    return 0;
}