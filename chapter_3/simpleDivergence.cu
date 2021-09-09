#include <stdio.h>

using namespace std;


__global__ void mathKernel(float *c){
    int idx = threadIdx.x + threadIdx.y*blockDim.x;
    float a,b;
    a = b = 0.0f;
    if(idx % 2 == 0) a = 100.f;
    else b = 200.f;
    c[idx] = a + b;
}


__global__ void mathKernel2(float *c){
    int idx = threadIdx.x + threadIdx.y * blockDim.x;
    float a = 0.0f, b = a;
    if((idx/warpSize)%2 == 0){
        a = 100.f;
    }else b = 200.f;
    c[idx] = a + b;
}

int main(){

    int dev = 1;
    cudaSetDevice(dev);

    int nElem = 64;
    int size = nElem*sizeof(float);
    float *h_c;
    h_c = (float *)malloc(size);
    
    float *d_c;
    cudaMalloc((void **)&d_c,size);

    int blockSize  = 64;
    dim3 block(blockSize,1);
    dim3 grid((size + block.x - 1)/block.x,1);
    mathKernel<<<block,grid>>>(d_c);
    cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost);
    mathKernel2<<<block,grid>>>(d_c);
    cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost);

    free(h_c);
    cudaFree(d_c);
    return 0;
}