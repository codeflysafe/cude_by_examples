#include <cuda_runtime.h>
#include <stdio.h>

// debug 模式启动
int main(){

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Device %d: %s \n",dev,deviceProp.name);
    printf("Total amount of global memory %2.f Mbytes\n",deviceProp.totalGlobalMem/(pow(1024.0,3)));

    return 0;
}