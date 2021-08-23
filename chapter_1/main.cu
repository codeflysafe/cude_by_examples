#include <stdio.h>
#include<stdlib.h>
#define N 1000

#define MAX_ERR 1e-6

__global__ void add(int *a, int b, int c) {
   *a = b + c;
}


int main(){

    int count = 0;
    cudaGetDevice(&count);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,count);
    int code = cudaChooseDevice(&count,&prop);
    printf("%d,%d\n",code,count);
    int a = 0;
    int *dev_a;
    cudaMalloc((void **)&dev_a,sizeof(int));

    add<<<1,1>>>(dev_a,3,5);
    cudaDeviceSynchronize();
    cudaMemcpy(&a,dev_a, sizeof(int),cudaMemcpyDeviceToHost);

    printf("3 + 5  = %d\n",a);

    cudaFree(dev_a);
    return 0;
}

