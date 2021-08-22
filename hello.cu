#include <stdio.h>


void hello(){
    printf("hello world!\n");
}

__global__ void say_hello(){
    printf("[say_hello] Hello World from GPU! \n");
}

__global__ void say_hello_multi(){
    
    int idx = threadIdx.x;
    if(idx == 5) printf("[say_hello_multi] hello world from gpu [%d]\n",idx);
}

int main(){
    hello();
    say_hello<<<1,1>>>();
    say_hello_multi<<<1,6>>>();
    // cudaDeviceReset();
    // printf("[%d]",cudaDeviceGetLimit());
    cudaDeviceSynchronize();
    return 0;
}