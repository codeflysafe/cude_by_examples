#include <stdlib.h>
#include <stdio.h>
#include <time.h>

__global__ void sumArrasOnDevice(float *A, float *B, float *C, const int N){
    int idx = threadIdx.x;
    C[idx] = A[idx] + B[idx];
}



void initialData(float *ip, int size){
    time_t t;
    srand((unsigned int) time(&t));

    for(int i =0; i < size; i++){
        ip[i] = (float)(rand() & 0xFF)/10.0f;
    }
}

int main(){
    int nElem = 10;
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    initialData(h_A,nElem);
    initialData(h_B,nElem);
     
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A,nBytes);
    cudaMalloc((float **)&d_B,nBytes);
    cudaMalloc((float **)&d_C,nBytes);


    // cp data from cpu to gpu
    cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice);
    
    // exec gpu 
    // sumArrasOnDevice(h_C,h_C,h_C,nElem);
    sumArrasOnDevice<<<1,nElem>>>(d_A,d_B,d_C,nElem);
    // wait all device finish! cudaMemcpy 内部还有同步，因此此处不需要加上 block
    // cudaDeviceReset();

    //
    cudaMemcpy(h_C,d_C,nBytes,cudaMemcpyDeviceToHost); 
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}