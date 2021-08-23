#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
// #define CHECK(call){
//     const cudaError_t error = call;
//     if(error != cudaSuccess){
//         printf("Error: %s:%d",__FILE__,__LINE__);
//         printf("code:%d, reason:%s\n",error,cudaGetErrorString(error));
//         exit(1);
//     }
// }

// block = N
__global__ void sumArraysOnDevice(float *A, float *B, float *C, const int N){
    int idx = threadIdx.x;
    if(idx < N) C[idx] = A[idx] + B[idx];
}

// grid = N
__global__ void sumArraysOnDeviceGrid(float *A,float *B,float *C, const int N){
    int idx = blockIdx.x;
    if(idx < N) C[idx] = A[idx] + B[idx];
}

__global__ void sumArraysOnDeviceCommon(float *A, float *B, float *C, const int N){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < N) C[idx] = A[idx] + B[idx];
}

void sumArraysOnHost(float *A, float *B, float *C, const int N){
    for(int i =0; i < N; i++){
        C[i] = A[i] + B[i];
    }
}

void checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1.0e-8;
    bool match = true;
    for(int i =0; i< N; i++){
        if(abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = false;
            printf("Result do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n",hostRef,gpuRef,i);
            break;
        }
    } 
    if(match) {
         printf("Result match!\n");
    }
}

void initialData(float *ip, int size){
    time_t t;
    srand((unsigned int) time(&t));

    for(int i =0; i < size; i++){
        ip[i] = (float)(rand() & 0xFF)/10.0f;
    }
}


// 计时操作
double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(){

    int dev = 0;
    cudaSetDevice(dev);
    
    // 扩大到 16M
    int nElem = 1<<24;
    printf("Vector size %d\n",nElem);
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *h_C,*h_Ref;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);
    h_Ref = (float *)malloc(nBytes);

    double iStart, iElaps;
    iStart = cpuSecond();
    initialData(h_A,nElem);
    initialData(h_B,nElem);
    iElaps = cpuSecond() - iStart;


    memset(h_C,0,nBytes);
    memset(h_Ref,0,nBytes);

    iStart = cpuSecond();
    // cpu exec
    sumArraysOnHost(h_A,h_B,h_Ref,nElem);
    iElaps = cpuSecond() - iStart;
    printf("sumArraysOnHost %d Time epapsed %f sec\n",nElem,iElaps);


    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A,nBytes);
    cudaMalloc((float **)&d_B,nBytes);
    cudaMalloc((float **)&d_C,nBytes);


    // cp data from cpu to gpu
    cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice);
    
    int iLen = 1024;
    dim3 block(iLen);
    dim3 grid((nElem + block.x -1)/block.x);

    // exec gpu 
    // sumArrasOnDevice(h_C,h_C,h_C,nElem);
    //sumArrasOnDevice<<<1,nElem>>>(d_A,d_B,d_C,nElem);
    // wait all device finish! cudaMemcpy 内部还有同步，因此此处不需要加上 block
    // cudaDeviceReset();
    iStart = cpuSecond();
    sumArraysOnDeviceCommon<<<grid,block>>>(d_A,d_B,d_C,nElem);
    cudaDeviceSynchronize();
    // why cudaDeviceReset 不行？ 对比一下
    iElaps = cpuSecond() - iStart;
    printf("sumArraysOnDeviceCommon <<<%d,%d>>> Time epapsed %f sec\n",grid.x,block.x,iElaps);

    //
    cudaMemcpy(h_C,d_C,nBytes,cudaMemcpyDeviceToHost); 

    checkResult(h_Ref,h_C,nElem);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}