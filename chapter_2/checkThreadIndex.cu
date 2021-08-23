#include <stdio.h>
#include <cuda_runtime.h>


void printMatrix(int *C, const int nx, const int ny){
    int *ic = C;
    printf("\n Matrix:(%d, %d)\n",nx,ny);
    for(int i =0; i < ny; i++){
        for(int j =0; j < nx; j++){
            printf("%3d",ic[j + i*nx]);
        }
        printf("\n");
    }
    printf("\n");
    
}

__global__ void printThreadIndex(int *A, const int nx, const int ny){
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int idx = ix + iy*nx;
    printf("thread_id:(%d,%d),block_id:(%d,%d), coordinate(%d,%d) global index %2d ival %2d\n",
    threadIdx.x,threadIdx.y, blockIdx.x,blockIdx.y,ix,iy,idx,A[idx]);
}

void initInt(int *ip, int size){
    for(int i =0; i < size; i++) ip[i] = i;
}

int main(){
    printf("Starting ....\n");

    // set device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaSetDevice(dev);

    // set matrix dimension
    int nx = 8, ny = 6;
    int nxy = nx * ny, nBytes = nxy*sizeof(int);

    // malloc host memory
    int *h_A;
    h_A = (int *)malloc(nBytes);

    // init host matrix with integer
    initInt(h_A,nxy);
    printMatrix(h_A,nx,ny);

    // malloc device memory
    int *d_A;
    cudaMalloc((void **)&d_A, nBytes);

    // cp data from host to device
    cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);

    // set up execution configuration
    dim3 block(4,2);
    dim3 grid((nx + block.x - 1)/block.x, (ny + block.y -1)/block.y);

    // invoke the kernel
    printThreadIndex<<<grid,block>>>(d_A,nx,ny);
    cudaDeviceSynchronize();

    // free host and device memory
    cudaFree(d_A);
    free(h_A);


    // reset device
    cudaDeviceReset();
    return 0;
}