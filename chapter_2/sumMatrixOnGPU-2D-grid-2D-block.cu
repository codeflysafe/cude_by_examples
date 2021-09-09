#include <stdio.h>
#include <sys/time.h>
// 计时操作
double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


void sumMatrixOnHost(float *h_A, float *h_B, float *h_C, const int nx, const int ny){
    for(int i = 0; i < ny; i++){
        for(int j = 0; j < nx; j++){
            int idx = j + i*nx;
            h_C[idx] = h_A[idx] + h_B[idx];
        }
    }
}


void initMatrix(float *ip, const int size){
     for(int i =0; i < size; i++) ip[i] = i;
}


__global__ void sumMatrixOnDevice2D(float *d_A, float *d_B, float *d_C, const int nx, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = ix + iy*nx;
    if(idx < ny*nx) d_C[idx] = d_A[idx] + d_B[idx];
}

__global__ void sumMatrixOnDevice1D(float *d_A, float *d_B, float *d_C, const int nx, const int ny){
    unsigned int ix = threadIdx.x + blockDim.x*blockIdx.x;
    if(ix < nx){
        for(int i = 0; i < ny; i++) {
            int idx = ix + i*nx;
            d_C[idx] = d_A[idx] + d_B[idx];
        }
    }
}

void checkRef(float *h_Ref, float *d_Ref, const int nx, const int ny){
    double epsilon = 1.0e-8;
    bool match = true;
    for(int i = 0; i < ny; i++){
        for(int j = 0; j < nx; j++){
            int idx = j + i*nx;
            if(abs(d_Ref[idx]  - h_Ref[idx]) > epsilon){
                match = false;
                printf("cpu and gpu not match,\n %d, %5.3f, %5.3f\n",idx,d_Ref[idx],h_Ref[idx]);
                break;
            }
        }
    }
    if(match){
        printf("Result Match !\n");
    }
}

int main(){

    int dev = 1;
    cudaSetDevice(dev);
    // set size of data
    int nx = 1<<14;
    int ny = 1<<14;
    int nxy = nx*ny;
    int nBytes = sizeof(float)*nxy;

    float *h_A, *h_B, *h_Ref, *g_Ref;
    // malloc data
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_Ref = (float *)malloc(nBytes);
    g_Ref = (float *)malloc(nBytes);

    // printf("%f %f",h_B[0],h_B[0]);

    initMatrix(h_A,nxy);
    initMatrix(h_B,nxy);

    memset(h_Ref,0,nBytes);
    memset(g_Ref,0,nBytes);

    double iStart, iEp;
    iStart = cpuSecond();
    // 
    sumMatrixOnHost(h_A,h_B,h_Ref,nx, ny);
    // printf("h_Ref[1]: h_A[1] %f + h_B[1] %f = %f\n",h_A[1],h_B[1],h_Ref[1]);
    // return 0;
    iEp = cpuSecond() - iStart;
    printf(" sumMatrixOnHost time [%.3f]sec\n",iEp);

    // define d_A d_B, d_Ref
    float *d_A, *d_B, *d_Ref;
    cudaMalloc((void **)&d_A,nBytes);
    cudaMalloc((void **)&d_B,nBytes);
    cudaMalloc((void **)&d_Ref,nBytes);

    // 
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    
    int dimx = 128, dimy = 1;
    dim3 block(dimx,dimy);
    dim3 grid((nx + dimx - 1)/dimx, (ny + dimy - 1)/dimy);

    iStart = cpuSecond();
    sumMatrixOnDevice1D<<<grid,block>>>(d_A,d_B,d_Ref,nx,ny);
    cudaDeviceSynchronize();
    iEp = cpuSecond() - iStart;
    printf("sumMatrixOnDevice2D time <<<(%d,%d),(%d,%d)>>> [%.3f]sec\n",
    grid.x,grid.y,block.x,block.y,iEp);

    // cp data from device to host
    cudaMemcpy(g_Ref,d_Ref,nBytes,cudaMemcpyDeviceToHost);

    checkRef(g_Ref,h_Ref,nx,ny);

    free(h_A);
    free(h_B);
    free(h_Ref);
    free(g_Ref);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_Ref);

    cudaDeviceReset();

    return 0;
}