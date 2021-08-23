//
// Created by sjhuang on 2021/8/21.
//
#include<stdio.h>
#include<stdlib.h>
#define N 100000

__global__ void vector_add(const float *a, const float *b, float *output,int n){
    for(int i =0; i < n; i++){
        output[i] = a[i] + b[i];
    }
}

void vector_output(float *output, int n){
    for(int i =0; i < n; i++){
        printf("output[%d] is %f",i,output[i]);
    }
}

int main(){
    // malloc memory
    float *a,*b,*output;
    a = (float *)malloc(sizeof (float )*N);
    b = (float *)malloc(sizeof (float )*N);
    output = (float*)malloc(sizeof(float) * N);
    // Initialize array
    for(int i =0; i < N; i++){
        a[i] = 1.0f, b[i] = 2.0f;
    }

    float *d_a,*d_b,*d_output;
    // Device Memory malloc
    cudaMalloc(&d_a,sizeof (float )*N);
    cudaMalloc(&d_b,sizeof (float )*N);
    cudaMalloc(&d_output,sizeof (float )*N);
    // Transfer host data to device data
    cudaMemcpy(d_a,a,sizeof (float )*N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,sizeof (float )*N,cudaMemcpyHostToDevice);

    // operate
    vector_add<<<1,1>>>(d_a,d_b,d_output,N);
    cudaDeviceSynchronize();
    cudaMemcpy(output,d_output,sizeof (float )*N,cudaMemcpyDeviceToHost);

    // output
    printf("out[0] = %f\n", output[0]);
    printf("PASSED\n");
//
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_output);
    free(a);
    free(b);
    free(output);
    return 0;
}