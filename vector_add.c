//
// Created by sjhuang on 2021/8/21.
//
#include<stdio.h>
#include<stdlib.h>
#define N 100000

void vector_add(const float *a, const float *b, float *output,int n){
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
    output = (float *)malloc(sizeof (float )*N);

    // Initialize array
    for(int i =0; i < N; i++){
        a[i] = 1.0f,b[i] = 2.0f;
    }

    // operate
    vector_add(a,b,output,N);

    // output
    vector_output(output,100);

    free(a);
    free(b);
    free(output);
}