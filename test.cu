//
// Created by sjhuang on 2021/8/21.
//

/************************************************************************/
/* cuda测试                                                                     */
/************************************************************************/
#include <iostream>
bool InitCUDA()
{
    int count;

    cudaGetDeviceCount(&count);//获得cuda设备的数量

    if(count == 0)
    {
        std::cout<<"There is no device.\n" ;
        return false;
    }

    int i;

    for(i = 0; i < count; i++)
    {
        cudaDeviceProp prop;//cuda设备属性对象

        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess)
        {
            std::cout<<"设备名称："<<prop.name<<"\n" ;
            std::cout<<"计算能力的主代号："<<prop.major<<"\t"<<"计算能力的次代号："<<prop.minor<<"\n" ;
            std::cout<<"时钟频率："<<prop.clockRate<<"\n" ;

            std::cout<<"设备上多处理器的数量："<<prop.multiProcessorCount<<"\n" ;
            std::cout<<"GPU是否支持同时执行多个核心程序:"<<prop.concurrentKernels<<"\n" ;
        }
    }

    cudaSetDevice(i);//启动设备

    return true;
}

int main()
{
    if(!InitCUDA())
    {
        return 0;
    }

    std::cout<<"cuda配置成功！\n" ;
    return 0;
}