#include <stdio.h>
#include <iostream>

#include "csrc/utils/device_mem_utils.h"


int main(int argc, char* argv[]){

    if(argc <= 3){
        std::cout << "at least give this bench 3 args, like 1024 2048 512, which means M, N, K are 1024 2048 512" << std::endl;
        return 0;
    }

    int32_t M, N, K;
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);

    bool transA = (argc >= 5 ? (atoi(argv[4]) != 0) : false);
    bool transB = (argc >= 6 ? (atoi(argv[5]) != 0) : false);

    printf("\nM = %d, N = %d, K = %d \n\n", M, N, K);

    int64_t size_A = M * K, size_B = K * N, size_C = M * N, size_Bias = N, size_Gain = N;

    half *d_A_fp16, *d_B_fp16, *d_C_fp16, *d_Bias_fp16, *d_Gain_fp16;
    deviceMalloc(&d_A_fp16, size_A);
    deviceMalloc(&d_B_fp16, size_B);
    deviceMalloc(&d_C_fp16, size_C);



    deviceFree(d_A_fp16);
    deviceFree(d_B_fp16);
    deviceFree(d_C_fp16);


    std::cout << "gemv test" << std::endl;
    return 0;

}
