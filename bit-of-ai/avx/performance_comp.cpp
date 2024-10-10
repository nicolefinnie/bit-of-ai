#include <iostream>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include "avx_mul.hpp"

int main() {
    int M = 8, K = 16, N = 32;
    // row-major matrix A and column-major matrix B
    std::vector<float, AlignedAllocator<float, 32>> A(M*K), B(N*K), C(M*N);

    for(int i=0; i<M; i++){
        for (int j=0; j<K; j++){
            A[i*K+j] = static_cast<float>(i*K+j);
        }
    }
    for(int i=0; i<N; i++){
        for (int j=0; j<K; j++){
            B[i*K+j] = static_cast<float>(i*K+j);
        }
    }

    // std::cout << "Matrix A: ";
    // print_matrix(B, M, K);
    
 
    // std::cout << "Column-major Matrix B: ";
    // print_matrix(B, N, K);
    
    transpose(B, N, K);
    // std::cout << "Row-major Matrix B: ";
    // print_matrix(B, K, N);


    mamul_basic(A, B, C, M, N, K);
    std::cout << "Row-major Matrix C: ";
    print_matrix(C, M, N);

    return 0;
}
