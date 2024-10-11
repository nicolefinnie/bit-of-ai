#include <iostream>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include "avx_mul.hpp"
#include <cstring>


bool compare_matrices(const std::vector<float>& C_native, const std::vector<float, AlignedAllocator<float, 32>>& C_SIMD){
    if (C_native.size() != C_SIMD.size()){
        return false;
    }
    for (int i=0; i<C_native.size(); i++){
        if (std::abs(C_native[i] - C_SIMD[i]) > 1e-5){
            return false;
        }
    }
    return true;
}

void measure_execution_time(const std::vector<float, AlignedAllocator<float, 32>>& A,
                            const std::vector<float, AlignedAllocator<float, 32>>& B, 
                            std::vector<float, AlignedAllocator<float, 32>>& C_SIMD, 
                            const std::vector<float>& A_native,
                            const std::vector<float>& B_native,
                            std::vector<float>& C_native,
                            const int M, const int N, const int K){

    auto start = std::chrono::high_resolution_clock::now();
    if (K % 8 == 0){
        mamul_SIMD(A, B, C_SIMD, M, N, K);
    } else {
        mamul_SIMD_unaligned(A, B, C_SIMD, M, N, K);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;

    std::cout << "SIMD matrix multiplication took: " << diff.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    mamul(A_native, B_native, C_native, M, N, K); 
    end = std::chrono::high_resolution_clock::now();
    diff = end-start;

    std::cout << "Native matrix multiplication took: " << diff.count() << " seconds\n";

    //std::cout << "Native" << std::endl;
    //print_native(C_native, M, N);

    //std::cout << "SIMD" << std::endl;
    //print_SIMD(C_SIMD, M, N);

    if (compare_matrices(C_native, C_SIMD)){
        std::cout << "Matrices are equal" << std::endl;
    } else {
        std::cout << "Matrices are not equal" << std::endl;
    }

}


int main() {
    int M = 99, K = 257, N = 99;
    // row-major matrix A and column-major matrix B
    std::vector<float, AlignedAllocator<float,32>> A(M * K), B(K * N), C_SIMD(M * N);
    std::vector<float> A_native(M * K), B_native(K*N), C_native(M*N);

    for(int i=0; i<M; i++){
        for (int j=0; j<K; j++){
            A[i*K+j] = static_cast<float>(j+1);
            A_native[i*K+j] = static_cast<float>(j+1);
            
        }
    }
    // column major
    for(int i=0; i<K; i++){
        for (int j=0; j<N; j++){
            B[i*N+j] = static_cast<float>(i+1);
            B_native[i*N+j] = static_cast<float>(i+1);
        }
    }

    transpose(B, K, N);
    measure_execution_time(A, B, C_SIMD, A_native, B_native, C_native, M, N, K);
    
    return 0;
}
