// AVX matrix multiplication in SIMD. Only for educational purpose.
// matrix A: 8x16, matrix B: 16x32, and we want to compute C = A*B in SIMD
// We assume the matrix B is stored in column-major so we transpose it to another
// row-major matrix, so it's more efficient to access continguously in memory
// for matrix multiplication
#include <iostream>
#include "avx_mul.hpp"


// Basic matrix multiplication using AVX SIMD without unrolling
// AVX handles 8 floats or 4 doubles at a time (256 bit reigster)
void mamul_basic(const std::vector<float, AlignedAllocator<float, 32>>& A,
                const std::vector<float, AlignedAllocator<float, 32>>& B,
                std::vector<float, AlignedAllocator<float, 32>>& C, 
        const int M, const int N, const int K){
    std::fill(C.begin(), C.end(), 0);

    for (int i=0; i<M; i++){
        for (int j=0; j<N; j++){
            __m256 sum = _mm256_setzero_ps();
            int k=0;
            for (; k < K; k+=8){
                __m256 a = _mm256_load_ps(&A[i*K+k]);
                __m256 b = _mm256_load_ps(&B[k*N+j]);
                sum = _mm256_fmadd_ps(a, b, sum);
            }

            float tail_sum = 0;
            // Handling remainder 
            for(; k < K; ++k){
                tail_sum = A[i*K+k]*B[k*N+k];
            }
            C[i*N+j] = reduce_sum_avx256(sum) + tail_sum;
        }
    }
}


// Sum all elements in 8 floats in a vector 
// f0, f1, f2, f3, f4, f5, f6, f7 -> f0+f4, f1+f5, f2+f6, f3+f7
// f0+f4, f1+f5, f2+f6, f3+f7 -> f0+f4+f2+f6, f1+f5+f3+f7
// return the first float as the result
float reduce_sum_avx256(__m256 vec8){
    __m128 vec4_low = _mm256_extractf128_ps(vec8, 0); // lower 128 bits f0, f1, f2, f3
    __m128 vec4_high = _mm256_extractf128_ps(vec8, 1); // upper 128 bits f4, f5, f6, f7
    // add lower and upper 128 bits (f0+f4, f1+f5, f2+f6, f3+f7)
    __m128 vec4_sum = _mm_add_ps(vec4_low, vec4_high);

    vec4_sum = _mm_hadd_ps(vec4_sum, vec4_sum); // f0+f4+f2+f6, f1+f5+f3+f7
    vec4_sum = _mm_hadd_ps(vec4_sum, vec4_sum); // f0+f4+f2+f6+f1+f5+f3+f7

    return _mm_cvtss_f32(vec4_sum);                 
}

// TODO matrix multiplication unrolled
void transpose(std::vector<float, AlignedAllocator<float, 32>>& B, int row, int column){
    std::vector<float, AlignedAllocator<float, 32>> temp(column*row);
    for(int i=0; i<row; i++){
        for (int j=0; j<column; j++){
            temp[j*row+i] = B[i*column + j];
        }
    }
    B = temp;
}


void print_matrix(const std::vector<float, AlignedAllocator<float, 32>>& matrix, const int row, const int column){
    for(int i=0; i<row; i++){
        std::cout << std::endl;
        for (int j=0; j<column; j++){
            std::cout << matrix[i*column+j] << " ";
        }
    }
}


