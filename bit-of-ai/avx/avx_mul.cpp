// AVX matrix multiplication in SIMD. Only for educational purpose.
// For example, matrix A: 8x16, matrix B: 16x32, and we want to compute C = A*B in SIMD
// we transpose the matrix B because the machine is row-major, it's more efficient to access 
// continguously in memory for matrix multiplication.
#include <iostream>
#include <cstring>
#include "avx_mul.hpp"


// Basic matrix multiplication using AVX SIMD without unrolling
// AVX handles 8 floats or 4 doubles at a time (256 bit reigster)
// K has to be a multiple of 8, hence no handling for remainder
void mamul_SIMD(const std::vector<float, AlignedAllocator<float, 32>>& A,
                const std::vector<float, AlignedAllocator<float, 32>>& B,
                std::vector<float, AlignedAllocator<float, 32>>& C, 
                const int M, const int N, const int K){
    std::fill(C.begin(), C.end(), 0);

    for (int i=0; i<M; i++){
        for (int j=0; j<N; j++){
            __m256 sum = _mm256_setzero_ps();
            int k=0;
            for (; k < K-7; k+=8){
                __m256 a = _mm256_load_ps(&A[i*K+k]);
                __m256 b = _mm256_load_ps(&B[j*K+k]);
                sum = _mm256_fmadd_ps(a, b, sum);
            }
        }
    }
}

// Unaligned memory load for matrix A and B, a bit slower than aligned load
// but it's more flexible because K may not be multiple of 8
void mamul_SIMD_unaligned(const std::vector<float, AlignedAllocator<float, 32>>& A,
                const std::vector<float, AlignedAllocator<float, 32>>& B,
                std::vector<float, AlignedAllocator<float, 32>>& C, 
                const int M, const int N, const int K){
    std::fill(C.begin(), C.end(), 0);

    for (int i=0; i<M; i++){
        for (int j=0; j<N; j++){
            __m256 sum = _mm256_setzero_ps();
            int k=0;
            for (; k < K-7; k+=8){
                __m256 a = _mm256_loadu_ps(&A[i*K+k]);
                __m256 b = _mm256_loadu_ps(&B[j*K+k]);
                sum = _mm256_fmadd_ps(a, b, sum);
            }

            float tail_sum = 0;
            //Handling remainder 
            for(; k < K; ++k){
                tail_sum += A[i*K+k]*B[j*K+k];
                
            }
            C[i*N+j] = reduce_sum_avx256(sum) + tail_sum;
        }
    }
}


void mamul(const float* A, const float* B, float* C, const int M, const int N, const int K){

    for (int i=0; i<M; i++){
        for (int j=0; j<N; j++){
            __m256 sum = _mm256_setzero_ps();
            int k=0;
            for (; k < K-7; k=k+8){
                // Use unaligned load because K may not be multiple of 8
                // so it's not aligned to 32 bytes
                __m256 a = _mm256_loadu_ps(&A[i*K+k]);
                __m256 b = _mm256_loadu_ps(&B[j*K+k]);
                sum = _mm256_fmadd_ps(a, b, sum);
            }
       
            float tail_sum = 0;
            //Handling remainder 
            for(; k < K; ++k){
                tail_sum += A[i*K+k]*B[j*K+k];
                
            }
            C[i*N+j] = reduce_sum_avx256(sum) + tail_sum;
        }
    }
}

// Native matrix multiplication without SIMD
void mamul(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                const int M, const int N, const int K){
    std::fill(C.begin(), C.end(), 0);
    for (int i=0; i<M; i++){
        for (int j=0; j<N; j++){
            for (int k=0; k<K; k++){
                C[i*N+j] += A[i*K+k]*B[k*N+j];
            }
        }
    }                
}

// Sum all elements in 8 floats in a vector 
// f0, f1, f2, f3, f4, f5, f6, f7 -> f0+f4, f1+f5, f2+f6, f3+f7
// f0+f4, f1+f5, f2+f6, f3+f7 -> f0+f4+f2+f6, f1+f5+f3+f7
// return the first float as the result
float reduce_sum_avx256(__m256 vec8){
    __m128 vec4_low = _mm256_castps256_ps128(vec8); // lower 128 bits f0, f1, f2, f3
    __m128 vec4_high = _mm256_extractf128_ps(vec8, 1); // upper 128 bits f4, f5, f6, f7
    // add lower and upper 128 bits (f0+f4, f1+f5, f2+f6, f3+f7)
    __m128 vec4_sum = _mm_add_ps(vec4_low, vec4_high);

    vec4_sum = _mm_hadd_ps(vec4_sum, vec4_sum); // f0+f4+f2+f6, f1+f5+f3+f7
    vec4_sum = _mm_hadd_ps(vec4_sum, vec4_sum); // f0+f4+f2+f6+f1+f5+f3+f7

    return _mm_cvtss_f32(vec4_sum);                 
}

void transpose(std::vector<float, AlignedAllocator<float,32>>& B, int row, int column){
    std::vector<float, AlignedAllocator<float,32>> temp(column * row);
    for(int i=0; i<row; i++){
        for (int j=0; j<column; j++){
            temp[j*row+i] = B[i*column + j];
        }
    }
    B = std::move(temp);
}

void transpose(float *B, int row, int column){
    float temp[row * column];
    for(int i=0; i<row; i++){
        for (int j=0; j<column; j++){
            temp[j*row+i] = B[i*column + j];
        }
    }
    memcpy(B, temp, row*column*sizeof(float));
}


void print(const std::vector<float>& matrix, const int row, const int column){
    for(int i=0; i<row; i++){
        std::cout << std::endl;
        for (int j=0; j<column; j++){
            std::cout << matrix[i*column+j] << " ";
        }
    }
}


void print(const std::vector<float, AlignedAllocator<float, 32>>& matrix, const int row, const int column){
    for(int i=0; i<row; i++){
        std::cout << std::endl;
        for (int j=0; j<column; j++){
            std::cout << matrix[i*column+j] << " ";
        }
    }
}


void print(const float* matrix, const int row, const int column){
    for(int i=0; i<row; i++){
        std::cout << std::endl;
        for (int j=0; j<column; j++){
            std::cout << matrix[i*column+j] << " ";
        }
    }
}

void print(__m256 vec) {
    float values[8]; // Ensure the array is properly aligned to 32 bytes for AVX

    // Store the contents of the __m256 register into the array
    _mm256_storeu_ps(values, vec);
    
    // Print the values
    std::cout << "Loaded values: ";
    for (int i = 0; i < 8; ++i) {
        std::cout << values[i] << " ";
    }
    std::cout << std::endl;
}

