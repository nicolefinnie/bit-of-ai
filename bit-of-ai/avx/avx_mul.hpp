#include <cstdlib>
#include <vector>
#include <immintrin.h>


// Define a custom allocator that aligns AVX memory with container elements such as std::vector
// to a specific byte boundary, e.g. 32 bytes required by AVX SIMD instructions.
template<typename T, size_t Alignment>
class AlignedAllocator {
public:
    using value_type = T;
    AlignedAllocator() noexcept = default;
    
    T* allocate(std::size_t num){
        void* ptr = std::aligned_alloc(Alignment, num*sizeof(T));
        if(!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    void deallocate(T* ptr, std::size_t num) noexcept { 
        std::free(ptr);
    }

    // Rebind so it can be compatible with other types such as std::vector
    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    template <typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept {
        return true;
    }

    template <typename U>
    bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept {
        return false;
    }
};

void mamul_SIMD(const std::vector<float, AlignedAllocator<float, 32>>& A,
    const std::vector<float, AlignedAllocator<float, 32>>& B, 
    std::vector<float, AlignedAllocator<float, 32>>& C, const int M, const int N, const int K);
void mamul_SIMD_unaligned(const std::vector<float, AlignedAllocator<float, 32>>& A,
    const std::vector<float, AlignedAllocator<float, 32>>& B, 
    std::vector<float, AlignedAllocator<float, 32>>& C, const int M, const int N, const int K);
void mamul(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                const int M, const int N, const int K);
void mamul(const float* A, const float* B, const float* C, const int M, const int N, const int K);
float reduce_sum_avx256(__m256 vec8);
void transpose(std::vector<float, AlignedAllocator<float, 32>>& B, int row, int column);
void transpose(float *B, int row, int column);
void print(const std::vector<float, AlignedAllocator<float, 32>>& matrix, const int row, const int column);
void print(const std::vector<float>& matrix, const int row, const int column);
void print(const float* matrix, const int row, const int column);
void print(__m256 vec);


