#include <cstdlib>
#include <vector>
#include <immintrin.h>


// Define a custom allocator that aligns AVX memory
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

float reduce_sum_avx256(__m256 vec8);
void print_matrix(const std::vector<float, AlignedAllocator<float, 32>>& matrix, const int row, const int column);
void mamul_basic(const std::vector<float, AlignedAllocator<float, 32>>& A, const std::vector<float, AlignedAllocator<float, 32>>& B, std::vector<float, AlignedAllocator<float, 32>>& C, const int M, const int N, const int K);
void transpose(const std::vector<float, AlignedAllocator<float, 32>>& B, int row, int column);


