#pragma once
#include <omp.h>
#include <arm_neon.h>
#include <stdint.h>
#include "Montgomery.h"

// SIMD + OpenMP混合并行的Montgomery32实现
class Montgomery32 {
public:
    uint32_t N;          // 模数
    uint32_t R;          // 2^32
    uint32_t R2;         // R² mod N
    uint32_t N_prime;    // -N⁻¹ mod R
    uint32_t R_mod_N;    // R mod N
    
    // 构造函数
    Montgomery32(uint32_t N) : N(N) {
        R = 0;  // 2^32, 用0表示(因为uint32_t溢出)
        
        // 计算N的模逆元
        uint64_t inv = 1;
        uint64_t temp = N;
        for (int i = 0; i < 31; i++) {
            inv *= 2 - temp * inv;
            temp *= temp;
        }
        N_prime = -inv;  // -N⁻¹ mod 2^32
        
        // 计算R² mod N
        uint64_t r_mod_n = ((1ULL << 32) % N);
        R2 = (r_mod_n * r_mod_n) % N;
        R_mod_N = r_mod_n;
    }
    
    // Montgomery约简
    uint32_t REDC(uint64_t T) const {
        uint32_t m = (uint32_t)T * N_prime;
        uint64_t t = (T + (uint64_t)m * N) >> 32;
        return t >= N ? t - N : t;
    }
    
    // SIMD版本的Montgomery约简 - 4个数据并行处理
    uint32x4_t REDC_neon(uint64_t T[4]) const {
        uint32_t results[4];
        for (int i = 0; i < 4; i++) {
            results[i] = REDC(T[i]);
        }
        return vld1q_u32(results);
    }
};

// SIMD + OpenMP混合并行的NTT函数声明
void NTT_base4_simd_openmp(uint32_t *a, int n, bool invert, int p, int num_threads = 4);
void NTT_multiply_simd_openmp(int *a, int *b, int *result, int n, int p, int num_threads = 4);

// 辅助函数声明
void reverse_base4_parallel(uint32_t *a, int n, int num_threads);
void butterfly_simd_optimized(uint32_t *data, int start, int len, 
                              uint32_t g_n, uint32_t g_n2, uint32_t g_n3, 
                              uint32_t g_pow_step, const Montgomery32 &mont, int p);
