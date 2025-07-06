#include "simd_openmp_ntt.h"
#include <algorithm>
#include <iostream>

// 快速幂函数
int pow(int base, int exp, int mod) {
    int res = 1;
    while (exp > 0) {
        if (exp & 1) {
            res = 1LL * res * base % mod;
        }
        base = 1LL * base * base % mod;
        exp >>= 1;
    }
    return res;
}

// 并行的基4位逆序置换
void reverse_base4_parallel(uint32_t *a, int n, int num_threads) {
    int log4n = 0;
    int temp = n;
    while (temp > 1) {
        temp >>= 2;
        log4n++;
    }
    
    // 预计算位逆序表
    int *rev = new int[n];
    
    // 使用OpenMP并行计算位逆序表
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; i++) {
        int reversed = 0;
        int num = i;
        for (int j = 0; j < log4n; j++) {
            reversed = (reversed << 2) | (num & 3);
            num >>= 2;
        }
        rev[i] = reversed;
    }
    
    // 并行执行位逆序置换
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; i++) {
        if (i < rev[i]) {
            #pragma omp critical
            {
                if (i < rev[i]) {  // 双重检查避免重复交换
                    std::swap(a[i], a[rev[i]]);
                }
            }
        }
    }
    
    delete[] rev;
}

// SIMD优化的蝶形操作
void butterfly_simd_optimized(uint32_t *data, int start, int len, 
                              uint32_t g_n, uint32_t g_n2, uint32_t g_n3, 
                              uint32_t g_pow_step, const Montgomery32 &mont, int p) {
    int step = len >> 2;
    uint64_t temp64[4];
    uint32_t w[4] = {mont.R_mod_N, mont.R_mod_N, mont.R_mod_N, mont.R_mod_N};
    
    for (int j = 0; j < len / 4; j++) {
        uint32_t u[4];
        
        if (j == 0) {
            // 第一次迭代，w都是1
            for (int k = 0; k < 4; k++) {
                u[k] = data[start + j + k * step];
            }
        } else {
            // 使用SIMD进行并行乘法
            for (int k = 0; k < 4; k++) {
                temp64[k] = (uint64_t)data[start + j + k * step] * w[k];
            }
            uint32x4_t u_vec = mont.REDC_neon(temp64);
            vst1q_u32(u, u_vec);
        }
        
        // 计算旋转因子
        uint32_t j_1 = mont.REDC((uint64_t)u[1] * g_pow_step);
        uint32_t j_3 = mont.REDC((uint64_t)u[3] * g_pow_step);
        
        // 蝶形操作
        data[start + j] = (u[0] + u[1] + u[2] + u[3]) % p;
        data[start + j + step] = (u[0] + j_1 + p - u[2] + p - j_3) % p;
        data[start + j + 2 * step] = (u[0] + p - u[1] + u[2] + p - u[3]) % p;
        data[start + j + 3 * step] = (u[0] + p - j_1 + p - u[2] + j_3) % p;
        
        // 使用SIMD更新旋转因子
        temp64[0] = (uint64_t)w[0] * mont.R_mod_N;
        temp64[1] = (uint64_t)w[1] * g_n;
        temp64[2] = (uint64_t)w[2] * g_n2;
        temp64[3] = (uint64_t)w[3] * g_n3;
        
        uint32x4_t w_vec = mont.REDC_neon(temp64);
        vst1q_u32(w, w_vec);
    }
}

// SIMD + OpenMP混合并行的基4 NTT实现
void NTT_base4_simd_openmp(uint32_t *a, int n, bool invert, int p, int num_threads) {
    Montgomery32 mont(p);
    int g = 3;
    
    // 设置OpenMP线程数
    omp_set_num_threads(num_threads);
    
    // 并行位逆序置换
    reverse_base4_parallel(a, n, num_threads);
    
    // 蝶形操作 - 分层并行化
    for (int len = 4; len <= n; len <<= 2) {
        int g_n_normal = invert ? pow(g, (p - 1) - (p - 1) / len, p)
                                : pow(g, (p - 1) / len, p);
        
        uint32_t g_n = mont.REDC((uint64_t)g_n_normal * mont.R2);
        uint32_t g_n2 = mont.REDC((uint64_t)g_n * g_n);
        uint32_t g_n3 = mont.REDC((uint64_t)g_n2 * g_n);
        
        int step = len >> 2;
        uint32_t g_pow_step_normal = pow(g_n_normal, step, p);
        uint32_t g_pow_step = mont.REDC((uint64_t)g_pow_step_normal * mont.R2);
        
        // 关键：根据len的大小选择并行策略
        if (len <= 64) {
            // 小块：串行处理但内部使用SIMD
            for (int i = 0; i < n; i += len) {
                butterfly_simd_optimized(a, i, len, g_n, g_n2, g_n3, 
                                       g_pow_step, mont, p);
            }
        } else {
            // 大块：OpenMP并行 + SIMD
            #pragma omp parallel for schedule(static) num_threads(num_threads)
            for (int i = 0; i < n; i += len) {
                butterfly_simd_optimized(a, i, len, g_n, g_n2, g_n3, 
                                       g_pow_step, mont, p);
            }
        }
    }
    
    // 逆变换系数处理 - 并行化
    if (invert) {
        int inv_n = pow(n, p - 2, p);
        uint32_t inv_n_mont = mont.REDC((uint64_t)inv_n * mont.R2);
        
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < n; i++) {
            a[i] = mont.REDC((uint64_t)a[i] * inv_n_mont);
        }
    }
}

// SIMD + OpenMP混合并行的多项式乘法
void NTT_multiply_simd_openmp(int *a, int *b, int *result, int n, int p, int num_threads) {
    // 计算长度
    int len = 1;
    while (len < 2 * n) {
        if (len < 4) len = 4;
        else len <<= 2;
    }
    
    Montgomery32 mont(p);
    uint32_t *fa = new uint32_t[len];
    uint32_t *fb = new uint32_t[len];
    
    // 设置OpenMP线程数
    omp_set_num_threads(num_threads);
    
    // 数据转换 - OpenMP + SIMD并行化
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            // 处理数组a
            #pragma omp parallel for
            for (int i = 0; i < n; i += 4) {
                uint64_t a_mul[4];
                
                // 处理边界情况
                int actual_size = std::min(4, n - i);
                for (int j = 0; j < actual_size; j++) {
                    a_mul[j] = (uint64_t)a[i + j] * mont.R2;
                }
                for (int j = actual_size; j < 4; j++) {
                    a_mul[j] = 0;
                }
                
                if (actual_size == 4) {
                    uint32x4_t fa_vec = mont.REDC_neon(a_mul);
                    vst1q_u32(&fa[i], fa_vec);
                } else {
                    // 处理边界
                    for (int j = 0; j < actual_size; j++) {
                        fa[i + j] = mont.REDC(a_mul[j]);
                    }
                }
            }
            
            // 填充零
            #pragma omp parallel for
            for (int i = n; i < len; i++) {
                fa[i] = 0;
            }
        }
        
        #pragma omp section
        {
            // 处理数组b
            #pragma omp parallel for
            for (int i = 0; i < n; i += 4) {
                uint64_t b_mul[4];
                
                // 处理边界情况
                int actual_size = std::min(4, n - i);
                for (int j = 0; j < actual_size; j++) {
                    b_mul[j] = (uint64_t)b[i + j] * mont.R2;
                }
                for (int j = actual_size; j < 4; j++) {
                    b_mul[j] = 0;
                }
                
                if (actual_size == 4) {
                    uint32x4_t fb_vec = mont.REDC_neon(b_mul);
                    vst1q_u32(&fb[i], fb_vec);
                } else {
                    // 处理边界
                    for (int j = 0; j < actual_size; j++) {
                        fb[i + j] = mont.REDC(b_mul[j]);
                    }
                }
            }
            
            // 填充零
            #pragma omp parallel for
            for (int i = n; i < len; i++) {
                fb[i] = 0;
            }
        }
    }
    
    // 正向NTT - 可以并行执行
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            NTT_base4_simd_openmp(fa, len, false, p, num_threads / 2);
        }
        
        #pragma omp section
        {
            NTT_base4_simd_openmp(fb, len, false, p, num_threads / 2);
        }
    }
    
    // 点值乘法 - OpenMP + SIMD并行化
    #pragma omp parallel for
    for (int i = 0; i < len; i += 4) {
        if (i + 4 <= len) {
            uint64_t temp[4];
            for (int j = 0; j < 4; j++) {
                temp[j] = (uint64_t)fa[i + j] * fb[i + j];
            }
            uint32x4_t result_vec = mont.REDC_neon(temp);
            vst1q_u32(&fa[i], result_vec);
        } else {
            // 处理边界
            for (int j = 0; j < len - i; j++) {
                fa[i + j] = mont.REDC((uint64_t)fa[i + j] * fb[i + j]);
            }
        }
    }
    
    // 逆向NTT
    NTT_base4_simd_openmp(fa, len, true, p, num_threads);
    
    // 结果转换 - 并行化
    #pragma omp parallel for
    for (int i = 0; i < 2 * n - 1; i++) {
        result[i] = mont.REDC(fa[i]);
    }
    
    delete[] fa;
    delete[] fb;
}
