#include <omp.h>
#include <sys/time.h>
#include <immintrin.h>  // X86 SIMD指令集 (SSE/AVX)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#define NUM_THREADS 8

// NTT友好的4个固定模数
const uint64_t GLOBAL_MOD_LIST[4] = {1004535809, 1224736769, 469762049, 998244353};
const int GLOBAL_MOD_COUNT = 4;

// 预分配的结果数组，避免动态分配
uint64_t *GLOBAL_MOD_RESULTS[4] = {nullptr, nullptr, nullptr, nullptr};

// 预计算模数乘积及逆元相关值
__uint128_t GLOBAL_M = 0;          // 模数乘积
__uint128_t GLOBAL_MI_VALUES[4];   // 各模数的"M/模数"值
uint64_t GLOBAL_MI_INV_VALUES[4];  // 各模数的逆元值

void fRead(uint64_t *a, uint64_t *b, int *n, uint64_t *p, int input_id) {
    // 数据输入函数
    std::string str1 = "./nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    fin >> *n >> *p;
    for (uint64_t i = 0; i < *n; i++) {
        fin >> a[i];
    }
    for (uint64_t i = 0; i < *n; i++) {
        fin >> b[i];
    }
}

void fCheck(uint64_t *ab, int n, int input_id) {
    // 判断多项式乘法结果是否正确
    std::string str1 = "./nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++) {
        uint64_t x;
        fin >> x;
        if (x != ab[i]) {
            std::cout << "多项式乘法结果错误" << std::endl;
            return;
        }
    }
    std::cout << "多项式乘法结果正确" << std::endl;
    return;
}

void fWrite(uint64_t *ab, int n, int input_id) {
    // 数据输出函数, 可以用来输出最终结果, 也可用于调试时输出中间数组
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (int i = 0; i < n * 2 - 1; i++) {
        fout << ab[i] << '\n';
    }
}

/**
 * @brief 快速幂函数
 * @param base 底数
 * @param exp 指数
 * @param mod 模数
 */
uint64_t pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) {
            res = res * base % mod;
        }
        base = base * base % mod;
        exp >>= 1;
    }
    return res;
}

/**
 * @brief SIMD版本的蝶形运算 - 使用AVX2指令集
 * @param u_ptr 指向u值的指针
 * @param v_ptr 指向v值的指针  
 * @param len 处理长度
 * @param mod 模数
 */
inline void simd_butterfly_avx2(uint64_t *u_ptr, uint64_t *v_ptr, int len, uint64_t mod) {
    const __m256i mod_vec = _mm256_set1_epi64x(mod);
    
    // 处理4个元素为一组（AVX2可以同时处理4个64位整数）
    int simd_len = (len / 4) * 4;
    
    for (int i = 0; i < simd_len; i += 4) {
        // 加载数据
        __m256i u_vec = _mm256_loadu_si256((__m256i*)(u_ptr + i));
        __m256i v_vec = _mm256_loadu_si256((__m256i*)(v_ptr + i));
        
        // 计算 sum = u + v
        __m256i sum_vec = _mm256_add_epi64(u_vec, v_vec);
        
        // 计算 diff = u - v (需要处理负数情况)
        __m256i diff_vec = _mm256_sub_epi64(u_vec, v_vec);
        
        // 对sum进行模运算 (简化版本，实际需要更复杂的模运算)
        // 这里使用标量版本的模运算
        uint64_t sum_tmp[4], diff_tmp[4], u_tmp[4], v_tmp[4];
        _mm256_storeu_si256((__m256i*)sum_tmp, sum_vec);
        _mm256_storeu_si256((__m256i*)diff_tmp, diff_vec);
        _mm256_storeu_si256((__m256i*)u_tmp, u_vec);
        _mm256_storeu_si256((__m256i*)v_tmp, v_vec);
        
        for (int j = 0; j < 4; j++) {
            uint64_t sum = sum_tmp[j];
            uint64_t diff = diff_tmp[j];
            
            // 模运算
            if (sum >= mod) sum -= mod;
            if (diff_tmp[j] > u_tmp[j]) diff = diff + mod;
            
            u_ptr[i + j] = sum;
            v_ptr[i + j] = diff;
        }
    }
    
    // 处理剩余元素
    for (int i = simd_len; i < len; i++) {
        uint64_t u = u_ptr[i];
        uint64_t v = v_ptr[i];
        uint64_t sum = u + v;
        if (sum >= mod) sum -= mod;
        uint64_t diff = u >= v ? u - v : u + mod - v;
        u_ptr[i] = sum;
        v_ptr[i] = diff;
    }
}

/**
 * @brief SIMD+OpenMP优化的位逆序置换函数
 * @param a 输入数组
 * @param n 数组长度
 * @param bit log2(n)向上取整
 * @param revT 预分配的逆序表
 * @param num_threads 线程数
 */
void reverse_simd_openmp(uint64_t *a, int n, int bit, int *revT, int num_threads = NUM_THREADS) {
    int *rev = revT;
    
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int i = 0; i < n; i += 4) {
        // 使用SIMD指令同时处理4个位置的交换
        int end = std::min(i + 4, n);
        for (int j = i; j < end; j++) {
            if (j < rev[j]) {
                uint64_t tmp = a[j];
                a[j] = a[rev[j]];
                a[rev[j]] = tmp;
            }
        }
    }
}

/**
 * @brief SIMD+OpenMP优化的NTT变换函数
 * @param a 输入数组
 * @param n 数组长度
 * @param invert 是否进行逆变换
 * @param p 模数
 * @param num_threads 线程数
 * @param g 原根
 */
void NTT_simd_openmp(uint64_t *a, uint64_t n, bool invert, uint64_t p, int num_threads = NUM_THREADS, int g = 3) {
    for (int len = 2; len <= n; len <<= 1) {
        uint64_t g_n = invert ? pow(g, (p - 1) - (p - 1) / len, p)
                              : pow(g, (p - 1) / len, p);
        
        int step = len >> 1;
        
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int i = 0; i < n; i += len) {
            uint64_t gk = 1;
            
            // SIMD优化的蝶形运算
            for (int j = 0; j < step; j += 4) {
                int end = std::min(j + 4, step);
                
                // 使用AVX2指令集处理4个蝶形运算
                if (end - j == 4) {
                    // 加载数据
                    __m256i u_vec = _mm256_loadu_si256((__m256i*)(a + i + j));
                    __m256i v_vec = _mm256_loadu_si256((__m256i*)(a + i + j + step));
                    
                    // 计算v_vec * gk_vec % p (需要4个不同的gk值)
                    uint64_t gk_vals[4];
                    for (int k = 0; k < 4; k++) {
                        gk_vals[k] = gk;
                        gk = gk * g_n % p;
                    }
                    __m256i gk_vec = _mm256_loadu_si256((__m256i*)gk_vals);
                    
                    // 由于模乘运算比较复杂，这里使用标量处理
                    uint64_t u_vals[4], v_vals[4];
                    _mm256_storeu_si256((__m256i*)u_vals, u_vec);
                    _mm256_storeu_si256((__m256i*)v_vals, v_vec);
                    
                    for (int k = 0; k < 4; k++) {
                        uint64_t u = u_vals[k];
                        uint64_t v = v_vals[k] * gk_vals[k] % p;
                        uint64_t sum = u + v;
                        if (sum >= p) sum -= p;
                        uint64_t diff = u >= v ? u - v : u + p - v;
                        a[i + j + k] = sum;
                        a[i + j + k + step] = diff;
                    }
                } else {
                    // 处理剩余元素
                    for (int k = j; k < end; k++) {
                        uint64_t u = a[i + k];
                        uint64_t v = a[i + k + step] * gk % p;
                        uint64_t sum = u + v;
                        if (sum >= p) sum -= p;
                        uint64_t diff = u >= v ? u - v : u + p - v;
                        a[i + k] = sum;
                        a[i + k + step] = diff;
                        gk = gk * g_n % p;
                    }
                }
            }
        }
    }
    
    if (invert) {
        uint64_t inv_n = pow(n, p - 2, p);
        const __m256i inv_n_vec = _mm256_set1_epi64x(inv_n);
        const __m256i p_vec = _mm256_set1_epi64x(p);
        
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int i = 0; i < n; i += 4) {
            int end = std::min(i + 4, (int)n);
            
            if (end - i == 4) {
                // 使用SIMD处理4个元素
                __m256i a_vec = _mm256_loadu_si256((__m256i*)(a + i));
                
                // 由于模乘运算复杂，使用标量处理
                uint64_t a_vals[4];
                _mm256_storeu_si256((__m256i*)a_vals, a_vec);
                
                for (int j = 0; j < 4; j++) {
                    a_vals[j] = a_vals[j] * inv_n % p;
                }
                
                a_vec = _mm256_loadu_si256((__m256i*)a_vals);
                _mm256_storeu_si256((__m256i*)(a + i), a_vec);
            } else {
                // 处理剩余元素
                for (int j = i; j < end; j++) {
                    a[j] = a[j] * inv_n % p;
                }
            }
        }
    }
}

/**
 * @brief SIMD+OpenMP优化的点值乘法
 * @param fa 多项式A的点值表示
 * @param fb 多项式B的点值表示
 * @param len 长度
 * @param p 模数
 * @param num_threads 线程数
 */
void pointwise_multiply_simd_openmp(uint64_t *fa, uint64_t *fb, int len, uint64_t p, int num_threads = NUM_THREADS) {
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int i = 0; i < len; i += 4) {
        int end = std::min(i + 4, len);
        
        if (end - i == 4) {
            // 使用SIMD处理4个元素的乘法
            __m256i fa_vec = _mm256_loadu_si256((__m256i*)(fa + i));
            __m256i fb_vec = _mm256_loadu_si256((__m256i*)(fb + i));
            
            // 由于64位模乘运算比较复杂，使用标量处理
            uint64_t fa_vals[4], fb_vals[4];
            _mm256_storeu_si256((__m256i*)fa_vals, fa_vec);
            _mm256_storeu_si256((__m256i*)fb_vals, fb_vec);
            
            for (int j = 0; j < 4; j++) {
                fa_vals[j] = fa_vals[j] * fb_vals[j] % p;
            }
            
            fa_vec = _mm256_loadu_si256((__m256i*)fa_vals);
            _mm256_storeu_si256((__m256i*)(fa + i), fa_vec);
        } else {
            // 处理剩余元素
            for (int j = i; j < end; j++) {
                fa[j] = fa[j] * fb[j] % p;
            }
        }
    }
}

/**
 * @brief 使用SIMD+OpenMP的NTT多项式乘法
 * @param a 输入多项式A
 * @param b 输入多项式B
 * @param result 结果数组
 * @param n 多项式长度
 * @param p 模数
 * @param threads 线程数
 */
void NTT_multiply_simd_openmp(uint64_t *a, uint64_t *b, uint64_t *result, int n, uint64_t p, int threads = NUM_THREADS) {
    int len = (n << 1);
    uint64_t *fa = new uint64_t[len];
    uint64_t *fb = new uint64_t[len];
    
    // SIMD+OpenMP优化的数组初始化
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < n; i += 4) {
        int end = std::min(i + 4, n);
        if (end - i == 4) {
            __m256i a_vec = _mm256_loadu_si256((__m256i*)(a + i));
            __m256i b_vec = _mm256_loadu_si256((__m256i*)(b + i));
            _mm256_storeu_si256((__m256i*)(fa + i), a_vec);
            _mm256_storeu_si256((__m256i*)(fb + i), b_vec);
        } else {
            for (int j = i; j < end; j++) {
                fa[j] = a[j];
                fb[j] = b[j];
            }
        }
    }
    
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = n; i < len; i += 4) {
        int end = std::min(i + 4, len);
        if (end - i == 4) {
            __m256i zero_vec = _mm256_setzero_si256();
            _mm256_storeu_si256((__m256i*)(fa + i), zero_vec);
            _mm256_storeu_si256((__m256i*)(fb + i), zero_vec);
        } else {
            for (int j = i; j < end; j++) {
                fa[j] = fb[j] = 0;
            }
        }
    }
    
    int g = 3;
    
    // 计算位逆序表
    int bit = 0;
    while ((1 << bit) < len) bit++;
    int *rev = new int[len];
    rev[0] = 0;
    for (int i = 0; i < len; i++) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    }
    
    // 使用SIMD+OpenMP优化的NTT变换
    reverse_simd_openmp(fa, len, bit, rev, threads);
    reverse_simd_openmp(fb, len, bit, rev, threads);
    NTT_simd_openmp(fa, len, false, p, threads);
    NTT_simd_openmp(fb, len, false, p, threads);
    
    // SIMD+OpenMP优化的点值乘法
    pointwise_multiply_simd_openmp(fa, fb, len, p, threads);
    
    // 逆NTT
    reverse_simd_openmp(fa, len, bit, rev, threads);
    NTT_simd_openmp(fa, len, true, p, threads);
    
    // 复制结果
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < (n << 1) - 1; i++) {
        result[i] = fa[i];
    }
    
    delete[] fa;
    delete[] fb;
    delete[] rev;
}

/**
 * @brief 初始化全局变量
 */
void init_global_crt_values() {
    // 计算模数乘积
    GLOBAL_M = 1;
    for (int i = 0; i < GLOBAL_MOD_COUNT; i++) {
        GLOBAL_M *= GLOBAL_MOD_LIST[i];
    }

    // 预计算Mi和Mi_inv值
    for (int i = 0; i < GLOBAL_MOD_COUNT; i++) {
        uint64_t mod_num = GLOBAL_MOD_LIST[i];
        GLOBAL_MI_VALUES[i] = GLOBAL_M / mod_num;
        uint64_t Mi_mod = GLOBAL_MI_VALUES[i] % mod_num;
        GLOBAL_MI_INV_VALUES[i] = pow(Mi_mod, mod_num - 2, mod_num);
    }

    // 分配结果数组
    constexpr int MAX_RESULT_LEN = 1 << 18;
    for (int i = 0; i < GLOBAL_MOD_COUNT; i++) {
        GLOBAL_MOD_RESULTS[i] = new uint64_t[MAX_RESULT_LEN];
    }
}

/**
 * @brief 释放全局资源
 */
void cleanup_global_crt_values() {
    for (int i = 0; i < GLOBAL_MOD_COUNT; i++) {
        if (GLOBAL_MOD_RESULTS[i]) {
            delete[] GLOBAL_MOD_RESULTS[i];
            GLOBAL_MOD_RESULTS[i] = nullptr;
        }
    }
}

/**
 * @brief 为CRT服务的SIMD+OpenMP多项式乘法
 */
void NTT_multiply_simd_openmp_big(uint64_t *a, uint64_t *b, uint64_t *result, int n, uint64_t p) {
    int len = (n << 1);
    uint64_t *fa = new uint64_t[len];
    uint64_t *fb = new uint64_t[len];

    int inner_threads = std::max(1, NUM_THREADS / GLOBAL_MOD_COUNT);

    // SIMD+OpenMP优化的数组初始化和模运算
#pragma omp parallel for schedule(static) num_threads(inner_threads)
    for (int i = 0; i < n; i += 4) {
        int end = std::min(i + 4, n);
        if (end - i == 4) {
            // 使用SIMD处理模运算
            uint64_t a_vals[4], b_vals[4];
            for (int j = 0; j < 4; j++) {
                a_vals[j] = a[i + j] % p;
                b_vals[j] = b[i + j] % p;
            }
            __m256i a_vec = _mm256_loadu_si256((__m256i*)a_vals);
            __m256i b_vec = _mm256_loadu_si256((__m256i*)b_vals);
            _mm256_storeu_si256((__m256i*)(fa + i), a_vec);
            _mm256_storeu_si256((__m256i*)(fb + i), b_vec);
        } else {
            for (int j = i; j < end; j++) {
                fa[j] = a[j] % p;
                fb[j] = b[j] % p;
            }
        }
    }

#pragma omp parallel for schedule(static) num_threads(inner_threads)
    for (int i = n; i < len; i += 4) {
        int end = std::min(i + 4, len);
        if (end - i == 4) {
            __m256i zero_vec = _mm256_setzero_si256();
            _mm256_storeu_si256((__m256i*)(fa + i), zero_vec);
            _mm256_storeu_si256((__m256i*)(fb + i), zero_vec);
        } else {
            for (int j = i; j < end; j++) {
                fa[j] = fb[j] = 0;
            }
        }
    }

    int g = 3;

    int bit = 0;
    while ((1 << bit) < len) bit++;
    int *rev = new int[len];
    rev[0] = 0;
    for (int i = 0; i < len; i++) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    }

    reverse_simd_openmp(fa, len, bit, rev, inner_threads);
    reverse_simd_openmp(fb, len, bit, rev, inner_threads);
    NTT_simd_openmp(fa, len, false, p, inner_threads);
    NTT_simd_openmp(fb, len, false, p, inner_threads);
    pointwise_multiply_simd_openmp(fa, fb, len, p, inner_threads);
    reverse_simd_openmp(fa, len, bit, rev, inner_threads);
    NTT_simd_openmp(fa, len, true, p, inner_threads);

#pragma omp parallel for schedule(static) num_threads(inner_threads)
    for (int i = 0; i < (n << 1) - 1; i++) {
        result[i] = fa[i];
    }

    delete[] fa;
    delete[] fb;
    delete[] rev;
}

/**
 * @brief 使用SIMD+OpenMP的CRT多项式乘法
 */
void CRT_NTT_multiply_simd_openmp(uint64_t *a, uint64_t *b, uint64_t *result, int n, uint64_t p) {
    int result_len = (n << 1) - 1;

    // 清零结果数组
#pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (int i = 0; i < result_len; i += 4) {
        int end = std::min(i + 4, result_len);
        if (end - i == 4) {
            __m256i zero_vec = _mm256_setzero_si256();
            for (int mod_idx = 0; mod_idx < 4; mod_idx++) {
                _mm256_storeu_si256((__m256i*)(GLOBAL_MOD_RESULTS[mod_idx] + i), zero_vec);
            }
        } else {
            for (int j = i; j < end; j++) {
                for (int mod_idx = 0; mod_idx < 4; mod_idx++) {
                    GLOBAL_MOD_RESULTS[mod_idx][j] = 0;
                }
            }
        }
    }

    // 逐个模数计算NTT - 可并行执行
#pragma omp parallel for num_threads(GLOBAL_MOD_COUNT) schedule(static)
    for (int i = 0; i < GLOBAL_MOD_COUNT; i++) {
        NTT_multiply_simd_openmp_big(a, b, GLOBAL_MOD_RESULTS[i], n, GLOBAL_MOD_LIST[i]);
    }

    // 使用CRT合并结果 - SIMD+OpenMP优化
#pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (int j = 0; j < result_len; j += 4) {
        int end = std::min(j + 4, result_len);
        
        if (end - j == 4) {
            // 使用SIMD处理4个CRT计算
            uint64_t results[4];
            
            for (int k = 0; k < 4; k++) {
                __uint128_t term0 = GLOBAL_MI_VALUES[0] * ((GLOBAL_MOD_RESULTS[0][j + k] * GLOBAL_MI_INV_VALUES[0]) % GLOBAL_MOD_LIST[0]);
                __uint128_t term1 = GLOBAL_MI_VALUES[1] * ((GLOBAL_MOD_RESULTS[1][j + k] * GLOBAL_MI_INV_VALUES[1]) % GLOBAL_MOD_LIST[1]);
                __uint128_t term2 = GLOBAL_MI_VALUES[2] * ((GLOBAL_MOD_RESULTS[2][j + k] * GLOBAL_MI_INV_VALUES[2]) % GLOBAL_MOD_LIST[2]);
                __uint128_t term3 = GLOBAL_MI_VALUES[3] * ((GLOBAL_MOD_RESULTS[3][j + k] * GLOBAL_MI_INV_VALUES[3]) % GLOBAL_MOD_LIST[3]);

                __uint128_t sum = (term0 + term1 + term2 + term3) % GLOBAL_M;
                results[k] = sum % p;
            }
            
            __m256i result_vec = _mm256_loadu_si256((__m256i*)results);
            _mm256_storeu_si256((__m256i*)(result + j), result_vec);
        } else {
            // 处理剩余元素
            for (int k = j; k < end; k++) {
                __uint128_t term0 = GLOBAL_MI_VALUES[0] * ((GLOBAL_MOD_RESULTS[0][k] * GLOBAL_MI_INV_VALUES[0]) % GLOBAL_MOD_LIST[0]);
                __uint128_t term1 = GLOBAL_MI_VALUES[1] * ((GLOBAL_MOD_RESULTS[1][k] * GLOBAL_MI_INV_VALUES[1]) % GLOBAL_MOD_LIST[1]);
                __uint128_t term2 = GLOBAL_MI_VALUES[2] * ((GLOBAL_MOD_RESULTS[2][k] * GLOBAL_MI_INV_VALUES[2]) % GLOBAL_MOD_LIST[2]);
                __uint128_t term3 = GLOBAL_MI_VALUES[3] * ((GLOBAL_MOD_RESULTS[3][k] * GLOBAL_MI_INV_VALUES[3]) % GLOBAL_MOD_LIST[3]);

                __uint128_t sum = (term0 + term1 + term2 + term3) % GLOBAL_M;
                result[k] = sum % p;
            }
        }
    }
}

uint64_t a[300000], b[300000], ab[300000];

int main(int argc, char *argv[]) {
    // 初始化OpenMP线程数
    omp_set_num_threads(NUM_THREADS);
    omp_set_nested(1);                // 启用嵌套并行
    init_global_crt_values();
    
    int test_begin = 0;
    int test_end = 4;
    for (int i = test_begin; i <= test_end; ++i) {
        long double ans = 0;
        int n_;
        uint64_t p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));

        auto Start = std::chrono::high_resolution_clock::now();
        
        // 根据模数大小选择算法
        if (p_ > (1ULL << 32)) {
            CRT_NTT_multiply_simd_openmp(a, b, ab, n_, p_);
        } else {
            NTT_multiply_simd_openmp(a, b, ab, n_, p_);
        }

        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000>> elapsed = End - Start;
        ans += elapsed.count();

        fCheck(ab, n_, i);
        std::cout << "SIMD+OpenMP - average latency for n = " << n_ << " p = " << p_ << " : "
                  << ans << " (ms) " << std::endl;
        fWrite(ab, n_, i);
    }
    
    cleanup_global_crt_values();
    return 0;
}