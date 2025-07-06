#include <omp.h>
#include <pthread.h>
#include <sys/time.h>

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
// 可以自行添加需要的头文件

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

// 预分配内存池，避免重复动态分配
constexpr int MAX_LEN = 1 << 18;  // 根据最大处理规模设置

/**
 * @brief 位逆序置换函数
 * @param a 输入数组
 * @param n 数组长度
 * @param bit log2(n)向上取整
 * @param revT 预分配的逆序表
 * @param num_threads NUM_THREADS 线程数
 * @details 该函数实现了位逆序置换，使用OpenMP并行化处理。
 */
void reverse(uint64_t *a, int n, int bit, int *revT, int num_threads = NUM_THREADS) {
    int *rev = revT;
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int i = 0; i < n; i++) {
        if (i < rev[i]) {
            uint64_t tmp = a[i];
            a[i] = a[rev[i]];
            a[rev[i]] = tmp;
        }
    }
}

/**
 * @brief NTT变换函数
 * @param a 输入数组
 * @param n 数组长度
 * @param invert 是否进行逆变换
 * @param p 模数
 * @param num_threads NUM_THREADS  线程数
 * @param g  3 原根
 */
void NTT_parallel(uint64_t *a, uint64_t n, bool invert, uint64_t p, int num_threads = NUM_THREADS, int g = 3 ) {
    for (int len = 2; len <= n; len <<= 1) {
        uint64_t g_n = invert ? pow(g, (p - 1) - (p - 1) / len, p)
                              : pow(g, (p - 1) / len, p);
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int i = 0; i < n; i += len) {
            uint64_t gk = 1;
            int step = len >> 1;
            for (int j = 0; j < step; j++) {
                uint64_t u = a[i + j];
                uint64_t v = a[i + j + step] * gk % p;
                uint64_t sum = u + v;
                if (sum >= p)
                    sum -= p;
                uint64_t diff = u >= v ? u - v : u + p - v;
                a[i + j] = sum;
                a[i + j + step] = diff;
                gk = gk * g_n % p;
            }
        }
    }
    if (invert) {
        uint64_t inv_n = pow(n, p - 2, p);
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int i = 0; i < n; i++) {
            a[i] = a[i] * inv_n % p;
        }
    }
}

/**
 * @brief 使用NTT的多项式乘法,使用openMP并行化处理
 * @param a 输入多项式A
 * @param b 输入多项式B
 * @param result 结果数组
 * @param n 多项式长度
 * @param p 模数
 * @details 该函数实现了NTT变换下的多项式乘法，包括正向NTT、点值乘法和逆向NTT。
 */
void NTT_multiply_parallel(uint64_t *a, uint64_t *b, uint64_t *result, int n, uint64_t p ,int threads = NUM_THREADS) {
    int len = (n << 1);
    uint64_t *fa = new uint64_t[len];
    uint64_t *fb = new uint64_t[len];
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < n; i++) {
        fa[i] = a[i];
        fb[i] = b[i];
    }
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = n; i < len; i++) {
        fa[i] = fb[i] = 0;
    }
    int g = 3;

    int bit = 0;
    while ((1 << bit) < len)
        bit++;
    int *rev = new int[len];
    rev[0] = 0;
    for (int i = 0; i < len; i++) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    }

    reverse(fa, len, bit, rev);
    reverse(fb, len, bit, rev);
    NTT_parallel(fa, len, false, p);
    NTT_parallel(fb, len, false, p);
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < len; i++) {
        fa[i] = fa[i] * fb[i] % p;
    }
    reverse(fa, len, bit, rev);
    NTT_parallel(fa, len, true, p);
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < (n << 1) - 1; i++) {
        result[i] = fa[i];
    }
    delete[] fa;
    delete[] fb;
    delete[] rev;
}

/**
 * @brief 为CRT服务的多项式乘法
 * @param a 输入多项式A 
 * @param b 输入多项式B
 * @param result 结果数组
 * @param n 多项式长度
 * @param p 模数
 * @details 该函数实现了NTT变换下的多项式乘法，包括正向NTT、点值乘法和逆向NTT。
 */
void NTT_multiply_parallel_big(uint64_t *a, uint64_t *b, uint64_t *result, int n, uint64_t p) {
    int len = (n << 1);
    uint64_t *fa = new uint64_t[len];
    uint64_t *fb = new uint64_t[len];

    int inner_threads = std::max(1, NUM_THREADS / GLOBAL_MOD_COUNT); // 8 / 4 = 2

#pragma omp parallel for schedule(static) num_threads(inner_threads) // 使用 inner_threads (2)
    for (int i = 0; i < n; i++) {
        fa[i] = a[i] % p;
        fb[i] = b[i] % p;
    }
#pragma omp parallel for schedule(static) num_threads(inner_threads) // 使用 inner_threads (2)
    for (int i = n; i < len; i++) {
        fa[i] = fb[i] = 0;
    }
    int g = 3;

    int bit = 0;
    while ((1 << bit) < len)
        bit++;
    int *rev = new int[len];
    rev[0] = 0;
    // 这个循环是串行的，用于计算rev表，它本身不并行
    for (int i = 0; i < len; i++) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    }
    // 调用 reverse 和 NTT_parallel 时，会传入 inner_threads
    reverse(fa, len, bit, rev, inner_threads); 
    reverse(fb, len, bit, rev, inner_threads);
    NTT_parallel(fa, len, false, p, inner_threads); 
    NTT_parallel(fb, len, false, p, inner_threads);
#pragma omp parallel for schedule(static) num_threads(inner_threads) // 使用 inner_threads (2)
    for (int i = 0; i < len; i++) {
        fa[i] = fa[i] * fb[i] % p;
    }
    reverse(fa, len, bit, rev, inner_threads);
    NTT_parallel(fa, len, true, p, inner_threads);
#pragma omp parallel for schedule(static) num_threads(inner_threads) // 使用 inner_threads (2)
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

    // 分配结果数组 - 使用最大可能的多项式长度
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
 * @brief 使用持久线程的CRT多项式乘法
 * @param a 输入多项式A
 * @param b 输入多项式B
 * @param result 结果数组
 * @param n 多项式长度
 * @param p 模数
 * @details 该函数实现了使用NTT的多项式乘法，使用OpenMP并行化处理。
 */
void CRT_NTT_multiply_openmp(uint64_t *a, uint64_t *b, uint64_t *result, int n, uint64_t p) {
    // 使用全局预分配的模数和结果数组
    int result_len = (n << 1) - 1;

// 清零结果数组的必要部分
#pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for(int i = 0;i<result_len;i++){
        GLOBAL_MOD_RESULTS[0][i] = 0;
        GLOBAL_MOD_RESULTS[1][i] = 0;
        GLOBAL_MOD_RESULTS[2][i] = 0;
        GLOBAL_MOD_RESULTS[3][i] = 0;
    }

// 逐个模数计算NTT - 可并行执行
#pragma omp parallel for num_threads(GLOBAL_MOD_COUNT) schedule(static)
    for (int i = 0; i < GLOBAL_MOD_COUNT; i++) {
        NTT_multiply_parallel_big(a, b, GLOBAL_MOD_RESULTS[i], n, GLOBAL_MOD_LIST[i]);
    }

// 使用CRT合并结果
#pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (int j = 0; j < result_len; j++) {
        // 完全展开4个模数的循环
        __uint128_t term0 = GLOBAL_MI_VALUES[0] * ((GLOBAL_MOD_RESULTS[0][j] * GLOBAL_MI_INV_VALUES[0]) % GLOBAL_MOD_LIST[0]);
        __uint128_t term1 = GLOBAL_MI_VALUES[1] * ((GLOBAL_MOD_RESULTS[1][j] * GLOBAL_MI_INV_VALUES[1]) % GLOBAL_MOD_LIST[1]);
        __uint128_t term2 = GLOBAL_MI_VALUES[2] * ((GLOBAL_MOD_RESULTS[2][j] * GLOBAL_MI_INV_VALUES[2]) % GLOBAL_MOD_LIST[2]);
        __uint128_t term3 = GLOBAL_MI_VALUES[3] * ((GLOBAL_MOD_RESULTS[3][j] * GLOBAL_MI_INV_VALUES[3]) % GLOBAL_MOD_LIST[3]);

        __uint128_t sum = (term0 + term1 + term2 + term3) % GLOBAL_M;

        result[j] = sum % p;
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
            CRT_NTT_multiply_openmp(a, b, ab, n_, p_);
        } else {
            NTT_multiply_parallel(a, b, ab, n_, p_);
        }

        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000>> elapsed = End - Start;
        ans += elapsed.count();

        fCheck(ab, n_, i);
        std::cout << "average latency for n = " << n_ << " p = " << p_ << " : "
                  << ans << " (us) " << std::endl;
        fWrite(ab, n_, i);
    }
    cleanup_global_crt_values();
    return 0;
}