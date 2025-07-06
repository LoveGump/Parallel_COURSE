/**
 * 使用多线程 + MPI 来并行计算NTT
 */
#include <mpi.h>
#include <pthread.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
// 全局数组声明
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

uint64_t a[300000], b[300000];

// Barrett reduction struct
struct Barrett {
    uint64_t mod;
    uint64_t k;
    uint64_t mu;

    Barrett(uint64_t m) : mod(m) {
        k = 31;
        mu = (1ull << 62) / m;
    }

    uint64_t reduce(uint64_t x) const {
        uint64_t q = ((__uint128_t)x * mu) >> 62;
        uint64_t r = x - q * mod;
        return r >= mod ? r - mod : r;
    }
};

void fRead(uint64_t *a, uint64_t *b, int *n, uint64_t *p, int input_id) {
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

uint64_t pow(uint64_t base, uint64_t exp, const Barrett &br) {
    uint64_t res = 1;
    base = br.reduce(base);
    while (exp > 0) {
        if (exp & 1) {
            res = br.reduce(res * base);
        }
        base = br.reduce(base * base);
        exp >>= 1;
    }
    return res;
}
// 预分配内存池，避免重复动态分配
constexpr int MAX_LEN = 1 << 18;  // 根据最大处理规模设置

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

void cleanup_global_crt_values() {
    for (int i = 0; i < GLOBAL_MOD_COUNT; i++) {
        if (GLOBAL_MOD_RESULTS[i]) {
            delete[] GLOBAL_MOD_RESULTS[i];
            GLOBAL_MOD_RESULTS[i] = nullptr;
        }
    }
}

void CRT_NTT_multiply_parallel(uint64_t *a, uint64_t *b, uint64_t *result, int n, uint64_t p, int rank, int size) {
    int result_len = (2 * n) - 1;
    // 分配临时数组
    uint64_t *ta = new uint64_t[n];
    uint64_t *tb = new uint64_t[n];
    // 每个MPI进程处理分配给它的模数，其实我们一共就四个线程
    for (int i = rank; i < GLOBAL_MOD_COUNT; i += size) {
        Barrett br(GLOBAL_MOD_LIST[i]);
        int j = 0;
        for (; j +3 < n; j += 4) {
            ta[j] = br.reduce(a[j]);
            ta[j + 1] = br.reduce(a[j + 1]);
            ta[j + 2] = br.reduce(a[j + 2]);
            ta[j + 3] = br.reduce(a[j + 3]);
            
            tb[j] = br.reduce(b[j]);
            tb[j + 1] = br.reduce(b[j + 1]);
            tb[j + 2] = br.reduce(b[j + 2]);
            tb[j + 3] = br.reduce(b[j + 3]);
        }
         // 处理剩余元素
        for (; j < n; j++) {
            ta[j] = br.reduce(a[j]);
            tb[j] = br.reduce(b[j]);
        }
        // 使用openmp优化的NTT乘法
        NTT_multiply_parallel_big(ta, tb, GLOBAL_MOD_RESULTS[i], n, GLOBAL_MOD_LIST[i]);
    }

    // 同步所有进程，确保NTT计算完成
    MPI_Barrier(MPI_COMM_WORLD);

    // 收集所有模数的结果到所有进程
    for (int i = 0; i < GLOBAL_MOD_COUNT; i++) {
        int owner = i % size;  // 确定哪个进程拥有这个模数的结果
        MPI_Bcast(GLOBAL_MOD_RESULTS[i], result_len, MPI_UINT64_T, owner, MPI_COMM_WORLD);
    }

    // 分配部分结果缓冲区
    uint64_t *partial_result = new uint64_t[result_len]();
    // 计算每个进程需要处理的元素范围
    int elements_per_process = (result_len + size - 1) / size;
    int start_idx = rank * elements_per_process;
    int end_idx = std::min(start_idx + elements_per_process, result_len);

    // 每个进程处理自己负责的部分
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (int j = start_idx; j < end_idx; j++) {
        __uint128_t sum = 0;
        // 完全展开4个模数的循环
        __uint128_t term0 = GLOBAL_MI_VALUES[0] * ((GLOBAL_MOD_RESULTS[0][j] * GLOBAL_MI_INV_VALUES[0]) % GLOBAL_MOD_LIST[0]);
        __uint128_t term1 = GLOBAL_MI_VALUES[1] * ((GLOBAL_MOD_RESULTS[1][j] * GLOBAL_MI_INV_VALUES[1]) % GLOBAL_MOD_LIST[1]);
        __uint128_t term2 = GLOBAL_MI_VALUES[2] * ((GLOBAL_MOD_RESULTS[2][j] * GLOBAL_MI_INV_VALUES[2]) % GLOBAL_MOD_LIST[2]);
        __uint128_t term3 = GLOBAL_MI_VALUES[3] * ((GLOBAL_MOD_RESULTS[3][j] * GLOBAL_MI_INV_VALUES[3]) % GLOBAL_MOD_LIST[3]);

        sum = (term0 + term1 + term2 + term3) % GLOBAL_M;
        partial_result[j] = sum % p;
    }
    // 使用MPI_Reduce将所有部分结果合并到进程0
    MPI_Reduce(partial_result, result, result_len, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    delete[] ta;
    delete[] tb;
    delete[] partial_result;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 初始化全局CRT变量
    init_global_crt_values();

    if (rank == 0) {
        std::cout << "使用 " << size << " 个MPI进程，每个进程 " << NUM_THREADS << " 个工作线程进行并行计算" << std::endl;
    }
    int test_begin = 0;
    int test_end = 4;
    for (int i = test_begin; i <= test_end; ++i) {
        long double ans = 0;
        int n_;
        uint64_t p_;
        

        if (rank == 0) {
            fRead(a, b, &n_, &p_, i);
        }
        
        // Broadcast input data to all processes
        MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p_, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(a, n_, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(b, n_, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        
        uint64_t *ab = new uint64_t[2 * n_ ]();
        
        auto Start = std::chrono::high_resolution_clock::now();
        
        CRT_NTT_multiply_parallel(a, b, ab, n_, p_, rank, size);

      
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000>> elapsed = End - Start;
        ans += elapsed.count();
        
        // 确保所有进程完成计算
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Only rank 0 checks and writes results
        if (rank == 0) {
            fCheck(ab, n_, i);
            std::cout << "average latency for n = " << n_ << " p = " << p_ << " : "
                    << ans << " (us) " << std::endl;
            fWrite(ab, n_, i);
        }
        
        // 同步所有进程
        MPI_Barrier(MPI_COMM_WORLD);
        
        delete[] ab;
    }
    // 清理全局CRT变量
    cleanup_global_crt_values();
    
    MPI_Finalize();
    return 0;
}
