#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// 常量内存优化：存储频繁访问的Montgomery参数
__constant__ uint64_t c_mod;        // 模数
__constant__ uint64_t c_r;          // Montgomery R = 2^64 mod m
__constant__ uint64_t c_r2;         // R^2 mod m
__constant__ uint64_t c_r_inv;      // R^(-1) mod m
__constant__ uint64_t c_n_inv;      // -m^(-1) mod 2^64



// ---------------- 优化的64位Montgomery结构体 ----------------
struct __align__(16) MontgomeryOpt {
    uint64_t mod;     // 模数 (必须为奇数)
    uint64_t r;       // R = 2^64 mod m
    uint64_t r2;      // R^2 mod m
    uint64_t r_inv;   // R^(-1) mod m
    uint64_t n_inv;   // -m^(-1) mod 2^64
    uint64_t mask;    // 2^32 - 1

    __host__ __device__ explicit MontgomeryOpt(uint64_t m) : mod(m), mask(0xFFFFFFFFULL) {
        init_montgomery_params();
    }

    // 使用扩展欧几里得算法计算模逆元
    __host__ __device__ uint64_t extended_gcd(uint64_t a, uint64_t b, int64_t& x, int64_t& y) const {
        if (a == 0) {
            x = 0; y = 1;
            return b;
        }
        int64_t x1, y1;
        uint64_t gcd = extended_gcd(b % a, a, x1, y1);
        x = y1 - (int64_t)(b / a) * x1;
        y = x1;
        return gcd;
    }

    __host__ __device__ uint64_t mod_inverse(uint64_t a, uint64_t m) const {
        int64_t x, y;
        uint64_t g = extended_gcd(a % m, m, x, y);
        if (g != 1) return 0; // 逆元不存在
        return (x % (int64_t)m + (int64_t)m) % (int64_t)m;
    }

    __host__ __device__ void init_montgomery_params() {
        // 计算 -m^(-1) mod 2^64 使用牛顿迭代法
        uint64_t inv = mod;
        for (int i = 0; i < 6; i++) { // 6次迭代对64位足够
            inv = inv * (2 - mod * inv);
        }
        n_inv = ~inv + 1; // -inv mod 2^64

        // 计算 R = 2^64 mod m
        // 使用重复平方法
        r = 1;
        for (int i = 0; i < 64; i++) {
            r = (r * 2) % mod;
        }

        // 计算 R^2 mod m
        __uint128_t r_128 = r;
        r2 = (r_128 * r_128) % mod;

        // 计算 R^(-1) mod m
        r_inv = mod_inverse(r, mod);
    }

    // 优化的Montgomery约简：CIOS算法的64位版本
    __host__ __device__ uint64_t mont_reduce(uint64_t T) const {
        // Montgomery约简
        uint64_t m = T * n_inv;
        __uint128_t tm = (__uint128_t)m * mod + T ;
        uint64_t result = tm >> 64;
        
        return (result >= mod) ? result - mod : result;
    }


    // 使用常量内存的快速Montgomery乘法
    __device__ uint64_t mont_mul_fast(uint64_t a, uint64_t b) const {
        __uint128_t ab = (__uint128_t)a * b;
        uint64_t lo = (uint64_t)ab;
        uint64_t m = lo * c_n_inv;
        __uint128_t tm = (__uint128_t)m * c_mod;
        __uint128_t t = ab + tm;
        uint64_t result = t >> 64;
        return (result >= c_mod) ? result - c_mod : result;
    }



    // Montgomery域中的加法
    __host__ __device__ uint64_t mont_add(uint64_t a, uint64_t b) const {
        uint64_t sum = a + b;
        return (sum >= mod) ? sum - mod : sum;
    }

    // Montgomery域中的减法
    __host__ __device__ uint64_t mont_sub(uint64_t a, uint64_t b) const {
        return (a >= b) ? a - b : a + mod - b;
    }

    // 快速模加法（使用常量内存）
    __device__ uint64_t mont_add_fast(uint64_t a, uint64_t b) const {
        uint64_t sum = a + b;
        return (sum >= c_mod) ? sum - c_mod : sum;
    }

    // 快速模减法（使用常量内存）
    __device__ uint64_t mont_sub_fast(uint64_t a, uint64_t b) const {
        return (a >= b) ? a - b : a + c_mod - b;
    }
};

// Montgomery域中的幂运算
__host__ __device__ uint64_t mont_pow(uint64_t base, uint64_t exp, const MontgomeryOpt& mont) {
    uint64_t result = mont.mont_reduce(mont.r2);
    uint64_t base_mont = mont.mont_reduce(base * mont.r2);
    
    while (exp > 0) {
        if (exp & 1) {
            result = mont.mont_reduce(result * base_mont);
        }
        base_mont = mont.mont_reduce(base_mont * base_mont);
        exp >>= 1;
    }
    return mont.mont_reduce(result);
}

// ---------------- CUDA Kernels ----------------


// 传统位反转kernel（备用）
__global__ void bit_reverse_kernel(uint64_t *a, const int *rev, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && idx < rev[idx]) {
        uint64_t tmp = a[idx];
        a[idx] = a[rev[idx]];
        a[rev[idx]] = tmp;
    }
}

// 高度优化的Montgomery点值乘法kernel
__global__ void multiply_pointwise_mont_optimized(uint64_t *fa, const uint64_t *fb, 
                                                  MontgomeryOpt mont, int len) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // 向量化处理：每个线程处理多个元素
    for (int base = gid * 4; base < len; base += stride * 4) {
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            int idx = base + i;
            if (idx < len) {
                uint64_t a_val = fa[idx];
                uint64_t b_val = fb[idx];
                fa[idx] = mont.mont_reduce(a_val * b_val);
            }
        }
    }
}


// Montgomery归一化kernel
__global__ void apply_inv_mont(uint64_t *a, uint64_t inv_n_mont, MontgomeryOpt mont, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = idx; i < n; i += stride) {
        a[i] = mont.mont_mul_fast(a[i], inv_n_mont);
    }
}

// 预计算旋转因子kernel（Montgomery域）
__global__ void fill_twiddle_mont(uint64_t* d_fwd, uint64_t* d_inv,
                                  int offset, int half,
                                  uint64_t g_n, uint64_t g_n_inv,
                                  MontgomeryOpt mont) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half) return;
    
    uint64_t val_f, val_i;
    if (j == 0) {
        val_f = mont.mont_reduce(mont.r2);
        val_i = mont.mont_reduce(mont.r2);
    } else {
        val_f = mont_pow(g_n, j, mont);
        val_i = mont_pow(g_n_inv, j, mont);
        val_f = mont.mont_reduce(val_f * mont.r2);
        val_i = mont.mont_reduce(val_i * mont.r2);
    }
    d_fwd[offset + j] = val_f;
    d_inv[offset + j] = val_i;
}

// 高性能NTT butterfly kernel（warp优化版本）
__global__ void ntt_butterfly_mont_warp(uint64_t* a, const uint64_t* twiddles, 
                                        int len, MontgomeryOpt mont, int n) {
    int half = len >> 1;
    int lane = threadIdx.x & 31;
    if (lane >= half) return;
    
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int num_warps = (gridDim.x * blockDim.x) >> 5;
    int segments = n / len;
    
    uint64_t twiddle = twiddles[lane];
    
    for (int seg = warp_id; seg < segments; seg += num_warps) {
        int base = seg * len;
        int pos1 = base + lane;
        int pos2 = pos1 + half;
        
        uint64_t u = a[pos1];
        uint64_t v = mont.mont_mul_fast(a[pos2], twiddle);
        
        a[pos1] = mont.mont_add_fast(u, v);
        a[pos2] = mont.mont_sub_fast(u, v);
    }
}

// 标准NTT butterfly kernel
__global__ void ntt_butterfly_mont(uint64_t *a, const uint64_t *twiddles,
                                   int len, MontgomeryOpt mont, int n) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    int half = len >> 1;
    int segments = n / len;
    long long total_work = (long long)segments * half;
    
    for (long long work_id = global_tid; work_id < total_work; work_id += total_threads) {
        int seg_id = work_id / half;
        int j = work_id % half;
        
        int base = seg_id * len;
        int pos1 = base + j;
        int pos2 = pos1 + half;
        
        uint64_t twiddle = twiddles[j];
        uint64_t u = a[pos1];
        uint64_t v = mont.mont_reduce(a[pos2] * twiddle);
        
        a[pos1] = mont.mont_add(u, v);
        a[pos2] = mont.mont_sub(u, v);
    }
}

// ---------------- 内存池管理 ----------------
// 优化的内存池，避免频繁分配/释放GPU内存，提高性能
// 支持多个缓冲区和异步内存操作
struct MemoryPool {
    uint64_t *d_buffer1 = nullptr;  // GPU缓冲区1
    uint64_t *d_buffer2 = nullptr;  // GPU缓冲区2
    int *d_rev_buffer = nullptr;    // 位反转表缓冲区
    uint64_t *d_temp_buffer = nullptr; // 临时缓冲区
    size_t max_size = 0;            // 最大内存大小
    size_t max_rev_size = 0;        // 最大位反转表大小

    void init(size_t size) {
        if (size > max_size) {
            // 只在需要更大内存时才重新分配
            if (d_buffer1) cudaFree(d_buffer1);
            if (d_buffer2) cudaFree(d_buffer2);
            if (d_temp_buffer) cudaFree(d_temp_buffer);

            cudaMalloc(&d_buffer1, size);
            cudaMalloc(&d_buffer2, size);
            cudaMalloc(&d_temp_buffer, size);
            max_size = size;
        }
    }
    
    void init_rev_buffer(size_t size) {
        if (size > max_rev_size) {
            if (d_rev_buffer) cudaFree(d_rev_buffer);
            cudaMalloc(&d_rev_buffer, size);
            max_rev_size = size;
        }
    }

    // 重置缓冲区（只重置需要的部分以提高性能）
    void reset_buffers(size_t size, cudaStream_t stream = 0) {
        if (d_buffer1 && size <= max_size) {
            cudaMemsetAsync(d_buffer1, 0, size, stream);
            cudaMemsetAsync(d_buffer2, 0, size, stream);
        }
    }

    ~MemoryPool() {
        if (d_buffer1) cudaFree(d_buffer1);
        if (d_buffer2) cudaFree(d_buffer2);
        if (d_rev_buffer) cudaFree(d_rev_buffer);
        if (d_temp_buffer) cudaFree(d_temp_buffer);
    }
};

static MemoryPool g_mem_pool_mont;

// ---------------- 旋转因子池管理 ----------------
struct TwiddlePool {
    uint64_t *d_fwd = nullptr;
    uint64_t *d_inv = nullptr;
    std::vector<int> offset;
    int log_n = 0;
    int len = 0;
};

void build_twiddle_pool(int len, uint64_t p, uint64_t g, TwiddlePool &pool) {
    pool.len = len;
    pool.log_n = 0;
    while ((1 << pool.log_n) < len) ++pool.log_n;
    
    int total = len - 1;
    pool.offset.assign(pool.log_n + 1, 0);
    
    MontgomeryOpt mont(p);
    cudaMalloc(&pool.d_fwd, total * sizeof(uint64_t));
    cudaMalloc(&pool.d_inv, total * sizeof(uint64_t));
    
    int cur = 0;
    for (int stage = 1; stage <= pool.log_n; ++stage) {
        int seg_len = 1 << stage;
        int half = seg_len >> 1;
        pool.offset[stage] = cur;
        
        uint64_t g_n = mont_pow(g, (p - 1) / seg_len, mont);
        uint64_t g_n_inv = mont_pow(g, (p - 1) - (p - 1) / seg_len, mont);
        
        int block = 256;
        int grid = (half + block - 1) / block;
        fill_twiddle_mont<<<grid, block>>>(pool.d_fwd, pool.d_inv, cur, half, g_n, g_n_inv, mont);
        cur += half;
    }
    cudaDeviceSynchronize();
}

void destroy_twiddle_pool(TwiddlePool &pool) {
    if (pool.d_fwd) cudaFree(pool.d_fwd);
    if (pool.d_inv) cudaFree(pool.d_inv);
    pool.d_fwd = pool.d_inv = nullptr;
    pool.offset.clear();
}

// ---------------- 主要NTT函数 ----------------
void compute_bit_reverse(int *h_rev, int n) {
    int bit = 0;
    while ((1 << bit) < n) ++bit;
    h_rev[0] = 0;
    for (int i = 1; i < n; ++i) {
        h_rev[i] = (h_rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    }
}

void NTT_montgomery_cuda(uint64_t *d_a, int n, bool invert, uint64_t p, 
                        const TwiddlePool &pool, cudaStream_t stream = 0) {
    MontgomeryOpt mont(p);
    
    // 设置常量内存
    cudaMemcpyToSymbol(c_mod, &mont.mod, sizeof(uint64_t));
    cudaMemcpyToSymbol(c_r, &mont.r, sizeof(uint64_t));
    cudaMemcpyToSymbol(c_r2, &mont.r2, sizeof(uint64_t));
    cudaMemcpyToSymbol(c_r_inv, &mont.r_inv, sizeof(uint64_t));
    cudaMemcpyToSymbol(c_n_inv, &mont.n_inv, sizeof(uint64_t));

    
    // 位反转（使用内存池）
    size_t rev_bytes = n * sizeof(int);
    g_mem_pool_mont.init_rev_buffer(rev_bytes);
    int *d_rev = g_mem_pool_mont.d_rev_buffer;
    
    int *h_rev = new int[n];
    compute_bit_reverse(h_rev, n);
    cudaMemcpyAsync(d_rev, h_rev, rev_bytes, cudaMemcpyHostToDevice, stream);
    delete[] h_rev;
    
    int block = 256;
    int grid = (n + block - 1) / block;
    bit_reverse_kernel<<<grid, block, 0, stream>>>(d_a, d_rev, n);
    
    
    // NTT主循环
    for (int stage = 1, len = 2; len <= n; ++stage, len <<= 1) {
        int half = len >> 1;
        const uint64_t *d_twiddle = (invert ? pool.d_inv : pool.d_fwd) + pool.offset[stage];
        
        
            // 使用标准butterfly
            int threads = 256;
            long long total_work = (static_cast<long long>(n) / len) * half;
            int blocks = (total_work + threads - 1) / threads;
            blocks = std::min(blocks, 32768);
            ntt_butterfly_mont<<<blocks, threads, 0, stream>>>(d_a, d_twiddle, len, mont, n);
        
        cudaStreamSynchronize(stream);
    }
    
    // 逆变换归一化
    if (invert) {
        uint64_t inv_n = mont_pow(n, p - 2, mont);
        uint64_t inv_n_mont = mont.mont_reduce(inv_n * mont.r2);
        apply_inv_mont<<<grid, block, 0, stream>>>(d_a, inv_n_mont, mont, n);
        cudaStreamSynchronize(stream);
    }
}

// Montgomery多项式乘法
void NTT_multiply_montgomery_cuda(const uint64_t *h_a, const uint64_t *h_b, uint64_t *h_res,
                                  int n, uint64_t p, cudaStream_t stream = 0) {
    int len = n << 1;
    size_t bytes = len * sizeof(uint64_t);

    // 使用内存池避免频繁分配
    g_mem_pool_mont.init(bytes);
    uint64_t *d_fa = g_mem_pool_mont.d_buffer1;
    uint64_t *d_fb = g_mem_pool_mont.d_buffer2;
    
    // 异步重置GPU内存
    g_mem_pool_mont.reset_buffers(bytes, stream);
    
    // 转换到Montgomery域
    MontgomeryOpt mont(p);
    uint64_t *h_fa_mont = new uint64_t[n];
    uint64_t *h_fb_mont = new uint64_t[n];

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        h_fa_mont[i] = mont.mont_reduce(h_a[i] * mont.r2);
        h_fb_mont[i] = mont.mont_reduce(h_b[i] * mont.r2);
    }
    
    cudaMemcpyAsync(d_fa, h_fa_mont, n * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_fb, h_fb_mont, n * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    
    delete[] h_fa_mont;
    delete[] h_fb_mont;
    
    // 构建旋转因子池
    TwiddlePool pool;
    build_twiddle_pool(len, p, 3, pool);
    
    // 前向NTT
    NTT_montgomery_cuda(d_fa, len, false, p, pool, stream);
    NTT_montgomery_cuda(d_fb, len, false, p, pool, stream);
    
    // 点值乘法
        int block = 256;
        int grid = (len + block * 4 - 1) / (block * 4);
        multiply_pointwise_mont_optimized<<<grid, block, 0, stream>>>(d_fa, d_fb, mont, len);
    cudaStreamSynchronize(stream);
    
    // 逆向NTT
    NTT_montgomery_cuda(d_fa, len, true, p, pool, stream);
    
    // 从Montgomery域转换并复制结果
    uint64_t *h_temp = new uint64_t[len];
    cudaMemcpyAsync(h_temp, d_fa, len * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
#pragma omp parallel for
    for (int i = 0; i < 2 * n - 1; i++) {
        h_res[i] = mont.mont_reduce(h_temp[i]);
    }
    
    delete[] h_temp;
    destroy_twiddle_pool(pool);
}

// CRT版本的Montgomery多项式乘法
void CRT_NTT_multiply_montgomery_cuda(uint64_t *a, uint64_t *b, uint64_t *result, int n, uint64_t p) {
    constexpr int MOD_COUNT = 4;
    const uint64_t MOD_LIST[MOD_COUNT] = {1004535809ULL, 1224736769ULL, 469762049ULL, 998244353ULL};
    int result_len = (n << 1) - 1;
    
    
    uint64_t *mod_results[MOD_COUNT];
    for (int i = 0; i < MOD_COUNT; ++i) {
        cudaMallocHost(&mod_results[i], result_len * sizeof(uint64_t));
        memset(mod_results[i], 0, result_len * sizeof(uint64_t));
    }
    
    __uint128_t M = 1;
    for (int i = 0; i < MOD_COUNT; ++i) M *= MOD_LIST[i];
    
    __uint128_t MI_VALUES[MOD_COUNT];
    uint64_t MI_INV_VALUES[MOD_COUNT];
    
    
    for (int i = 0; i < MOD_COUNT; ++i) {
        MI_VALUES[i] = M / MOD_LIST[i];
        MontgomeryOpt mont(MOD_LIST[i]);
        uint64_t Mi_mod = MI_VALUES[i] % MOD_LIST[i];
        MI_INV_VALUES[i] = mont_pow(Mi_mod, MOD_LIST[i] - 2, mont);
    }
    
    cudaStream_t streams[MOD_COUNT];
    for (int i = 0; i < MOD_COUNT; ++i) cudaStreamCreate(&streams[i]);
    
    uint64_t *h_ta[MOD_COUNT], *h_tb[MOD_COUNT];
    for (int i = 0; i < MOD_COUNT; ++i) {
        cudaMallocHost(&h_ta[i], n * sizeof(uint64_t));
        cudaMallocHost(&h_tb[i], n * sizeof(uint64_t));
    }
    
    for (int idx = 0; idx < MOD_COUNT; ++idx) {
        for (int j = 0; j < n; ++j) {
            h_ta[idx][j] = a[j] % MOD_LIST[idx];
            h_tb[idx][j] = b[j] % MOD_LIST[idx];
        }
        NTT_multiply_montgomery_cuda(h_ta[idx], h_tb[idx], mod_results[idx], n, MOD_LIST[idx], streams[idx]);
    }
    
    for (int i = 0; i < MOD_COUNT; ++i) cudaStreamSynchronize(streams[i]);
    for (int i = 0; i < MOD_COUNT; ++i) cudaStreamDestroy(streams[i]);
    
    // CRT合并
#pragma omp parallel for
    for (int j = 0; j < result_len; ++j) {
        __uint128_t sum = 0;
        for (int i = 0; i < MOD_COUNT; ++i) {
            __uint128_t term = MI_VALUES[i] * ((mod_results[i][j] * MI_INV_VALUES[i]) % MOD_LIST[i]);
            sum += term;
        }
        sum %= M;
        result[j] = static_cast<uint64_t>(sum % p);
    }
    
    // 清理内存
    for (int i = 0; i < MOD_COUNT; ++i) {
        cudaFreeHost(mod_results[i]);
        cudaFreeHost(h_ta[i]);
        cudaFreeHost(h_tb[i]);
    }
}

// ---------------- 文件I/O和测试函数 ----------------
void fRead(uint64_t *a, uint64_t *b, int *n, uint64_t *p, int input_id) {
    std::string str1 = "./nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    std::ifstream fin(strin);
    if (!fin.is_open()) {
        std::cerr << "无法打开输入文件 " << strin << std::endl;
        exit(EXIT_FAILURE);
    }
    fin >> *n >> *p;
    for (int i = 0; i < *n; ++i) fin >> a[i];
    for (int i = 0; i < *n; ++i) fin >> b[i];
}

void fCheck(uint64_t *ab, int n, int input_id) {
    std::string str1 = "./nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    std::ifstream fin(strout);
    if (!fin.is_open()) {
        std::cerr << "无法打开校验文件 " << strout << std::endl;
        return;
    }
    for (int i = 0; i < 2 * n - 1; ++i) {
        uint64_t x; fin >> x;
        if (x != ab[i]) {
            std::cout << "多项式乘法结果错误" << std::endl;
            return;
        }
    }
    std::cout << "多项式乘法结果正确" << std::endl;
}

void fWrite(uint64_t *ab, int n, int input_id) {
    std::string str1 = "./files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    std::ofstream fout(strout);
    for (int i = 0; i < 2 * n - 1; ++i) fout << ab[i] << '\n';
}

// ---------------- 主函数 ----------------
int main() {
    cudaSetDevice(0);
    cudaFree(0); // 预热GPU
    
    static uint64_t a[300000], b[300000], ab[300000];
    
    int test_begin = 0;
    int test_end = 4;
    
    for (int idx = test_begin; idx <= test_end; ++idx) {
        
        int n;
        uint64_t p;
        fRead(a, b, &n, &p, idx);
        std::fill(ab, ab + (2 * n - 1), 0ULL);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        if (p > (1ULL << 32)) {
            CRT_NTT_multiply_montgomery_cuda(a, b, ab, n, p);
        } else {
            NTT_multiply_montgomery_cuda(a, b, ab, n, p);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        
        fCheck(ab, n, idx);
        std::cout << "n = " << n << ", p = " << p << " 的平均延迟: "
                  << elapsed.count() << " (ms)" << std::endl;
        fWrite(ab, n, idx);
    }
    
    return 0;
} 