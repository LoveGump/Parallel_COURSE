#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

//Barrett 结构体
struct __align__(16) Barrett {
    uint64_t mod;
    uint64_t k;
    uint64_t mu;

    __host__ __device__ explicit Barrett(uint64_t m)
        : mod(m) {
        k = 31;
        mu = (1ull << 62) / m;
    }

    __host__ __device__ uint64_t reduce(uint64_t x) const {
        uint64_t q = ((__uint128_t)x * mu) >> 62;
        uint64_t r = x - q * mod;
        return r >= mod ? r - mod : r;
    }
};

// 设备通用函数
__host__ __device__ uint64_t pow_mod(uint64_t base, uint64_t exp, const Barrett &br) {
    uint64_t res = 1;
    base = br.reduce(base);
    while (exp) {
        if (exp & 1)
            res = br.reduce(res * base);
        base = br.reduce(base * base);
        exp >>= 1;
    }
    return res;
}

// CUDA Kernels
// 位反转：将数组 a 中的元素按照 rev 数组指定的顺序进行反转
__global__ void bit_reverse_kernel(uint64_t *a, const int *rev, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && idx < rev[idx]) {
        uint64_t tmp = a[idx];
        a[idx] = a[rev[idx]];
        a[rev[idx]] = tmp;
    }
}

// 128bit 合并访存的点值乘：每线程处理2个64位
__global__ void multiply_pointwise_kernel_vec(uint64_t *fa, const uint64_t *fb, Barrett br, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int vecLen = len >> 1;

    ulonglong2 *fa_vec = reinterpret_cast<ulonglong2 *>(fa);
    const ulonglong2 *fb_vec = reinterpret_cast<const ulonglong2 *>(fb);

    for (int i = idx; i < vecLen; i += stride) {
        ulonglong2 a = fa_vec[i];
        ulonglong2 b = fb_vec[i];
        a.x = br.reduce(a.x * b.x);
        a.y = br.reduce(a.y * b.y);
        fa_vec[i] = a;
    }

    if ((len & 1) && idx == 0) {
        int last = len - 1;
        fa[last] = br.reduce(fa[last] * fb[last]);
    }
}

// 128bit 合并访存的归一化 kernel
__global__ void apply_inv_kernel_vec(uint64_t *a, uint64_t inv_n, Barrett br,
                                     int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int vecLen = n >> 1;  // 两个元素一向量

    ulonglong2 *a_vec = reinterpret_cast<ulonglong2 *>(a);
    for (int i = idx; i < vecLen; i += stride) {
        ulonglong2 v = a_vec[i];
        v.x = br.reduce(v.x * inv_n);
        v.y = br.reduce(v.y * inv_n);
        a_vec[i] = v;
    }

    if ((n & 1) && idx == 0) {
        int last = n - 1;
        a[last] = br.reduce(a[last] * inv_n);
    }
}

// GPU 端生成旋转因子表的 kernel
__global__ void fill_twiddle_kernel(uint64_t *d_fwd, uint64_t *d_inv, int offset, int half, uint64_t g_n, uint64_t g_n_inv, Barrett br) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half)
        return;
    uint64_t val_f = (j == 0) ? 1ull : pow_mod(g_n, j, br);
    uint64_t val_i = (j == 0) ? 1ull : pow_mod(g_n_inv, j, br);
    d_fwd[offset + j] = val_f;
    d_inv[offset + j] = val_i;
}

// --------------- TwiddlePool ---------------
struct TwiddlePool {
    uint64_t *d_fwd = nullptr;  // 正向旋转因子
    uint64_t *d_inv = nullptr;  // 逆向旋转因子
    std::vector<int> offset;    // host 端每 stage 起始下标
    int logN = 0;
    int len = 0;
};

static void build_twiddle_pool(int len, uint64_t p, uint64_t g, TwiddlePool &pool) {
    pool.len = len;
    pool.logN = 0;
    while ((1 << pool.logN) < len)
        ++pool.logN;
    int total = len - 1;
    pool.offset.assign(pool.logN + 1, 0);

    Barrett br(p);
    // 直接在 GPU 上生成旋转因子表
    cudaMalloc(&pool.d_fwd, total * sizeof(uint64_t));
    cudaMalloc(&pool.d_inv, total * sizeof(uint64_t));

    int cur = 0;
    for (int stage = 1; stage <= pool.logN; ++stage) {
        int segLen = 1 << stage;  //
        int half = segLen >> 1;
        pool.offset[stage] = cur;

        uint64_t g_n = pow_mod(g, (p - 1) / segLen, br);
        uint64_t g_n_inv = pow_mod(g, (p - 1) - (p - 1) / segLen, br);

        int block = 256;
        int grid = (half + block - 1) / block;
        fill_twiddle_kernel<<<grid, block>>>(pool.d_fwd, pool.d_inv, cur, half, g_n, g_n_inv, br);
        cur += half;
    }
    cudaDeviceSynchronize();
}

static void destroy_twiddle_pool(TwiddlePool &pool) {
    if (pool.d_fwd)
        cudaFree(pool.d_fwd);
    if (pool.d_inv)
        cudaFree(pool.d_inv);
    pool.d_fwd = pool.d_inv = nullptr;
    pool.offset.clear();
}

__global__ void ntt_stage_kernel_table(uint64_t *a, const uint64_t *g_pows,
                                       int len, Barrett br, int n) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    int half = len >> 1;
    int blocks = n / len;  // 蝶形块数

    // 将二维循环(base,j) 展平为一维 workID = baseIdx*half + j
    long long totalWork = (long long)blocks * half;  // 总蝶形单元数

    for (long long workId = global_tid; workId < totalWork;
         workId += totalThreads) {
        int baseIdx = workId / half;                 // 第几个 len 段
        int j = workId - (long long)baseIdx * half;  // 段内索引

        int base = baseIdx * len;
        int pos1 = base + j;
        int pos2 = pos1 + half;

        uint64_t g_pow = g_pows[j];
        uint64_t u = a[pos1];
        uint64_t v = br.reduce(a[pos2] * g_pow);
        uint64_t sum = u + v;
        if (sum >= br.mod)
            sum -= br.mod;
        uint64_t diff = (u >= v) ? u - v : u + br.mod - v;
        a[pos1] = sum;
        a[pos2] = diff;
    }
}

// Host 端 GPU NTT - OpenMP 优化版本
void compute_bitrev(int *h_rev, int n) {
    int bit = 0;
    while ((1 << bit) < n)
        ++bit;
    h_rev[0] = 0;
    
    // OpenMP 并行化位反转表计算
    #pragma omp parallel for schedule(static) if(n > 1024)
    for (int i = 1; i < n; ++i) {
        h_rev[i] = (h_rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    }
}

// 使用预分配内存池
struct MemoryPool {
    uint64_t *d_buffer1 = nullptr;  // GPU缓冲区1
    uint64_t *d_buffer2 = nullptr;  // GPU缓冲区2
    int *d_rev_buffer = nullptr;    // 位反转表
    size_t max_size = 0;            // 最大内存大小

    void init(size_t size) {
        if (size > max_size) {
            cudaFree(d_buffer1);
            cudaFree(d_buffer2);
            cudaFree(d_rev_buffer);

            cudaMalloc(&d_buffer1, size);
            cudaMalloc(&d_buffer2, size);
            cudaMalloc(&d_rev_buffer, size / sizeof(uint64_t) * sizeof(int));
            max_size = size;
        }
    }

    ~MemoryPool() {
        if (d_buffer1)
            cudaFree(d_buffer1);
        if (d_buffer2)
            cudaFree(d_buffer2);
        if (d_rev_buffer)
            cudaFree(d_rev_buffer);
    }
};

static MemoryPool g_mem_pool;

void NTT_cuda(uint64_t *d_a, int n, bool invert, uint64_t p,
              const TwiddlePool &pool, cudaStream_t stream = 0) {
    Barrett br(p);

    // 计算位反转表
    int *d_rev;
    int *h_rev = new int[n];
    int bit = 0;
    while ((1 << bit) < n)
        ++bit;
    h_rev[0] = 0;
    for (int i = 1; i < n; ++i) {
        h_rev[i] = (h_rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    }

    cudaMalloc(&d_rev, n * sizeof(int));                                             // 分配GPU内存
    cudaMemcpyAsync(d_rev, h_rev, n * sizeof(int), cudaMemcpyHostToDevice, stream);  // 拷贝位反转表到GPU
    delete[] h_rev;

    // 位反转
    int block = 256;
    int grid = (n + block - 1) / block;
    bit_reverse_kernel<<<grid, block, 0, stream>>>(d_a, d_rev, n);
    cudaFree(d_rev);
    cudaStreamSynchronize(stream);

    // 蝶形运算
    for (int stage = 1, len = 2; len <= n; ++stage, len <<= 1) {
        int half = len >> 1;
        const uint64_t *d_twiddle = (invert ? pool.d_inv : pool.d_fwd) + pool.offset[stage];

        int threads = 256;
        long long totalWork = (static_cast<long long>(n) / len) * half;
        int blocks = (totalWork + threads - 1) / threads;
        blocks = std::min(blocks, 32768);
        ntt_stage_kernel_table<<<blocks, threads, 0, stream>>>(d_a, d_twiddle, len, br, n);

        cudaStreamSynchronize(stream);
    }
    if (invert) {
        uint64_t inv_n = pow_mod(n, p - 2, br);
        apply_inv_kernel_vec<<<grid, block, 0, stream>>>(d_a, inv_n, br, n);
        cudaStreamSynchronize(stream);
    }
}

void NTT_multiply_cuda(const uint64_t *h_a, const uint64_t *h_b, uint64_t *h_res, int n, uint64_t p, cudaStream_t stream = 0) {
    int len = n << 1;
    size_t bytes = len * sizeof(uint64_t);  // 分配内存大小（字节数）

    // 使用内存池避免频繁分配
    g_mem_pool.init(bytes);
    uint64_t *d_fa = g_mem_pool.d_buffer1;
    uint64_t *d_fb = g_mem_pool.d_buffer2;

    // 初始化GPU内存
    cudaMemset(d_fa, 0, bytes);
    cudaMemset(d_fb, 0, bytes);

    // 数据传输
    cudaMemcpyAsync(d_fa, h_a, n * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_fb, h_b, n * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

    // 构建旋转因子池
    TwiddlePool pool;
    build_twiddle_pool(len, p, 3, pool);

    // 前向 NTT
    NTT_cuda(d_fa, len, false, p, pool, stream);
    NTT_cuda(d_fb, len, false, p, pool, stream);

    Barrett br(p);
    // 点值相乘
    int block = 256;
    int grid = (len + block - 1) / block;
    multiply_pointwise_kernel_vec<<<grid, block, 0, stream>>>(d_fa, d_fb, br, len);  // kernel调用 点值相乘

    cudaStreamSynchronize(stream);  // 等待 同步

    // 逆向 NTT
    NTT_cuda(d_fa, len, true, p, pool, stream);

    cudaMemcpyAsync(h_res, d_fa, (2 * n - 1) * sizeof(uint64_t),
                    cudaMemcpyDeviceToHost, stream);  //将结果拷贝回主机

    destroy_twiddle_pool(pool);
}

// OpenMP + CUDA 混合优化版本
void NTT_multiply_cuda_openmp(const uint64_t *h_a, const uint64_t *h_b, uint64_t *h_res, int n, uint64_t p, cudaStream_t stream = 0) {
    int len = n << 1;
    size_t bytes = len * sizeof(uint64_t);

    // 使用内存池避免频繁分配
    g_mem_pool.init(bytes);
    uint64_t *d_fa = g_mem_pool.d_buffer1;
    uint64_t *d_fb = g_mem_pool.d_buffer2;

    // 根据数据规模决定是否使用OpenMP并行初始化
    if (n > 8192) {
        // 大规模数据使用OpenMP并行初始化GPU内存
        #pragma omp parallel sections
        {
            #pragma omp section
            cudaMemset(d_fa, 0, bytes);
            #pragma omp section 
            cudaMemset(d_fb, 0, bytes);
        }
    } else {
        // 小规模数据串行处理避免开销
        cudaMemset(d_fa, 0, bytes);
        cudaMemset(d_fb, 0, bytes);
    }

    // 数据传输
    cudaMemcpyAsync(d_fa, h_a, n * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_fb, h_b, n * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

    // 构建旋转因子池
    TwiddlePool pool;
    build_twiddle_pool(len, p, 3, pool);

    // 前向 NTT
    NTT_cuda(d_fa, len, false, p, pool, stream);
    NTT_cuda(d_fb, len, false, p, pool, stream);

    Barrett br(p);
    // 点值相乘
    int block = 256;
    int grid = (len + block - 1) / block;
    multiply_pointwise_kernel_vec<<<grid, block, 0, stream>>>(d_fa, d_fb, br, len);

    cudaStreamSynchronize(stream);

    // 逆向 NTT
    NTT_cuda(d_fa, len, true, p, pool, stream);

    // 异步数据传输回主机
    cudaMemcpyAsync(h_res, d_fa, (2 * n - 1) * sizeof(uint64_t),
                    cudaMemcpyDeviceToHost, stream);

    destroy_twiddle_pool(pool);
}

// CRT + CUDA 
void CRT_NTT_multiply_cuda(uint64_t *a, uint64_t *b, uint64_t *result, int n, uint64_t p) {
    constexpr int MOD_COUNT = 4;
    const uint64_t MOD_LIST[MOD_COUNT] = {1004535809ULL, 1224736769ULL, 469762049ULL, 998244353ULL};
    int result_len = (n << 1) - 1;

    // 使用对齐内存分配提高缓存性能
    uint64_t *mod_results[MOD_COUNT];  // 模数结果
    for (int i = 0; i < MOD_COUNT; ++i) {
        // 分配内存，并初始化为0
        cudaMallocHost(&mod_results[i], result_len * sizeof(uint64_t));
        memset(mod_results[i], 0, result_len * sizeof(uint64_t));
    }

    // 总模数 M
    __uint128_t M = 1;
    for (int i = 0; i < MOD_COUNT; ++i)
        M *= MOD_LIST[i];

    __uint128_t MI_VALUES[MOD_COUNT];
    uint64_t MI_INV_VALUES[MOD_COUNT];

    for (int i = 0; i < MOD_COUNT; ++i) {
        MI_VALUES[i] = M / MOD_LIST[i];
        Barrett br(MOD_LIST[i]);
        uint64_t Mi_mod = MI_VALUES[i] % MOD_LIST[i];
        MI_INV_VALUES[i] = pow_mod(Mi_mod, MOD_LIST[i] - 2, br);
    }

    // 为 4 个模数创建独立 CUDA stream
    cudaStream_t streams[MOD_COUNT];
    for (int i = 0; i < MOD_COUNT; ++i)
        cudaStreamCreate(&streams[i]);

    // 使用页锁定内存避免拷贝开销
    uint64_t *h_ta[MOD_COUNT], *h_tb[MOD_COUNT];
    for (int i = 0; i < MOD_COUNT; ++i) {
        cudaMallocHost(&h_ta[i], n * sizeof(uint64_t));
        cudaMallocHost(&h_tb[i], n * sizeof(uint64_t));
    }

    for (int i = 0; i < MOD_COUNT; ++i) {
        Barrett br(MOD_LIST[i]);
        for (int j = 0; j < n; ++j) {
            h_ta[i][j] = br.reduce(a[j]);
            h_tb[i][j] = br.reduce(b[j]);
        }
        NTT_multiply_cuda(h_ta[i], h_tb[i], mod_results[i], n, MOD_LIST[i], streams[i]);
    }

    // 等待所有 stream 完成
    for (int i = 0; i < MOD_COUNT; ++i)
        cudaStreamSynchronize(streams[i]);
    for (int i = 0; i < MOD_COUNT; ++i)
        cudaStreamDestroy(streams[i]);

// CRT 合并 (CPU)  opm并行化
#pragma omp parallel for
    for (int j = 0; j < result_len; ++j) {
        __uint128_t sum = 0;
        for (int i = 0; i < MOD_COUNT; ++i) {
            __uint128_t term =
                MI_VALUES[i] * ((mod_results[i][j] * MI_INV_VALUES[i]) % MOD_LIST[i]);
            sum += term;
        }
        sum %= M;
        result[j] = static_cast<uint64_t>(sum % p);
    }

    // 清理页锁定内存
    for (int i = 0; i < MOD_COUNT; ++i) {
        cudaFreeHost(mod_results[i]);
        cudaFreeHost(h_ta[i]);
        cudaFreeHost(h_tb[i]);
    }
}

// 多GPU + OpenMP 终极优化版本
void CRT_NTT_multiply_multigpu_openmp(uint64_t *a, uint64_t *b, uint64_t *result, int n, uint64_t p) {
    constexpr int MOD_COUNT = 4;
    const uint64_t MOD_LIST[MOD_COUNT] = {1004535809ULL, 1224736769ULL, 469762049ULL, 998244353ULL};
    int result_len = (n << 1) - 1;
    
    // 获取GPU数量
    int gpu_count = 1;
    cudaGetDeviceCount(&gpu_count);
    gpu_count = std::min(gpu_count, MOD_COUNT);
    
    // 设置OpenMP线程数为逻辑核心数
    omp_set_num_threads(omp_get_max_threads());
    
    // 使用对齐内存分配提高缓存性能
    uint64_t *mod_results[MOD_COUNT];
    #pragma omp parallel for
    for (int i = 0; i < MOD_COUNT; ++i) {
        cudaMallocHost(&mod_results[i], result_len * sizeof(uint64_t));
        memset(mod_results[i], 0, result_len * sizeof(uint64_t));
    }

    // 预计算CRT常数
    __uint128_t M = 1;
    for (int i = 0; i < MOD_COUNT; ++i)
        M *= MOD_LIST[i];

    __uint128_t MI_VALUES[MOD_COUNT];
    uint64_t MI_INV_VALUES[MOD_COUNT];

    #pragma omp parallel for
    for (int i = 0; i < MOD_COUNT; ++i) {
        MI_VALUES[i] = M / MOD_LIST[i];
        Barrett br(MOD_LIST[i]);
        uint64_t Mi_mod = MI_VALUES[i] % MOD_LIST[i];
        MI_INV_VALUES[i] = pow_mod(Mi_mod, MOD_LIST[i] - 2, br);
    }

    // 多GPU + 多stream 并行处理
    cudaStream_t streams[MOD_COUNT];
    uint64_t *h_ta[MOD_COUNT], *h_tb[MOD_COUNT];
    
    // 序列化初始化，避免竞争条件
    for (int i = 0; i < MOD_COUNT; ++i) {
        // 设置GPU设备（轮询分配）
        cudaSetDevice(i % gpu_count);
        cudaStreamCreate(&streams[i]);
        
        cudaMallocHost(&h_ta[i], n * sizeof(uint64_t));
        cudaMallocHost(&h_tb[i], n * sizeof(uint64_t));
    }
    
    // 并行预处理数据
    #pragma omp parallel for if(n > 8192)
    for (int i = 0; i < MOD_COUNT; ++i) {
        // 模数约简预处理
        Barrett br(MOD_LIST[i]);
        for (int j = 0; j < n; ++j) {
            h_ta[i][j] = br.reduce(a[j]);
            h_tb[i][j] = br.reduce(b[j]);
        }
    }
    
    // 串行执行NTT运算避免GPU竞争
    for (int i = 0; i < MOD_COUNT; ++i) {
        cudaSetDevice(i % gpu_count);
        NTT_multiply_cuda_openmp(h_ta[i], h_tb[i], mod_results[i], n, MOD_LIST[i], streams[i]);
    }

    // 同步所有streams
    for (int i = 0; i < MOD_COUNT; ++i) {
        cudaSetDevice(i % gpu_count);
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // CRT合并 - 保守的OpenMP优化版本
    #pragma omp parallel for schedule(static, 64) num_threads(omp_get_max_threads()) if(result_len > 1024)
    for (int j = 0; j < result_len; ++j) {
        __uint128_t sum = 0;
        
        // 手动展开内循环以提高性能
        for (int i = 0; i < MOD_COUNT; ++i) {
            __uint128_t term = MI_VALUES[i] * ((mod_results[i][j] * MI_INV_VALUES[i]) % MOD_LIST[i]);
            sum += term;
        }
        
        sum %= M;
        result[j] = static_cast<uint64_t>(sum % p);
    }

    // 清理资源
    #pragma omp parallel for
    for (int i = 0; i < MOD_COUNT; ++i) {
        cudaFreeHost(mod_results[i]);
        cudaFreeHost(h_ta[i]);
        cudaFreeHost(h_tb[i]);
    }
    
    // 恢复GPU 0
    cudaSetDevice(0);
}

void fRead(uint64_t *a, uint64_t *b, int *n, uint64_t *p, int input_id) {
    std::string str1 = "./nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    std::ifstream fin(strin);
    if (!fin.is_open()) {
        std::cerr << "无法打开输入文件 " << strin << "，退出" << std::endl;
        exit(EXIT_FAILURE);
    }
    fin >> *n >> *p;
    for (int i = 0; i < *n; ++i)
        fin >> a[i];
    for (int i = 0; i < *n; ++i)
        fin >> b[i];
}

void fCheck(uint64_t *ab, int n, int input_id) {
    std::string str1 = "./nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    std::ifstream fin(strout);
    if (!fin.is_open()) {
        std::cerr << "无法打开输出校验文件 " << strout << std::endl;
        return;
    }
    for (int i = 0; i < 2 * n - 1; ++i) {
        uint64_t x;
        fin >> x;
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
    for (int i = 0; i < 2 * n - 1; ++i)
        fout << ab[i] << '\n';
}

// 主函数 - OpenMP + CUDA 优化版本
int main() {
    // 选择 GPU 0
    cudaSetDevice(0);
    // 预热：触发 CUDA 上下文初始化，避免计时包含启动开销
    cudaFree(0);
    
    // 设置OpenMP参数
    omp_set_num_threads(omp_get_max_threads());
    omp_set_nested(1);  // 启用嵌套并行
    
    std::cout << "使用 " << omp_get_max_threads() << " 个OpenMP线程" << std::endl;
    
    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    std::cout << "检测到 " << gpu_count << " 个GPU设备" << std::endl;

    static uint64_t a[300000], b[300000], ab[300000], ab_openmp[300000];

    int test_begin = 0;
    int test_end = 4;

    
    for (int idx = test_begin; idx <= test_end; ++idx) {
        int n;
        uint64_t p;
        fRead(a, b, &n, &p, idx);
        std::fill(ab, ab + (2 * n), 0ull);
        std::fill(ab_openmp, ab_openmp + (2 * n ), 0ull);

        std::cout << "\n测试用例 " << idx << ": n = " << n << ", p = " << p << std::endl;
        
        // OpenMP + CUDA 优化版本
        fRead(a, b, &n, &p, idx);
        auto start2 = std::chrono::high_resolution_clock::now();
        if (p > (1ULL << 32)) {
            if (gpu_count > 1) {
                CRT_NTT_multiply_multigpu_openmp(a, b, ab_openmp, n, p);
            } else {
                CRT_NTT_multiply_cuda(a, b, ab_openmp, n, p);  // 使用已优化的单GPU版本
            }
        } else {
            NTT_multiply_cuda_openmp(a, b, ab_openmp, n, p);
        }
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed2 = end2 - start2;
        std::cout << "OpenMP+CUDA时间: " << std::fixed << std::setprecision(3) << elapsed2.count() << " ms" << std::endl;
       
        
        fCheck(ab_openmp, n, idx);
        fWrite(ab_openmp, n, idx);
    }
    
    return 0;
}
