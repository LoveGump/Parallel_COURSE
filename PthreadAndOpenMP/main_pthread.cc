#include <pthread.h>
#include <sys/time.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#define NUM_THREADS 8
// 全局变量和屏障
pthread_barrier_t barrier;  // 屏障

void fRead(uint64_t *a, uint64_t *b, int *n, uint64_t *p, int input_id) {
    // 数据输入函数
    std::string str1 = "/nttdata/";
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
    std::string str1 = "/nttdata/";
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
 * @brief 快速幂取模
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

// 任务类型枚举
enum TaskType {
    TASK_REVERSE,      // 位逆序置换任务
    TASK_NTT,          // NTT任务
    TASK_MULTIPLY,     // 多项式点值乘法
    TASK_APPLY_INV_N,  // 应用逆元
    TASK_CRT_COMBINE,  // CRT合并
    TASK_EXIT          // 退出任务
};

// 全局共享任务变量
struct GlobalTaskState {
    TaskType current_task;

    // 通用参数
    uint64_t *a;       // 输入多项式A
    uint64_t *b;       // 输入多项式B
    uint64_t *result;  // 结果数组

    int n;        // 多项式长度
    uint64_t p;   // 模数
    int g = 3;    // 原根
    bool invert;  // 是否进行逆变换

    // NTT任务参数
    int current_len;  // 当前蝶形长度
    uint64_t g_n;     // 当前单位根

    // 逆序置换参数
    int bit;   // 二进制位数
    int *rev;  // 逆序表

    // 逆元相关
    uint64_t inv_n;  // n的逆元

    // CRT相关
    uint64_t **mod_results;  // 各模数下的结果
    uint64_t *mod_list;      // 模数列表
    int mod_count;           // 模数个数
    int mod_index;           // 当前处理的模数索引

    // 数据复制任务参数
    uint64_t *dst_a;  // 目标数组A
    uint64_t *dst_b;  // 目标数组B
    int src_n;        // 源数组长度
    int dst_len;      // 目标数组长度(包括填充部分)

    __uint128_t M;  // 源数组A
    __uint128_t *Mi;
    uint64_t *Mi_inv;
} task_state;
/**
 * @brief 位逆序置换函数
 * @param a 输入数组
 * @param n 数组长度
 * @param bit log_2(n)向上取整
 */
void reverse(uint64_t *a, int n, int bit) {
    // a 是输入数组，n 是数组长度，bit 是二进制位数
    int *rev = new int[n];
    // i 的反转 = (i/2) 的反转去掉最高位后，再补上 i 的最低位作为最高位。
    rev[0] = 0;
    for (int i = 0; i < n; i++) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    }
    for (int i = 0; i < n; i++) {
        if (i < rev[i]) {
            uint64_t temp = a[i];
            a[i] = a[rev[i]];  // 位逆序置换
            a[rev[i]] = temp;
        }
    }
    delete[] rev;  // 释放逆序表内存
}

/**
 * @brief NTT变换函数
 * @param a 输入数组
 * @param n 数组长度
 * @param invert 是否进行逆变换
 * @param p 模数
 * @param g 原根
 * @details 该函数串行实现了NTT变换的核心算法，包括位逆序置换和蝶形操作。
 */
void NTT(uint64_t *a, int n, bool invert, uint64_t p, uint64_t g) {
    // 计算二进制位数
    int bit = 0;
    while ((1 << bit) < n)
        bit++;

    // 位逆序置换
    reverse(a, n, bit);

    // 蝶形操作
    for (int len = 2; len <= n; len <<= 1) {
        uint64_t g_n = invert ? pow(g, (p - 1) - (p - 1) / len, p)
                              : pow(g, (p - 1) / len, p);
        // 单位根
        // 处理每个块
        for (int i = 0; i < n; i += len) {
            uint64_t g_pow = 1;  // 当前单元所需的单位根的幂次
            // 处理每个蝶形单元
            for (int j = 0; j < len / 2; j++) {
                uint64_t u = a[i + j];
                uint64_t v = (a[i + j + len / 2] * g_pow) % p;

                uint64_t sum = u + v;
                if (sum >= p)
                    sum -= p;
                uint64_t diff = u >= v ? u - v : u + p - v;
                // 高效的模加法
                a[i + j] = sum;
                a[i + j + len / 2] = diff;
                g_pow = (g_pow * g_n) % p;
            }
        }
    }

    // 如果是逆变换，需要除以n（即乘以n的模p逆元）
    if (invert) {
        uint64_t inv_n = pow(n, p - 2, p);  // 使用费马小定理计算逆元
        for (int i = 0; i < n; i++) {
            a[i] = (a[i] * inv_n) % p;
        }
    }
}

/**
 * @brief 使用NTT的多项式乘法
 * @param a 输入多项式A
 * @param b 输入多项式B
 * @param result 结果数组
 * @param n 多项式长度
 * @param p 模数
 * @details 该函数实现了NTT变换下的多项式乘法，包括正向NTT、点值乘法和逆向NTT。
 */
void NTT_multiply(uint64_t *a, uint64_t *b, uint64_t *result, int n, uint64_t p) {
    // 计算可以容纳结果的2的幂次长度
    int len = n << 1;
    // 创建临时数组
    uint64_t *fa = new uint64_t[len];
    uint64_t *fb = new uint64_t[len];

    // 复制输入数组并填充0
    for (int i = 0; i < n; i++) {
        fa[i] = a[i];
        fb[i] = b[i];
    }
    for (int i = n; i < len; i++) {
        fa[i] = fb[i] = 0;
    }

    // 确定原根
    int g = 3;

    // 正向NTT
    NTT(fa, len, false, p, g);
    NTT(fb, len, false, p, g);

    // 点值表示下的乘法
    for (int i = 0; i < len; i++) {
        fa[i] = (1LL * fa[i] * fb[i]) % p;
    }

    // 逆向NTT得到结果
    NTT(fa, len, true, p, g);

    // 复制结果
    for (int i = 0; i < 2 * n - 1; i++) {
        result[i] = fa[i];
    }

    delete[] fa;
    delete[] fb;
}
// 预分配内存池，避免重复动态分配
constexpr int MAX_LEN = 1 << 18;  // 根据最大处理规模设置
uint64_t fa_pool[MAX_LEN];
uint64_t fb_pool[MAX_LEN];
int rev_pool[MAX_LEN];
// 位逆序置换函数
void parallel_reverse(uint64_t *a, int n, int bit) {
    // 分配并初始化逆序表
    task_state.rev = rev_pool;

    // 设置逆序置换任务参数
    task_state.current_task = TASK_REVERSE;
    task_state.a = a;
    task_state.n = n;
    task_state.bit = bit;

    // 唤醒工作线程执行逆序置换
    pthread_barrier_wait(&barrier);

    // 等待所有线程完成
    pthread_barrier_wait(&barrier);
}

/**
 * @brief 工作线程函数
 * @param arg 线程参数
 */
void *worker_thread(void *arg) {
    int thread_id = *((int *)arg);  // 线程ID
    delete (int *)arg;              // 释放分配的内存

    // 持久线程主循环
    while (true) {
        // 等待主线程分配任务
        pthread_barrier_wait(&barrier);

        // 根据当前任务类型执行相应操作
        switch (task_state.current_task) {
            case TASK_REVERSE: {
                uint64_t *a = task_state.a;
                int n = task_state.n;
                int *rev = task_state.rev;

                // 每个线程处理一部分逆序置换
                for (int i = thread_id; i < n; i += NUM_THREADS) {
                    if (i < rev[i]) {
                        uint64_t temp = a[i];
                        a[i] = a[rev[i]];
                        a[rev[i]] = temp;
                    }
                }
                break;
            }

            case TASK_NTT: {
                uint64_t *a = task_state.a;
                int len = task_state.current_len;  // 当前蝶形长度
                uint64_t p = task_state.p;         // 模数
                uint64_t g_n = task_state.g_n;     // 当前单位根
                int n = task_state.n;              // 多项式长度

                // 优化的块分配 - 每个线程处理均等数量的蝶形块
                int blocks = (n + len - 1) / len;                                       // 计算块数向上取整
                int blocks_per_thread = (blocks + NUM_THREADS - 1) / NUM_THREADS;       // 每个线程处理的块数 向上取整
                int start_block = thread_id * blocks_per_thread;                        // 每个线程的起始块
                int end_block = std::min((thread_id + 1) * blocks_per_thread, blocks);  // 结束块
                int step = len >> 1;                                                    // 蝶形单元的步长

                for (int block = start_block; block < end_block; block++) {
                    int i = block * len;
                    uint64_t g_pow = 1;  // 当前单元所需的单位根的幂次
                    // 处理每个蝶形单元
                    for (int j = 0; j < step; j++) {
                        uint64_t u = a[i + j];
                        uint64_t v = (a[i + j + step] * g_pow) % p;
                        uint64_t sum = u + v;
                        if (sum > p)
                            sum -= p;
                        uint64_t diff = u <= v ? u + p - v : u - v;
                        a[i + j] = sum;
                        a[i + j + step] = diff;
                        g_pow = (g_pow * g_n) % p;  // 更新单位根
                    }
                }
                break;
            }
            case TASK_MULTIPLY: {
                uint64_t *fa = task_state.a;  // 输入多项式A
                uint64_t *fb = task_state.b;  // 输入多项式B
                int len = task_state.n;       // 多项式长度
                uint64_t p = task_state.p;    // 模数

                // 使用连续块处理而非跨步处理
                int block_size = (len + NUM_THREADS - 1) / NUM_THREADS;
                int start = thread_id * block_size;
                int end = std::min(start + block_size, len);

                // 每个线程处理连续的内存块
                for (int i = start; i + 3 < end; i += 4) {
                    fa[i] = (fa[i] * fb[i]) % p;
                    fa[i + 1] = (fa[i + 1] * fb[i + 1]) % p;
                    fa[i + 2] = (fa[i + 2] * fb[i + 2]) % p;
                    fa[i + 3] = (fa[i + 3] * fb[i + 3]) % p;
                }
                break;
            }

            case TASK_APPLY_INV_N: {
                uint64_t *a = task_state.a;
                int n = task_state.n;
                uint64_t p = task_state.p;
                uint64_t inv_n = task_state.inv_n;

                // 使用连续块处理而非跨步处理
                int block_size = (n + NUM_THREADS - 1) / NUM_THREADS;
                int start = thread_id * block_size;
                int end = std::min(start + block_size, n);

                for (int i = start; i + 3 < end; i += 4) {
                    // 一次处理4个元素
                    a[i] = (a[i] * inv_n) % p;
                    a[i + 1] = (a[i + 1] * inv_n) % p;
                    a[i + 2] = (a[i + 2] * inv_n) % p;
                    a[i + 3] = (a[i + 3] * inv_n) % p;
                }
                break;
            }

            case TASK_CRT_COMBINE: {
                // CRT合并任务，每个线程处理部分结果
                uint64_t **results = task_state.mod_results;
                uint64_t *mod_list = task_state.mod_list;
                uint64_t *final_result = task_state.result;
                int result_length = task_state.n;
                uint64_t p = task_state.p;

                // 计算模数乘积
                __uint128_t M = task_state.M;

                // 预计算所有Mi和Mi_inv (针对4个固定模数)
                __uint128_t *Mi_values = task_state.Mi;
                uint64_t *Mi_inv_values = task_state.Mi_inv;

                // 使用连续块处理
                int block_size = (result_length + NUM_THREADS - 1) / NUM_THREADS;
                int start = thread_id * block_size;
                int end = std::min(start + block_size, result_length);

                // 处理连续内存块
                for (int j = start; j < end; j++) {
                    // 完全展开4个模数的循环
                    __uint128_t term0 = Mi_values[0] * ((results[0][j] * Mi_inv_values[0]) % mod_list[0]);
                    __uint128_t term1 = Mi_values[1] * ((results[1][j] * Mi_inv_values[1]) % mod_list[1]);
                    __uint128_t term2 = Mi_values[2] * ((results[2][j] * Mi_inv_values[2]) % mod_list[2]);
                    __uint128_t term3 = Mi_values[3] * ((results[3][j] * Mi_inv_values[3]) % mod_list[3]);

                    // 分批次累加避免中间溢出
                    __uint128_t sum = (term0 + term1 + term2 + term3) % M;
                    final_result[j] = sum % p;
                }
                break;
            }
            case TASK_EXIT:
                return NULL;  // 退出任务
                break;
            default:
                break;
        }
        // 等待所有线程完成当前任务
        pthread_barrier_wait(&barrier);
    }
    return NULL;
}

/**
 * @brief NTT变换函数，使用持久线程
 * @param a 输入数组
 * @param n 数组长度
 * @param invert 是否进行逆变换
 * @param p 模数
 * @param bit 二进制位数
 * @param g 原根
 */
void NTT_persistent(uint64_t *a, int n, bool invert, uint64_t p, int bit,
                    int g = 3) {
    // 设置基本任务参数
    task_state.a = a;
    task_state.n = n;
    task_state.p = p;
    task_state.invert = invert;

    // 蝶形操作
    for (int len = 2; len <= n; len <<= 1) {
        // 更新当前任务状态
        task_state.current_task = TASK_NTT;
        task_state.current_len = len;
        task_state.g_n = invert ? pow(g, (p - 1) - (p - 1) / len, p)
                                : pow(g, (p - 1) / len, p);

        // 唤醒工作线程
        pthread_barrier_wait(&barrier);

        // 等待所有线程完成当前任务
        pthread_barrier_wait(&barrier);
    }

    // 如果是逆变换，需要除以n（即乘以n的模p逆元），并行处理
    if (invert) {
        task_state.current_task = TASK_APPLY_INV_N;
        task_state.inv_n = pow(n, p - 2, p);  // 使用费马小定理计算逆元

        // 唤醒工作线程应用逆元
        pthread_barrier_wait(&barrier);

        // 等待所有线程完成
        pthread_barrier_wait(&barrier);
    }
}

/**
 * @brief 使用持久线程的多项式乘法
 * @param a 输入多项式A
 * @param b 输入多项式B
 * @param result 结果数组
 * @param n 多项式长度
 * @param p 模数
 * @param threads 线程数组
 * @details 该函数实现了NTT变换下的多项式乘法，包括正向NTT、点值乘法和逆向NTT。
 */
void NTT_multiply_persistent(uint64_t *a, uint64_t *b, uint64_t *result, int n,
                             uint64_t p, pthread_t *threads) {
    // 计算可以容纳结果的2的幂次长度 可以保证n为4的倍数
    int len = (n << 1);

    // 使用预分配的内存池
    uint64_t *fa = fa_pool;
    uint64_t *fb = fb_pool;

    for (int i = 0; i + 3 < n; i += 4) {
        fa[i] = a[i];
        fa[i + 1] = a[i + 1];
        fa[i + 2] = a[i + 2];
        fa[i + 3] = a[i + 3];
        fb[i] = b[i];
        fb[i + 1] = b[i + 1];
        fb[i + 2] = b[i + 2];
        fb[i + 3] = b[i + 3];
    }
    for (int i = n; i + 3 < len; i += 4) {
        fa[i] = fb[i] = 0;
        fa[i + 1] = fb[i + 1] = 0;
        fa[i + 2] = fb[i + 2] = 0;
        fa[i + 3] = fb[i + 3] = 0;
    }

    // 计算二进制位数
    int bit = 0;
    while ((1 << bit) < len)
        bit++;

    // 正向NTT
    auto rev = rev_pool;
    rev[0] = 0;

    // 计算逆序表
    for (int i = 0; i + 3 < len; i += 4) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
        rev[i + 1] = (rev[(i + 1) >> 1] >> 1) | (((i + 1) & 1) << (bit - 1));
        rev[i + 2] = (rev[(i + 2) >> 1] >> 1) | (((i + 2) & 1) << (bit - 1));
        rev[i + 3] = (rev[(i + 3) >> 1] >> 1) | (((i + 3) & 1) << (bit - 1));
    }
    // 位逆序置换
    parallel_reverse(fa, len, bit);
    // 位逆序置换
    parallel_reverse(fb, len, bit);
    NTT_persistent(fa, len, false, p, bit);
    NTT_persistent(fb, len, false, p, bit);

    // 点值乘法
    task_state.current_task = TASK_MULTIPLY;
    task_state.a = fa;
    task_state.b = fb;
    task_state.n = len;
    task_state.p = p;

    // 唤醒工作线程执行点值乘法
    pthread_barrier_wait(&barrier);

    // 等待所有线程完成
    pthread_barrier_wait(&barrier);

    // 逆向NTT
    // 位逆序置换
    parallel_reverse(fa, len, bit);
    NTT_persistent(fa, len, true, p, bit);

    // 复制结果
    for (int i = 0; i + 3 < len; i += 4) {
        result[i] = fa[i];
        result[i + 1] = fa[i + 1];
        result[i + 2] = fa[i + 2];
        result[i + 3] = fa[i + 3];
    }
}

/**
 * @brief 使用持久线程的多项式乘法,在赋值的时候多了一项模取操作，相当于预处理，适合CRT的时候使用
 * @param a 输入多项式A
 * @param b 输入多项式B
 * @param result 结果数组
 * @param n 多项式长度
 * @param p 模数
 * @param threads 线程数组
 */
void NTT_multiply_persistent_big(uint64_t *a, uint64_t *b, uint64_t *result, int n,
                                 uint64_t p, pthread_t *threads) {
    /// 计算可以容纳结果的2的幂次长度 可以保证n为4的倍数
    int len = (n << 1);

    // 使用预分配的内存池
    uint64_t *fa = fa_pool;
    uint64_t *fb = fb_pool;

    for (int i = 0; i + 3 < n; i += 4) {
        fa[i] = a[i] % p;
        fa[i + 1] = a[i + 1] % p;
        fa[i + 2] = a[i + 2] % p;
        fa[i + 3] = a[i + 3] % p;
        fb[i] = b[i] % p;
        fb[i + 1] = b[i + 1] % p;
        fb[i + 2] = b[i + 2] % p;
        fb[i + 3] = b[i + 3] % p;
    }
    for (int i = n; i + 3 < len; i += 4) {
        fa[i] = fb[i] = 0;
        fa[i + 1] = fb[i + 1] = 0;
        fa[i + 2] = fb[i + 2] = 0;
        fa[i + 3] = fb[i + 3] = 0;
    }

    // 计算二进制位数
    int bit = 0;
    while ((1 << bit) < len)
        bit++;

    // 正向NTT
    auto rev = rev_pool;
    rev[0] = 0;

    // 计算逆序表
    for (int i = 0; i + 3 < len; i += 4) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
        rev[i + 1] = (rev[(i + 1) >> 1] >> 1) | (((i + 1) & 1) << (bit - 1));
        rev[i + 2] = (rev[(i + 2) >> 1] >> 1) | (((i + 2) & 1) << (bit - 1));
        rev[i + 3] = (rev[(i + 3) >> 1] >> 1) | (((i + 3) & 1) << (bit - 1));
    }
    // 位逆序置换
    parallel_reverse(fa, len, bit);
    // 位逆序置换
    parallel_reverse(fb, len, bit);
    NTT_persistent(fa, len, false, p, bit);
    NTT_persistent(fb, len, false, p, bit);

    // 点值乘法
    task_state.current_task = TASK_MULTIPLY;
    task_state.a = fa;
    task_state.b = fb;
    task_state.n = len;
    task_state.p = p;

    // 唤醒工作线程执行点值乘法
    pthread_barrier_wait(&barrier);

    // 等待所有线程完成
    pthread_barrier_wait(&barrier);

    // 逆向NTT
    // 位逆序置换
    parallel_reverse(fa, len, bit);
    NTT_persistent(fa, len, true, p, bit);

    // 复制结果
    for (int i = 0; i + 3 < len; i += 4) {
        result[i] = fa[i];
        result[i + 1] = fa[i + 1];
        result[i + 2] = fa[i + 2];
        result[i + 3] = fa[i + 3];
    }
}
// 添加全局静态变量，避免重复创建和销毁
// NTT友好的4个固定模数
const uint64_t GLOBAL_MOD_LIST[4] = {1004535809, 1224736769, 469762049, 998244353};
const int GLOBAL_MOD_COUNT = 4;

// 预分配的结果数组，避免动态分配
uint64_t *GLOBAL_MOD_RESULTS[4] = {nullptr, nullptr, nullptr, nullptr};

// 预计算模数乘积及逆元相关值
__uint128_t GLOBAL_M = 1;          // 模数乘积
__uint128_t GLOBAL_MI_VALUES[4];   // 各模数的"M/模数"值
uint64_t GLOBAL_MI_INV_VALUES[4];  // 各模数的逆元值

/**
 * @brief 初始化全局CRT变量
 * @details 预计算模数乘积和逆元值，避免重复计算
 */
void init_global_crt_values() {
    // 计算模数乘积
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
    constexpr int MAX_RESULT_LEN = 1 << 18;  // 2*max_n
    for (int i = 0; i < GLOBAL_MOD_COUNT; i++) {
        GLOBAL_MOD_RESULTS[i] = new uint64_t[MAX_RESULT_LEN];
    }
}

// 释放全局资源
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
 * @param threads 线程数组
 */
void CRT_NTT_multiply_persistent(uint64_t *a, uint64_t *b, uint64_t *result,
                                 int n, uint64_t p, pthread_t *threads) {
    // 使用全局预分配的模数和结果数组
    task_state.M = GLOBAL_M;  // 设置模数乘积

    // 清零结果数组的必要部分
    int result_len = (n << 1) - 1;
    for (int i = 0; i < GLOBAL_MOD_COUNT; i++) {
        memset(GLOBAL_MOD_RESULTS[i], 0, sizeof(uint64_t) * result_len);
    }

    task_state.Mi = GLOBAL_MI_VALUES;          // 使用预计算的Mi值
    task_state.Mi_inv = GLOBAL_MI_INV_VALUES;  // 使用预计算的Mi_inv值

    // 逐个模数计算NTT
    for (int i = 0; i < GLOBAL_MOD_COUNT; i++) {
        NTT_multiply_persistent_big(a, b, GLOBAL_MOD_RESULTS[i], n, GLOBAL_MOD_LIST[i], threads);
    }

    // 使用CRT合并结果
    task_state.current_task = TASK_CRT_COMBINE;
    task_state.mod_results = GLOBAL_MOD_RESULTS;
    task_state.mod_list = (uint64_t *)GLOBAL_MOD_LIST;
    task_state.mod_count = GLOBAL_MOD_COUNT;
    task_state.result = result;
    task_state.n = result_len;
    task_state.p = p;

    // 唤醒工作线程执行CRT合并
    pthread_barrier_wait(&barrier);

    // 等待所有线程完成
    pthread_barrier_wait(&barrier);
}

uint64_t a[300000], b[300000], ab[300000];
pthread_t threads[NUM_THREADS];
int main(int argc, char *argv[]) {
    // 测试用例
    init_global_crt_values();
    pthread_barrier_init(&barrier, NULL, NUM_THREADS + 1);

            // 创建线程
            for (int j = 0; j < NUM_THREADS; j++) {
                int *thread_id = new int(j);
                pthread_create(&threads[j], NULL, worker_thread, thread_id);
            }
    int test_begin = 0;
    int test_end = 4;
    for (int i = test_begin; i <= test_end; ++i) {
        long double ans = 0;
        int n_;
        uint64_t p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));
        auto Start = std::chrono::high_resolution_clock::now();
        // 每轮重置屏障和退出标志
        if (n_ == 4) {
            // 直接使用单线程计算
            NTT_multiply(a, b, ab, n_, p_);
        } else {
            

            if (p_ > (1ULL << 32)) {
                CRT_NTT_multiply_persistent(a, b, ab, n_, p_, threads);
            } else {
                NTT_multiply_persistent(a, b, ab, n_, p_, threads);
            }

            
        }
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000>> elapsed =
            End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);
        std::cout << "average latency for n = " << n_ << " p = " << p_ << " : "
                  << ans << " (us) " << std::endl;

        // 结果输出
        fWrite(ab, n_, i);
    }
    // 终止线程
            task_state.current_task = TASK_EXIT;
            pthread_barrier_wait(&barrier);

            for (int j = 0; j < NUM_THREADS; j++) {
                pthread_join(threads[j], NULL);
            }

            pthread_barrier_destroy(&barrier);
    cleanup_global_crt_values();


    return 0;
}