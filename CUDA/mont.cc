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

#include "Montgomery.h"

#define NUM_THREADS 8

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

// 幂取模
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

// 全局变量和屏障
pthread_barrier_t barrier;  // 屏障
std::atomic<bool> thread_exit(false);

// 任务类型枚举
enum TaskType {
    TASK_REVERSE,      // 位逆序置换任务
    TASK_NTT,          // NTT任务
    TASK_MULTIPLY,     // 多项式点值乘法
    TASK_APPLY_INV_N,  // 应用逆元
    TASK_CRT_COMBINE,  // CRT合并
    TASK_COPY_DATA,    // 新增：数据复制任务
    TASK_COPY_RESULT,  // 新增：结果复制任务
    TASK_EXIT          // 退出任务
};

// 全局共享任务状态
struct GlobalTaskState {
    TaskType current_task;
    Montgomery *montgomery;  // Montgomery模乘器

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
} task_state;

// 预分配内存池，避免重复动态分配
constexpr int MAX_LEN = 1 << 22;  // 根据最大处理规模设置
uint64_t fa_pool[MAX_LEN];
uint64_t fb_pool[MAX_LEN];
int rev_pool[MAX_LEN];
// 位逆序置换函数
void parallel_reverse(uint64_t *a, int n, int bit) {
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

// 工作线程函数
void *worker_thread(void *arg) {
    int thread_id = *((int *)arg);  // 线程ID
    delete (int *)arg;              // 释放分配的内存

    // 持久线程主循环
    while (!thread_exit.load(std::memory_order_acquire)) {
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
                        // 使用异或进行就地交换，避免临时变量
                        a[i] ^= a[rev[i]];
                        a[rev[i]] ^= a[i];
                        a[i] ^= a[rev[i]];
                    }
                }
                break;
            }
            case TASK_NTT: {
                uint64_t *a = task_state.a;
                int len = task_state.current_len;
                uint64_t p = task_state.p;
                uint64_t g_n = task_state.g_n;
                int n = task_state.n;
                Montgomery *m = task_state.montgomery;
                int step = len >> 1;

                // 每个线程处理一部分块
                for (int i = thread_id * len; i < n; i += len * NUM_THREADS) {
                    uint64_t g = m->R_mon_N;  // 初始单位根为1

                    // 处理每个蝶形单元
                    for (int j = 0; j < step; j++) {
                        uint64_t u = a[i + j];
                        uint64_t v = m->REDC(a[i + j + step] * g);

                        a[i + j] = (u + v) % p;
                        a[i + j + step] = (u - v + p) % p;

                        g = m->REDC(g * g_n);  // 更新单位根
                    }
                }
                break;
            }
            case TASK_MULTIPLY: {
                uint64_t *fa = task_state.a;
                uint64_t *fb = task_state.b;
                int len = task_state.n;
                uint64_t p = task_state.p;
                auto m = task_state.montgomery;

                // 每个线程处理一部分点值乘法
                for (int i = thread_id; i < len; i += NUM_THREADS) {
                    fa[i] = m->REDC(fa[i] * fb[i]);
                }
                break;
            }
            case TASK_APPLY_INV_N: {
                uint64_t *a = task_state.a;
                int n = task_state.n;
                uint64_t p = task_state.p;
                uint64_t inv_n = task_state.inv_n;
                auto m = task_state.montgomery;

                auto inv_n_mont = m->REDC(inv_n * m->R2);

                // 每个线程处理一部分应用逆元
                for (int i = thread_id; i < n; i += NUM_THREADS) {
                    a[i] = m->REDC(a[i] * inv_n_mont);
                }
                break;
            }
            case TASK_CRT_COMBINE: {
                // CRT合并任务，每个线程处理部分结果
                uint64_t **results = task_state.mod_results;
                uint64_t *mod_list = task_state.mod_list;
                uint64_t *final_result = task_state.result;
                int mod_count = task_state.mod_count;
                int result_length = task_state.n;
                uint64_t p = task_state.p;

                // 计算模数乘积
                __uint128_t M = 1;
                for (int i = 0; i < mod_count; i++) {
                    M *= mod_list[i];
                }

                // 每个线程处理一部分结果
                for (int j = thread_id; j < result_length; j += NUM_THREADS) {
                    __uint128_t sum = 0;
                    for (int i = 0; i < mod_count; i++) {
                        uint64_t mod_num = mod_list[i];

                        __uint128_t Mi = M / mod_num;
                        uint64_t Mi_mod = Mi % mod_num;
                        uint64_t Mi_inv = pow(Mi_mod, mod_num - 2, mod_num);

                        sum =
                            (sum + (Mi * ((results[i][j] * Mi_inv) % mod_num))) % M;
                    }
                    final_result[j] = sum % p;
                }

                break;
            }
            case TASK_COPY_DATA: {
                // 转入蒙哥马利
                uint64_t *a = task_state.a;       // 源数组A
                uint64_t *b = task_state.b;       // 源数组B
                uint64_t *fa = task_state.dst_a;  // 目标数组A
                uint64_t *fb = task_state.dst_b;  // 目标数组B
                int n = task_state.src_n;         // 源数组长度
                int len = task_state.dst_len;     // 目标数组长度
                Montgomery *m = task_state.montgomery;

                // 每个线程复制一部分源数据
                for (int i = thread_id; i < n; i += NUM_THREADS) {
                    fa[i] = m->REDC(a[i] * m->R2);
                    fb[i] = m->REDC(b[i] * m->R2);
                }

                // 每个线程填充一部分零
                for (int i = n + thread_id; i < len; i += NUM_THREADS) {
                    fa[i] = fb[i] = 0;
                }
                break;
            }
            case TASK_COPY_RESULT: {
                uint64_t *src = task_state.a;       // 源数组
                uint64_t *dst = task_state.result;  // 目标数组
                int len = task_state.n;             // 数组长度
                auto m = task_state.montgomery;

                // 每个线程复制一部分结果
                for (int i = thread_id; i < len; i += NUM_THREADS) {
                    dst[i] = m->REDC(src[i]);
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

// 使用持久线程
void NTT_persistent(uint64_t *a, int n, bool invert, uint64_t p, int bit,
                    int g = 3) {
    // 设置基本任务参数
    task_state.a = a;
    task_state.n = n;
    task_state.p = p;
    task_state.invert = invert;
    Montgomery *mont = task_state.montgomery;
    // 蝶形操作
    for (int len = 2; len <= n; len <<= 1) {
        // 更新当前任务状态
        task_state.current_task = TASK_NTT;
        task_state.current_len = len;
        auto g_n = invert ? pow(g, (p - 1) - (p - 1) / len, p)
                          : pow(g, (p - 1) / len, p);
        task_state.g_n = mont->REDC((uint64_t)g_n * mont->R2);
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

void NTT_multiply_persistent(uint64_t *a, uint64_t *b, uint64_t *result, int n,
                             uint64_t p, pthread_t *threads) {
    // 计算可以容纳结果的2的幂次长度
    int len = (n << 1);

    // 使用预分配的内存池
    uint64_t *fa = fa_pool;
    uint64_t *fb = fb_pool;
    auto m = task_state.montgomery;

    for (int i = 0; i < n; i++) {
        fa[i] = m->REDC(a[i] * m->R2);
        fb[i] = m->REDC(b[i] * m->R2);
    }
    for (int i = n; i < len; i++) {
        fa[i] = fb[i] = 0;
    }

    // 计算长度的二进制位数
    int bit = 0;
    while ((1 << bit) < len) {
        bit++;
    }
    // 位逆序置换
    // 分配并初始化逆序表
    int *rev = rev_pool;
    // 计算逆序表
    rev[0] = 0;
    for (int i = 0; i < len; i++) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    }
    parallel_reverse(fa, len, bit);
    parallel_reverse(fb, len, bit);

    // 正向NTT
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

    // 逆序置换
    parallel_reverse(fa, len, bit);

    // 逆向NTT
    NTT_persistent(fa, len, true, p, bit);

    // // 设置复制任务参数
    // task_state.current_task = TASK_COPY_RESULT;
    // task_state.a = fa;
    // task_state.result = result;
    // task_state.n = len;

    // // 唤醒工作线程执行复制任务
    // pthread_barrier_wait(&barrier);

    // // 等待所有线程完成
    // pthread_barrier_wait(&barrier);
    for (int i = 0; i < len; i++) {
        result[i] = m->REDC(fa[i]);
    }
}

// 使用CRT和持久线程的多项式乘法
void CRT_NTT_multiply_persistent(uint64_t *a, uint64_t *b, uint64_t *result,
                                 int n, uint64_t p, pthread_t *threads) {
    // 使用多个小模数，确保它们都是NTT友好的（形如k·2^m+1）
    uint64_t mod_list[4] = {1004535809, 1224736769, 469762049, 998244353};
    int mod_count = 4;

    // 分配结果数组
    uint64_t **mod_results = new uint64_t *[mod_count];
    for (int i = 0; i < mod_count; i++) {
        mod_results[i] = new uint64_t[2 * n - 1];
        memset(mod_results[i], 0, sizeof(uint64_t) * (2 * n - 1));
    }

    // 逐个模数计算NTT
    for (int i = 0; i < mod_count; i++) {
        NTT_multiply_persistent(a, b, mod_results[i], n, mod_list[i], threads);
    }

    // 使用CRT合并结果
    task_state.current_task = TASK_CRT_COMBINE;
    task_state.mod_results = mod_results;
    task_state.mod_list = mod_list;
    task_state.mod_count = mod_count;
    task_state.result = result;
    task_state.n = 2 * n - 1;
    task_state.p = p;

    // 唤醒工作线程执行CRT合并
    pthread_barrier_wait(&barrier);

    // 等待所有线程完成
    pthread_barrier_wait(&barrier);

    // 释放内存
    for (int i = 0; i < mod_count; i++) {
        delete[] mod_results[i];
    }
    delete[] mod_results;
}

uint64_t a[300000], b[300000], ab[300000];
pthread_t threads[NUM_THREADS];
int main(int argc, char *argv[]) {
    // 测试用例
    int test_begin = 0;
    int test_end = 4;
    for (int i = test_begin; i <= test_end; ++i) {
        std::cout << "test " << i << std::endl;
        long double ans = 0;
        int n_;
        uint64_t p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));
        auto Start = std::chrono::high_resolution_clock::now();
        // 每轮重置屏障和退出标志
        pthread_barrier_init(&barrier, NULL, NUM_THREADS + 1);
        thread_exit.store(false, std::memory_order_release);
        task_state.montgomery = new Montgomery(p_);
        // 创建线程

        for (int j = 0; j < NUM_THREADS; j++) {
            int *thread_id = new int(j);
            pthread_create(&threads[j], NULL, worker_thread, thread_id);
        }

        // 算法执行部分不变...
        if (p_ > (1ULL << 32)) {
            CRT_NTT_multiply_persistent(a, b, ab, n_, p_, threads);
        } else {
            NTT_multiply_persistent(a, b, ab, n_, p_, threads);
        }
        delete task_state.montgomery;
        // 正确终止线程
        task_state.current_task = TASK_EXIT;

        pthread_barrier_wait(&barrier);

        for (int j = 0; j < NUM_THREADS; j++) {
            pthread_join(threads[j], NULL);
        }

        pthread_barrier_destroy(&barrier);

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

    return 0;
}