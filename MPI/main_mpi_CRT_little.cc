/** 
 *  小模数版本的CRT多项式乘法,emmmm
 *  使用Barrett reduction和多线程优化
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

#define NUM_THREADS 8
pthread_barrier_t barrier;

// 全局数组声明
uint64_t a[300000], b[300000];

// Barrett reduction struct
struct Barrett {
    uint64_t mod;
    uint64_t k;
    uint64_t mu;

    Barrett(uint64_t m) : mod(m) {
        // 计算k值：找到最小的k使得2^k > m
        k = 0;
        uint64_t temp = m;
        while (temp > 0) {
            temp >>= 1;
            k++;
        }
        // 确保2^k > m
        if ((1ull << k) <= m) k++;
        // 计算mu = floor(2^(2k)/m)
        mu = (1ull << (2 * k)) / m;
    }

    uint64_t reduce(uint64_t x) const {
        uint64_t q = ((__uint128_t)x * mu) >> (2 * k);
        
        uint64_t r = x - q * mod;
        return r >= mod ? r - mod : r;
    }
};

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
    Barrett *br_ptr;

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
} task_state;

void fRead(uint64_t *a, uint64_t *b, int *n, uint64_t *p, int input_id) {
    std::string str1 = "./filetest/";
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
    std::string str1 = "./filetest/";
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

void NTT_persistent(uint64_t *a, int n, bool invert, uint64_t p,  int g = 3) {
    task_state.a = a;
    task_state.n = n;
    task_state.p = p;
    task_state.invert = invert;

    for (int len = 2; len <= n; len <<= 1) {
        task_state.current_task = TASK_NTT;
        task_state.current_len = len;
        task_state.g_n = invert ? pow(g, (p - 1) - (p - 1) / len, *task_state.br_ptr)
                               : pow(g, (p - 1) / len, *task_state.br_ptr);

        pthread_barrier_wait(&barrier);
        pthread_barrier_wait(&barrier);
    }

    if (invert) {
        task_state.current_task = TASK_APPLY_INV_N;
        task_state.inv_n = pow(n, p - 2, *task_state.br_ptr);
        pthread_barrier_wait(&barrier);
        pthread_barrier_wait(&barrier);
    }
}

void NTT_multiply_persistent(uint64_t *a, uint64_t *b, uint64_t *result, int n, uint64_t p, pthread_t *threads,Barrett *br_ptr) {
    
    task_state.br_ptr = br_ptr;

    int len = (n << 1);
    uint64_t *fa = new uint64_t[len]();  // 使用()初始化所有元素为0
    uint64_t *fb = new uint64_t[len]();  // 使用()初始化所有元素为0
    int i = 0;

    // 复制输入数据，确保不会越界
    for (i = 0; i < n; i += 4) {
        fa[i] = a[i];
        fa[i + 1] = a[i + 1];
        fa[i + 2] = a[i + 2];
        fa[i + 3] = a[i + 3];
        fb[i] = b[i];
        fb[i + 1] = b[i + 1];
        fb[i + 2] = b[i + 2];
        fb[i + 3] = b[i + 3];
    }

    int bit = 0;
    while ((1 << bit) < len) bit++;

    auto rev = new int[len]();  // 使用()初始化所有元素为0
    rev[0] = 0;

    // 计算逆序表，确保不会越界
    for (i = 0; i < len; i += 4) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
        rev[i + 1] = (rev[(i + 1) >> 1] >> 1) | (((i + 1) & 1) << (bit - 1));
        rev[i + 2] = (rev[(i + 2) >> 1] >> 1) | (((i + 2) & 1) << (bit - 1));
        rev[i + 3] = (rev[(i + 3) >> 1] >> 1) | (((i + 3) & 1) << (bit - 1));
    }
    // 位逆序置换
    task_state.current_task = TASK_REVERSE;
    task_state.a = fa;
    task_state.n = len;
    task_state.rev = rev;
    pthread_barrier_wait(&barrier);
    pthread_barrier_wait(&barrier);

    task_state.a = fb;
    pthread_barrier_wait(&barrier);
    pthread_barrier_wait(&barrier);


    NTT_persistent(fa, len, false, p, 3);
    NTT_persistent(fb, len, false, p, 3);

    // 点值乘法
    task_state.current_task = TASK_MULTIPLY;
    task_state.a = fa;
    task_state.b = fb;
    task_state.n = len;
    task_state.p = p;
    pthread_barrier_wait(&barrier);
    pthread_barrier_wait(&barrier);

    // 逆向NTT
    task_state.current_task = TASK_REVERSE;
    task_state.a = fa;
    pthread_barrier_wait(&barrier);
    pthread_barrier_wait(&barrier);

    NTT_persistent(fa, len, true, p, 3);
    
    // 复制结果，确保不会越界
    for (i = 0; i < len - 1; i++) {
        result[i] = fa[i];
    }
    delete[] fa;
    delete[] fb;
    delete[] rev;
}

const uint64_t GLOBAL_MOD_LIST[3] = {65537, 163841, 114689};
const int GLOBAL_MOD_COUNT = 3;

// 预分配的结果数组
uint64_t *GLOBAL_MOD_RESULTS[3] = {nullptr, nullptr, nullptr};

// 预计算模数乘积及逆元相关值
uint64_t GLOBAL_M = 1;
uint64_t GLOBAL_MI_VALUES[3];
uint64_t GLOBAL_MI_INV_VALUES[3];

// 初始化全局CRT变量
void init_global_crt_values() {
    // 计算模数乘积
    for (int i = 0; i < GLOBAL_MOD_COUNT; i++) {
        GLOBAL_M *= GLOBAL_MOD_LIST[i];
    }

    // 预计算Mi和Mi_inv值
    for (int i = 0; i < GLOBAL_MOD_COUNT; i++) {
        uint64_t mod_num = GLOBAL_MOD_LIST[i];
        GLOBAL_MI_VALUES[i] = GLOBAL_M / mod_num;
        Barrett br(mod_num);
        uint64_t Mi_mod = GLOBAL_MI_VALUES[i] % mod_num;
        GLOBAL_MI_INV_VALUES[i] = pow(Mi_mod, mod_num - 2, br);
    }

    // 分配结果数组
    constexpr int MAX_RESULT_LEN = 1 << 18;
    for (int i = 0; i < GLOBAL_MOD_COUNT; i++) {
        GLOBAL_MOD_RESULTS[i] = new uint64_t[MAX_RESULT_LEN]();
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

// 添加CRT合并任务到worker_thread
void *worker_thread(void *arg) {
    int thread_id = *((int *)arg);
    delete (int *)arg;

    while (true) {
        pthread_barrier_wait(&barrier);

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
                Barrett *br = task_state.br_ptr;

                // 使用连续块处理而非跨步处理
                int block_size = (len + NUM_THREADS - 1) / NUM_THREADS;
                int start = thread_id * block_size;
                int end = std::min(start + block_size, len);
                int i = start;
                for (; i + 3 < end; i += 4) {  // 问题：当end-start < 4时可能跳过所有元素
                    fa[i] = br->reduce(fa[i] * fb[i]);
                    fa[i + 1] = br->reduce(fa[i + 1] * fb[i + 1]);
                    fa[i + 2] = br->reduce(fa[i + 2] * fb[i + 2]);
                    fa[i + 3] = br->reduce(fa[i + 3] * fb[i + 3]);
                }
                // 需要处理剩余元素
                for (; i < end; i++) {
                    fa[i] = br->reduce(fa[i] * fb[i]);
                }
                break;
            }

            case TASK_APPLY_INV_N: {
                uint64_t *a = task_state.a;
                int n = task_state.n;
                uint64_t p = task_state.p;
                uint64_t inv_n = task_state.inv_n;
                Barrett *br = task_state.br_ptr;

                // 使用连续块处理而非跨步处理
                int block_size = (n + NUM_THREADS - 1) / NUM_THREADS;
                int start = thread_id * block_size;
                int end = std::min(start + block_size, n);
                int i = start;
                for (; i + 3 < end; i += 4) {
                    // 一次处理4个元素
                    a[i] = br->reduce(a[i] * inv_n);
                    a[i + 1] = br->reduce(a[i + 1] * inv_n);
                    a[i + 2] = br->reduce(a[i + 2] * inv_n);
                    a[i + 3] = br->reduce(a[i + 3] * inv_n);
                }
                // 处理剩余元素
                for (; i < end; i++) {
                    a[i] = br->reduce(a[i] * inv_n);
                }
                break;
            }
           
            case TASK_EXIT:
                return NULL;  // 退出任务
                break;
            default:
                break;
        }
        
        pthread_barrier_wait(&barrier);
    }
    return NULL;
}

// 修改后的CRT多项式乘法函数
void CRT_NTT_multiply_parallel(uint64_t *a, uint64_t *b, uint64_t *result, int n, uint64_t p, int rank, int size, pthread_t *threads) {
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
        // 使用pthread优化的NTT乘法
        NTT_multiply_persistent(ta, tb, GLOBAL_MOD_RESULTS[i], n, GLOBAL_MOD_LIST[i], threads, &br);
        
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
    for (int j = start_idx; j < end_idx; j++) {
        uint64_t sum = 0;
        // 完全展开4个模数的循环
        uint64_t term0 = GLOBAL_MI_VALUES[0] * ((GLOBAL_MOD_RESULTS[0][j] * GLOBAL_MI_INV_VALUES[0]) % GLOBAL_MOD_LIST[0]);
        uint64_t term1 = GLOBAL_MI_VALUES[1] * ((GLOBAL_MOD_RESULTS[1][j] * GLOBAL_MI_INV_VALUES[1]) % GLOBAL_MOD_LIST[1]);
        uint64_t term2 = GLOBAL_MI_VALUES[2] * ((GLOBAL_MOD_RESULTS[2][j] * GLOBAL_MI_INV_VALUES[2]) % GLOBAL_MOD_LIST[2]);

        sum = (term0 + term1 + term2) % GLOBAL_M;
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

    // 初始化pthread屏障和创建线程
    pthread_barrier_init(&barrier, NULL, NUM_THREADS + 1);
    pthread_t threads[NUM_THREADS];
    
    // 创建工作线程
    for (int j = 0; j < NUM_THREADS; j++) {
        int *thread_id = new int(j);
        pthread_create(&threads[j], NULL, worker_thread, thread_id);
    }

    int test_begin = 0;
    int test_end = 3;
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
        
        CRT_NTT_multiply_parallel(a, b, ab, n_, p_, rank, size, threads);

        
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
    
    // 清理线程
    task_state.current_task = TASK_EXIT;
    pthread_barrier_wait(&barrier);

    for (int j = 0; j < NUM_THREADS; j++) {
        pthread_join(threads[j], NULL);
    }

    pthread_barrier_destroy(&barrier);
    
    // 清理全局CRT变量
    cleanup_global_crt_values();
    
    MPI_Finalize();
    return 0;
}
