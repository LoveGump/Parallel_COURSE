/**
 *   使用多进程来 并行计算小模数的NTT，但是效果不是很好，因为多进程之间通信开销太大了
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
void fCheck1(uint64_t *ab, int n, int input_id) {
    std::string str1 = "files/";
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

void reverse(uint64_t *a, int n, int bit) {
    int *rev = new int[n];
    rev[0] = 0;
    for (int i = 0; i < n; i++) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    }
    for (int i = 0; i < n; i++) {
        if (i < rev[i]) {
            std::swap(a[i], a[rev[i]]);
        }
    }
    delete[] rev;
}

void MPI_NTT(uint64_t *a, int n, bool invert, uint64_t p, int rank, int size   , Barrett &br) {
    int bit = 0;
    while ((1 << bit) < n) bit++;

    // 在rank 0进程上进行逆序置换
    if (rank == 0) {
        int *rev = new int[n];
        rev[0] = 0;
        for (int i = 1; i < n; i++) {
            rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
        }

        // 执行逆序置换
        for (int i = 0; i < n; i++) {
            if (i < rev[i]) {
                std::swap(a[i], a[rev[i]]);
            }
        }
        delete[] rev;
    }

    // 广播逆序后的数据到所有进程
    MPI_Bcast(a, n, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    // 分布式NTT计算
    for (int len = 2; len <= n; len <<= 1) {
        uint64_t g_n = invert ? pow(3, p - 1 - (p - 1) / len, br)
                             : pow(3, (p - 1) / len, br);

        int step = len >> 1;
        int total_blocks = n / len;
        
        // 每个进程处理一部分蝶形块
        int blocks_per_process = (total_blocks + size - 1) / size;
        int start_block = rank * blocks_per_process;
        int end_block = std::min(start_block + blocks_per_process, total_blocks);
        
        // 本地结果数组，只存储本进程计算的部分
        uint64_t *local_result = new uint64_t[n]();
        // 处理分配给本进程的蝶形块
        for (int block = start_block; block < end_block; block++) {
            int i = block * len;
            uint64_t g_pow = 1;
            
            for (int j = 0; j < step; j++) {
                uint64_t u = a[i + j];
                uint64_t v = br.reduce(a[i + j + step] * g_pow);
                
                uint64_t sum = u + v;
                if (sum >= p) sum -= p;
                uint64_t diff = u >= v ? u - v : u + p - v;
                
                local_result[i + j] = sum;
                local_result[i + j + step] = diff;
                
                g_pow = br.reduce(g_pow * g_n);
            }
        }
        
        // 合并所有进程的结果
        MPI_Allreduce(local_result, a, n, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        
        delete[] local_result;
        
       
    }

    // 如果是逆变换，分布式应用逆元
    if (invert) {
        uint64_t inv_n = pow(n, p - 2, br);
        
        // 每个进程处理一部分元素的逆元
        int elements_per_process = (n + size - 1) / size;
        int start_idx = rank * elements_per_process;
        int end_idx = std::min(start_idx + elements_per_process, n);
        
        uint64_t *local_inv = new uint64_t[n]();
        
        for (int i = start_idx; i < end_idx; i++) {
            local_inv[i] = br.reduce(a[i] * inv_n);
        }
        
        // 收集所有进程的逆元结果
        MPI_Allreduce(local_inv, a, n, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

        delete[] local_inv;
    }
    

}


void MPI_NTT_multiply(uint64_t *a, uint64_t *b, uint64_t *result, int n, uint64_t p, int rank, int size, Barrett &br) {
    int len = n << 1;
    uint64_t *fa = new uint64_t[len]();
    uint64_t *fb = new uint64_t[len]();

    // 复制输入数据
    for (int i = 0; i < n; i+=4) {
        fa[i] = a[i];
        fa[i + 1] = a[i + 1];
        fa[i + 2] = a[i + 2];
        fa[i + 3] = a[i + 3];
        fb[i] = b[i];
        fb[i + 1] = b[i + 1];
        fb[i + 2] = b[i + 2];
        fb[i + 3] = b[i + 3];
    }

    // 执行NTT变换
    MPI_NTT(fa, len, false, p, rank, size, br);
    MPI_NTT(fb, len, false, p, rank, size, br);

    
    // 使用更简单的 MPI_Scatter
    int local_n = len / size;
    uint64_t *local_fa = new uint64_t[local_n];
    uint64_t *local_fb = new uint64_t[local_n];

    // 分发数据
    MPI_Scatter(fa, local_n, MPI_UINT64_T, 
                local_fa, local_n, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Scatter(fb, local_n, MPI_UINT64_T, 
                local_fb, local_n, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    for (int i = 0; i+3 < local_n; i+=4) {
        local_fa[i] = br.reduce(local_fa[i] * local_fb[i]);
        local_fa[i + 1] = br.reduce(local_fa[i + 1] * local_fb[i + 1]);
        local_fa[i + 2] = br.reduce(local_fa[i + 2] * local_fb[i + 2]);
        local_fa[i + 3] = br.reduce(local_fa[i + 3] * local_fb[i + 3]);
    }

    // 收集结果
    MPI_Gather(local_fa, local_n, MPI_UINT64_T,
                fa, local_n, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    delete[] local_fa;
    delete[] local_fb;
        
   
    // 执行逆NTT变换
    MPI_NTT(fa, len, true, p, rank, size, br);

    // 复制结果
    if (rank == 0) {
        for (int i = 0; i <  len - 1; i++) {
            result[i] = fa[i];
        }
    }

    delete[] fa;
    delete[] fb;
}

// 修改main函数
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "使用 " << size << " 个MPI进程，每个进程 " << NUM_THREADS << " 个工作线程进行并行计算" << std::endl;
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
        
        uint64_t *ab = new uint64_t[2 * n_ - 1]();
        
        auto Start = std::chrono::high_resolution_clock::now();
        

        Barrett br(p_);
        MPI_NTT_multiply(a, b, ab, n_, p_, rank, size,br);

        
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
    MPI_Finalize();
    return 0;
}
