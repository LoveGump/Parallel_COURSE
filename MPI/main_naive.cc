#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

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
            std::swap(a[i], a[rev[i]]);  // 位逆序置换
        }
    }
}
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
// 使用NTT的多项式乘法
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
void CRT_NTT_multiply_serial(uint64_t *a, uint64_t *b, uint64_t *result, int n, uint64_t p) {
    constexpr int MOD_COUNT = 4;
    const uint64_t MOD_LIST[MOD_COUNT] = {1004535809, 1224736769, 469762049, 998244353};
    int result_len = (n << 1) - 1;

    // 堆区二维数组
    uint64_t **mod_results = new uint64_t *[MOD_COUNT];
    for (int i = 0; i < MOD_COUNT; ++i) {
        mod_results[i] = new uint64_t[result_len]();
    }

    __uint128_t M = 1;
    for (int i = 0; i < MOD_COUNT; ++i)
        M *= MOD_LIST[i];

    // 预计算Mi和Mi_inv
    __uint128_t MI_VALUES[MOD_COUNT];
    uint64_t MI_INV_VALUES[MOD_COUNT];
    for (int i = 0; i < MOD_COUNT; ++i) {
        MI_VALUES[i] = M / MOD_LIST[i];
        uint64_t Mi_mod = MI_VALUES[i] % MOD_LIST[i];
        MI_INV_VALUES[i] = pow(Mi_mod, MOD_LIST[i] - 2, MOD_LIST[i]);
    }

    // 逐个模数串行计算NTT
    for (int i = 0; i < MOD_COUNT; i++) {
        uint64_t *ta = new uint64_t[n];
        uint64_t *tb = new uint64_t[n];
        for (int j = 0; j < n; j++) {
            ta[j] = a[j] % MOD_LIST[i];
            tb[j] = b[j] % MOD_LIST[i];
        }
        NTT_multiply(ta, tb, mod_results[i], n, MOD_LIST[i]);
        delete[] ta;
        delete[] tb;
    }

    // 串行CRT合并
    for (int j = 0; j < result_len; j++) {
        __uint128_t sum = 0;
        for (int i = 0; i < MOD_COUNT; i++) {
            __uint128_t term = MI_VALUES[i] * ((mod_results[i][j] * MI_INV_VALUES[i]) % MOD_LIST[i]);
            sum = (sum + term) % M;
        }
        result[j] = sum % p;
    }

    // 释放二维数组
    for (int i = 0; i < MOD_COUNT; ++i) {
        delete[] mod_results[i];
    }
    delete[] mod_results;
}
uint64_t a[300000], b[300000], ab[300000];
int main(int argc, char *argv[]) {
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
            CRT_NTT_multiply_serial(a, b, ab, n_, p_);
        } else {
            NTT_multiply(a, b, ab, n_, p_);
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
    return 0;
}
