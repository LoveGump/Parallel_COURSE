#include <sys/time.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
// #include <omp.h>
#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <complex>

#include "Mentgomery32.h"
#include "Montgomery.h"

// 可以自行添加需要的头文件

void fRead(int *a, int *b, int *n, int *p, int input_id) {
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
    for (int i = 0; i < *n; i++) {
        fin >> a[i];
    }
    for (int i = 0; i < *n; i++) {
        fin >> b[i];
    }
}

void fCheck(int *ab, int n, int input_id) {
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
        int x;
        fin >> x;
        if (x != ab[i]) {
            std::cout << "多项式乘法结果错误" << std::endl;
            return;
        }
    }
    std::cout << "多项式乘法结果正确" << std::endl;
    return;
}

void fWrite(int *ab, int n, int input_id) {
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

void fWrite1(uint32_t *ab, int n, int input_id) {
    // 数据输出函数, 可以用来输出最终结果, 也可用于调试时输出中间数组
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (int i = 0; i < n; i++) {
        fout << ab[i] << '\n';
    }
}

void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            ab[i + j] = (1LL * a[i] * b[j] % p + ab[i + j]) % p;
        }
    }
}

// 1. 位逆序置换函数

// 快速幂
int pow(int base, int exp, int mod) {
    int res = 1;
    while (exp > 0) {
        if (exp & 1) {
            res = 1LL * res * base % mod;
        }
        base = 1LL * base * base % mod;
        exp >>= 1;
    }
    return res;
}

using Comp = std::complex<double>;  // 复数类型
constexpr Comp I(0, 1);             // i
constexpr int MAX_N = 1 << 20;
Comp tmp[MAX_N];  // 临时数组

// DFT（离散傅立叶变换） 来实现FFT（快速傅立叶变换）

// 递归 DFT
// a 是输入数组，n 是数组长度，rev 是正向还是逆向变换
// rev = 1 表示正向变换，rev = -1 表示逆向变换
void DFT(Comp *a, int n, int rev) {
    if (n == 1)
        return;  // 递归终止条件

    for (int i = 0; i < n; ++i)
        tmp[i] = a[i];

    // 偶数放在左面 ，奇数放在右面
    for (int i = 0; i < n; ++i) {
        if (i & 1)  // 奇数
            a[n / 2 + i / 2] = tmp[i];
        else  // 偶数
            a[i / 2] = tmp[i];
    }
    // 分别对偶数和奇数部分进行递归 DFT
    DFT(a, n / 2, rev);
    DFT(a + n / 2, n / 2, rev);

    Comp cur(1, 0), step(cos(2 * M_PI / n), sin(2 * M_PI * rev / n));
    // 这里的 cur 是当前单位复根，对于 k = 0 而言，它对应的单位复根 omega^0_n =
    // 1。
    for (int k = 0; k < n / 2; ++k) {
        tmp[k] = a[k] + cur * a[k + n / 2];
        tmp[k + n / 2] = a[k] - cur * a[k + n / 2];
        cur *= step;
    }
    for (int i = 0; i < n; ++i)
        a[i] = tmp[i];
}

// 迭代FFT
// on = 1 表示正向变换，on = -1 表示逆向变换
void reverse_fft(Comp *a, int n, int bit) {
    // a 是输入数组，n 是数组长度，bit 是二进制位数
    int *rev = new int[n];
    rev[0] = 0;
    for (int i = 0; i < n; i++) {
        // 二进制反转, rev[i] = 0;
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    }
    for (int i = 0; i < n; i++) {
        if (i < rev[i]) {
            std::swap(a[i], a[rev[i]]);  // 位逆序置换
        }
    }
    // 位逆序置换
}
void FFT_Iteration(Comp *a, int n, int on) {
    // 迭代FFT实现
    // 计算二进制位数
    int bit = 0;
    while ((1 << bit) < n)
        bit++;

    reverse_fft(a, n, bit);  // 20 是二进制位数，n <= 2^20

    // 蝶形操作
    for (int len = 2; len <= n; len <<= 1) {
        // len 是当前的蝶形长度，len = 2^k
        // wlen 是当前的单位复根，wlen = exp(2 * pi * i / len) ,单位圆的 len
        // 等分 如果是逆变换，则 wlen = exp(-2 * pi * i / len)
        Comp wlen = on == -1 ? std::exp(Comp(0, -2 * M_PI / len))
                             : std::exp(Comp(0, 2 * M_PI / len));  // 原根
        // 合并 一共合并 n / len 次
        for (int i = 0; i < n; i += len) {
            Comp w(1, 0);  // 当前单位复根 omega^k_n ，初始 k = 0 ，值为1

            // 每一部分 合并 len/2 次
            for (int j = 0; j < len / 2; j++) {
                Comp u = a[i + j];
                Comp v = a[i + j + len / 2] * w;
                // F(omega^k*n) = G(omega^k\_{n/2}) +
                // omega^k_n*H(omega^k\_{n/2})
                a[i + j] = u + v;
                // F(omega^{k+n/2}*n) = G(omega^k*{n/2}) -
                // omega^k_n*H(omega^k\_{n/2})
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    //逆向fft
    if (on == -1) {
        for (int i = 0; i < n; i++) {
            a[i] /= n;  // 逆变换结果除以 n
        }
    }
}
void FFT_multiply(int *a, int *b, int *result, int n, int p) {
    // 计算可以容纳结果的2的幂次长度
    int len = 1;
    while (len < 2 * n)
        len <<= 1;

    // 创建临时数组
    Comp *fa = new Comp[len];
    Comp *fb = new Comp[len];

    // 复制输入数组并填充0
    for (int i = 0; i < n; i++) {
        fa[i] = a[i];
        fb[i] = b[i];
    }
    for (int i = n; i < len; i++) {
        fa[i] = fb[i] = 0;
    }

    // 正向FFT
    FFT_Iteration(fa, len, 1);
    FFT_Iteration(fb, len, 1);

    // 点值表示下的乘法
    for (int i = 0; i < len; i++) {
        fa[i] *= fb[i];
    }

    // 逆向FFT得到结果
    FFT_Iteration(fa, len, -1);

    // 复制结果
    for (int i = 0; i < 2 * n - 1; i++) {
        result[i] = static_cast<int>(fa[i].real() + 0.5) % p;
        if (result[i] < 0)
            result[i] += p;  // 确保结果非负
    }

    delete[] fa;
    delete[] fb;
}

// NTT函数（数论变换）
// 迭代NTT实现，模仿FFT的实现 将 w 替换为原根 g
// a 是输入数组，n 是数组长度，invert 是是否进行逆变换
// p 是模数，g 是原根
void reverse(int *a, int n, int bit) {
    // a 是输入数组，n 是数组长度，bit 是二进制位数
    int *rev = new int[n];
    // i 的反转 = (i/2) 的反转去掉最高位后，再补上 i 的最低位作为最高位。
    rev[0] = 0;
    for (int i = 0; i < n; i++) {
        // 二进制反转, rev[i] = 0;
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    }
    for (int i = 0; i < n; i++) {
        if (i < rev[i]) {
            std::swap(a[i], a[rev[i]]);  // 位逆序置换
        }
    }
}
void NTT(int *a, int n, bool invert, int p, int g) {
    // 计算二进制位数
    int bit = 0;
    while ((1 << bit) < n)
        bit++;

    // 位逆序置换
    reverse(a, n, bit);

    // 蝶形操作
    for (int len = 2; len <= n; len <<= 1) {
        // len 是当前的蝶形长度，len = 2^k
        // wlen 是当前的单位根，wlen = g^(p-1)/len
        // 计算单位根，原根为g，G^((p-1)/len)
        // 如果invert为true，则单位根为g^((p-1)/len) 的逆元
        int g_n = invert ? pow(g, (p - 1) - (p - 1) / len, p)
                         : pow(g, (p - 1) / len, p);

        // 处理每个块
        for (int i = 0; i < n; i += len) {
            int g = 1;  // 初始单位根为1 k = 0 ，g = 1

            // 处理每个蝶形单元
            for (int j = 0; j < len / 2; j++) {
                int u = a[i + j];
                int v = (1LL * a[i + j + len / 2] * g) % p;
                // F(omega^k*n) = G(omega^k\_{n/2}) +
                // omega^k_n*H(omega^k\_{n/2})
                a[i + j] = (u + v) % p;
                // F(omega^{k+n/2}*n) = G(omega^k*{n/2}) -
                // omega^k_n*H(omega^k\_{n/2})
                a[i + j + len / 2] =
                    (u - v + p) % p;  // 注意要加上p再取模，保证结果非负

                g = (1LL * g * g_n) % p;  // 更新单位根
            }
        }
    }

    // 如果是逆变换，需要除以n（即乘以n的模p逆元）
    if (invert) {
        int inv_n = pow(n, p - 2, p);  // 使用费马小定理计算逆元
        for (int i = 0; i < n; i++) {
            a[i] = (1LL * a[i] * inv_n) % p;
        }
    }
}
// 使用NTT的多项式乘法
void NTT_multiply(int *a, int *b, int *result, int n, int p) {
    // 计算可以容纳结果的2的幂次长度
    int len = 1;
    while (len < 2 * n)
        len <<= 1;

    // 创建临时数组
    int *fa = new int[len];
    int *fb = new int[len];

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

void NTT_Montgomery(int *a, int n, bool invert, int p, int g = 3) {
    // 创建Montgomery实例
    Montgomery mont(p);

    // 计算二进制位数
    int bit = 0;
    while ((1 << bit) < n)
        bit++;

    // 位逆序置换(不变)

    reverse(a, n, bit);

    // 蝶形操作
    for (int len = 2; len <= n; len <<= 1) {
        int g_n = invert ? pow(g, (p - 1) - (p - 1) / len, p)
                         : pow(g, (p - 1) / len, p);

        int step = len >> 1;
        // step 是当前的蝶形步长，step = len / 2
        for (int i = 0; i < n; i += len) {
            uint64_t g_pow = 1;  // 当前单位根幂次

            for (int j = 0; j < step; j++) {
                uint64_t u = a[i + j];
                // 使用Montgomery乘法
                uint64_t v = mont.multiply(a[i + j + step], g_pow);

                a[i + j] = (u + v) % p;
                a[i + j + step] = (u - v + p) % p;

                // 更新单位根
                g_pow = mont.multiply(g_pow, g_n);
            }
        }
    }

    // 处理逆变换的系数
    if (invert) {
        int inv_n = pow(n, p - 2, p);
        for (int i = 0; i < n; i++) {
            a[i] = mont.multiply(a[i], inv_n);
        }
    }
}

void NTT_multiply_Montgomery(int *a, int *b, int *result, int n, int p) {
    // 准备工作不变
    int len = 1;
    while (len < 2 * n)
        len <<= 1;

    int *fa = new int[len];
    int *fb = new int[len];

    for (int i = 0; i < n; i++) {
        fa[i] = a[i];
        fb[i] = b[i];
    }
    for (int i = n; i < len; i++) {
        fa[i] = fb[i] = 0;
    }
    // 使用Montgomery优化的NTT
    NTT_Montgomery(fa, len, false, p);
    NTT_Montgomery(fb, len, false, p);

    // 点值乘法也使用Montgomery优化
    Montgomery mont(p);
    for (int i = 0; i < len; i++) {
        fa[i] = mont.multiply(fa[i], fb[i]);
    }

    // 逆NTT
    NTT_Montgomery(fa, len, true, p);

    // 复制结果
    for (int i = 0; i < 2 * n - 1; i++) {
        result[i] = fa[i];
    }

    delete[] fa;
    delete[] fb;
}

void NTT_Montgomerybase2(int *a, int n, bool invert, int p, Montgomery32 mont,
                         int g = 3) {
    // 计算二进制位数
    int bit = 0;
    while ((1 << bit) < n)
        bit++;
    // 位逆序置换(不变)
    reverse(a, n, bit);
    // 蝶形操作
    for (int len = 2; len <= n; len <<= 1) {
        // g_n^1
        int g_n_noamal = invert ? pow(g, (p - 1) - (p - 1) / len, p)
                                : pow(g, (p - 1) / len, p);

        int g_n = mont.REDC((uint64_t)g_n_noamal * mont.R2);
        int step = len >> 1;
        // step 是当前的蝶形步长，step = len / 2
        for (int i = 0; i < n; i += len) {
            uint64_t g_pow = mont.R_mod_N;  // 当前单位根幂次
            uint64_t u, v;
            for (int j = 0; j < step; j++) {
                u = a[i + j];
                // 使用Montgomery乘法
                v = mont.REDC((uint64_t)a[i + j + step] * g_pow);

                a[i + j] = (u + v) % p;
                a[i + j + step] = (u - v + p) % p;

                // 更新单位根
                g_pow = mont.REDC((uint64_t)g_pow * g_n);
            }
        }
    }

    // 处理逆变换的系数
    if (invert) {
        int inv_n = pow(n, p - 2, p);

        uint32_t inv_n_mont = mont.REDC((uint64_t)inv_n * mont.R2);

        for (int i = 0; i < n; i++) {
            a[i] = mont.REDC((uint64_t)a[i] * inv_n_mont);
        }
    }
}

void NTT_multiply_Montgomerybase2(int *a, int *b, int *result, int n, int p) {
    // 准备工作不变
    int len = 1;
    while (len < 2 * n)
        len <<= 1;

    Montgomery32 mont(p);

    int *fa = new int[len];
    int *fb = new int[len];

    // 转换为Montgomery域
    for (int i = 0; i < n; i++) {
        fa[i] = mont.REDC((uint64_t)a[i] * mont.R2);
        fb[i] = mont.REDC((uint64_t)b[i] * mont.R2);
    }
    for (int i = n; i < len; i++) {
        fa[i] = fb[i] = 0;
    }

    // 使用Montgomery优化的NTT
    NTT_Montgomerybase2(fa, len, false, p, mont);
    NTT_Montgomerybase2(fb, len, false, p, mont);

    for (int i = 0; i < len; i++) {
        fa[i] = mont.REDC((uint64_t)fa[i] * fb[i]);
    }

    // 逆NTT
    NTT_Montgomerybase2(fa, len, true, p, mont);

    // 复制结果
    for (int i = 0; i < 2 * n - 1; i++) {
        result[i] = mont.REDC((uint64_t)fa[i]);
    }

    delete[] fa;
    delete[] fb;
}

// 基4NTT
// 这里的基4NTT是基于基2NTT的优化版本
// 通过将输入数据分成4个部分来减少蝶形操作的次数
// 基4的位逆序置换函数
void reverse_base4(int *a, int n) {
    int log4n = 0;  // log4n是n的基4对数
    int temp = n;   // 临时变量
    while (temp > 1) {
        temp >>= 2;
        log4n++;
    }
    // 计算n的基4对数
    int *rev = new int[n];
    for (int i = 0; i < n; i++) {
        int reversed = 0;
        int num = i;
        for (int j = 0; j < log4n; j++) {
            reversed = (reversed << 2) | (num & 3);  // 每次取最低两位并左移
            num >>= 2;
        }
        rev[i] = reversed;
    }

    // 位逆序置换
    for (int i = 0; i < n; i++) {
        if (i < rev[i]) {
            std::swap(a[i], a[rev[i]]);
        }
    }
    delete[] rev;
}

// ------------------------------------------------------------------------------------

// 基4 NTT/INTT核心函数
void NTT_base4_naive(int *a, int n, bool invert, int p, int g = 3) {
    reverse_base4(a, n);

    // 蝶形操作
    for (int len = 4; len <= n; len <<= 2) {
        // len 是当前的蝶形长度，len = 4^k
        // wlen 是当前的单位根，wlen = g^(p-1)/len
        // 计算单位根，原根为g，G^((p-1)/len)
        // 如果invert为true，则单位根为g^((p-1)/len) 的逆元
        // 使用Montgomery计算单位根
        // g_n^1
        int g_n = invert ? pow(g, (p - 1) - (p - 1) / len, p)
                         : pow(g, (p - 1) / len, p);

        // 预计算单位根幂
        uint64_t g_n2 = (1LL * g_n * g_n) % p;
        uint64_t g_n3 = (1LL * g_n2 * g_n) % p;

        int step = len >> 2;
        uint64_t g_pow_step = pow(g_n, step, p);

        for (int i = 0; i < n; i += len) {
            // 处理每个蝴蝶形单元
            // 这里的len是4的幂次

            uint64_t w[4] = {1, 1, 1, 1};  // 当前单位根幂次

            // 当前单位根幂次
            uint64_t u[4];
            for (int j = 0; j < len / 4; j++) {
                if (j == 0) {
                    for (int k = 0; k < 4; k++) {
                        u[k] = a[i + j + k * step];
                    }
                } else {
                    for (int k = 0; k < 4; k++) {
                        u[k] = 1LL * a[i + j + k * step] * w[k] % p;
                    }
                }
                uint64_t j_1 = 1LL * u[1] * g_pow_step % p;
                uint64_t j_3 = 1LL * u[3] * g_pow_step % p;

                a[i + j] = (u[0] + u[1] + u[2] + u[3]) % p;
                a[i + j + step] = (u[0] + j_1 + p - u[2] + p - j_3) % p;
                a[i + j + 2 * step] = (u[0] + p - u[1] + u[2] + p - u[3]) % p;
                a[i + j + 3 * step] = (u[0] + p - j_1 + p - u[2] + j_3) % p;
                w[1] = 1LL * w[1] * g_n % p;
                w[2] = 1LL * w[2] * g_n2 % p;
                w[3] = 1LL * w[3] * g_n3 % p;
            }
        }
    }
    // 处理逆变换的系数
    if (invert) {
        int inv_n = pow(n, p - 2, p);
        for (int i = 0; i < n; i++) {
            a[i] = 1LL * a[i] * inv_n % p;
        }
    }
}

// 基4多项式乘法
void NTT_multiply_base4_Naive(int *a, int *b, int *result, int n, int p) {
    // 准备工作不变
    // len 为4的幂次
    int len = 1;
    while (len < 2 * n)
        len <<= 2;

    int *fa = new int[len];
    int *fb = new int[len];

    for (int i = 0; i < n; i++) {
        fa[i] = a[i];
        fb[i] = b[i];
    }
    for (int i = n; i < len; i++) {
        fa[i] = fb[i] = 0;
    }

    // 使用基4 NTT
    NTT_base4_naive(fa, len, false, p);
    NTT_base4_naive(fb, len, false, p);

    for (int i = 0; i < len; i++) {
        fa[i] = (1LL * fa[i] * fb[i]) % p;
    }

    // 逆NTT
    NTT_base4_naive(fa, len, true, p);

    // 复制结果
    for (int i = 0; i < 2 * n - 1; i++) {
        result[i] = fa[i];
    }

    delete[] fa;
    delete[] fb;
}

// ------------------------------------------------------------------------------------

void NTT_base4_m(int *a, int n, bool invert, int p, int g = 3) {
    // n 是数组长度，invert 是是否进行逆变换
    Montgomery mont(p);
    // 计算二进制位数

    reverse_base4(a, n);

    // 蝶形操作
    for (int len = 4; len <= n; len <<= 2) {
        // len 是当前的蝶形长度，len = 4^k
        // wlen 是当前的单位根，wlen = g^(p-1)/len
        // 计算单位根，原根为g，G^((p-1)/len)
        // 如果invert为true，则单位根为g^((p-1)/len) 的逆元
        // 使用Montgomery计算单位根
        // g_n^1
        int g_n = invert ? pow(g, (p - 1) - (p - 1) / len, p)
                         : pow(g, (p - 1) / len, p);

        // 预计算单位根幂
        uint64_t g_n2 = mont.multiply(g_n, g_n);
        uint64_t g_n3 = mont.multiply(g_n2, g_n);
        int step = len >> 2;
        uint64_t g_pow_step = pow(g_n, step, p);

        for (int i = 0; i < n; i += len) {
            // 处理每个蝴蝶形单元
            // 这里的len是4的幂次

            uint64_t w[4] = {1, 1, 1, 1};  // 当前单位根幂次

            // 当前单位根幂次
            uint64_t u[4];
            for (int j = 0; j < len / 4; j++) {
                if (j == 0) {
                    for (int k = 0; k < 4; k++) {
                        u[k] = a[i + j + k * step];
                    }
                } else {
                    for (int k = 0; k < 4; k++) {
                        u[k] = mont.multiply(a[i + j + k * step], w[k]);
                    }
                }
                uint64_t j_1 = mont.multiply(u[1], g_pow_step);
                uint64_t j_3 = mont.multiply(u[3], g_pow_step);

                a[i + j] = (u[0] + u[1] + u[2] + u[3]) % p;
                a[i + j + step] = (u[0] + j_1 + p - u[2] + p - j_3) % p;
                a[i + j + 2 * step] = (u[0] + p - u[1] + u[2] + p - u[3]) % p;
                a[i + j + 3 * step] = (u[0] + p - j_1 + p - u[2] + j_3) % p;
                w[1] = mont.multiply(w[1], g_n);
                w[2] = mont.multiply(w[2], g_n2);
                w[3] = mont.multiply(w[3], g_n3);
            }
        }
    }
    // 处理逆变换的系数
    if (invert) {
        int inv_n = pow(n, p - 2, p);
        for (int i = 0; i < n; i++) {
            a[i] = mont.multiply(a[i], inv_n);
        }
    }
}
void NTT_multiply_base4_m(int *a, int *b, int *result, int n, int p) {
    // 准备工作不变
    // len 为4的幂次
    int len = 1;
    while (len < 2 * n)
        len <<= 2;

    int *fa = new int[len];
    int *fb = new int[len];

    for (int i = 0; i < n; i++) {
        fa[i] = a[i];
        fb[i] = b[i];
    }
    for (int i = n; i < len; i++) {
        fa[i] = fb[i] = 0;
    }

    // 使用基4 NTT
    NTT_base4_m(fa, len, false, p);
    NTT_base4_m(fb, len, false, p);

    // 点值乘法也使用Montgomery优化
    Montgomery mont(p);
    for (int i = 0; i < len; i++) {
        fa[i] = mont.multiply(fa[i], fb[i]);
    }

    // 逆NTT
    NTT_base4_m(fa, len, true, p);

    // 复制结果
    for (int i = 0; i < 2 * n - 1; i++) {
        result[i] = fa[i];
    }

    delete[] fa;
    delete[] fb;
}
// ------------------------------------------------------------------------------------

// 使用蒙哥马利域的基4 NTT
void NTT_base4_Montgomery_domain(int *a, int n, bool invert, int p, int g = 3) {
    Montgomery mont(p);
    // 位逆序置换
    reverse_base4(a, n);
    // 蝶形操作
    for (int len = 4; len <= n; len <<= 2) {
        int g_n_normal = invert ? pow(g, (p - 1) - (p - 1) / len, p)
                                : pow(g, (p - 1) / len, p);

        // 将g_n转换为蒙哥马利域
        uint64_t g_n = mont.REDC(static_cast<uint32_t>(g_n_normal) * mont.R2);
        // 预计算单位根幂次
        uint64_t g_n2 = mont.REDC(g_n * g_n);
        uint64_t g_n3 = mont.REDC(g_n2 * g_n);

        int step = len >> 2;
        uint64_t g_pow_step_normal = pow(g_n_normal, step, p);
        uint64_t g_pow_step =
            mont.REDC(static_cast<uint64_t>(g_pow_step_normal) * mont.R2);
        uint64_t w[4] = {1, 1, 1, 1};
        for (int i = 0; i < n; i += len) {
            // 将1转换到蒙哥马利域
            w[0] = mont.REDC(1ULL * mont.R2);
            w[1] = w[0];
            w[2] = w[0];
            w[3] = w[0];
            uint64_t u[4];
            for (int j = 0; j < len / 4; j++) {
                if (j == 0) {
                    for (int k = 0; k < 4; k++) {
                        u[k] = a[i + j + k * step];  // 已经在蒙哥马利域
                    }
                } else {
                    for (int k = 0; k < 4; k++) {
                        // a和w都在蒙哥马利域，直接用REDC计算乘积
                        u[k] = mont.REDC(
                            static_cast<uint64_t>(a[i + j + k * step]) * w[k]);
                    }
                }

                // 计算旋转因子
                uint64_t j_1 = mont.REDC(u[1] * g_pow_step);
                uint64_t j_3 = mont.REDC(u[3] * g_pow_step);

                a[i + j] = (u[0] + u[1] + u[2] + u[3]) % p;
                a[i + j + step] = (u[0] + j_1 + p - u[2] + p - j_3) % p;
                a[i + j + 2 * step] = (u[0] + p - u[1] + u[2] + p - u[3]) % p;
                a[i + j + 3 * step] = (u[0] + p - j_1 + p - u[2] + j_3) % p;

                // 更新旋转因子
                w[1] = mont.REDC(w[1] * g_n);
                w[2] = mont.REDC(w[2] * g_n2);
                w[3] = mont.REDC(w[3] * g_n3);
            }
        }
    }

    // 处理逆变换的系数
    if (invert) {
        int inv_n = pow(n, p - 2, p);
        uint64_t inv_n_mont =
            mont.REDC(static_cast<uint64_t>(inv_n) * mont.R2);  // 转到蒙哥马利域

        for (int i = 0; i < n; i++) {
            a[i] = mont.REDC(static_cast<uint64_t>(a[i]) * inv_n_mont);
        }
    }
}
void NTT_multiply_base4_Montgomery_domain(int *a, int *b, int *result, int n,
                                          int p) {
    // 准备工作不变，但len需要是4的幂次
    int len = 1;
    while (len < 2 * n) {
        if (len < 4)
            len = 4;
        else
            len <<= 2;
    }
    // 初始化Montgomery实例
    Montgomery mont(p);

    int *fa = new int[len];
    int *fb = new int[len];

    // 复制数据
    for (int i = 0; i < n; i++) {
        fa[i] = mont.REDC(a[i] * mont.R2);
        fb[i] = mont.REDC(b[i] * mont.R2);
    }
    // 填充0
    for (int i = n; i < len; i++) {
        fa[i] = fb[i] = 0;
    }
    // 使用蒙哥马利优化的基4 NTT
    NTT_base4_Montgomery_domain(fa, len, false, p);
    NTT_base4_Montgomery_domain(fb, len, false, p);

    // 点值乘法（在蒙哥马利域中）
    for (int i = 0; i < len; i++) {
        fa[i] = mont.REDC(static_cast<uint64_t>(fa[i]) * fb[i]);
    }

    // 逆NTT
    NTT_base4_Montgomery_domain(fa, len, true, p);

    // 复制结果
    // 将结果从[0, p-1]调整为适当范围
    for (int i = 0; i < 2 * n - 1; i++) {
        result[i] = mont.REDC(fa[i]);
    }
    delete[] fa;
    delete[] fb;
}

// ------------------------------------------------------------------------------------

void reverse_base4neon(uint32_t *a, int n) {
    int log4n = 0;  // log4n是n的基4对数
    int temp = n;   // 临时变量
    while (temp > 1) {
        temp >>= 2;
        log4n++;
    }
    // 位逆序置换
    // 这里的位逆序置换是基于基4的
    // 通过将每个数的二进制位进行反转来实现
    // 预计算位逆序表
    int *rev = new int[n];
    for (int i = 0; i < n; i++) {
        int reversed = 0;
        int num = i;
        for (int j = 0; j < log4n; j++) {
            reversed = (reversed << 2) | (num & 3);
            num >>= 2;
        }
        rev[i] = reversed;
    }

    // 位逆序置换
    for (int i = 0; i < n; i++) {
        if (i < rev[i]) {
            std::swap(a[i], a[rev[i]]);
        }
    }
    delete[] rev;
}

// 使用优化的蒙哥马利域的基4 NTT
void NTT_base4_Montgomery32(int *a, int n, bool invert, int p) {
    Montgomery32 mont(p);
    // 位逆序置换
    reverse_base4(a, n);
    // 蝶形操作
    for (int len = 4; len <= n; len <<= 2) {
        int g = 3;  // 原根
        int g_n_normal = invert ? pow(g, (p - 1) - (p - 1) / len, p)
                                : pow(g, (p - 1) / len, p);

        // 将g_n转换为蒙哥马利域
        uint32_t g_n = mont.REDC((uint64_t)g_n_normal * mont.R2);
        // 预计算单位根幂次
        uint32_t g_n2 = mont.REDC((uint64_t)g_n * g_n);
        uint32_t g_n3 = mont.REDC((uint64_t)g_n2 * g_n);

        int step = len >> 2;
        uint32_t g_pow_step_normal = pow(g_n_normal, step, p);
        uint32_t g_pow_step = mont.REDC((uint64_t)g_pow_step_normal * mont.R2);

        for (int i = 0; i < n; i += len) {
            // 将1转换到蒙哥马利域
            uint32_t w[4];
            w[0] = mont.REDC((uint64_t)1 * mont.R2);
            w[1] = w[0];
            w[2] = w[0];
            w[3] = w[0];

            uint32_t u[4];
            for (int j = 0; j < len / 4; j++) {
                if (j == 0) {
                    for (int k = 0; k < 4; k++) {
                        u[k] = a[i + j + k * step];  // 已经在蒙哥马利域
                    }
                } else {
                    for (int k = 0; k < 4; k++) {
                        // a和w都在蒙哥马利域，使用32位乘法
                        u[k] = mont.REDC((uint64_t)a[i + j + k * step] * w[k]);
                    }
                }

                // 计算旋转因子
                uint32_t j_1 = mont.REDC((uint64_t)u[1] * g_pow_step);
                uint32_t j_3 = mont.REDC((uint64_t)u[3] * g_pow_step);

                // 蝶形操作结果
                a[i + j] = (u[0] + u[1] + u[2] + u[3]) % p;
                a[i + j + step] = (u[0] + j_1 + p - u[2] + p - j_3) % p;
                a[i + j + 2 * step] = (u[0] + p - u[1] + u[2] + p - u[3]) % p;
                a[i + j + 3 * step] = (u[0] + p - j_1 + p - u[2] + j_3) % p;

                // 更新旋转因子
                w[1] = mont.REDC((uint64_t)w[1] * g_n);
                w[2] = mont.REDC((uint64_t)w[2] * g_n2);
                w[3] = mont.REDC((uint64_t)w[3] * g_n3);
            }
        }
    }

    // 处理逆变换的系数
    if (invert) {
        int inv_n = pow(n, p - 2, p);
        uint32_t inv_n_mont =
            mont.REDC((uint64_t)inv_n * mont.R2);  // 转到蒙哥马利域

        for (int i = 0; i < n; i++) {
            a[i] = mont.REDC((uint64_t)a[i] * inv_n_mont);
        }
    }
}

//使用优化的蒙哥马利域的基4 NTT
void NTT_base4_Montgomery32neon(uint32_t *a, int n, bool invert, int p,
                                int g = 3) {
    Montgomery32 mont(p);
    // 位逆序置换
    reverse_base4neon(a, n);
    // 蝶形操作
    for (int len = 4; len <= n; len <<= 2) {
        int g_n_normal = invert ? pow(g, (p - 1) - (p - 1) / len, p)
                                : pow(g, (p - 1) / len, p);

        // 将g_n转换为蒙哥马利域
        uint32_t g_n = mont.REDC((uint64_t)g_n_normal * mont.R2);
        uint32_t g_n2 = mont.REDC((uint64_t)g_n * g_n);
        uint32_t g_n3 = mont.REDC((uint64_t)g_n2 * g_n);

        uint32_t g_n_array[4] = {mont.R_mod_N, g_n2, g_n3, g_n};
        // uint32x4_t g_n_vec = vld1q_u32(g_n_array);  // 加载g_n的4个值

        int step = len >> 2;
        uint32_t g_pow_step_normal = pow(g_n_normal, step, p);
        uint32_t g_pow_step = mont.REDC((uint64_t)g_pow_step_normal * mont.R2);

        uint64_t temp64[4];
        uint32_t temp32[4];

        for (int i = 0; i < n; i += len) {
            // 将1转换到蒙哥马利域
            uint32_t w[4];
            w[0] = mont.R_mod_N;
            w[1] = w[0];
            w[2] = w[0];
            w[3] = w[0];

            uint32_t u[4];
            for (int j = 0; j < len / 4; j++) {
                if (j == 0) {
                    for (int k = 0; k < 4; k++) {
                        u[k] = a[i + j + k * step];  // 已经在蒙哥马利域
                    }
                } else {
                    for (int k = 0; k < 4; k++) {
                        // a和w都在蒙哥马利域，使用32位乘法

                        u[k] = mont.REDC((uint64_t)a[i + j + k * step] * w[k]);

                        // //储存在temp64之中
                        //  temp64[k] = (uint64_t)a[i + j + k * step] * w[k];
                    }
                    // 会变慢 不如直接运算的结果

                    // // 使用NEON指令进行并行乘法
                    // uint32x4_t temp32x4 = mont.REDC_neon(temp64);
                    // // 将寄存器加载到u数组中
                    // vst1q_u32(u, temp32x4);
                }

                // // 计算旋转因子
                // temp64[0] = u[0] * mont.R_mod_N;
                // temp64[1] = u[1] * g_pow_step;
                // temp64[2] = u[2] * mont.R_mod_N;
                // temp64[3] = u[3] * g_pow_step;

                // // 使用NEON指令进行并行乘法

                //  uint32x4_t j_vec = mont.REDC_neon(temp64);
                // // 将数据加载回数组
                //  vst1q_u32(temp32, j_vec);

                //  uint32_t j_1 = temp32[1];
                // uint32_t j_3 = temp32[3];

                uint32_t j_1 = mont.REDC((uint64_t)u[1] * g_pow_step);
                uint32_t j_3 = mont.REDC((uint64_t)u[3] * g_pow_step);

                // 蝶形操作结果
                a[i + j] = (u[0] + u[1] + u[2] + u[3]) % p;
                a[i + j + step] = (u[0] + j_1 + p - u[2] + p - j_3) % p;
                a[i + j + 2 * step] = (u[0] + p - u[1] + u[2] + p - u[3]) % p;
                a[i + j + 3 * step] = (u[0] + p - j_1 + p - u[2] + j_3) % p;

                // 更新旋转因子
                // // 使用NEON指令进行并行乘法
                // temp64[0] = w[0] * mont.R_mod_N;
                // temp64[1] = w[1] * g_n;
                // temp64[2] = w[2] * g_n2;
                // temp64[3] = w[3] * g_n3;
                // // 使用NEON指令进行并行乘法
                // uint32x4_t w_vec = mont.REDC_neon(temp64);
                // // 将寄存器加载到w数组中
                // vst1q_u32(w, w_vec);

                w[1] = mont.REDC((uint64_t)w[1] * g_n);
                w[2] = mont.REDC((uint64_t)w[2] * g_n2);
                w[3] = mont.REDC((uint64_t)w[3] * g_n3);
            }
        }
    }

    // 处理逆变换的系数
    if (invert) {
        int inv_n = pow(n, p - 2, p);
        uint32_t inv_n_mont =
            mont.REDC((uint64_t)inv_n * mont.R2);  // 转到蒙哥马利域

        for (int i = 0; i < n; i++) {
            a[i] = mont.REDC((uint64_t)a[i] * inv_n_mont);
        }
    }
}
// ------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------

// void NTT_multiply_base4_Montgomery32(int *a, int *b, int *result, int n, int
// p) {
//    // 准备工作不变，但len需要是4的幂次
//    int len = 1;
//    while (len < 2 * n) {
//        if (len < 4)
//            len = 4;
//        else
//            len <<= 2;
//    }

//    // 初始化优化的32位Montgomery实例
//    Montgomery32 mont(p);

//     uint32_t *fa = new uint32_t[len];
//     uint32_t *fb = new uint32_t[len];

//    // 复制数据并转换到蒙哥马利域
//    for (int i = 0; i < n; i++) {
//        fa[i] = mont.REDC((uint64_t)a[i] * mont.R2);
//        fb[i] = mont.REDC((uint64_t)b[i] * mont.R2);
//    }

//    // 填充0
//    for (int i = n; i < len; i++) {
//        fa[i] = fb[i] = 0;
//    }

//    // 使用32位优化的蒙哥马利NTT
//    NTT_base4_Montgomery32neon(fa, len, false, p);
//    NTT_base4_Montgomery32neon(fb, len, false, p);

//    // 点值乘法（在蒙哥马利域中）
//    for (int i = 0; i < len; i++) {
//        fa[i] = mont.REDC((uint64_t)fa[i] * fb[i]);
//    }

//    // 逆NTT
//    NTT_base4_Montgomery32neon(fa, len, true, p);

//    // 复制结果，将结果从蒙哥马利域转回普通域
//    for (int i = 0; i < 2 * n - 1; i++) {
//        result[i] = mont.REDC(fa[i]);
//    }

//    delete[] fa;
//    delete[] fb;
// }

void NTT_multiply_base4_Montgomery32neon(int *a, int *b, int *result, int n,
                                         int p) {
    // 准备工作不变，但len需要是4的幂次
    int len = 1;
    while (len < 2 * n) {
        if (len < 4)
            len = 4;
        else
            len <<= 2;
    }

    // 初始化优化的32位Montgomery实例
    Montgomery32 mont(p);

    uint32_t *fa = new uint32_t[len];
    uint32_t *fb = new uint32_t[len];

    uint64_t a_mul[4], b_mul[4];
    uint32x4_t fa_vec, fb_vec;
    // 复制数据并转换到蒙哥马利域
    for (int i = 0; i < n; i += 4) {
        for (int j = 0; j < 4; j++) {
            a_mul[j] = (uint64_t)a[i + j] * mont.R2;
            b_mul[j] = (uint64_t)b[i + j] * mont.R2;
        }
        fa_vec = mont.REDC_neon(a_mul);
        fb_vec = mont.REDC_neon(b_mul);

        // Store results
        vst1q_u32(&fa[i], fa_vec);
        vst1q_u32(&fb[i], fb_vec);
    }

    // 填充0
    for (int i = n; i < len; i++) {
        fa[i] = fb[i] = 0;
    }

    // 使用32位优化的蒙哥马利NTT
    NTT_base4_Montgomery32neon(fa, len, false, p);
    NTT_base4_Montgomery32neon(fb, len, false, p);

    // 点值乘法（在蒙哥马利域中）
    uint64_t temp[4];
    for (int i = 0; i < len; i += 4) {
        for (int j = 0; j < 4; j++) {
            temp[j] = (uint64_t)fa[i + j] * fb[i + j];
        }
        fa_vec = mont.REDC_neon(temp);
        vst1q_u32(&fa[i], fa_vec);
        //  // 使用NEON指令进行点值乘法
        //  uint32x4_t fa_vec = vld1q_u32(&fa[i]);
        //  uint32x4_t fb_vec = vld1q_u32(&fb[i]);
        //  // 由于需要64位完整结果，我们分解为低位和高位分别计算
        // uint32x2_t fa_low = vget_low_u32(fa_vec);   // m0 m1
        // uint32x2_t fa_high = vget_high_u32(fa_vec); // m2 m3
        // uint32x2_t fb_low = vget_low_u32(fb_vec);  // N0 N1
        // uint32x2_t fb_high = vget_high_u32(fa_vec);// N2 N3

        // // 完整的乘法结果
        // uint64x2_t mul_low = vmull_u32(fa_low, fb_low);
        // uint64x2_t mul_high = vmull_u32(fa_high, fb_high);

        // // 分别保留high和low部分
        // uint32x4_t mul_result_high = vcombine_u32(
        //     vshrn_n_u64(mul_low, 32),
        //     vshrn_n_u64(mul_high, 32)
        // );
        // uint32x4_t mul_result_low = vcombine_u32(
        //     vmovn_u64(mul_low),
        //     vmovn_u64(mul_high)
        // );
        // // 计算最终结果

        // // 将结果存储到fa中
        // vst1q_u32(&fa[i],  mont.REDC_neon(mul_result_high, mul_result_low));
    }

    // 逆NTT
    NTT_base4_Montgomery32neon(fa, len, true, p);

    // 复制结果，将结果从蒙哥马利域转回普通域
    for (int i = 0; i < 2 * n - 1; i++) {
        result[i] = mont.REDC(fa[i]);
    }

    delete[] fa;
    delete[] fb;
}

int a[300000], b[300000], ab[300000];
int main(int argc, char *argv[]) {
    // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1
    // 的形式 输入模数分别为 7340033 104857601 469762049 1337006139375617
    // 第四个模数超过了整型表示范围,
    // 如果实现此模数意义下的多项式乘法需要修改框架
    // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT,
    // 请在完成前三个模数的基础代码及优化后实现大模数 NTT 输入文件共五个,
    // 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
    // 在实现快速数论变化前, 后四个测试样例运行时间较久,
    // 推荐调试正确性时只使用输入文件 1
    int test_begin = 0;
    int test_end = 4;
    for (int i = test_begin; i <= test_end; ++i) {
        long double ans = 0;
        int n_, p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));
        auto Start = std::chrono::high_resolution_clock::now();
        // TODO : 将 poly_multiply 函数替换成你写的 ntt
        poly_multiply(a, b, ab, n_, p_);
        // FFT_multiply(a, b, ab, n_, p_);
        // ntt 初始版本
        NTT_multiply(a, b, ab, n_, p_);
        // ntt 使用蒙哥马利 模乘 的版本
        // NTT_multiply_Montgomery(a, b, ab, n_, p_);
        // ntt使用蒙哥马利 域 的版本
        // NTT_multiply_Montgomerybase2(a, b, ab, n_, p_);

        // NTT_multiply_base4_(a, b, ab, n_, p_);
        // ntt使用基4 - 蒙哥马利模乘的版本
        // NTT_multiply_base4(a, b, ab, n_, p_);
        // NTT_multiply_base4_m(a, b, ab, n_, p_);
        // NTT_multiply_base4_NEON(a, b, ab, n_, p_);
        // ntt 使用基4 - 蒙哥马利域的版本
        // NTT_multiply_base4_Montgomery(a, b, ab, n_, p_);
        // 小于32位优化的蒙哥马利域基4 NTT
        // NTT_multiply_base4_Montgomery_domain(a, b, ab, n_, p_);
        // NTT_multiply_base4_Montgomery32(a, b, ab, n_, p_);
        // NTT_multiply_base4_Montgomery32neon(a, b, ab, n_, p_);

        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000>> elapsed =
            End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);
        std::cout << "average latency for n = " << n_ << " p = " << p_ << " : "
                  << ans << " (us) " << std::endl;
        // 可以使用 fWrite 函数将 ab 的输出结果打印到 files 文件夹下
        // 禁止使用 cout 一次性输出大量文件内容
        fWrite(ab, n_, i);
    }
    std::cout << "all test cases passed!34" << std::endl;
    return 0;
}
