#include <pthread.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

// Barrett reduction 结构体
struct Barrett {
    uint64_t mod;
    uint64_t k;
    __uint128_t mu;

    Barrett(uint64_t m) : mod(m) {
        k = 32;
        mu = ((__uint128_t)1 << 64) / m;
    }

    uint64_t reduce(uint64_t x) const {
        uint64_t q = (x * mu) >> 64;
        uint64_t r = x - q * mod;
        return r >= mod ? r - mod : r;
    }

    uint64_t multiply(uint64_t a, uint64_t b) const {
        return reduce((__uint128_t)a * b);
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

// 使用Barrett reduction的快速幂
uint64_t pow(uint64_t base, uint64_t exp, const Barrett &br) {
    uint64_t res = 1;
    base = br.reduce(base);
    while (exp > 0) {
        if (exp & 1) {
            res = br.multiply(res, base);
        }
        base = br.multiply(base, base);
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

void NTT(uint64_t *a, int n, bool invert, const Barrett &br, uint64_t g) {
    int bit = 0;
    while ((1 << bit) < n)
        bit++;

    reverse(a, n, bit);

    for (int len = 2; len <= n; len <<= 1) {
        uint64_t g_n = invert ? pow(g, br.mod - 1 - (br.mod - 1) / len, br)
                              : pow(g, (br.mod - 1) / len, br);

        for (int i = 0; i < n; i += len) {
            uint64_t g_pow = 1;
            for (int j = 0; j < len / 2; j++) {
                uint64_t u = a[i + j];
                uint64_t v = br.multiply(a[i + j + len / 2], g_pow);

                uint64_t sum = u + v;
                if (sum >= br.mod)
                    sum -= br.mod;
                uint64_t diff = u >= v ? u - v : u + br.mod - v;

                a[i + j] = sum;
                a[i + j + len / 2] = diff;
                g_pow = br.multiply(g_pow, g_n);
            }
        }
    }

    if (invert) {
        uint64_t inv_n = pow(n, br.mod - 2, br);
        for (int i = 0; i < n; i++) {
            a[i] = br.multiply(a[i], inv_n);
        }
    }
}

void NTT_multiply(uint64_t *a, uint64_t *b, uint64_t *result, int n, uint64_t p) {
    Barrett br(p);
    int len = n << 1;
    uint64_t *fa = new uint64_t[len];
    uint64_t *fb = new uint64_t[len];

    for (int i = 0; i < n; i++) {
        fa[i] = a[i];
        fb[i] = b[i];
    }
    for (int i = n; i < len; i++) {
        fa[i] = fb[i] = 0;
    }

    uint64_t g = 3;

    NTT(fa, len, false, br, g);
    NTT(fb, len, false, br, g);

    for (int i = 0; i < len; i++) {
        fa[i] = br.multiply(fa[i], fb[i]);
    }

    NTT(fa, len, true, br, g);

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

    uint64_t **mod_results = new uint64_t *[MOD_COUNT];
    for (int i = 0; i < MOD_COUNT; ++i) {
        mod_results[i] = new uint64_t[result_len]();
    }

    __uint128_t M = 1;
    for (int i = 0; i < MOD_COUNT; ++i)
        M *= MOD_LIST[i];

    __uint128_t MI_VALUES[MOD_COUNT];
    uint64_t MI_INV_VALUES[MOD_COUNT];
    for (int i = 0; i < MOD_COUNT; ++i) {
        MI_VALUES[i] = M / MOD_LIST[i];
        Barrett br(MOD_LIST[i]);
        uint64_t Mi_mod = MI_VALUES[i] % MOD_LIST[i];
        MI_INV_VALUES[i] = pow(Mi_mod, MOD_LIST[i] - 2, br);
    }

    for (int i = 0; i < MOD_COUNT; i++) {
        uint64_t *ta = new uint64_t[n];
        uint64_t *tb = new uint64_t[n];
        Barrett br(MOD_LIST[i]);
        for (int j = 0; j < n; j++) {
            ta[j] = br.reduce(a[j]);
            tb[j] = br.reduce(b[j]);
        }
        NTT_multiply(ta, tb, mod_results[i], n, MOD_LIST[i]);
        delete[] ta;
        delete[] tb;
    }

    for (int j = 0; j < result_len; j++) {
        __uint128_t sum = 0;
        for (int i = 0; i < MOD_COUNT; i++) {
            __uint128_t term = MI_VALUES[i] * ((mod_results[i][j] * MI_INV_VALUES[i]) % MOD_LIST[i]);
            sum = (sum + term) % M;
        }
        result[j] = sum % p;
    }

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
        if (p_ > (1ULL << 32)) {
            CRT_NTT_multiply_serial(a, b, ab, n_, p_);
        } else {
            NTT_multiply(a, b, ab, n_, p_);
        }
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000>> elapsed = End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);

        std::cout << "average latency for n = " << n_ << " p = " << p_ << " : "
                  << ans << " (us) " << std::endl;
        fWrite(ab, n_, i);
    }
    return 0;
}
