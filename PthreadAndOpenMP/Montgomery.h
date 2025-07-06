#pragma once
#include <stdint.h>

#include <stdexcept>

// Montgomery乘法类
class Montgomery {
public:
    uint64_t N;          // 模数
    uint64_t R;          // 通常选择2^k且满足 R > N, gcd(R,N)=1
    uint64_t logR;       // R的二进制位数（如R=2^64 → logR=64）
    uint64_t N_inv_neg;  // -N⁻¹ mod R
    uint64_t R2;         // R² mod N
    uint64_t R_mon_N;

    // 幂取模
    static uint64_t pow(uint64_t base, uint64_t exp, uint64_t mod) {
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

    // 扩展欧几里得算法求模逆元u
    static uint64_t extendedGCD(uint64_t a, uint64_t b, uint64_t &x,
                                uint64_t &y) {
        if (a == 0) {
            x = 0;
            y = 1;
            return b;
        }
        uint64_t x1, y1;
        uint64_t gcd = extendedGCD(b % a, a, x1, y1);
        x = y1 - (b / a) * x1;
        y = x1;
        return gcd;
    }

    // 计算模逆元 a⁻¹ mod m
    static uint64_t modinv(uint64_t a, uint64_t m) {
        uint64_t x, y;
        uint64_t gcd = extendedGCD(a, m, x, y);

        return (x < 0) ? x + m : x;
    }

public:
    // 构造函数
    Montgomery(uint64_t N)
        : N(N) {
        // 计算R为大于N的最小2^k
        this->logR = 32;           // 使用32位整数
        this->R = (1ULL << logR);  // R = 2^32

        // 计算N⁻¹ mod R
        uint64_t N_inv = modinv(N, R);
        this->N_inv_neg = R - N_inv;  // -N⁻¹ mod R

        // 预计算R² mod N
        this->R2 = (R % N) * (R % N) % N;
        this->R_mon_N = R % N;
    }

    // Montgomery约简算法
    // T 是待约简的数
    // 返回值是约简后的结果
    // 结果范围在[0, N-1]之间
    uint64_t REDC(uint64_t T) const {
        // m = (T mod R) * (-N⁻¹) mod R
        uint64_t m = ((T & (R - 1)) * N_inv_neg) & (R - 1);
        // t = (T + mN)/R
        uint64_t t = (T + m * N) >> logR;
        // 结果规约到[0, N-1]
        return (t >= N) ? t - N : t;
    }

    // Montgomery乘法
    uint64_t multiply(uint64_t a, uint64_t b) const {
        // 转换为Montgomery形式
        uint64_t aR = REDC(a * R2);
        uint64_t bR = REDC(b * R2);
        // 约简并转换回普通形式
        return REDC(REDC(aR * bR));
    }
};