/*
 * Compress.hpp
 *
 */

#ifndef COMPRESS_HPP_
#define COMPRESS_HPP_

#include <iostream>
#include <cassert>
#include <cmath>
#include <string.h>

uint32_t WORDS_COUNT(uint32_t v) {
    return (v != 0) ? (int) (log(v) / log(2)) : 0;
}

void bit_flush(uint32_t *&data, uint32_t &Fill, long &buffer, uint32_t &written) {
    if (Fill == 0) {
        return;
    }
    if (Fill > 32) {
        buffer <<= 64 - Fill;
        *data++ = buffer >> 32;
        *data++ = buffer & ((1ULL << 32) - 1);
        written += 2;
        Fill = 0;
    }
    if (Fill > 0) {
        *data++ = buffer << (32 - Fill) & ((1ULL << 32) - 1);
        written++;
    }
    buffer = 0;
    Fill = 0;
}

void bit_writer(uint32_t value, uint32_t bits, uint32_t *&data, uint32_t &Fill, long &buffer, uint32_t &written) {
    buffer = (buffer << bits) | (value & ((1ULL << bits) - 1));
    Fill += bits;
    if (Fill >= 32) {
        *data++ = (buffer >> (Fill - 32)) & ((1ULL << 32) - 1);
        written++;
        Fill -= 32;
    }
}

uint32_t bit_reader(uint32_t bits, uint32_t *&data, uint32_t &Fill, long &buffer) {
    if (bits == 0) {
        return 0;
    }
    if (Fill < bits) {
        buffer = (buffer << 32) | *data++;
        Fill += 32;
    }
    Fill -= bits;
    return (buffer >> Fill) & ((1ULL << bits) - 1);
}

#define SIMPLE_LOGDESC   4
#define SIMPLE_LEN    (1 << SIMPLE_LOGDESC)

#define SIMPLE16_DESC_FUNC1(num1, log1) \
    bool       \
    try##num1##_##log1##bit(uint32_t *n, uint32_t len) \
    {            \
        uint32_t  i;     \
        uint32_t  min;   \
        \
        min = (len < num1)? len : num1; \
        \
        for (i = 0; i < min; i++) {  \
            if (WORDS_COUNT(n[i]) > log1 - 1)  \
                return false;   \
        }      \
        \
        return true; \
    }

#define SIMPLE16_DESC_FUNC2(num1, log1, num2, log2)  \
    bool         \
    try##num1##_##log1##bit_##num2##_##log2##bit(uint32_t *n, uint32_t len)    \
    {            \
        uint32_t  i;     \
        uint32_t  base;     \
        uint32_t  min;   \
        \
        min = (len < num1)? len : num1; \
        \
        for (i = 0; i < min; i++) {  \
            if (WORDS_COUNT(n[i]) > log1 - 1)  \
                return false;   \
        }      \
        \
        base = min;  \
        len -= min;  \
        \
        min = (len < num2)? len: num2;        \
        \
        for (i = base; i < base + min; i++) {       \
            if (WORDS_COUNT(n[i]) > log2 - 1)  \
                return false;         \
        }      \
        \
        return true; \
    }

#define SIMPLE16_DESC_FUNC3(num1, log1, num2, log2, num3, log3) \
    bool         \
    try##num1##_##log1##bit_##num2##_##log2##bit_##num3##_##log3##bit(uint32_t *n, uint32_t len) \
    {            \
        uint32_t  i;     \
        uint32_t  base;     \
        uint32_t  min;   \
        \
        min = (len < num1)? len : num1; \
        \
        for (i = 0; i < min; i++) {  \
            if (WORDS_COUNT(n[i]) > log1 - 1)  \
                return false;   \
        }      \
        \
        base = min;  \
        len -= min;  \
        \
        min = (len < num2)? len: num2;        \
        \
        for (i = base; i < base + min; i++) {       \
            if (WORDS_COUNT(n[i]) > log2 - 1)  \
                return false;         \
        }      \
        \
        base += min; \
        len -= min;  \
        \
        min = (len < num3)? len: num3;        \
        \
        for (i = base; i < base + min; i++) {       \
            if (WORDS_COUNT(n[i]) > log3 - 1)  \
                return false;         \
        }      \
        \
        return true; \
    }

/* Fuction difinition by macros in a trying order */
SIMPLE16_DESC_FUNC1(28, 1);
SIMPLE16_DESC_FUNC2(7, 2, 14, 1);
SIMPLE16_DESC_FUNC3(7, 1, 7, 2, 7, 1);
SIMPLE16_DESC_FUNC2(14, 1, 7, 2);
SIMPLE16_DESC_FUNC1(14, 2);
SIMPLE16_DESC_FUNC2(1, 4, 8, 3);
SIMPLE16_DESC_FUNC3(1, 3, 4, 4, 3, 3);
SIMPLE16_DESC_FUNC1(7, 4);
SIMPLE16_DESC_FUNC2(4, 5, 2, 4);
SIMPLE16_DESC_FUNC2(2, 4, 4, 5);
SIMPLE16_DESC_FUNC2(3, 6, 2, 5);
SIMPLE16_DESC_FUNC2(2, 5, 3, 6);
SIMPLE16_DESC_FUNC1(4, 7);
SIMPLE16_DESC_FUNC2(1, 10, 2, 9);
SIMPLE16_DESC_FUNC1(2, 14);

/* A set of unpacking functions */
static inline void __simple16_unpack1_28(uint32_t **out, uint32_t **in)
__attribute__((always_inline));
static inline void __simple16_unpack2_7_1_14(uint32_t **out, uint32_t **in)
__attribute__((always_inline));
static inline void __simple16_unpack1_7_2_7_1_7(uint32_t **out, uint32_t **in)
__attribute__((always_inline));
static inline void __simple16_unpack1_14_2_7(uint32_t **out, uint32_t **in)
__attribute__((always_inline));
static inline void __simple16_unpack2_14(uint32_t **out, uint32_t **in)
__attribute__((always_inline));
static inline void __simple16_unpack4_1_3_8(uint32_t **out, uint32_t **in)
__attribute__((always_inline));
static inline void __simple16_unpack3_1_4_4_3_3(uint32_t **out, uint32_t **in)
__attribute__((always_inline));
static inline void __simple16_unpack4_7(uint32_t **out, uint32_t **in)
__attribute__((always_inline));
static inline void __simple16_unpack5_4_4_2(uint32_t **out, uint32_t **in)
__attribute__((always_inline));
static inline void __simple16_unpack4_2_5_4(uint32_t **out, uint32_t **in)
__attribute__((always_inline));
static inline void __simple16_unpack6_3_5_2(uint32_t **out, uint32_t **in)
__attribute__((always_inline));
static inline void __simple16_unpack5_2_6_3(uint32_t **out, uint32_t **in)
__attribute__((always_inline));
static inline void __simple16_unpack7_4(uint32_t **out, uint32_t **in)
__attribute__((always_inline));
static inline void __simple16_unpack10_1_9_2(uint32_t **out, uint32_t **in)
__attribute__((always_inline));
static inline void __simple16_unpack14_2(uint32_t **out, uint32_t **in)
__attribute__((always_inline));
static inline void __simple16_unpack28_1(uint32_t **out, uint32_t **in)
__attribute__((always_inline));

/* A interface of unpacking functions above */
typedef void (*__simple16_unpacker)(uint32_t **out, uint32_t **in);

static __simple16_unpacker __simple16_unpack[SIMPLE_LEN] ={
    __simple16_unpack1_28, __simple16_unpack2_7_1_14,
    __simple16_unpack1_7_2_7_1_7, __simple16_unpack1_14_2_7,
    __simple16_unpack2_14, __simple16_unpack4_1_3_8,
    __simple16_unpack3_1_4_4_3_3, __simple16_unpack4_7,
    __simple16_unpack5_4_4_2, __simple16_unpack4_2_5_4,
    __simple16_unpack6_3_5_2, __simple16_unpack5_2_6_3,
    __simple16_unpack7_4, __simple16_unpack10_1_9_2,
    __simple16_unpack14_2, __simple16_unpack28_1
};

void
__simple16_unpack1_28(uint32_t **out, uint32_t **in) {
    uint32_t *pout;
    uint32_t *pin;
    pout = *out;
    pin = *in;
    pout[0] = (pin[0] >> 27) & 0x01;
    pout[1] = (pin[0] >> 26) & 0x01;
    pout[2] = (pin[0] >> 25) & 0x01;
    pout[3] = (pin[0] >> 24) & 0x01;
    pout[4] = (pin[0] >> 23) & 0x01;
    pout[5] = (pin[0] >> 22) & 0x01;
    pout[6] = (pin[0] >> 21) & 0x01;
    pout[7] = (pin[0] >> 20) & 0x01;
    pout[8] = (pin[0] >> 19) & 0x01;
    pout[9] = (pin[0] >> 18) & 0x01;
    pout[10] = (pin[0] >> 17) & 0x01;
    pout[11] = (pin[0] >> 16) & 0x01;
    pout[12] = (pin[0] >> 15) & 0x01;
    pout[13] = (pin[0] >> 14) & 0x01;
    pout[14] = (pin[0] >> 13) & 0x01;
    pout[15] = (pin[0] >> 12) & 0x01;
    pout[16] = (pin[0] >> 11) & 0x01;
    pout[17] = (pin[0] >> 10) & 0x01;
    pout[18] = (pin[0] >> 9) & 0x01;
    pout[19] = (pin[0] >> 8) & 0x01;
    pout[20] = (pin[0] >> 7) & 0x01;
    pout[21] = (pin[0] >> 6) & 0x01;
    pout[22] = (pin[0] >> 5) & 0x01;
    pout[23] = (pin[0] >> 4) & 0x01;
    pout[24] = (pin[0] >> 3) & 0x01;
    pout[25] = (pin[0] >> 2) & 0x01;
    pout[26] = (pin[0] >> 1) & 0x01;
    pout[27] = pin[0] & 0x01;
    *in = pin + 1;
    *out = pout + 28;
}

void
__simple16_unpack2_7_1_14(uint32_t **out, uint32_t **in) {
    uint32_t *pout;
    uint32_t *pin;
    pout = *out;
    pin = *in;
    pout[0] = (pin[0] >> 26) & 0x03;
    pout[1] = (pin[0] >> 24) & 0x03;
    pout[2] = (pin[0] >> 22) & 0x03;
    pout[3] = (pin[0] >> 20) & 0x03;
    pout[4] = (pin[0] >> 18) & 0x03;
    pout[5] = (pin[0] >> 16) & 0x03;
    pout[6] = (pin[0] >> 14) & 0x03;
    pout[7] = (pin[0] >> 13) & 0x01;
    pout[8] = (pin[0] >> 12) & 0x01;
    pout[9] = (pin[0] >> 11) & 0x01;
    pout[10] = (pin[0] >> 10) & 0x01;
    pout[11] = (pin[0] >> 9) & 0x01;
    pout[12] = (pin[0] >> 8) & 0x01;
    pout[13] = (pin[0] >> 7) & 0x01;
    pout[14] = (pin[0] >> 6) & 0x01;
    pout[15] = (pin[0] >> 5) & 0x01;
    pout[16] = (pin[0] >> 4) & 0x01;
    pout[17] = (pin[0] >> 3) & 0x01;
    pout[18] = (pin[0] >> 2) & 0x01;
    pout[19] = (pin[0] >> 1) & 0x01;
    pout[20] = pin[0] & 0x01;
    *in = pin + 1;
    *out = pout + 21;
}

void
__simple16_unpack1_7_2_7_1_7(uint32_t **out, uint32_t **in) {
    uint32_t *pout;
    uint32_t *pin;
    pout = *out;
    pin = *in;
    pout[0] = (pin[0] >> 27) & 0x01;
    pout[1] = (pin[0] >> 26) & 0x01;
    pout[2] = (pin[0] >> 25) & 0x01;
    pout[3] = (pin[0] >> 24) & 0x01;
    pout[4] = (pin[0] >> 23) & 0x01;
    pout[5] = (pin[0] >> 22) & 0x01;
    pout[6] = (pin[0] >> 21) & 0x01;
    pout[7] = (pin[0] >> 19) & 0x03;
    pout[8] = (pin[0] >> 17) & 0x03;
    pout[9] = (pin[0] >> 15) & 0x03;
    pout[10] = (pin[0] >> 13) & 0x03;
    pout[11] = (pin[0] >> 11) & 0x03;
    pout[12] = (pin[0] >> 9) & 0x03;
    pout[13] = (pin[0] >> 7) & 0x03;
    pout[14] = (pin[0] >> 6) & 0x01;
    pout[15] = (pin[0] >> 5) & 0x01;
    pout[16] = (pin[0] >> 4) & 0x01;
    pout[17] = (pin[0] >> 3) & 0x01;
    pout[18] = (pin[0] >> 2) & 0x01;
    pout[19] = (pin[0] >> 1) & 0x01;
    pout[20] = pin[0] & 0x01;
    *in = pin + 1;
    *out = pout + 21;
}

void
__simple16_unpack1_14_2_7(uint32_t **out, uint32_t **in) {
    uint32_t *pout;
    uint32_t *pin;
    pout = *out;
    pin = *in;
    pout[0] = (pin[0] >> 27) & 0x01;
    pout[1] = (pin[0] >> 26) & 0x01;
    pout[2] = (pin[0] >> 25) & 0x01;
    pout[3] = (pin[0] >> 24) & 0x01;
    pout[4] = (pin[0] >> 23) & 0x01;
    pout[5] = (pin[0] >> 22) & 0x01;
    pout[6] = (pin[0] >> 21) & 0x01;
    pout[7] = (pin[0] >> 20) & 0x01;
    pout[8] = (pin[0] >> 19) & 0x01;
    pout[9] = (pin[0] >> 18) & 0x01;
    pout[10] = (pin[0] >> 17) & 0x01;
    pout[11] = (pin[0] >> 16) & 0x01;
    pout[12] = (pin[0] >> 15) & 0x01;
    pout[13] = (pin[0] >> 14) & 0x01;
    pout[14] = (pin[0] >> 12) & 0x03;
    pout[15] = (pin[0] >> 10) & 0x03;
    pout[16] = (pin[0] >> 8) & 0x03;
    pout[17] = (pin[0] >> 6) & 0x03;
    pout[18] = (pin[0] >> 4) & 0x03;
    pout[19] = (pin[0] >> 2) & 0x03;
    pout[20] = pin[0] & 0x03;
    *in = pin + 1;
    *out = pout + 21;
}

void
__simple16_unpack2_14(uint32_t **out, uint32_t **in) {
    uint32_t *pout;
    uint32_t *pin;
    pout = *out;
    pin = *in;
    pout[0] = (pin[0] >> 26) & 0x03;
    pout[1] = (pin[0] >> 24) & 0x03;
    pout[2] = (pin[0] >> 22) & 0x03;
    pout[3] = (pin[0] >> 20) & 0x03;
    pout[4] = (pin[0] >> 18) & 0x03;
    pout[5] = (pin[0] >> 16) & 0x03;
    pout[6] = (pin[0] >> 14) & 0x03;
    pout[7] = (pin[0] >> 12) & 0x03;
    pout[8] = (pin[0] >> 10) & 0x03;
    pout[9] = (pin[0] >> 8) & 0x03;
    pout[10] = (pin[0] >> 6) & 0x03;
    pout[11] = (pin[0] >> 4) & 0x03;
    pout[12] = (pin[0] >> 2) & 0x03;
    pout[13] = pin[0] & 0x03;
    *in = pin + 1;
    *out = pout + 14;
}

void
__simple16_unpack4_1_3_8(uint32_t **out, uint32_t **in) {
    uint32_t *pout;
    uint32_t *pin;
    pout = *out;
    pin = *in;
    pout[0] = (pin[0] >> 24) & 0x0f;
    pout[1] = (pin[0] >> 21) & 0x07;
    pout[2] = (pin[0] >> 18) & 0x07;
    pout[3] = (pin[0] >> 15) & 0x07;
    pout[4] = (pin[0] >> 12) & 0x07;
    pout[5] = (pin[0] >> 9) & 0x07;
    pout[6] = (pin[0] >> 6) & 0x07;
    pout[7] = (pin[0] >> 3) & 0x07;
    pout[8] = pin[0] & 0x07;
    *in = pin + 1;
    *out = pout + 9;
}

void
__simple16_unpack3_1_4_4_3_3(uint32_t **out, uint32_t **in) {
    uint32_t *pout;
    uint32_t *pin;
    pout = *out;
    pin = *in;
    pout[0] = (pin[0] >> 25) & 0x07;
    pout[1] = (pin[0] >> 21) & 0x0f;
    pout[2] = (pin[0] >> 17) & 0x0f;
    pout[3] = (pin[0] >> 13) & 0x0f;
    pout[4] = (pin[0] >> 9) & 0x0f;
    pout[5] = (pin[0] >> 6) & 0x07;
    pout[6] = (pin[0] >> 3) & 0x07;
    pout[7] = pin[0] & 0x07;
    *in = pin + 1;
    *out = pout + 8;
}

void
__simple16_unpack4_7(uint32_t **out, uint32_t **in) {
    uint32_t *pout;
    uint32_t *pin;
    pout = *out;
    pin = *in;
    pout[0] = (pin[0] >> 24) & 0x0f;
    pout[1] = (pin[0] >> 20) & 0x0f;
    pout[2] = (pin[0] >> 16) & 0x0f;
    pout[3] = (pin[0] >> 12) & 0x0f;
    pout[4] = (pin[0] >> 8) & 0x0f;
    pout[5] = (pin[0] >> 4) & 0x0f;
    pout[6] = pin[0] & 0x0f;
    *in = pin + 1;
    *out = pout + 7;
}

void
__simple16_unpack5_4_4_2(uint32_t **out, uint32_t **in) {
    uint32_t *pout;
    uint32_t *pin;
    pout = *out;
    pin = *in;
    pout[0] = (pin[0] >> 23) & 0x1f;
    pout[1] = (pin[0] >> 18) & 0x1f;
    pout[2] = (pin[0] >> 13) & 0x1f;
    pout[3] = (pin[0] >> 8) & 0x1f;
    pout[4] = (pin[0] >> 4) & 0x0f;
    pout[5] = pin[0] & 0x0f;
    *in = pin + 1;
    *out = pout + 6;
}

void
__simple16_unpack4_2_5_4(uint32_t **out, uint32_t **in) {
    uint32_t *pout;
    uint32_t *pin;
    pout = *out;
    pin = *in;
    pout[0] = (pin[0] >> 24) & 0x0f;
    pout[1] = (pin[0] >> 20) & 0x0f;
    pout[2] = (pin[0] >> 15) & 0x1f;
    pout[3] = (pin[0] >> 10) & 0x1f;
    pout[4] = (pin[0] >> 5) & 0x1f;
    pout[5] = pin[0] & 0x1f;
    *in = pin + 1;
    *out = pout + 6;
}

void
__simple16_unpack6_3_5_2(uint32_t **out, uint32_t **in) {
    uint32_t *pout;
    uint32_t *pin;
    pout = *out;
    pin = *in;
    pout[0] = (pin[0] >> 22) & 0x3f;
    pout[1] = (pin[0] >> 16) & 0x3f;
    pout[2] = (pin[0] >> 10) & 0x3f;
    pout[3] = (pin[0] >> 5) & 0x1f;
    pout[4] = pin[0] & 0x1f;
    *in = pin + 1;
    *out = pout + 5;
}

void
__simple16_unpack5_2_6_3(uint32_t **out, uint32_t **in) {
    uint32_t *pout;
    uint32_t *pin;
    pout = *out;
    pin = *in;
    pout[0] = (pin[0] >> 23) & 0x1f;
    pout[1] = (pin[0] >> 18) & 0x1f;
    pout[2] = (pin[0] >> 12) & 0x3f;
    pout[3] = (pin[0] >> 6) & 0x3f;
    pout[4] = pin[0] & 0x3f;
    *in = pin + 1;
    *out = pout + 5;
}

void
__simple16_unpack7_4(uint32_t **out, uint32_t **in) {
    uint32_t *pout;
    uint32_t *pin;
    pout = *out;
    pin = *in;
    pout[0] = (pin[0] >> 21) & 0x7f;
    pout[1] = (pin[0] >> 14) & 0x7f;
    pout[2] = (pin[0] >> 7) & 0x7f;
    pout[3] = pin[0] & 0x7f;
    *in = pin + 1;
    *out = pout + 4;
}

void
__simple16_unpack10_1_9_2(uint32_t **out, uint32_t **in) {
    uint32_t *pout;
    uint32_t *pin;
    pout = *out;
    pin = *in;
    pout[0] = (pin[0] >> 18) & 0x03ff;
    pout[1] = (pin[0] >> 9) & 0x01ff;
    pout[2] = pin[0] & 0x01ff;
    *in = pin + 1;
    *out = pout + 3;
}

void
__simple16_unpack14_2(uint32_t **out, uint32_t **in) {
    uint32_t *pout;
    uint32_t *pin;
    pout = *out;
    pin = *in;
    pout[0] = (pin[0] >> 14) & 0x3fff;
    pout[1] = pin[0] & 0x3fff;
    *in = pin + 1;
    *out = pout + 2;
}

void
__simple16_unpack28_1(uint32_t **out, uint32_t **in) {
    uint32_t *pout;
    uint32_t *pin;
    pout = *out;
    pin = *in;
    pout[0] = pin[0] & 0x0fffffff;
    *in = pin + 1;
    *out = pout + 1;
}

void Simple16Encode(uint32_t *in, uint32_t len, uint32_t *out, uint32_t &nvalue) {
    uint32_t i, min, base, fill = 0, written = 0;
    long buffer = 0;
    while (len > 0) {
        if (try28_1bit(in, len)) {
            bit_writer(0, 4, out, fill, buffer, written);
            min = (len < 28) ? len : 28;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 1, out, fill, buffer, written);
            }
        } else if (try7_2bit_14_1bit(in, len)) {
            bit_writer(1, 4, out, fill, buffer, written);
            min = (len < 7) ? len : 7;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 2, out, fill, buffer, written);
            }
            base = min;
            min = (len - base < 14) ? len - base : 14;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 1, out, fill, buffer, written);
            }
            min += base;
        } else if (try7_1bit_7_2bit_7_1bit(in, len)) {
            bit_writer(2, 4, out, fill, buffer, written);
            min = (len < 7) ? len : 7;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 1, out, fill, buffer, written);
            }
            base = min;
            min = (len - base < 7) ? len - base : 7;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 2, out, fill, buffer, written);
            }
            base += min;
            min = (len - base < 7) ? len - base : 7;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 1, out, fill, buffer, written);
            }
            min += base;
        } else if (try14_1bit_7_2bit(in, len)) {
            bit_writer(3, 4, out, fill, buffer, written);
            min = (len < 14) ? len : 14;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 1, out, fill, buffer, written);
            }
            base = min;
            min = (len - base < 7) ? len - base : 7;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 2, out, fill, buffer, written);
            }
            min += base;
        } else if (try14_2bit(in, len)) {
            bit_writer(4, 4, out, fill, buffer, written);
            min = (len < 14) ? len : 14;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 2, out, fill, buffer, written);
            }
        } else if (try1_4bit_8_3bit(in, len)) {
            bit_writer(5, 4, out, fill, buffer, written);
            min = (len < 1) ? len : 1;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 4, out, fill, buffer, written);
            }
            base = min;
            min = (len - base < 8) ? len - base : 8;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 3, out, fill, buffer, written);
            }
            min += base;
        } else if (try1_3bit_4_4bit_3_3bit(in, len)) {
            bit_writer(6, 4, out, fill, buffer, written);
            min = (len < 1) ? len : 1;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 3, out, fill, buffer, written);
            }
            base = min;
            min = (len - base < 4) ? len - base : 4;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 4, out, fill, buffer, written);
            }
            base += min;
            min = (len - base < 3) ? len - base : 3;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 3, out, fill, buffer, written);
            }
            min += base;
        } else if (try7_4bit(in, len)) {
            bit_writer(7, 4, out, fill, buffer, written);
            min = (len < 7) ? len : 7;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 4, out, fill, buffer, written);
            }
        } else if (try4_5bit_2_4bit(in, len)) {
            bit_writer(8, 4, out, fill, buffer, written);
            min = (len < 4) ? len : 4;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 5, out, fill, buffer, written);
            }
            base = min;
            min = (len - base < 2) ? len - base : 2;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 4, out, fill, buffer, written);
            }
            min += base;
        } else if (try2_4bit_4_5bit(in, len)) {
            bit_writer(9, 4, out, fill, buffer, written);
            min = (len < 2) ? len : 2;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 4, out, fill, buffer, written);
            }
            base = min;
            min = (len - base < 4) ? len - base : 4;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 5, out, fill, buffer, written);
            }
            min += base;
        } else if (try3_6bit_2_5bit(in, len)) {
            bit_writer(10, 4, out, fill, buffer, written);
            min = (len < 3) ? len : 3;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 6, out, fill, buffer, written);
            }
            base = min;
            min = (len - base < 2) ? len - base : 2;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 5, out, fill, buffer, written);
            }
            min += base;
        } else if (try2_5bit_3_6bit(in, len)) {
            bit_writer(11, 4, out, fill, buffer, written);
            min = (len < 2) ? len : 2;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 5, out, fill, buffer, written);
            }
            base = min;
            min = (len - base < 3) ? len - base : 3;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 6, out, fill, buffer, written);
            }
            min += base;
        } else if (try4_7bit(in, len)) {
            bit_writer(12, 4, out, fill, buffer, written);
            min = (len < 4) ? len : 4;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 7, out, fill, buffer, written);
            }
        } else if (try1_10bit_2_9bit(in, len)) {
            bit_writer(13, 4, out, fill, buffer, written);
            min = (len < 1) ? len : 1;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 10, out, fill, buffer, written);
            }
            base = min;
            min = (len - base < 2) ? len - base : 2;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 9, out, fill, buffer, written);
            }
            min += base;
        } else if (try2_14bit(in, len)) {
            bit_writer(14, 4, out, fill, buffer, written);
            min = (len < 2) ? len : 2;
            for (i = 0; i < min; i++) {
                bit_writer(*in++, 14, out, fill, buffer, written);
            }
        } else {
            if ((*in >> 28) > 0) {
                fprintf(stderr, "Input's out of range: %u", *in);
            }
            bit_writer(15, 4, out, fill, buffer, written);
            min = 1;
            bit_writer(*in++, 28, out, fill, buffer, written);
        }
        bit_flush(out, fill, buffer, written);
        len -= min;
    }
    nvalue = written;
}

void Simple16Decode(uint32_t *in, uint32_t len, uint32_t *out, uint32_t nvalue) {
    uint32_t *end;
    end = out + nvalue;
    while (end > out) {
        (__simple16_unpack[((unsigned) *in) >> (32 - SIMPLE_LOGDESC)])(&out, &in);
    }
}

int
div_roundup(uint32_t v, uint32_t div) {
    return (v + (div - 1)) / div;
}

#define PFORDELTA_RATIO   0.1
#define PFORDELTA_USE_HARDCODE_SIMPLE16   1
#define PFORDELTA_B    6
#define PFORDELTA_NEXCEPT    10
#define PFORDELTA_EXCEPTSZ   16
#define __array_size(x)   (sizeof(x) / sizeof(x[0]))
#define PFORDELTA_NBLOCK  1
#define PFORDELTA_BLOCKSZ    (32 * PFORDELTA_NBLOCK)
#define TAIL_MERGIN  128

#define __p4delta_copy(src, dest)    \
    __asm__ __volatile__(     \
                              "movdqu %4, %%xmm0\n\t"   \
                              "movdqu %5, %%xmm1\n\t"   \
                              "movdqu %6, %%xmm2\n\t"   \
                              "movdqu %7, %%xmm3\n\t"   \
                              "movdqu %%xmm0, %0\n\t"   \
                              "movdqu %%xmm1, %1\n\t"   \
                              "movdqu %%xmm2, %2\n\t"   \
                              "movdqu %%xmm3, %3\n\t"   \
                              :"=m" (dest[0]), "=m" (dest[4]), "=m" (dest[8]), "=m" (dest[12])  \
                              :"m" (src[0]), "m" (src[4]), "m" (src[8]), "m" (src[12])    \
                              :"memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3")

#define __p4delta_zero32(dest)    \
    __asm__ __volatile__(     \
                              "pxor   %%xmm0, %%xmm0\n\t"  \
                              "movdqu %%xmm0, %0\n\t"   \
                              "movdqu %%xmm0, %1\n\t"   \
                              "movdqu %%xmm0, %2\n\t"   \
                              "movdqu %%xmm0, %3\n\t"   \
                              "movdqu %%xmm0, %4\n\t"   \
                              "movdqu %%xmm0, %5\n\t"   \
                              "movdqu %%xmm0, %6\n\t"   \
                              "movdqu %%xmm0, %7\n\t"   \
                              :"=m" (dest[0]), "=m" (dest[4]), "=m" (dest[8]), "=m" (dest[12]) ,      \
                              "=m" (dest[16]), "=m" (dest[20]), "=m" (dest[24]), "=m" (dest[28])   \
                              ::"memory", "%xmm0")

/* A set of unpacking functions */
static void __p4delta_unpack0(uint32_t *out, uint32_t *in);
static void __p4delta_unpack1(uint32_t *out, uint32_t *in);
static void __p4delta_unpack2(uint32_t *out, uint32_t *in);
static void __p4delta_unpack3(uint32_t *out, uint32_t *in);
static void __p4delta_unpack4(uint32_t *out, uint32_t *in);
static void __p4delta_unpack5(uint32_t *out, uint32_t *in);
static void __p4delta_unpack6(uint32_t *out, uint32_t *in);
static void __p4delta_unpack7(uint32_t *out, uint32_t *in);
static void __p4delta_unpack8(uint32_t *out, uint32_t *in);
static void __p4delta_unpack9(uint32_t *out, uint32_t *in);
static void __p4delta_unpack10(uint32_t *out, uint32_t *in);
static void __p4delta_unpack11(uint32_t *out, uint32_t *in);
static void __p4delta_unpack12(uint32_t *out, uint32_t *in);
static void __p4delta_unpack13(uint32_t *out, uint32_t *in);
static void __p4delta_unpack16(uint32_t *out, uint32_t *in);
static void __p4delta_unpack20(uint32_t *out, uint32_t *in);
static void __p4delta_unpack32(uint32_t *out, uint32_t *in);

/* A interface of unpacking functions above */
typedef void (*__p4delta_unpacker)(uint32_t *out, uint32_t *in);

static __p4delta_unpacker __p4delta_unpack[] ={
    __p4delta_unpack0,
    __p4delta_unpack1,
    __p4delta_unpack2,
    __p4delta_unpack3,
    __p4delta_unpack4,
    __p4delta_unpack5,
    __p4delta_unpack6,
    __p4delta_unpack7,
    __p4delta_unpack8,
    __p4delta_unpack9,
    __p4delta_unpack10,
    __p4delta_unpack11,
    __p4delta_unpack12,
    __p4delta_unpack13,
    NULL,
    NULL,
    __p4delta_unpack16,
    NULL,
    NULL,
    NULL,
    __p4delta_unpack20,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    __p4delta_unpack32
};

/* A hard-corded Simple16 decoder wirtten in the original code */
static inline void __p4delta_simple16_decode(uint32_t *in, uint32_t len,
        uint32_t *out, uint32_t nvalue) __attribute__((always_inline));

static uint32_t __p4delta_possLogs[] ={
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 20, 32
};

uint32_t
tryB(uint32_t b, uint32_t *in, uint32_t len) {
    uint32_t i;
    uint32_t curExcept;
    assert(b <= 32);
    if (b == 32) {
        return 0;
    }
    for (i = 0, curExcept = 0; i < len; i++) {
        if (in[i] >= (1ULL << b)) {
            curExcept++;
        }
    }
    return curExcept;
}

uint32_t
findBestB(uint32_t *in, uint32_t len) {
    uint32_t i;
    uint32_t nExceptions;
    for (i = 0; i < __array_size(__p4delta_possLogs) - 1; i++) {
        nExceptions = tryB(__p4delta_possLogs[i], in, len);
        if (nExceptions <= len * PFORDELTA_RATIO) {
            return __p4delta_possLogs[i];
        }
    }
    return __p4delta_possLogs[__array_size(__p4delta_possLogs) - 1];
}

void
encodeBlock(uint32_t *in, uint32_t len,
        uint32_t *out, uint32_t &nvalue,
        uint32_t(*find)(uint32_t *in, uint32_t len)) {
    uint32_t i;
    uint32_t b;
    uint32_t e;
    uint32_t *codewords;
    uint32_t *codewords_tmp;
    uint32_t codewords_sz;
    uint32_t curExcept;
    uint32_t *exceptionsPositions;
    uint32_t *exceptionsValues;
    uint32_t *exceptions;
    uint32_t *encodedExceptions;
    uint32_t encodedExceptions_sz;
    uint32_t excPos;
    uint32_t excVal;
    if (len > 0) {
        uint32_t fill = 0, written = 0;
        long buffer = 0;
        codewords = new uint32_t[len];
        codewords_tmp = codewords;
        exceptionsPositions = new uint32_t[len];
        exceptionsValues = new uint32_t[len];
        exceptions = new uint32_t[2 * len];
        encodedExceptions = new uint32_t[2 * len + 2];
        if (codewords == NULL || exceptionsPositions == NULL ||
                exceptionsValues == NULL ||
                exceptions == NULL || encodedExceptions == NULL) {
            fprintf(stderr, "Can't allocate memory");
        }
        b = find(in, len);
        curExcept = 0;
        encodedExceptions_sz = 0;
        if (b < 32) {
            for (i = 0; i < len; i++) {
                bit_writer(in[i], b, codewords, fill, buffer, written);
                if (in[i] >= (1U << b)) {
                    e = in[i] >> b;
                    exceptionsPositions[curExcept] = i;
                    exceptionsValues[curExcept] = e;
                    curExcept++;
                }
            }
            if (curExcept > 0) {
                uint32_t cur;
                uint32_t prev;
                for (i = curExcept - 1; i > 0; i--) {
                    cur = exceptionsPositions[i];
                    prev = exceptionsPositions[i - 1];
                    exceptionsPositions[i] = cur - prev;
                }
                for (i = 0; i < curExcept; i++) {
                    excPos = (i > 0) ? exceptionsPositions[i] - 1 :
                            exceptionsPositions[i];
                    excVal = exceptionsValues[i] - 1;
                    exceptions[i] = excPos;
                    exceptions[i + curExcept] = excVal;
                }
                Simple16Encode(exceptions, 2 * curExcept,
                        encodedExceptions, encodedExceptions_sz);
            }
        } else {
            for (i = 0; i < len; i++) {
                bit_writer(in[i], 32, codewords, fill, buffer, written);
            }
            bit_flush(codewords, fill, buffer, written);
        }
        bit_flush(codewords, fill, buffer, written);
        // Write a header following the format
        *out++ = (b << (PFORDELTA_NEXCEPT + PFORDELTA_EXCEPTSZ)) |
                (curExcept << PFORDELTA_EXCEPTSZ) | encodedExceptions_sz;
        nvalue = 1;
        // Write exceptional values
        memcpy(out, encodedExceptions, encodedExceptions_sz * sizeof (int));
        out += encodedExceptions_sz;
        nvalue += encodedExceptions_sz;
        // Write fix-length values
        codewords_sz = written;
        memcpy(out, codewords_tmp, codewords_sz * sizeof (int));
        nvalue += codewords_sz;
        /* Finalization */
        delete[] exceptions;
        delete[] exceptionsPositions;
        delete[] exceptionsValues;
        delete[] encodedExceptions;
        delete[] codewords_tmp;
    }
}

/* --- Intra functions below --- */

void
__p4delta_simple16_decode(uint32_t *in, uint32_t len,
        uint32_t *out, uint32_t nvalue) {
    uint32_t hd;
    uint32_t nlen;
    nlen = 0;
    while (len > nlen) {
        hd = *in >> 28;
        switch (hd) {
            case 0:
                *out++ = (*in >> 27) & 0x01;
                *out++ = (*in >> 26) & 0x01;
                *out++ = (*in >> 25) & 0x01;
                *out++ = (*in >> 24) & 0x01;
                *out++ = (*in >> 23) & 0x01;
                *out++ = (*in >> 22) & 0x01;
                *out++ = (*in >> 21) & 0x01;
                *out++ = (*in >> 20) & 0x01;
                *out++ = (*in >> 19) & 0x01;
                *out++ = (*in >> 18) & 0x01;
                *out++ = (*in >> 17) & 0x01;
                *out++ = (*in >> 16) & 0x01;
                *out++ = (*in >> 15) & 0x01;
                *out++ = (*in >> 14) & 0x01;
                *out++ = (*in >> 13) & 0x01;
                *out++ = (*in >> 12) & 0x01;
                *out++ = (*in >> 11) & 0x01;
                *out++ = (*in >> 10) & 0x01;
                *out++ = (*in >> 9) & 0x01;
                *out++ = (*in >> 8) & 0x01;
                *out++ = (*in >> 7) & 0x01;
                *out++ = (*in >> 6) & 0x01;
                *out++ = (*in >> 5) & 0x01;
                *out++ = (*in >> 4) & 0x01;
                *out++ = (*in >> 3) & 0x01;
                *out++ = (*in >> 2) & 0x01;
                *out++ = (*in >> 1) & 0x01;
                *out++ = *in++ & 0x01;
                nlen += 28;
                break;
            case 1:
                *out++ = (*in >> 26) & 0x03;
                *out++ = (*in >> 24) & 0x03;
                *out++ = (*in >> 22) & 0x03;
                *out++ = (*in >> 20) & 0x03;
                *out++ = (*in >> 18) & 0x03;
                *out++ = (*in >> 16) & 0x03;
                *out++ = (*in >> 14) & 0x03;
                *out++ = (*in >> 13) & 0x01;
                *out++ = (*in >> 12) & 0x01;
                *out++ = (*in >> 11) & 0x01;
                *out++ = (*in >> 10) & 0x01;
                *out++ = (*in >> 9) & 0x01;
                *out++ = (*in >> 8) & 0x01;
                *out++ = (*in >> 7) & 0x01;
                *out++ = (*in >> 6) & 0x01;
                *out++ = (*in >> 5) & 0x01;
                *out++ = (*in >> 4) & 0x01;
                *out++ = (*in >> 3) & 0x01;
                *out++ = (*in >> 2) & 0x01;
                *out++ = (*in >> 1) & 0x01;
                *out++ = *in++ & 0x01;
                nlen += 21;
                break;
            case 2:
                *out++ = (*in >> 27) & 0x01;
                *out++ = (*in >> 26) & 0x01;
                *out++ = (*in >> 25) & 0x01;
                *out++ = (*in >> 24) & 0x01;
                *out++ = (*in >> 23) & 0x01;
                *out++ = (*in >> 22) & 0x01;
                *out++ = (*in >> 21) & 0x01;
                *out++ = (*in >> 19) & 0x03;
                *out++ = (*in >> 17) & 0x03;
                *out++ = (*in >> 15) & 0x03;
                *out++ = (*in >> 13) & 0x03;
                *out++ = (*in >> 11) & 0x03;
                *out++ = (*in >> 9) & 0x03;
                *out++ = (*in >> 7) & 0x03;
                *out++ = (*in >> 6) & 0x01;
                *out++ = (*in >> 5) & 0x01;
                *out++ = (*in >> 4) & 0x01;
                *out++ = (*in >> 3) & 0x01;
                *out++ = (*in >> 2) & 0x01;
                *out++ = (*in >> 1) & 0x01;
                *out++ = *in++ & 0x01;
                nlen += 21;
                break;
            case 3:
                *out++ = (*in >> 27) & 0x01;
                *out++ = (*in >> 26) & 0x01;
                *out++ = (*in >> 25) & 0x01;
                *out++ = (*in >> 24) & 0x01;
                *out++ = (*in >> 23) & 0x01;
                *out++ = (*in >> 22) & 0x01;
                *out++ = (*in >> 21) & 0x01;
                *out++ = (*in >> 20) & 0x01;
                *out++ = (*in >> 19) & 0x01;
                *out++ = (*in >> 18) & 0x01;
                *out++ = (*in >> 17) & 0x01;
                *out++ = (*in >> 16) & 0x01;
                *out++ = (*in >> 15) & 0x01;
                *out++ = (*in >> 14) & 0x01;
                *out++ = (*in >> 12) & 0x03;
                *out++ = (*in >> 10) & 0x03;
                *out++ = (*in >> 8) & 0x03;
                *out++ = (*in >> 6) & 0x03;
                *out++ = (*in >> 4) & 0x03;
                *out++ = (*in >> 2) & 0x03;
                *out++ = *in++ & 0x03;
                nlen += 21;
                break;
            case 4:
                *out++ = (*in >> 26) & 0x03;
                *out++ = (*in >> 24) & 0x03;
                *out++ = (*in >> 22) & 0x03;
                *out++ = (*in >> 20) & 0x03;
                *out++ = (*in >> 18) & 0x03;
                *out++ = (*in >> 16) & 0x03;
                *out++ = (*in >> 14) & 0x03;
                *out++ = (*in >> 12) & 0x03;
                *out++ = (*in >> 10) & 0x03;
                *out++ = (*in >> 8) & 0x03;
                *out++ = (*in >> 6) & 0x03;
                *out++ = (*in >> 4) & 0x03;
                *out++ = (*in >> 2) & 0x03;
                *out++ = *in++ & 0x03;
                nlen += 14;
                break;
            case 5:
                *out++ = (*in >> 24) & 0x0f;
                *out++ = (*in >> 21) & 0x07;
                *out++ = (*in >> 18) & 0x07;
                *out++ = (*in >> 15) & 0x07;
                *out++ = (*in >> 12) & 0x07;
                *out++ = (*in >> 9) & 0x07;
                *out++ = (*in >> 6) & 0x07;
                *out++ = (*in >> 3) & 0x07;
                *out++ = *in++ & 0x07;
                nlen += 9;
                break;
            case 6:
                *out++ = (*in >> 25) & 0x07;
                *out++ = (*in >> 21) & 0x0f;
                *out++ = (*in >> 17) & 0x0f;
                *out++ = (*in >> 13) & 0x0f;
                *out++ = (*in >> 9) & 0x0f;
                *out++ = (*in >> 6) & 0x07;
                *out++ = (*in >> 3) & 0x07;
                *out++ = *in++ & 0x07;
                nlen += 8;
                break;
            case 7:
                *out++ = (*in >> 24) & 0x0f;
                *out++ = (*in >> 20) & 0x0f;
                *out++ = (*in >> 16) & 0x0f;
                *out++ = (*in >> 12) & 0x0f;
                *out++ = (*in >> 8) & 0x0f;
                *out++ = (*in >> 4) & 0x0f;
                *out++ = *in++ & 0x0f;
                nlen += 7;
                break;
            case 8:
                *out++ = (*in >> 23) & 0x1f;
                *out++ = (*in >> 18) & 0x1f;
                *out++ = (*in >> 13) & 0x1f;
                *out++ = (*in >> 8) & 0x1f;
                *out++ = (*in >> 4) & 0x0f;
                *out++ = *in++ & 0x0f;
                nlen += 6;
                break;
            case 9:
                *out++ = (*in >> 24) & 0x0f;
                *out++ = (*in >> 20) & 0x0f;
                *out++ = (*in >> 15) & 0x1f;
                *out++ = (*in >> 10) & 0x1f;
                *out++ = (*in >> 5) & 0x1f;
                *out++ = *in++ & 0x1f;
                nlen += 6;
                break;
            case 10:
                *out++ = (*in >> 22) & 0x3f;
                *out++ = (*in >> 16) & 0x3f;
                *out++ = (*in >> 10) & 0x3f;
                *out++ = (*in >> 5) & 0x1f;
                *out++ = *in++ & 0x1f;
                nlen += 5;
                break;
            case 11:
                *out++ = (*in >> 23) & 0x1f;
                *out++ = (*in >> 18) & 0x1f;
                *out++ = (*in >> 12) & 0x3f;
                *out++ = (*in >> 6) & 0x3f;
                *out++ = *in++ & 0x3f;
                nlen += 5;
                break;
            case 12:
                *out++ = (*in >> 21) & 0x7f;
                *out++ = (*in >> 14) & 0x7f;
                *out++ = (*in >> 7) & 0x7f;
                *out++ = *in++ & 0x7f;
                nlen += 4;
                break;
            case 13:
                *out++ = (*in >> 18) & 0x03ff;
                *out++ = (*in >> 9) & 0x01ff;
                *out++ = *in++ & 0x01ff;
                nlen += 3;
                break;
            case 14:
                *out++ = (*in >> 14) & 0x3fff;
                *out++ = *in++ & 0x3fff;
                nlen += 2;
                break;
            case 15:
                *out++ = *in++ & 0x0fffffff;
                nlen += 1;
                break;
        }
    }
}

void
__p4delta_unpack0(uint32_t *out, uint32_t *in) {
    uint32_t i;
    for (i = 0; i < PFORDELTA_BLOCKSZ;
            i += 32, out += 32) {
        __p4delta_zero32(out);
    }
}

void
__p4delta_unpack1(uint32_t *out, uint32_t *in) {
    uint32_t i;
    for (i = 0; i < PFORDELTA_BLOCKSZ;
            i += 32, out += 32, in += 1) {
        out[0] = in[0] >> 31;
        out[1] = (in[0] >> 30) & 0x01;
        out[2] = (in[0] >> 29) & 0x01;
        out[3] = (in[0] >> 28) & 0x01;
        out[4] = (in[0] >> 27) & 0x01;
        out[5] = (in[0] >> 26) & 0x01;
        out[6] = (in[0] >> 25) & 0x01;
        out[7] = (in[0] >> 24) & 0x01;
        out[8] = (in[0] >> 23) & 0x01;
        out[9] = (in[0] >> 22) & 0x01;
        out[10] = (in[0] >> 21) & 0x01;
        out[11] = (in[0] >> 20) & 0x01;
        out[12] = (in[0] >> 19) & 0x01;
        out[13] = (in[0] >> 18) & 0x01;
        out[14] = (in[0] >> 17) & 0x01;
        out[15] = (in[0] >> 16) & 0x01;
        out[16] = (in[0] >> 15) & 0x01;
        out[17] = (in[0] >> 14) & 0x01;
        out[18] = (in[0] >> 13) & 0x01;
        out[19] = (in[0] >> 12) & 0x01;
        out[20] = (in[0] >> 11) & 0x01;
        out[21] = (in[0] >> 10) & 0x01;
        out[22] = (in[0] >> 9) & 0x01;
        out[23] = (in[0] >> 8) & 0x01;
        out[24] = (in[0] >> 7) & 0x01;
        out[25] = (in[0] >> 6) & 0x01;
        out[26] = (in[0] >> 5) & 0x01;
        out[27] = (in[0] >> 4) & 0x01;
        out[28] = (in[0] >> 3) & 0x01;
        out[29] = (in[0] >> 2) & 0x01;
        out[30] = (in[0] >> 1) & 0x01;
        out[31] = in[0] & 0x01;
    }
}

void
__p4delta_unpack2(uint32_t *out, uint32_t *in) {
    uint32_t i;
    for (i = 0; i < PFORDELTA_BLOCKSZ;
            i += 32, out += 32, in += 2) {
        out[0] = in[0] >> 30;
        out[1] = (in[0] >> 28) & 0x03;
        out[2] = (in[0] >> 26) & 0x03;
        out[3] = (in[0] >> 24) & 0x03;
        out[4] = (in[0] >> 22) & 0x03;
        out[5] = (in[0] >> 20) & 0x03;
        out[6] = (in[0] >> 18) & 0x03;
        out[7] = (in[0] >> 16) & 0x03;
        out[8] = (in[0] >> 14) & 0x03;
        out[9] = (in[0] >> 12) & 0x03;
        out[10] = (in[0] >> 10) & 0x03;
        out[11] = (in[0] >> 8) & 0x03;
        out[12] = (in[0] >> 6) & 0x03;
        out[13] = (in[0] >> 4) & 0x03;
        out[14] = (in[0] >> 2) & 0x03;
        out[15] = in[0] & 0x03;
        out[16] = in[1] >> 30;
        out[17] = (in[1] >> 28) & 0x03;
        out[18] = (in[1] >> 26) & 0x03;
        out[19] = (in[1] >> 24) & 0x03;
        out[20] = (in[1] >> 22) & 0x03;
        out[21] = (in[1] >> 20) & 0x03;
        out[22] = (in[1] >> 18) & 0x03;
        out[23] = (in[1] >> 16) & 0x03;
        out[24] = (in[1] >> 14) & 0x03;
        out[25] = (in[1] >> 12) & 0x03;
        out[26] = (in[1] >> 10) & 0x03;
        out[27] = (in[1] >> 8) & 0x03;
        out[28] = (in[1] >> 6) & 0x03;
        out[29] = (in[1] >> 4) & 0x03;
        out[30] = (in[1] >> 2) & 0x03;
        out[31] = in[1] & 0x03;
    }
}

void
__p4delta_unpack3(uint32_t *out, uint32_t *in) {
    uint32_t i;
    for (i = 0; i < PFORDELTA_BLOCKSZ;
            i += 32, out += 32, in += 3) {
        out[0] = in[0] >> 29;
        out[1] = (in[0] >> 26) & 0x07;
        out[2] = (in[0] >> 23) & 0x07;
        out[3] = (in[0] >> 20) & 0x07;
        out[4] = (in[0] >> 17) & 0x07;
        out[5] = (in[0] >> 14) & 0x07;
        out[6] = (in[0] >> 11) & 0x07;
        out[7] = (in[0] >> 8) & 0x07;
        out[8] = (in[0] >> 5) & 0x07;
        out[9] = (in[0] >> 2) & 0x07;
        out[10] = (in[0] << 1) & 0x07;
        out[10] |= in[1] >> 31;
        out[11] = (in[1] >> 28) & 0x07;
        out[12] = (in[1] >> 25) & 0x07;
        out[13] = (in[1] >> 22) & 0x07;
        out[14] = (in[1] >> 19) & 0x07;
        out[15] = (in[1] >> 16) & 0x07;
        out[16] = (in[1] >> 13) & 0x07;
        out[17] = (in[1] >> 10) & 0x07;
        out[18] = (in[1] >> 7) & 0x07;
        out[19] = (in[1] >> 4) & 0x07;
        out[20] = (in[1] >> 1) & 0x07;
        out[21] = (in[1] << 2) & 0x07;
        out[21] |= in[2] >> 30;
        out[22] = (in[2] >> 27) & 0x07;
        out[23] = (in[2] >> 24) & 0x07;
        out[24] = (in[2] >> 21) & 0x07;
        out[25] = (in[2] >> 18) & 0x07;
        out[26] = (in[2] >> 15) & 0x07;
        out[27] = (in[2] >> 12) & 0x07;
        out[28] = (in[2] >> 9) & 0x07;
        out[29] = (in[2] >> 6) & 0x07;
        out[30] = (in[2] >> 3) & 0x07;
        out[31] = in[2] & 0x07;
    }
}

void
__p4delta_unpack4(uint32_t *out, uint32_t *in) {
    uint32_t i;
    for (i = 0; i < PFORDELTA_BLOCKSZ;
            i += 32, out += 32, in += 4) {
        out[0] = in[0] >> 28;
        out[1] = (in[0] >> 24) & 0x0f;
        out[2] = (in[0] >> 20) & 0x0f;
        out[3] = (in[0] >> 16) & 0x0f;
        out[4] = (in[0] >> 12) & 0x0f;
        out[5] = (in[0] >> 8) & 0x0f;
        out[6] = (in[0] >> 4) & 0x0f;
        out[7] = in[0] & 0x0f;
        out[8] = in[1] >> 28;
        out[9] = (in[1] >> 24) & 0x0f;
        out[10] = (in[1] >> 20) & 0x0f;
        out[11] = (in[1] >> 16) & 0x0f;
        out[12] = (in[1] >> 12) & 0x0f;
        out[13] = (in[1] >> 8) & 0x0f;
        out[14] = (in[1] >> 4) & 0x0f;
        out[15] = in[1] & 0x0f;
        out[16] = in[2] >> 28;
        out[17] = (in[2] >> 24) & 0x0f;
        out[18] = (in[2] >> 20) & 0x0f;
        out[19] = (in[2] >> 16) & 0x0f;
        out[20] = (in[2] >> 12) & 0x0f;
        out[21] = (in[2] >> 8) & 0x0f;
        out[22] = (in[2] >> 4) & 0x0f;
        out[23] = in[2] & 0x0f;
        out[24] = in[3] >> 28;
        out[25] = (in[3] >> 24) & 0x0f;
        out[26] = (in[3] >> 20) & 0x0f;
        out[27] = (in[3] >> 16) & 0x0f;
        out[28] = (in[3] >> 12) & 0x0f;
        out[29] = (in[3] >> 8) & 0x0f;
        out[30] = (in[3] >> 4) & 0x0f;
        out[31] = in[3] & 0x0f;
    }
}

void
__p4delta_unpack5(uint32_t *out, uint32_t *in) {
    uint32_t i;
    for (i = 0; i < PFORDELTA_BLOCKSZ;
            i += 32, out += 32, in += 5) {
        out[0] = in[0] >> 27;
        out[1] = (in[0] >> 22) & 0x1f;
        out[2] = (in[0] >> 17) & 0x1f;
        out[3] = (in[0] >> 12) & 0x1f;
        out[4] = (in[0] >> 7) & 0x1f;
        out[5] = (in[0] >> 2) & 0x1f;
        out[6] = (in[0] << 3) & 0x1f;
        out[6] |= in[1] >> 29;
        out[7] = (in[1] >> 24) & 0x1f;
        out[8] = (in[1] >> 19) & 0x1f;
        out[9] = (in[1] >> 14) & 0x1f;
        out[10] = (in[1] >> 9) & 0x1f;
        out[11] = (in[1] >> 4) & 0x1f;
        out[12] = (in[1] << 1) & 0x1f;
        out[12] |= in[2] >> 0x1f;
        out[13] = (in[2] >> 26) & 0x1f;
        out[14] = (in[2] >> 21) & 0x1f;
        out[15] = (in[2] >> 16) & 0x1f;
        out[16] = (in[2] >> 11) & 0x1f;
        out[17] = (in[2] >> 6) & 0x1f;
        out[18] = (in[2] >> 1) & 0x1f;
        out[19] = (in[2] << 4) & 0x1f;
        out[19] |= in[3] >> 28;
        out[20] = (in[3] >> 23) & 0x1f;
        out[21] = (in[3] >> 18) & 0x1f;
        out[22] = (in[3] >> 13) & 0x1f;
        out[23] = (in[3] >> 8) & 0x1f;
        out[24] = (in[3] >> 3) & 0x1f;
        out[25] = (in[3] << 2) & 0x1f;
        out[25] |= in[4] >> 30;
        out[26] = (in[4] >> 25) & 0x1f;
        out[27] = (in[4] >> 20) & 0x1f;
        out[28] = (in[4] >> 15) & 0x1f;
        out[29] = (in[4] >> 10) & 0x1f;
        out[30] = (in[4] >> 5) & 0x1f;
        out[31] = in[4] & 0x1f;
    }
}

void
__p4delta_unpack6(uint32_t *out, uint32_t *in) {
    uint32_t i;
    for (i = 0; i < PFORDELTA_BLOCKSZ;
            i += 32, out += 32, in += 6) {
        out[0] = in[0] >> 26;
        out[1] = (in[0] >> 20) & 0x3f;
        out[2] = (in[0] >> 14) & 0x3f;
        out[3] = (in[0] >> 8) & 0x3f;
        out[4] = (in[0] >> 2) & 0x3f;
        out[5] = (in[0] << 4) & 0x3f;
        out[5] |= in[1] >> 28;
        out[6] = (in[1] >> 22) & 0x3f;
        out[7] = (in[1] >> 16) & 0x3f;
        out[8] = (in[1] >> 10) & 0x3f;
        out[9] = (in[1] >> 4) & 0x3f;
        out[10] = (in[1] << 2) & 0x3f;
        out[10] |= in[2] >> 30;
        out[11] = (in[2] >> 24) & 0x3f;
        out[12] = (in[2] >> 18) & 0x3f;
        out[13] = (in[2] >> 12) & 0x3f;
        out[14] = (in[2] >> 6) & 0x3f;
        out[15] = in[2] & 0x3f;
        out[16] = in[3] >> 26;
        out[17] = (in[3] >> 20) & 0x3f;
        out[18] = (in[3] >> 14) & 0x3f;
        out[19] = (in[3] >> 8) & 0x3f;
        out[20] = (in[3] >> 2) & 0x3f;
        out[21] = (in[3] << 4) & 0x3f;
        out[21] |= in[4] >> 28;
        out[22] = (in[4] >> 22) & 0x3f;
        out[23] = (in[4] >> 16) & 0x3f;
        out[24] = (in[4] >> 10) & 0x3f;
        out[25] = (in[4] >> 4) & 0x3f;
        out[26] = (in[4] << 2) & 0x3f;
        out[26] |= in[5] >> 30;
        out[27] = (in[5] >> 24) & 0x3f;
        out[28] = (in[5] >> 18) & 0x3f;
        out[29] = (in[5] >> 12) & 0x3f;
        out[30] = (in[5] >> 6) & 0x3f;
        out[31] = in[5] & 0x3f;
    }
}

void
__p4delta_unpack7(uint32_t *out, uint32_t *in) {
    uint32_t i;
    for (i = 0; i < PFORDELTA_BLOCKSZ;
            i += 32, out += 32, in += 7) {
        out[0] = in[0] >> 25;
        out[1] = (in[0] >> 18) & 0x7f;
        out[2] = (in[0] >> 11) & 0x7f;
        out[3] = (in[0] >> 4) & 0x7f;
        out[4] = (in[0] << 3) & 0x7f;
        out[4] |= in[1] >> 29;
        out[5] = (in[1] >> 22) & 0x7f;
        out[6] = (in[1] >> 15) & 0x7f;
        out[7] = (in[1] >> 8) & 0x7f;
        out[8] = (in[1] >> 1) & 0x7f;
        out[9] = (in[1] << 6) & 0x7f;
        out[9] |= in[2] >> 26;
        out[10] = (in[2] >> 19) & 0x7f;
        out[11] = (in[2] >> 12) & 0x7f;
        out[12] = (in[2] >> 5) & 0x7f;
        out[13] = (in[2] << 2) & 0x7f;
        out[13] |= in[3] >> 30;
        out[14] = (in[3] >> 23) & 0x7f;
        out[15] = (in[3] >> 16) & 0x7f;
        out[16] = (in[3] >> 9) & 0x7f;
        out[17] = (in[3] >> 2) & 0x7f;
        out[18] = (in[3] << 5) & 0x7f;
        out[18] |= in[4] >> 27;
        out[19] = (in[4] >> 20) & 0x7f;
        out[20] = (in[4] >> 13) & 0x7f;
        out[21] = (in[4] >> 6) & 0x7f;
        out[22] = (in[4] << 1) & 0x7f;
        out[22] |= in[5] >> 31;
        out[23] = (in[5] >> 24) & 0x7f;
        out[24] = (in[5] >> 17) & 0x7f;
        out[25] = (in[5] >> 10) & 0x7f;
        out[26] = (in[5] >> 3) & 0x7f;
        out[27] = (in[5] << 4) & 0x7f;
        out[27] |= in[6] >> 28;
        out[28] = (in[6] >> 21) & 0x7f;
        out[29] = (in[6] >> 14) & 0x7f;
        out[30] = (in[6] >> 7) & 0x7f;
        out[31] = in[6] & 0x7f;
    }
}

void
__p4delta_unpack8(uint32_t *out, uint32_t *in) {
    uint32_t i;
    for (i = 0; i < PFORDELTA_BLOCKSZ;
            i += 32, out += 32, in += 8) {
        out[0] = in[0] >> 24;
        out[1] = (in[0] >> 16) & 0xff;
        out[2] = (in[0] >> 8) & 0xff;
        out[3] = in[0] & 0xff;
        out[4] = in[1] >> 24;
        out[5] = (in[1] >> 16) & 0xff;
        out[6] = (in[1] >> 8) & 0xff;
        out[7] = in[1] & 0xff;
        out[8] = in[2] >> 24;
        out[9] = (in[2] >> 16) & 0xff;
        out[10] = (in[2] >> 8) & 0xff;
        out[11] = in[2] & 0xff;
        out[12] = in[3] >> 24;
        out[13] = (in[3] >> 16) & 0xff;
        out[14] = (in[3] >> 8) & 0xff;
        out[15] = in[3] & 0xff;
        out[16] = in[4] >> 24;
        out[17] = (in[4] >> 16) & 0xff;
        out[18] = (in[4] >> 8) & 0xff;
        out[19] = in[4] & 0xff;
        out[20] = in[5] >> 24;
        out[21] = (in[5] >> 16) & 0xff;
        out[22] = (in[5] >> 8) & 0xff;
        out[23] = in[5] & 0xff;
        out[24] = in[6] >> 24;
        out[25] = (in[6] >> 16) & 0xff;
        out[26] = (in[6] >> 8) & 0xff;
        out[27] = in[6] & 0xff;
        out[28] = in[7] >> 24;
        out[29] = (in[7] >> 16) & 0xff;
        out[30] = (in[7] >> 8) & 0xff;
        out[31] = in[7] & 0xff;
    }
}

void
__p4delta_unpack9(uint32_t *out, uint32_t *in) {
    uint32_t i;
    for (i = 0; i < PFORDELTA_BLOCKSZ;
            i += 32, out += 32, in += 9) {
        out[0] = in[0] >> 23;
        out[1] = (in[0] >> 14) & 0x01ff;
        out[2] = (in[0] >> 5) & 0x01ff;
        out[3] = (in[0] << 4) & 0x01ff;
        out[3] |= in[1] >> 28;
        out[4] = (in[1] >> 19) & 0x01ff;
        out[5] = (in[1] >> 10) & 0x01ff;
        out[6] = (in[1] >> 1) & 0x01ff;
        out[7] = (in[1] << 8) & 0x01ff;
        out[7] |= in[2] >> 24;
        out[8] = (in[2] >> 15) & 0x01ff;
        out[9] = (in[2] >> 6) & 0x01ff;
        out[10] = (in[2] << 3) & 0x01ff;
        out[10] |= in[3] >> 29;
        out[11] = (in[3] >> 20) & 0x01ff;
        out[12] = (in[3] >> 11) & 0x01ff;
        out[13] = (in[3] >> 2) & 0x01ff;
        out[14] = (in[3] << 7) & 0x01ff;
        out[14] |= in[4] >> 25;
        out[15] = (in[4] >> 16) & 0x01ff;
        out[16] = (in[4] >> 7) & 0x01ff;
        out[17] = (in[4] << 2) & 0x01ff;
        out[17] |= in[5] >> 30;
        out[18] = (in[5] >> 21) & 0x01ff;
        out[19] = (in[5] >> 12) & 0x01ff;
        out[20] = (in[5] >> 3) & 0x01ff;
        out[21] = (in[5] << 6) & 0x01ff;
        out[21] |= in[6] >> 26;
        out[22] = (in[6] >> 17) & 0x01ff;
        out[23] = (in[6] >> 8) & 0x01ff;
        out[24] = (in[6] << 1) & 0x01ff;
        out[24] |= in[7] >> 31;
        out[25] = (in[7] >> 22) & 0x01ff;
        out[26] = (in[7] >> 13) & 0x01ff;
        out[27] = (in[7] >> 4) & 0x01ff;
        out[28] = (in[7] << 5) & 0x01ff;
        out[28] |= in[8] >> 27;
        out[29] = (in[8] >> 18) & 0x01ff;
        out[30] = (in[8] >> 9) & 0x01ff;
        out[31] = in[8] & 0x01ff;
    }
}

void
__p4delta_unpack10(uint32_t *out, uint32_t *in) {
    uint32_t i;
    for (i = 0; i < PFORDELTA_BLOCKSZ;
            i += 32, out += 32, in += 10) {
        out[0] = in[0] >> 22;
        out[1] = (in[0] >> 12) & 0x03ff;
        out[2] = (in[0] >> 2) & 0x03ff;
        out[3] = (in[0] << 8) & 0x03ff;
        out[3] |= in[1] >> 24;
        out[4] = (in[1] >> 14) & 0x03ff;
        out[5] = (in[1] >> 4) & 0x03ff;
        out[6] = (in[1] << 6) & 0x03ff;
        out[6] |= in[2] >> 26;
        out[7] = (in[2] >> 16) & 0x03ff;
        out[8] = (in[2] >> 6) & 0x03ff;
        out[9] = (in[2] << 4) & 0x03ff;
        out[9] |= in[3] >> 28;
        out[10] = (in[3] >> 18) & 0x03ff;
        out[11] = (in[3] >> 8) & 0x03ff;
        out[12] = (in[3] << 2) & 0x03ff;
        out[12] |= in[4] >> 30;
        out[13] = (in[4] >> 20) & 0x03ff;
        out[14] = (in[4] >> 10) & 0x03ff;
        out[15] = in[4] & 0x03ff;
        out[16] = in[5] >> 22;
        out[17] = (in[5] >> 12) & 0x03ff;
        out[18] = (in[5] >> 2) & 0x03ff;
        out[19] = (in[5] << 8) & 0x03ff;
        out[19] |= in[6] >> 24;
        out[20] = (in[6] >> 14) & 0x03ff;
        out[21] = (in[6] >> 4) & 0x03ff;
        out[22] = (in[6] << 6) & 0x03ff;
        out[22] |= in[7] >> 26;
        out[23] = (in[7] >> 16) & 0x03ff;
        out[24] = (in[7] >> 6) & 0x03ff;
        out[25] = (in[7] << 4) & 0x03ff;
        out[25] |= in[8] >> 28;
        out[26] = (in[8] >> 18) & 0x03ff;
        out[27] = (in[8] >> 8) & 0x03ff;
        out[28] = (in[8] << 2) & 0x03ff;
        out[28] |= in[9] >> 30;
        out[29] = (in[9] >> 20) & 0x03ff;
        out[30] = (in[9] >> 10) & 0x03ff;
        out[31] = in[9] & 0x03ff;
    }
}

void
__p4delta_unpack11(uint32_t *out, uint32_t *in) {
    uint32_t i;
    for (i = 0; i < PFORDELTA_BLOCKSZ;
            i += 32, out += 32, in += 11) {
        out[0] = in[0] >> 21;
        out[1] = (in[0] >> 10) & 0x07ff;
        out[2] = (in[0] << 1) & 0x07ff;
        out[2] |= in[1] >> 31;
        out[3] = (in[1] >> 20) & 0x07ff;
        out[4] = (in[1] >> 9) & 0x07ff;
        out[5] = (in[1] << 2) & 0x07ff;
        out[5] |= in[2] >> 30;
        out[6] = (in[2] >> 19) & 0x07ff;
        out[7] = (in[2] >> 8) & 0x07ff;
        out[8] = (in[2] << 3) & 0x07ff;
        out[8] |= in[3] >> 29;
        out[9] = (in[3] >> 18) & 0x07ff;
        out[10] = (in[3] >> 7) & 0x07ff;
        out[11] = (in[3] << 4) & 0x07ff;
        out[11] |= in[4] >> 28;
        out[12] = (in[4] >> 17) & 0x07ff;
        out[13] = (in[4] >> 6) & 0x07ff;
        out[14] = (in[4] << 5) & 0x07ff;
        out[14] |= in[5] >> 27;
        out[15] = (in[5] >> 16) & 0x07ff;
        out[16] = (in[5] >> 5) & 0x07ff;
        out[17] = (in[5] << 6) & 0x07ff;
        out[17] |= in[6] >> 26;
        out[18] = (in[6] >> 15) & 0x07ff;
        out[19] = (in[6] >> 4) & 0x07ff;
        out[20] = (in[6] << 7) & 0x07ff;
        out[20] |= in[7] >> 25;
        out[21] = (in[7] >> 14) & 0x07ff;
        out[22] = (in[7] >> 3) & 0x07ff;
        out[23] = (in[7] << 8) & 0x07ff;
        out[23] |= in[8] >> 24;
        out[24] = (in[8] >> 13) & 0x07ff;
        out[25] = (in[8] >> 2) & 0x07ff;
        out[26] = (in[8] << 9) & 0x07ff;
        out[26] |= in[9] >> 23;
        out[27] = (in[9] >> 12) & 0x07ff;
        out[28] = (in[9] >> 1) & 0x07ff;
        out[29] = (in[9] << 10) & 0x07ff;
        out[29] |= in[10] >> 22;
        out[30] = (in[10] >> 11) & 0x07ff;
        out[31] = in[10] & 0x07ff;
    }
}

void
__p4delta_unpack12(uint32_t *out, uint32_t *in) {
    uint32_t i;
    for (i = 0; i < PFORDELTA_BLOCKSZ;
            i += 32, out += 32, in += 12) {
        out[0] = in[0] >> 20;
        out[1] = (in[0] >> 8) & 0x0fff;
        out[2] = (in[0] << 4) & 0x0fff;
        out[2] |= in[1] >> 28;
        out[3] = (in[1] >> 16) & 0x0fff;
        out[4] = (in[1] >> 4) & 0x0fff;
        out[5] = (in[1] << 8) & 0x0fff;
        out[5] |= in[2] >> 24;
        out[6] = (in[2] >> 12) & 0x0fff;
        out[7] = in[2] & 0x0fff;
        out[8] = in[3] >> 20;
        out[9] = (in[3] >> 8) & 0x0fff;
        out[10] = (in[3] << 4) & 0x0fff;
        out[10] |= in[4] >> 28;
        out[11] = (in[4] >> 16) & 0x0fff;
        out[12] = (in[4] >> 4) & 0x0fff;
        out[13] = (in[4] << 8) & 0x0fff;
        out[13] |= in[5] >> 24;
        out[14] = (in[5] >> 12) & 0x0fff;
        out[15] = in[5] & 0x0fff;
        out[16] = in[6] >> 20;
        out[17] = (in[6] >> 8) & 0x0fff;
        out[18] = (in[6] << 4) & 0x0fff;
        out[18] |= in[7] >> 28;
        out[19] = (in[7] >> 16) & 0x0fff;
        out[20] = (in[7] >> 4) & 0x0fff;
        out[21] = (in[7] << 8) & 0x0fff;
        out[21] |= in[8] >> 24;
        out[22] = (in[8] >> 12) & 0x0fff;
        out[23] = in[8] & 0x0fff;
        out[24] = in[9] >> 20;
        out[25] = (in[9] >> 8) & 0x0fff;
        out[26] = (in[9] << 4) & 0x0fff;
        out[26] |= in[10] >> 28;
        out[27] = (in[10] >> 16) & 0x0fff;
        out[28] = (in[10] >> 4) & 0x0fff;
        out[29] = (in[10] << 8) & 0x0fff;
        out[29] |= in[11] >> 24;
        out[30] = (in[11] >> 12) & 0x0fff;
        out[31] = in[11] & 0x0fff;
    }
}

void
__p4delta_unpack13(uint32_t *out, uint32_t *in) {
    uint32_t i;
    for (i = 0; i < PFORDELTA_BLOCKSZ;
            i += 32, out += 32, in += 13) {
        out[0] = in[0] >> 19;
        out[1] = (in[0] >> 6) & 0x1fff;
        out[2] = (in[0] << 7) & 0x1fff;
        out[2] |= in[1] >> 25;
        out[3] = (in[1] >> 12) & 0x1fff;
        out[4] = (in[1] << 1) & 0x1fff;
        out[4] |= in[2] >> 31;
        out[5] = (in[2] >> 18) & 0x1fff;
        out[6] = (in[2] >> 5) & 0x1fff;
        out[7] = (in[2] << 8) & 0x1fff;
        out[7] |= in[3] >> 24;
        out[8] = (in[3] >> 11) & 0x1fff;
        out[9] = (in[3] << 2) & 0x1fff;
        out[9] |= in[4] >> 30;
        out[10] = (in[4] >> 17) & 0x1fff;
        out[11] = (in[4] >> 4) & 0x1fff;
        out[12] = (in[4] << 9) & 0x1fff;
        out[12] |= in[5] >> 23;
        out[13] = (in[5] >> 10) & 0x1fff;
        out[14] = (in[5] << 3) & 0x1fff;
        out[14] |= in[6] >> 29;
        out[15] = (in[6] >> 16) & 0x1fff;
        out[16] = (in[6] >> 3) & 0x1fff;
        out[17] = (in[6] << 10) & 0x1fff;
        out[17] |= in[7] >> 22;
        out[18] = (in[7] >> 9) & 0x1fff;
        out[19] = (in[7] << 4) & 0x1fff;
        out[19] |= in[8] >> 28;
        out[20] = (in[8] >> 15) & 0x1fff;
        out[21] = (in[8] >> 2) & 0x1fff;
        out[22] = (in[8] << 11) & 0x1fff;
        out[22] |= in[9] >> 21;
        out[23] = (in[9] >> 8) & 0x1fff;
        out[24] = (in[9] << 5) & 0x1fff;
        out[24] |= in[10] >> 27;
        out[25] = (in[10] >> 14) & 0x1fff;
        out[26] = (in[10] >> 1) & 0x1fff;
        out[27] = (in[10] << 12) & 0x1fff;
        out[27] |= in[11] >> 20;
        out[28] = (in[11] >> 7) & 0x1fff;
        out[29] = (in[11] << 6) & 0x1fff;
        out[29] |= in[12] >> 26;
        out[30] = (in[12] >> 13) & 0x1fff;
        out[31] = in[12] & 0x1fff;
    }
}

void
__p4delta_unpack16(uint32_t *out, uint32_t *in) {
    uint32_t i;
    for (i = 0; i < PFORDELTA_BLOCKSZ;
            i += 32, out += 32, in += 16) {
        out[0] = in[0] >> 16;
        out[1] = in[0] & 0xffff;
        out[2] = in[1] >> 16;
        out[3] = in[1] & 0xffff;
        out[4] = in[2] >> 16;
        out[5] = in[2] & 0xffff;
        out[6] = in[3] >> 16;
        out[7] = in[3] & 0xffff;
        out[8] = in[4] >> 16;
        out[9] = in[4] & 0xffff;
        out[10] = in[5] >> 16;
        out[11] = in[5] & 0xffff;
        out[12] = in[6] >> 16;
        out[13] = in[6] & 0xffff;
        out[14] = in[7] >> 16;
        out[15] = in[7] & 0xffff;
        out[16] = in[8] >> 16;
        out[17] = in[8] & 0xffff;
        out[18] = in[9] >> 16;
        out[19] = in[9] & 0xffff;
        out[20] = in[10] >> 16;
        out[21] = in[10] & 0xffff;
        out[22] = in[11] >> 16;
        out[23] = in[11] & 0xffff;
        out[24] = in[12] >> 16;
        out[25] = in[12] & 0xffff;
        out[26] = in[13] >> 16;
        out[27] = in[13] & 0xffff;
        out[28] = in[14] >> 16;
        out[29] = in[14] & 0xffff;
        out[30] = in[15] >> 16;
        out[31] = in[15] & 0xffff;
    }
}

void
__p4delta_unpack20(uint32_t *out, uint32_t *in) {
    uint32_t i;
    for (i = 0; i < PFORDELTA_BLOCKSZ;
            i += 32, out += 32, in += 20) {
        out[0] = in[0] >> 12;
        out[1] = (in[0] << 8) & 0x0fffff;
        out[1] |= in[1] >> 24;
        out[2] = (in[1] >> 4) & 0x0fffff;
        out[3] = (in[1] << 16) & 0x0fffff;
        out[3] |= in[2] >> 16;
        out[4] = (in[2] << 4) & 0x0fffff;
        out[4] |= in[3] >> 28;
        out[5] = (in[3] >> 8) & 0x0fffff;
        out[6] = (in[3] << 12) & 0x0fffff;
        out[6] |= in[4] >> 20;
        out[7] = in[4] & 0x0fffff;
        out[8] = in[5] >> 12;
        out[9] = (in[5] << 8) & 0x0fffff;
        out[9] |= in[6] >> 24;
        out[10] = (in[6] >> 4) & 0x0fffff;
        out[11] = (in[6] << 16) & 0x0fffff;
        out[11] |= in[7] >> 16;
        out[12] = (in[7] << 4) & 0x0fffff;
        out[12] |= in[8] >> 28;
        out[13] = (in[8] >> 8) & 0x0fffff;
        out[14] = (in[8] << 12) & 0x0fffff;
        out[14] |= in[9] >> 20;
        out[15] = in[9] & 0x0fffff;
        out[16] = in[10] >> 12;
        out[17] = (in[10] << 8) & 0x0fffff;
        out[17] |= in[11] >> 24;
        out[18] = (in[11] >> 4) & 0x0fffff;
        out[19] = (in[11] << 16) & 0x0fffff;
        out[19] |= in[12] >> 16;
        out[20] = (in[12] << 4) & 0x0fffff;
        out[20] |= in[13] >> 28;
        out[21] = (in[13] >> 8) & 0x0fffff;
        out[22] = (in[13] << 12) & 0x0fffff;
        out[22] |= in[14] >> 20;
        out[23] = in[14] & 0x0fffff;
        out[24] = in[15] >> 12;
        out[25] = (in[15] << 8) & 0x0fffff;
        out[25] |= in[16] >> 24;
        out[26] = (in[16] >> 4) & 0x0fffff;
        out[27] = (in[16] << 16) & 0x0fffff;
        out[27] |= in[17] >> 16;
        out[28] = (in[17] << 4) & 0x0fffff;
        out[28] |= in[18] >> 28;
        out[29] = (in[18] >> 8) & 0x0fffff;
        out[30] = (in[18] << 12) & 0x0fffff;
        out[30] |= in[19] >> 20;
        out[31] = in[19] & 0x0fffff;
    }
}

void
__p4delta_unpack32(uint32_t *out, uint32_t *in) {
    uint32_t i;
    for (i = 0; i < PFORDELTA_BLOCKSZ; i += 16, out += 16, in += 16) {
        __p4delta_copy(in, out);
    }
}

#define   likely(x)      __builtin_expect(!!(x),   1)
#define   unlikely(x)    __builtin_expect(!!(x),   0)

void PFDEncode(uint32_t *in, uint32_t len, uint32_t *out, uint32_t &nvalue) {
    uint32_t i;
    uint32_t numBlocks;
    uint32_t csize;
    numBlocks = div_roundup(len, PFORDELTA_BLOCKSZ);
    /* Output the number of blocks */
    *out++ = numBlocks;
    nvalue = 1;
    for (i = 0; i < numBlocks; i++) {
        if (likely(i != numBlocks - 1)) {
            encodeBlock(in, PFORDELTA_BLOCKSZ, out, csize, findBestB);
            in += PFORDELTA_BLOCKSZ;
            out += csize;
        } else {
            /*
             *       This is a code to pack gabage in the tail of lists.
             *       I think it couldn't be a bottleneck.
             *       */
            uint32_t nblk;
            nblk = ((len % PFORDELTA_BLOCKSZ) != 0) ?
                    len % PFORDELTA_BLOCKSZ : PFORDELTA_BLOCKSZ;
            encodeBlock(in, nblk, out, csize, findBestB);
        }
        nvalue += csize;
    }
}

void
PFDDecode(uint32_t *in, uint32_t len,
        uint32_t *out, uint32_t nvalue) {
    int32_t lpos;
    uint32_t i;
    uint32_t e;
    uint32_t numBlocks;
    uint32_t b;
    uint32_t excVal;
    uint32_t nExceptions;
    uint32_t encodedExceptionsSize;
    uint32_t except[2 * PFORDELTA_BLOCKSZ + TAIL_MERGIN + 1];
    numBlocks = *in++;
    for (i = 0; i < numBlocks; i++) {
        b = *in >> (32 - PFORDELTA_B);
        nExceptions = (*in >>
                (32 - (PFORDELTA_B + PFORDELTA_NEXCEPT))) &
                ((1 << PFORDELTA_NEXCEPT) - 1);
        encodedExceptionsSize = *in & ((1 << PFORDELTA_EXCEPTSZ) - 1);
        if (PFORDELTA_USE_HARDCODE_SIMPLE16) {
            __p4delta_simple16_decode(++in, 2 * nExceptions, except, 2 * nExceptions);
        } else {
            Simple16Decode(++in, 2 * nExceptions, except, 2 * nExceptions);
        }
        in += encodedExceptionsSize;
        __p4delta_unpack[b](out, in);
        for (e = 0, lpos = -1; e < nExceptions; e++) {
            lpos += except[e] + 1;
            excVal = except[e + nExceptions] + 1;
            excVal <<= b;
            out[lpos] |= excVal;
            assert(lpos < PFORDELTA_BLOCKSZ);
        }
        out += PFORDELTA_BLOCKSZ;
        in += b * PFORDELTA_NBLOCK;
    }
}

#endif /* COMPRESS_HPP_ */
