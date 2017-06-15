/*************************************************************************
	> File: util.h
	> Author: Naiyong Ao
	> Email: aonaiyong@gmail.com 
	> Time: Wed 28 Jan 2015 09:55:54 PM CST
 ************************************************************************/

#ifndef UTIL_H_
#define UTIL_H_

#include <stdint.h>

inline uint64_t div_roundup(uint64_t v, uint32_t divisor) {
	    return (v + (divisor - 1)) / divisor;
}

inline uint32_t gccbits(const uint32_t v) {
	return v == 0 ? 0 : 32 - __builtin_clz(v);
}

inline uint32_t maxbits(const uint32_t *in, uint32_t nvalue) {
	uint32_t accumulator = 0;
	for (uint32_t i = 0; i < nvalue; ++i) {
		accumulator |= in[i];
	}
	return gccbits(accumulator);
}

#endif /* UTIL_H_ */
