/*************************************************************************
	> File: unpack.h
	> Author: Naiyong Ao
	> Email: aonaiyong@gmail.com 
	> Time: Wed 28 Jan 2015 10:04:50 PM CST
 ************************************************************************/

#ifndef UNPACK_H_
#define UNPACK_H_

#include <stdint.h>
#include "util.h"

inline void unpack_generic(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out, 
		const uint32_t bit, uint32_t nvalue) {
	uint64_t mask = (1U << bit) - 1;
	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < nvalue; ++numberOfValuesUnpacked) {
		uint32_t idx = (numberOfValuesUnpacked * bit) >> 5;
		uint32_t shift = (numberOfValuesUnpacked * bit) & 0x1f;
		const uint64_t codeword = (reinterpret_cast<const uint64_t *>(in + idx))[0];
		out[numberOfValuesUnpacked] = static_cast<uint32_t>((codeword >> shift) & mask);
	}
}

#endif /* UNPACK_H_ */

