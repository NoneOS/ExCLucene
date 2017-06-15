/*************************************************************************
	> File: pack.h
	> Author: Naiyong Ao
	> Email: aonaiyong@gmail.com 
	> Time: Wed 28 Jan 2015 10:36:04 PM CST
 ************************************************************************/
#ifndef PACK_H_
#define PACK_H_

#include <stdint.h>
#include <string.h>
#include "util.h"

inline void packwithoutmask_generic(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
		const uint32_t bit, uint32_t nvalue) {
	uint32_t nwords = div_roundup(nvalue * bit, 32);
	memset(out, 0, nwords * sizeof(uint32_t));
	for (uint32_t numberOfValuesPacked = 0; numberOfValuesPacked < nvalue; ++numberOfValuesPacked) {
		uint32_t idx = (numberOfValuesPacked * bit) >> 5;
		uint32_t shift = (numberOfValuesPacked * bit) & 0x1f;
		uint64_t &codeword = (reinterpret_cast<uint64_t *>(out + idx))[0];
		codeword |= static_cast<uint64_t>(in[numberOfValuesPacked]) << shift;
	}
}

#endif /* PACK_H_ */

