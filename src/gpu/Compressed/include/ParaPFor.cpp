/*************************************************************************
	> File: ParaPFor.cpp
	> Author: Naiyong Ao
	> Email: aonaiyong@gmail.com 
	> Time: Wed 28 Jan 2015 11:36:41 PM CST
 ************************************************************************/


#include <stdint.h>
#include <assert.h>

#include <vector>

#include "ParaPFor.h"
#include "pack.h"
#include "unpack.h"
#include "util.h"

float FRAC = 0.0; 

int cnum[33] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};

#define MAX_LEN 40000000


uint32_t lower[MAX_LEN];
uint32_t exceptions[MAX_LEN];
uint32_t positions[MAX_LEN];


inline uint32_t tryB(uint32_t b, const uint32_t *in, uint32_t nvalue) {
	assert(b <= 32);

	if (b == 32)
		return 0;

	uint32_t nExceptions = 0;
	for (uint32_t i = 0; i < nvalue; ++i) {
		if (in[i] >= (1U << b))
			++nExceptions;
	}

	return nExceptions;
}

inline uint32_t findBestB(const uint32_t *in, uint32_t nvalue) {
	for (uint32_t i = 0; i < 32; ++i) {
		const uint32_t nExceptions = tryB(cnum[i], in, nvalue);
		if ((float)nExceptions <= FRAC * nvalue) 
			return cnum[i];
	}
	return 32;
}


uint32_t encodeBlock(const uint32_t *in, uint32_t nvalue, uint32_t *&out) { 
	uint32_t *initout = out;
	++out;

	uint32_t lb = findBestB(in, nvalue);
	uint32_t en = 0;
	for (uint32_t i = 0; i < nvalue; ++i) {
		if (in[i] >= (1U << lb)) {  
			lower[i] = in[i] & ((1U << lb) - 1);	
			exceptions[en] = in[i] >> lb;
			positions[en] = i;			
			++en;
		}
		else					
			lower[i] = in[i];
	} 

	// lower lb-bit part
	packwithoutmask_generic(lower, out, lb, nvalue);
	out += div_roundup(nvalue * lb, 32);

	if (en > 0) {
		// exceptions' positions
		uint32_t eb = gccbits(positions[en-1]);
		packwithoutmask_generic(positions, out, eb, en);
		out += div_roundup(en * eb, 32);

		// exceptions' higher bits
		uint32_t hb = maxbits(exceptions, en);
		packwithoutmask_generic(exceptions, out, hb, en);
		out += div_roundup(en * hb, 32);

		*initout = (lb << 26) + (eb << 21) + (hb << 16) + en;
	}
	else 
		*initout = (lb << 26);

	return out - initout;
}


uint32_t encodeArray(const uint32_t *in, uint32_t *out, uint32_t nvalue) {
	return encodeBlock(in, nvalue, out);
}

const uint32_t *decodeBlock(const uint32_t *in, uint32_t *out, uint32_t nvalue) {
	uint32_t descriptor = in[0];
	uint32_t lb = (descriptor >> 26) % 32;
	uint32_t eb = (descriptor >> 21) & 31;
	uint32_t hb = (descriptor >> 16) & 31;
	uint32_t en = (descriptor & 65535);
	++in;

	unpack_generic(in, out, lb, nvalue);
	in += div_roundup(nvalue * lb, 32);

	if (en > 0) {
		unpack_generic(in, positions, eb, en);
		in += div_roundup(en * eb, 32);

		unpack_generic(in, exceptions, hb, en);
		in += div_roundup(en * hb, 32);

		for (uint32_t i = 0; i < en; ++i) {
			exceptions[i] <<= lb;
			out[positions[i]] |= exceptions[i];
		}
	}

	return in;
}


uint32_t decodeArray(const uint32_t *in, uint32_t *out, uint32_t nvalue) {
	const uint32_t *const initin = in;
	in = decodeBlock(in, out, nvalue);

	return in - initin;	
}

