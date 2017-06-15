/*************************************************************************
	> File: NewPFor.cpp
	> Author: Naiyong Ao
	> Email: aonaiyong@gmail.com 
	> Time: Sun 01 Feb 2015 07:16:04 PM CST
 ************************************************************************/

#include <stdint.h>
#include <assert.h>

#include "NewPFor.h"
#include "Simple16.h"
#include "pack.h"
#include "unpack.h"
#include "util.h"

float FRAC = 0.0;

int cnum[33] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};

#define MAX_LEN 40000000

uint32_t lower[MAX_LEN];
uint32_t exceptions[MAX_LEN];
uint32_t positions[MAX_LEN];
uint32_t all[MAX_LEN];


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

	uint32_t b = findBestB(in, nvalue);
	uint32_t en = 0;
	for (uint32_t i = 0; i < nvalue; ++i) {
		if (in[i] >= (1U << b)) {  
			lower[i] = (in[i] & ((1U << b) - 1));
			exceptions[en] = in[i] >> b;
			positions[en] = i;
			++en;
		}
		else
			lower[i] = in[i];
	} 

	// lower lb-bit part
	packwithoutmask_generic(lower, out, b, nvalue);
	out += div_roundup(nvalue * b, 32);

	if (en > 0) {
		for (uint32_t i = en - 1; i > 0; --i) {
			positions[i] -= positions[i - 1];
			--positions[i];
		}

		for (uint32_t i = 0; i < en; ++i) {
			all[i] = positions[i];
			all[en + i] = exceptions[i];
		}

	
		uint32_t *moveout = out;
		for (uint32_t *moveall = all; moveall < &(all[2 * en]); )
			s16_encode(moveout, moveall, &(all[2 * en]) - moveall);

		out += (moveout - out);

		*initout = (b << 26) + en;
	}
	else
		*initout = (b << 26);

	return out - initout;
}


uint32_t encodeArray(const uint32_t *in, uint32_t *out, uint32_t nvalue) {
	return encodeBlock(in, nvalue, out);
}


const uint32_t *decodeBlock(const uint32_t *in, uint32_t *out, uint32_t nvalue) {
	uint32_t descriptor = in[0];
	uint32_t b = (descriptor >> 26) % 32;
	uint32_t en = (descriptor & 65535);
	++in;

	unpack_generic(in, out, b, nvalue);
	in += div_roundup(nvalue * b, 32);

	if (en > 0) {
		const uint32_t *movein = in; 
		for (uint32_t *moveall = all; moveall < &(all[2 * en]); )
			S16_DECODE(movein, moveall);
		in += (movein - in);

		uint32_t psum = all[0];
		for (uint32_t i = 0; i < en; ++i) {
			out[psum] += (all[en + i] << b);
			psum += all[i + 1] + 1;
		}
	}

	return in;
}


uint32_t decodeArray(const uint32_t *in, uint32_t *out, uint32_t nvalue) {
	const uint32_t *const initin = in;
	in = decodeBlock(in, out, nvalue);

	return in - initin;
}

