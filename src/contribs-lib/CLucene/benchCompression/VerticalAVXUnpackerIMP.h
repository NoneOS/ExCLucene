/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef VERTICALAVXUNPACKERIMP_H_
#define VERTICALAVXUNPACKERIMP_H_


#include "util.h"


// 0-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c0(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		uint32_t *out32 = reinterpret_cast<uint32_t *>(out);
		for (uint32_t numberofValuesUnpacked = 0; numberofValuesUnpacked < 256; numberofValuesUnpacked += 64) {
			memset64(out32);
			out32 += 64;
		}
	}
	else { // Rice Coding
		const uint32_t *quotient32 = reinterpret_cast<const uint32_t *>(quotient);
		uint32_t *out32 = reinterpret_cast<uint32_t *>(out);
		for (uint32_t numberofValuesUnpacked = 0; numberofValuesUnpacked < 256; numberofValuesUnpacked += 32) {
			memcpy32(quotient32, out32);
			quotient32 += 32;
			out32 += 32;
		}
	}
}


// 1-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c1(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c1_and_mask_m256i = _mm256_set1_epi32(0x01);
		__m256i c1_load_rslt_m256i, c1_rslt_m256i;


		c1_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c1_rslt_m256i = _mm256_and_si256( c1_load_rslt_m256i, c1_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 1), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 2), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 3), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 4), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 5), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 6), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 7), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 8), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 9), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 10), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 11), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 12), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 13), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 14), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 15), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 15, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 16), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 17), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 18), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 19), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 20), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 21), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 22), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 23), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 24), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 25), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 26), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 27), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 28), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 29), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 30), c1_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_srli_epi32(c1_load_rslt_m256i, 31);
		_mm256_storeu_si256(out + 31, c1_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c1_and_mask_m256i = _mm256_set1_epi32(0x01);
		__m256i c1_load_rslt_m256i, c1_rslt_m256i;


		c1_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c1_rslt_m256i = _mm256_and_si256( c1_load_rslt_m256i, c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 1) );
		_mm256_storeu_si256(out + 0, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 1), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 1) );
		_mm256_storeu_si256(out + 1, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 2), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 1) );
		_mm256_storeu_si256(out + 2, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 3), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 1) );
		_mm256_storeu_si256(out + 3, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 4), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 1) );
		_mm256_storeu_si256(out + 4, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 5), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 1) );
		_mm256_storeu_si256(out + 5, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 6), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 1) );
		_mm256_storeu_si256(out + 6, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 7), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 1) );
		_mm256_storeu_si256(out + 7, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 8), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 1) );
		_mm256_storeu_si256(out + 8, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 9), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 1) );
		_mm256_storeu_si256(out + 9, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 10), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 1) );
		_mm256_storeu_si256(out + 10, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 11), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 1) );
		_mm256_storeu_si256(out + 11, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 12), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 1) );
		_mm256_storeu_si256(out + 12, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 13), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 1) );
		_mm256_storeu_si256(out + 13, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 14), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 1) );
		_mm256_storeu_si256(out + 14, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 15), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 1) );
		_mm256_storeu_si256(out + 15, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 16), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 1) );
		_mm256_storeu_si256(out + 16, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 17), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 1) );
		_mm256_storeu_si256(out + 17, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 18), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 1) );
		_mm256_storeu_si256(out + 18, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 19), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 1) );
		_mm256_storeu_si256(out + 19, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 20), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 1) );
		_mm256_storeu_si256(out + 20, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 21), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 1) );
		_mm256_storeu_si256(out + 21, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 22), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 1) );
		_mm256_storeu_si256(out + 22, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 23), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 1) );
		_mm256_storeu_si256(out + 23, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 24), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 1) );
		_mm256_storeu_si256(out + 24, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 25), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 1) );
		_mm256_storeu_si256(out + 25, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 26), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 1) );
		_mm256_storeu_si256(out + 26, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 27), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 1) );
		_mm256_storeu_si256(out + 27, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 28), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 1) );
		_mm256_storeu_si256(out + 28, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 29), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 1) );
		_mm256_storeu_si256(out + 29, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c1_load_rslt_m256i, 30), c1_and_mask_m256i );
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 1) );
		_mm256_storeu_si256(out + 30, c1_rslt_m256i);

		c1_rslt_m256i = _mm256_srli_epi32(c1_load_rslt_m256i, 31);
		c1_rslt_m256i = _mm256_or_si256( c1_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 1) );
		_mm256_storeu_si256(out + 31, c1_rslt_m256i);
	}
}


// 2-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c2(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c2_and_mask_m256i = _mm256_set1_epi32(0x03);
		__m256i c2_load_rslt_m256i, c2_rslt_m256i;


		c2_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c2_rslt_m256i = _mm256_and_si256( c2_load_rslt_m256i, c2_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 2), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 4), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 6), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 8), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 10), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 12), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 14), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 16), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 18), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 20), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 22), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 24), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 26), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 28), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_srli_epi32(c2_load_rslt_m256i, 30);
		_mm256_storeu_si256(out + 15, c2_rslt_m256i);


		c2_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c2_rslt_m256i = _mm256_and_si256( c2_load_rslt_m256i, c2_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 2), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 4), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 6), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 8), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 10), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 12), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 14), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 16), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 18), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 20), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 22), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 24), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 26), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 28), c2_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_srli_epi32(c2_load_rslt_m256i, 30);
		_mm256_storeu_si256(out + 31, c2_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c2_and_mask_m256i = _mm256_set1_epi32(0x03);
		__m256i c2_load_rslt_m256i, c2_rslt_m256i;


		c2_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c2_rslt_m256i = _mm256_and_si256( c2_load_rslt_m256i, c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 2) );
		_mm256_storeu_si256(out + 0, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 2), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 2) );
		_mm256_storeu_si256(out + 1, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 4), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 2) );
		_mm256_storeu_si256(out + 2, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 6), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 2) );
		_mm256_storeu_si256(out + 3, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 8), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 2) );
		_mm256_storeu_si256(out + 4, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 10), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 2) );
		_mm256_storeu_si256(out + 5, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 12), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 2) );
		_mm256_storeu_si256(out + 6, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 14), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 2) );
		_mm256_storeu_si256(out + 7, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 16), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 2) );
		_mm256_storeu_si256(out + 8, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 18), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 2) );
		_mm256_storeu_si256(out + 9, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 20), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 2) );
		_mm256_storeu_si256(out + 10, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 22), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 2) );
		_mm256_storeu_si256(out + 11, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 24), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 2) );
		_mm256_storeu_si256(out + 12, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 26), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 2) );
		_mm256_storeu_si256(out + 13, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 28), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 2) );
		_mm256_storeu_si256(out + 14, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_srli_epi32(c2_load_rslt_m256i, 30);
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 2) );
		_mm256_storeu_si256(out + 15, c2_rslt_m256i);


		c2_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c2_rslt_m256i = _mm256_and_si256( c2_load_rslt_m256i, c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 2) );
		_mm256_storeu_si256(out + 16, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 2), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 2) );
		_mm256_storeu_si256(out + 17, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 4), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 2) );
		_mm256_storeu_si256(out + 18, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 6), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 2) );
		_mm256_storeu_si256(out + 19, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 8), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 2) );
		_mm256_storeu_si256(out + 20, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 10), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 2) );
		_mm256_storeu_si256(out + 21, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 12), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 2) );
		_mm256_storeu_si256(out + 22, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 14), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 2) );
		_mm256_storeu_si256(out + 23, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 16), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 2) );
		_mm256_storeu_si256(out + 24, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 18), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 2) );
		_mm256_storeu_si256(out + 25, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 20), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 2) );
		_mm256_storeu_si256(out + 26, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 22), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 2) );
		_mm256_storeu_si256(out + 27, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 24), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 2) );
		_mm256_storeu_si256(out + 28, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 26), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 2) );
		_mm256_storeu_si256(out + 29, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c2_load_rslt_m256i, 28), c2_and_mask_m256i );
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 2) );
		_mm256_storeu_si256(out + 30, c2_rslt_m256i);

		c2_rslt_m256i = _mm256_srli_epi32(c2_load_rslt_m256i, 30);
		c2_rslt_m256i = _mm256_or_si256( c2_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 2) );
		_mm256_storeu_si256(out + 31, c2_rslt_m256i);
	}
}


// 3-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c3(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c3_and_mask_m256i = _mm256_set1_epi32(0x07);
		__m256i c3_load_rslt_m256i, c3_rslt_m256i;


		c3_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c3_rslt_m256i = _mm256_and_si256( c3_load_rslt_m256i, c3_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 3), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 6), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 9), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 12), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 15), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 18), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 21), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 24), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 27), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c3_rslt_m256i);


		c3_rslt_m256i = _mm256_srli_epi32(c3_load_rslt_m256i, 30);
		c3_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c3_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 32 - 30)), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 1), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 4), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 7), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 10), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 13), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 15, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 16), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 19), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 22), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 25), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 28), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c3_rslt_m256i);


		c3_rslt_m256i = _mm256_srli_epi32(c3_load_rslt_m256i, 31);
		c3_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c3_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 32 - 31)), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 2), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 5), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 8), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 11), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 14), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 17), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 20), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 23), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 26), c3_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_srli_epi32(c3_load_rslt_m256i, 29);
		_mm256_storeu_si256(out + 31, c3_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c3_and_mask_m256i = _mm256_set1_epi32(0x07);
		__m256i c3_load_rslt_m256i, c3_rslt_m256i;


		c3_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c3_rslt_m256i = _mm256_and_si256( c3_load_rslt_m256i, c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 3) );
		_mm256_storeu_si256(out + 0, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 3), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 3) );
		_mm256_storeu_si256(out + 1, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 6), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 3) );
		_mm256_storeu_si256(out + 2, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 9), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 3) );
		_mm256_storeu_si256(out + 3, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 12), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 3) );
		_mm256_storeu_si256(out + 4, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 15), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 3) );
		_mm256_storeu_si256(out + 5, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 18), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 3) );
		_mm256_storeu_si256(out + 6, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 21), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 3) );
		_mm256_storeu_si256(out + 7, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 24), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 3) );
		_mm256_storeu_si256(out + 8, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 27), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 3) );
		_mm256_storeu_si256(out + 9, c3_rslt_m256i);


		c3_rslt_m256i = _mm256_srli_epi32(c3_load_rslt_m256i, 30);
		c3_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c3_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 32 - 30)), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 3) );
		_mm256_storeu_si256(out + 10, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 1), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 3) );
		_mm256_storeu_si256(out + 11, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 4), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 3) );
		_mm256_storeu_si256(out + 12, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 7), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 3) );
		_mm256_storeu_si256(out + 13, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 10), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 3) );
		_mm256_storeu_si256(out + 14, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 13), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 3) );
		_mm256_storeu_si256(out + 15, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 16), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 3) );
		_mm256_storeu_si256(out + 16, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 19), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 3) );
		_mm256_storeu_si256(out + 17, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 22), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 3) );
		_mm256_storeu_si256(out + 18, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 25), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 3) );
		_mm256_storeu_si256(out + 19, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 28), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 3) );
		_mm256_storeu_si256(out + 20, c3_rslt_m256i);


		c3_rslt_m256i = _mm256_srli_epi32(c3_load_rslt_m256i, 31);
		c3_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c3_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 32 - 31)), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 3) );
		_mm256_storeu_si256(out + 21, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 2), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 3) );
		_mm256_storeu_si256(out + 22, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 5), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 3) );
		_mm256_storeu_si256(out + 23, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 8), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 3) );
		_mm256_storeu_si256(out + 24, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 11), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 3) );
		_mm256_storeu_si256(out + 25, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 14), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 3) );
		_mm256_storeu_si256(out + 26, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 17), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 3) );
		_mm256_storeu_si256(out + 27, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 20), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 3) );
		_mm256_storeu_si256(out + 28, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 23), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 3) );
		_mm256_storeu_si256(out + 29, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c3_load_rslt_m256i, 26), c3_and_mask_m256i );
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 3) );
		_mm256_storeu_si256(out + 30, c3_rslt_m256i);

		c3_rslt_m256i = _mm256_srli_epi32(c3_load_rslt_m256i, 29);
		c3_rslt_m256i = _mm256_or_si256( c3_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 3) );
		_mm256_storeu_si256(out + 31, c3_rslt_m256i);
	}
}


// 4-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c4(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c4_and_mask_m256i = _mm256_set1_epi32(0x0f);
		__m256i c4_load_rslt_m256i, c4_rslt_m256i;


		c4_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c4_rslt_m256i = _mm256_and_si256( c4_load_rslt_m256i, c4_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 4), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 8), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 12), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 16), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 20), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 24), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_srli_epi32(c4_load_rslt_m256i, 28);
		_mm256_storeu_si256(out + 7, c4_rslt_m256i);


		c4_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c4_rslt_m256i = _mm256_and_si256( c4_load_rslt_m256i, c4_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 4), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 8), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 12), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 16), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 20), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 24), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_srli_epi32(c4_load_rslt_m256i, 28);
		_mm256_storeu_si256(out + 15, c4_rslt_m256i);


		c4_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c4_rslt_m256i = _mm256_and_si256( c4_load_rslt_m256i, c4_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 4), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 8), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 12), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 16), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 20), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 24), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_srli_epi32(c4_load_rslt_m256i, 28);
		_mm256_storeu_si256(out + 23, c4_rslt_m256i);


		c4_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c4_rslt_m256i = _mm256_and_si256( c4_load_rslt_m256i, c4_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 4), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 8), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 12), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 16), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 20), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 24), c4_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_srli_epi32(c4_load_rslt_m256i, 28);
		_mm256_storeu_si256(out + 31, c4_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c4_and_mask_m256i = _mm256_set1_epi32(0x0f);
		__m256i c4_load_rslt_m256i, c4_rslt_m256i;


		c4_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c4_rslt_m256i = _mm256_and_si256( c4_load_rslt_m256i, c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 4) );
		_mm256_storeu_si256(out + 0, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 4), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 4) );
		_mm256_storeu_si256(out + 1, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 8), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 4) );
		_mm256_storeu_si256(out + 2, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 12), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 4) );
		_mm256_storeu_si256(out + 3, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 16), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 4) );
		_mm256_storeu_si256(out + 4, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 20), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 4) );
		_mm256_storeu_si256(out + 5, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 24), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 4) );
		_mm256_storeu_si256(out + 6, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_srli_epi32(c4_load_rslt_m256i, 28);
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 4) );
		_mm256_storeu_si256(out + 7, c4_rslt_m256i);


		c4_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c4_rslt_m256i = _mm256_and_si256( c4_load_rslt_m256i, c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 4) );
		_mm256_storeu_si256(out + 8, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 4), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 4) );
		_mm256_storeu_si256(out + 9, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 8), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 4) );
		_mm256_storeu_si256(out + 10, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 12), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 4) );
		_mm256_storeu_si256(out + 11, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 16), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 4) );
		_mm256_storeu_si256(out + 12, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 20), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 4) );
		_mm256_storeu_si256(out + 13, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 24), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 4) );
		_mm256_storeu_si256(out + 14, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_srli_epi32(c4_load_rslt_m256i, 28);
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 4) );
		_mm256_storeu_si256(out + 15, c4_rslt_m256i);


		c4_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c4_rslt_m256i = _mm256_and_si256( c4_load_rslt_m256i, c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 4) );
		_mm256_storeu_si256(out + 16, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 4), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 4) );
		_mm256_storeu_si256(out + 17, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 8), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 4) );
		_mm256_storeu_si256(out + 18, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 12), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 4) );
		_mm256_storeu_si256(out + 19, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 16), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 4) );
		_mm256_storeu_si256(out + 20, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 20), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 4) );
		_mm256_storeu_si256(out + 21, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 24), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 4) );
		_mm256_storeu_si256(out + 22, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_srli_epi32(c4_load_rslt_m256i, 28);
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 4) );
		_mm256_storeu_si256(out + 23, c4_rslt_m256i);


		c4_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c4_rslt_m256i = _mm256_and_si256( c4_load_rslt_m256i, c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 4) );
		_mm256_storeu_si256(out + 24, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 4), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 4) );
		_mm256_storeu_si256(out + 25, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 8), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 4) );
		_mm256_storeu_si256(out + 26, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 12), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 4) );
		_mm256_storeu_si256(out + 27, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 16), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 4) );
		_mm256_storeu_si256(out + 28, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 20), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 4) );
		_mm256_storeu_si256(out + 29, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c4_load_rslt_m256i, 24), c4_and_mask_m256i );
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 4) );
		_mm256_storeu_si256(out + 30, c4_rslt_m256i);

		c4_rslt_m256i = _mm256_srli_epi32(c4_load_rslt_m256i, 28);
		c4_rslt_m256i = _mm256_or_si256( c4_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 4) );
		_mm256_storeu_si256(out + 31, c4_rslt_m256i);
	}
}


// 5-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c5(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c5_and_mask_m256i = _mm256_set1_epi32(0x1f);
		__m256i c5_load_rslt_m256i, c5_rslt_m256i;


		c5_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c5_rslt_m256i = _mm256_and_si256( c5_load_rslt_m256i, c5_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 5), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 10), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 15), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 20), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 25), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c5_rslt_m256i);


		c5_rslt_m256i = _mm256_srli_epi32(c5_load_rslt_m256i, 30);
		c5_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c5_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 32 - 30)), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 3), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 8), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 13), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 18), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 23), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c5_rslt_m256i);


		c5_rslt_m256i = _mm256_srli_epi32(c5_load_rslt_m256i, 28);
		c5_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c5_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 32 - 28)), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 1), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 6), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 11), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 15, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 16), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 21), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 26), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c5_rslt_m256i);


		c5_rslt_m256i = _mm256_srli_epi32(c5_load_rslt_m256i, 31);
		c5_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c5_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 32 - 31)), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 4), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 9), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 14), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 19), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 24), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c5_rslt_m256i);


		c5_rslt_m256i = _mm256_srli_epi32(c5_load_rslt_m256i, 29);
		c5_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c5_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 32 - 29)), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 2), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 7), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 12), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 17), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 22), c5_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_srli_epi32(c5_load_rslt_m256i, 27);
		_mm256_storeu_si256(out + 31, c5_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c5_and_mask_m256i = _mm256_set1_epi32(0x1f);
		__m256i c5_load_rslt_m256i, c5_rslt_m256i;


		c5_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c5_rslt_m256i = _mm256_and_si256( c5_load_rslt_m256i, c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 5) );
		_mm256_storeu_si256(out + 0, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 5), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 5) );
		_mm256_storeu_si256(out + 1, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 10), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 5) );
		_mm256_storeu_si256(out + 2, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 15), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 5) );
		_mm256_storeu_si256(out + 3, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 20), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 5) );
		_mm256_storeu_si256(out + 4, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 25), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 5) );
		_mm256_storeu_si256(out + 5, c5_rslt_m256i);


		c5_rslt_m256i = _mm256_srli_epi32(c5_load_rslt_m256i, 30);
		c5_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c5_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 32 - 30)), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 5) );
		_mm256_storeu_si256(out + 6, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 3), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 5) );
		_mm256_storeu_si256(out + 7, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 8), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 5) );
		_mm256_storeu_si256(out + 8, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 13), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 5) );
		_mm256_storeu_si256(out + 9, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 18), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 5) );
		_mm256_storeu_si256(out + 10, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 23), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 5) );
		_mm256_storeu_si256(out + 11, c5_rslt_m256i);


		c5_rslt_m256i = _mm256_srli_epi32(c5_load_rslt_m256i, 28);
		c5_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c5_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 32 - 28)), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 5) );
		_mm256_storeu_si256(out + 12, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 1), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 5) );
		_mm256_storeu_si256(out + 13, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 6), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 5) );
		_mm256_storeu_si256(out + 14, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 11), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 5) );
		_mm256_storeu_si256(out + 15, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 16), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 5) );
		_mm256_storeu_si256(out + 16, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 21), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 5) );
		_mm256_storeu_si256(out + 17, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 26), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 5) );
		_mm256_storeu_si256(out + 18, c5_rslt_m256i);


		c5_rslt_m256i = _mm256_srli_epi32(c5_load_rslt_m256i, 31);
		c5_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c5_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 32 - 31)), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 5) );
		_mm256_storeu_si256(out + 19, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 4), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 5) );
		_mm256_storeu_si256(out + 20, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 9), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 5) );
		_mm256_storeu_si256(out + 21, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 14), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 5) );
		_mm256_storeu_si256(out + 22, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 19), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 5) );
		_mm256_storeu_si256(out + 23, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 24), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 5) );
		_mm256_storeu_si256(out + 24, c5_rslt_m256i);


		c5_rslt_m256i = _mm256_srli_epi32(c5_load_rslt_m256i, 29);
		c5_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c5_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 32 - 29)), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 5) );
		_mm256_storeu_si256(out + 25, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 2), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 5) );
		_mm256_storeu_si256(out + 26, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 7), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 5) );
		_mm256_storeu_si256(out + 27, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 12), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 5) );
		_mm256_storeu_si256(out + 28, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 17), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 5) );
		_mm256_storeu_si256(out + 29, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c5_load_rslt_m256i, 22), c5_and_mask_m256i );
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 5) );
		_mm256_storeu_si256(out + 30, c5_rslt_m256i);

		c5_rslt_m256i = _mm256_srli_epi32(c5_load_rslt_m256i, 27);
		c5_rslt_m256i = _mm256_or_si256( c5_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 5) );
		_mm256_storeu_si256(out + 31, c5_rslt_m256i);
	}
}


// 6-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c6(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c6_and_mask_m256i = _mm256_set1_epi32(0x3f);
		__m256i c6_load_rslt_m256i, c6_rslt_m256i;


		c6_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c6_rslt_m256i = _mm256_and_si256( c6_load_rslt_m256i, c6_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 6), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 12), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 18), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 24), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c6_rslt_m256i);


		c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 30);
		c6_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c6_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 32 - 30)), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 4), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 10), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 16), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 22), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c6_rslt_m256i);


		c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 28);
		c6_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c6_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 32 - 28)), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 2), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 8), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 14), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 20), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 26);
		_mm256_storeu_si256(out + 15, c6_rslt_m256i);


		c6_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c6_rslt_m256i = _mm256_and_si256( c6_load_rslt_m256i, c6_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 6), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 12), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 18), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 24), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c6_rslt_m256i);


		c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 30);
		c6_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c6_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 32 - 30)), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 4), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 10), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 16), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 22), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c6_rslt_m256i);


		c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 28);
		c6_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c6_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 32 - 28)), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 2), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 8), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 14), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 20), c6_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 26);
		_mm256_storeu_si256(out + 31, c6_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c6_and_mask_m256i = _mm256_set1_epi32(0x3f);
		__m256i c6_load_rslt_m256i, c6_rslt_m256i;


		c6_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c6_rslt_m256i = _mm256_and_si256( c6_load_rslt_m256i, c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 6) );
		_mm256_storeu_si256(out + 0, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 6), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 6) );
		_mm256_storeu_si256(out + 1, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 12), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 6) );
		_mm256_storeu_si256(out + 2, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 18), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 6) );
		_mm256_storeu_si256(out + 3, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 24), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 6) );
		_mm256_storeu_si256(out + 4, c6_rslt_m256i);


		c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 30);
		c6_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c6_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 32 - 30)), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 6) );
		_mm256_storeu_si256(out + 5, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 4), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 6) );
		_mm256_storeu_si256(out + 6, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 10), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 6) );
		_mm256_storeu_si256(out + 7, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 16), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 6) );
		_mm256_storeu_si256(out + 8, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 22), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 6) );
		_mm256_storeu_si256(out + 9, c6_rslt_m256i);


		c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 28);
		c6_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c6_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 32 - 28)), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 6) );
		_mm256_storeu_si256(out + 10, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 2), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 6) );
		_mm256_storeu_si256(out + 11, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 8), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 6) );
		_mm256_storeu_si256(out + 12, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 14), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 6) );
		_mm256_storeu_si256(out + 13, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 20), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 6) );
		_mm256_storeu_si256(out + 14, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 26);
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 6) );
		_mm256_storeu_si256(out + 15, c6_rslt_m256i);


		c6_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c6_rslt_m256i = _mm256_and_si256( c6_load_rslt_m256i, c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 6) );
		_mm256_storeu_si256(out + 16, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 6), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 6) );
		_mm256_storeu_si256(out + 17, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 12), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 6) );
		_mm256_storeu_si256(out + 18, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 18), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 6) );
		_mm256_storeu_si256(out + 19, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 24), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 6) );
		_mm256_storeu_si256(out + 20, c6_rslt_m256i);


		c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 30);
		c6_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c6_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 32 - 30)), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 6) );
		_mm256_storeu_si256(out + 21, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 4), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 6) );
		_mm256_storeu_si256(out + 22, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 10), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 6) );
		_mm256_storeu_si256(out + 23, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 16), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 6) );
		_mm256_storeu_si256(out + 24, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 22), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 6) );
		_mm256_storeu_si256(out + 25, c6_rslt_m256i);


		c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 28);
		c6_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c6_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 32 - 28)), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 6) );
		_mm256_storeu_si256(out + 26, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 2), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 6) );
		_mm256_storeu_si256(out + 27, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 8), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 6) );
		_mm256_storeu_si256(out + 28, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 14), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 6) );
		_mm256_storeu_si256(out + 29, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c6_load_rslt_m256i, 20), c6_and_mask_m256i );
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 6) );
		_mm256_storeu_si256(out + 30, c6_rslt_m256i);

		c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 26);
		c6_rslt_m256i = _mm256_or_si256( c6_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 6) );
		_mm256_storeu_si256(out + 31, c6_rslt_m256i);
	}
}


// 7-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c7(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c7_and_mask_m256i = _mm256_set1_epi32(0x7f);
		__m256i c7_load_rslt_m256i, c7_rslt_m256i;


		c7_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c7_rslt_m256i = _mm256_and_si256( c7_load_rslt_m256i, c7_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 7), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 14), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 21), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c7_rslt_m256i);


		c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 28);
		c7_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c7_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 32 - 28)), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 3), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 10), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 17), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 24), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c7_rslt_m256i);


		c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 31);
		c7_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c7_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 32 - 31)), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 6), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 13), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 20), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c7_rslt_m256i);


		c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 27);
		c7_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c7_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 32 - 27)), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 2), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 9), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 15, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 16), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 23), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c7_rslt_m256i);


		c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 30);
		c7_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c7_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 32 - 30)), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 5), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 12), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 19), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c7_rslt_m256i);


		c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 26);
		c7_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c7_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 32 - 26)), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 1), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 8), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 15), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 22), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c7_rslt_m256i);


		c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 29);
		c7_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c7_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 32 - 29)), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 4), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 11), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 18), c7_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 25);
		_mm256_storeu_si256(out + 31, c7_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c7_and_mask_m256i = _mm256_set1_epi32(0x7f);
		__m256i c7_load_rslt_m256i, c7_rslt_m256i;


		c7_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c7_rslt_m256i = _mm256_and_si256( c7_load_rslt_m256i, c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 7) );
		_mm256_storeu_si256(out + 0, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 7), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 7) );
		_mm256_storeu_si256(out + 1, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 14), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 7) );
		_mm256_storeu_si256(out + 2, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 21), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 7) );
		_mm256_storeu_si256(out + 3, c7_rslt_m256i);


		c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 28);
		c7_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c7_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 32 - 28)), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 7) );
		_mm256_storeu_si256(out + 4, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 3), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 7) );
		_mm256_storeu_si256(out + 5, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 10), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 7) );
		_mm256_storeu_si256(out + 6, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 17), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 7) );
		_mm256_storeu_si256(out + 7, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 24), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 7) );
		_mm256_storeu_si256(out + 8, c7_rslt_m256i);


		c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 31);
		c7_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c7_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 32 - 31)), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 7) );
		_mm256_storeu_si256(out + 9, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 6), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 7) );
		_mm256_storeu_si256(out + 10, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 13), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 7) );
		_mm256_storeu_si256(out + 11, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 20), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 7) );
		_mm256_storeu_si256(out + 12, c7_rslt_m256i);


		c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 27);
		c7_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c7_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 32 - 27)), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 7) );
		_mm256_storeu_si256(out + 13, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 2), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 7) );
		_mm256_storeu_si256(out + 14, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 9), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 7) );
		_mm256_storeu_si256(out + 15, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 16), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 7) );
		_mm256_storeu_si256(out + 16, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 23), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 7) );
		_mm256_storeu_si256(out + 17, c7_rslt_m256i);


		c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 30);
		c7_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c7_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 32 - 30)), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 7) );
		_mm256_storeu_si256(out + 18, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 5), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 7) );
		_mm256_storeu_si256(out + 19, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 12), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 7) );
		_mm256_storeu_si256(out + 20, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 19), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 7) );
		_mm256_storeu_si256(out + 21, c7_rslt_m256i);


		c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 26);
		c7_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c7_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 32 - 26)), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 7) );
		_mm256_storeu_si256(out + 22, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 1), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 7) );
		_mm256_storeu_si256(out + 23, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 8), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 7) );
		_mm256_storeu_si256(out + 24, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 15), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 7) );
		_mm256_storeu_si256(out + 25, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 22), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 7) );
		_mm256_storeu_si256(out + 26, c7_rslt_m256i);


		c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 29);
		c7_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c7_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 32 - 29)), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 7) );
		_mm256_storeu_si256(out + 27, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 4), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 7) );
		_mm256_storeu_si256(out + 28, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 11), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 7) );
		_mm256_storeu_si256(out + 29, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c7_load_rslt_m256i, 18), c7_and_mask_m256i );
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 7) );
		_mm256_storeu_si256(out + 30, c7_rslt_m256i);

		c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 25);
		c7_rslt_m256i = _mm256_or_si256( c7_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 7) );
		_mm256_storeu_si256(out + 31, c7_rslt_m256i);
	}
}


// 8-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c8(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c8_and_mask_m256i = _mm256_set1_epi32(0xff);
		__m256i c8_load_rslt_m256i, c8_rslt_m256i;


		c8_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c8_rslt_m256i = _mm256_and_si256( c8_load_rslt_m256i, c8_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 8), c8_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 16), c8_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_srli_epi32(c8_load_rslt_m256i, 24);
		_mm256_storeu_si256(out + 3, c8_rslt_m256i);


		c8_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c8_rslt_m256i = _mm256_and_si256( c8_load_rslt_m256i, c8_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 8), c8_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 16), c8_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_srli_epi32(c8_load_rslt_m256i, 24);
		_mm256_storeu_si256(out + 7, c8_rslt_m256i);


		c8_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c8_rslt_m256i = _mm256_and_si256( c8_load_rslt_m256i, c8_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 8), c8_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 16), c8_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_srli_epi32(c8_load_rslt_m256i, 24);
		_mm256_storeu_si256(out + 11, c8_rslt_m256i);


		c8_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c8_rslt_m256i = _mm256_and_si256( c8_load_rslt_m256i, c8_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 8), c8_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 16), c8_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_srli_epi32(c8_load_rslt_m256i, 24);
		_mm256_storeu_si256(out + 15, c8_rslt_m256i);


		c8_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c8_rslt_m256i = _mm256_and_si256( c8_load_rslt_m256i, c8_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 8), c8_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 16), c8_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_srli_epi32(c8_load_rslt_m256i, 24);
		_mm256_storeu_si256(out + 19, c8_rslt_m256i);


		c8_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c8_rslt_m256i = _mm256_and_si256( c8_load_rslt_m256i, c8_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 8), c8_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 16), c8_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_srli_epi32(c8_load_rslt_m256i, 24);
		_mm256_storeu_si256(out + 23, c8_rslt_m256i);


		c8_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c8_rslt_m256i = _mm256_and_si256( c8_load_rslt_m256i, c8_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 8), c8_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 16), c8_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_srli_epi32(c8_load_rslt_m256i, 24);
		_mm256_storeu_si256(out + 27, c8_rslt_m256i);


		c8_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c8_rslt_m256i = _mm256_and_si256( c8_load_rslt_m256i, c8_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 8), c8_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 16), c8_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_srli_epi32(c8_load_rslt_m256i, 24);
		_mm256_storeu_si256(out + 31, c8_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c8_and_mask_m256i = _mm256_set1_epi32(0xff);
		__m256i c8_load_rslt_m256i, c8_rslt_m256i;


		c8_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c8_rslt_m256i = _mm256_and_si256( c8_load_rslt_m256i, c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 8) );
		_mm256_storeu_si256(out + 0, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 8), c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 8) );
		_mm256_storeu_si256(out + 1, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 16), c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 8) );
		_mm256_storeu_si256(out + 2, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_srli_epi32(c8_load_rslt_m256i, 24);
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 8) );
		_mm256_storeu_si256(out + 3, c8_rslt_m256i);


		c8_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c8_rslt_m256i = _mm256_and_si256( c8_load_rslt_m256i, c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 8) );
		_mm256_storeu_si256(out + 4, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 8), c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 8) );
		_mm256_storeu_si256(out + 5, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 16), c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 8) );
		_mm256_storeu_si256(out + 6, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_srli_epi32(c8_load_rslt_m256i, 24);
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 8) );
		_mm256_storeu_si256(out + 7, c8_rslt_m256i);


		c8_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c8_rslt_m256i = _mm256_and_si256( c8_load_rslt_m256i, c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 8) );
		_mm256_storeu_si256(out + 8, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 8), c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 8) );
		_mm256_storeu_si256(out + 9, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 16), c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 8) );
		_mm256_storeu_si256(out + 10, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_srli_epi32(c8_load_rslt_m256i, 24);
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 8) );
		_mm256_storeu_si256(out + 11, c8_rslt_m256i);


		c8_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c8_rslt_m256i = _mm256_and_si256( c8_load_rslt_m256i, c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 8) );
		_mm256_storeu_si256(out + 12, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 8), c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 8) );
		_mm256_storeu_si256(out + 13, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 16), c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 8) );
		_mm256_storeu_si256(out + 14, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_srli_epi32(c8_load_rslt_m256i, 24);
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 8) );
		_mm256_storeu_si256(out + 15, c8_rslt_m256i);


		c8_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c8_rslt_m256i = _mm256_and_si256( c8_load_rslt_m256i, c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 8) );
		_mm256_storeu_si256(out + 16, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 8), c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 8) );
		_mm256_storeu_si256(out + 17, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 16), c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 8) );
		_mm256_storeu_si256(out + 18, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_srli_epi32(c8_load_rslt_m256i, 24);
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 8) );
		_mm256_storeu_si256(out + 19, c8_rslt_m256i);


		c8_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c8_rslt_m256i = _mm256_and_si256( c8_load_rslt_m256i, c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 8) );
		_mm256_storeu_si256(out + 20, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 8), c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 8) );
		_mm256_storeu_si256(out + 21, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 16), c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 8) );
		_mm256_storeu_si256(out + 22, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_srli_epi32(c8_load_rslt_m256i, 24);
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 8) );
		_mm256_storeu_si256(out + 23, c8_rslt_m256i);


		c8_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c8_rslt_m256i = _mm256_and_si256( c8_load_rslt_m256i, c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 8) );
		_mm256_storeu_si256(out + 24, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 8), c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 8) );
		_mm256_storeu_si256(out + 25, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 16), c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 8) );
		_mm256_storeu_si256(out + 26, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_srli_epi32(c8_load_rslt_m256i, 24);
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 8) );
		_mm256_storeu_si256(out + 27, c8_rslt_m256i);


		c8_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c8_rslt_m256i = _mm256_and_si256( c8_load_rslt_m256i, c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 8) );
		_mm256_storeu_si256(out + 28, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 8), c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 8) );
		_mm256_storeu_si256(out + 29, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c8_load_rslt_m256i, 16), c8_and_mask_m256i );
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 8) );
		_mm256_storeu_si256(out + 30, c8_rslt_m256i);

		c8_rslt_m256i = _mm256_srli_epi32(c8_load_rslt_m256i, 24);
		c8_rslt_m256i = _mm256_or_si256( c8_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 8) );
		_mm256_storeu_si256(out + 31, c8_rslt_m256i);
	}
}


// 9-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c9(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c9_and_mask_m256i = _mm256_set1_epi32(0x01ff);
		__m256i c9_load_rslt_m256i, c9_rslt_m256i;


		c9_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c9_rslt_m256i = _mm256_and_si256( c9_load_rslt_m256i, c9_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 9), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 18), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c9_rslt_m256i);


		c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 27);
		c9_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c9_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 32 - 27)), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 4), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 13), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 22), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c9_rslt_m256i);


		c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 31);
		c9_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c9_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 32 - 31)), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 8), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 17), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c9_rslt_m256i);


		c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 26);
		c9_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c9_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 32 - 26)), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 3), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 12), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 21), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c9_rslt_m256i);


		c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 30);
		c9_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c9_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 32 - 30)), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 7), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 15, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 16), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c9_rslt_m256i);


		c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 25);
		c9_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c9_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 32 - 25)), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 2), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 11), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 20), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c9_rslt_m256i);


		c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 29);
		c9_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c9_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 32 - 29)), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 6), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 15), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c9_rslt_m256i);


		c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 24);
		c9_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c9_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 32 - 24)), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 1), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 10), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 19), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c9_rslt_m256i);


		c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 28);
		c9_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c9_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 32 - 28)), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 5), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 14), c9_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 23);
		_mm256_storeu_si256(out + 31, c9_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c9_and_mask_m256i = _mm256_set1_epi32(0x01ff);
		__m256i c9_load_rslt_m256i, c9_rslt_m256i;


		c9_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c9_rslt_m256i = _mm256_and_si256( c9_load_rslt_m256i, c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 9) );
		_mm256_storeu_si256(out + 0, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 9), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 9) );
		_mm256_storeu_si256(out + 1, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 18), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 9) );
		_mm256_storeu_si256(out + 2, c9_rslt_m256i);


		c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 27);
		c9_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c9_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 32 - 27)), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 9) );
		_mm256_storeu_si256(out + 3, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 4), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 9) );
		_mm256_storeu_si256(out + 4, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 13), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 9) );
		_mm256_storeu_si256(out + 5, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 22), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 9) );
		_mm256_storeu_si256(out + 6, c9_rslt_m256i);


		c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 31);
		c9_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c9_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 32 - 31)), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 9) );
		_mm256_storeu_si256(out + 7, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 8), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 9) );
		_mm256_storeu_si256(out + 8, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 17), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 9) );
		_mm256_storeu_si256(out + 9, c9_rslt_m256i);


		c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 26);
		c9_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c9_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 32 - 26)), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 9) );
		_mm256_storeu_si256(out + 10, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 3), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 9) );
		_mm256_storeu_si256(out + 11, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 12), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 9) );
		_mm256_storeu_si256(out + 12, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 21), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 9) );
		_mm256_storeu_si256(out + 13, c9_rslt_m256i);


		c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 30);
		c9_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c9_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 32 - 30)), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 9) );
		_mm256_storeu_si256(out + 14, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 7), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 9) );
		_mm256_storeu_si256(out + 15, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 16), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 9) );
		_mm256_storeu_si256(out + 16, c9_rslt_m256i);


		c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 25);
		c9_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c9_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 32 - 25)), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 9) );
		_mm256_storeu_si256(out + 17, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 2), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 9) );
		_mm256_storeu_si256(out + 18, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 11), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 9) );
		_mm256_storeu_si256(out + 19, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 20), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 9) );
		_mm256_storeu_si256(out + 20, c9_rslt_m256i);


		c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 29);
		c9_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c9_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 32 - 29)), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 9) );
		_mm256_storeu_si256(out + 21, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 6), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 9) );
		_mm256_storeu_si256(out + 22, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 15), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 9) );
		_mm256_storeu_si256(out + 23, c9_rslt_m256i);


		c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 24);
		c9_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c9_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 32 - 24)), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 9) );
		_mm256_storeu_si256(out + 24, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 1), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 9) );
		_mm256_storeu_si256(out + 25, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 10), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 9) );
		_mm256_storeu_si256(out + 26, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 19), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 9) );
		_mm256_storeu_si256(out + 27, c9_rslt_m256i);


		c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 28);
		c9_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c9_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 32 - 28)), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 9) );
		_mm256_storeu_si256(out + 28, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 5), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 9) );
		_mm256_storeu_si256(out + 29, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c9_load_rslt_m256i, 14), c9_and_mask_m256i );
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 9) );
		_mm256_storeu_si256(out + 30, c9_rslt_m256i);

		c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 23);
		c9_rslt_m256i = _mm256_or_si256( c9_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 9) );
		_mm256_storeu_si256(out + 31, c9_rslt_m256i);
	}
}


// 10-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c10(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c10_and_mask_m256i = _mm256_set1_epi32(0x03ff);
		__m256i c10_load_rslt_m256i, c10_rslt_m256i;


		c10_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c10_rslt_m256i = _mm256_and_si256( c10_load_rslt_m256i, c10_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 10), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 20), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c10_rslt_m256i);


		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 30);
		c10_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c10_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 32 - 30)), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 8), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 18), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c10_rslt_m256i);


		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 28);
		c10_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c10_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 32 - 28)), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 6), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 16), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c10_rslt_m256i);


		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 26);
		c10_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c10_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 32 - 26)), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 4), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 14), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c10_rslt_m256i);


		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 24);
		c10_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c10_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 32 - 24)), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 2), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 12), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 22);
		_mm256_storeu_si256(out + 15, c10_rslt_m256i);


		c10_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c10_rslt_m256i = _mm256_and_si256( c10_load_rslt_m256i, c10_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 10), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 20), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c10_rslt_m256i);


		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 30);
		c10_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c10_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 32 - 30)), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 8), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 18), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c10_rslt_m256i);


		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 28);
		c10_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c10_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 32 - 28)), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 6), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 16), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c10_rslt_m256i);


		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 26);
		c10_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c10_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 32 - 26)), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 4), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 14), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c10_rslt_m256i);


		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 24);
		c10_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c10_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 32 - 24)), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 2), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 12), c10_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 22);
		_mm256_storeu_si256(out + 31, c10_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c10_and_mask_m256i = _mm256_set1_epi32(0x03ff);
		__m256i c10_load_rslt_m256i, c10_rslt_m256i;


		c10_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c10_rslt_m256i = _mm256_and_si256( c10_load_rslt_m256i, c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 10) );
		_mm256_storeu_si256(out + 0, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 10), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 10) );
		_mm256_storeu_si256(out + 1, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 20), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 10) );
		_mm256_storeu_si256(out + 2, c10_rslt_m256i);


		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 30);
		c10_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c10_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 32 - 30)), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 10) );
		_mm256_storeu_si256(out + 3, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 8), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 10) );
		_mm256_storeu_si256(out + 4, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 18), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 10) );
		_mm256_storeu_si256(out + 5, c10_rslt_m256i);


		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 28);
		c10_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c10_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 32 - 28)), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 10) );
		_mm256_storeu_si256(out + 6, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 6), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 10) );
		_mm256_storeu_si256(out + 7, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 16), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 10) );
		_mm256_storeu_si256(out + 8, c10_rslt_m256i);


		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 26);
		c10_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c10_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 32 - 26)), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 10) );
		_mm256_storeu_si256(out + 9, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 4), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 10) );
		_mm256_storeu_si256(out + 10, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 14), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 10) );
		_mm256_storeu_si256(out + 11, c10_rslt_m256i);


		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 24);
		c10_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c10_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 32 - 24)), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 10) );
		_mm256_storeu_si256(out + 12, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 2), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 10) );
		_mm256_storeu_si256(out + 13, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 12), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 10) );
		_mm256_storeu_si256(out + 14, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 22);
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 10) );
		_mm256_storeu_si256(out + 15, c10_rslt_m256i);


		c10_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c10_rslt_m256i = _mm256_and_si256( c10_load_rslt_m256i, c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 10) );
		_mm256_storeu_si256(out + 16, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 10), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 10) );
		_mm256_storeu_si256(out + 17, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 20), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 10) );
		_mm256_storeu_si256(out + 18, c10_rslt_m256i);


		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 30);
		c10_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c10_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 32 - 30)), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 10) );
		_mm256_storeu_si256(out + 19, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 8), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 10) );
		_mm256_storeu_si256(out + 20, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 18), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 10) );
		_mm256_storeu_si256(out + 21, c10_rslt_m256i);


		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 28);
		c10_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c10_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 32 - 28)), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 10) );
		_mm256_storeu_si256(out + 22, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 6), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 10) );
		_mm256_storeu_si256(out + 23, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 16), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 10) );
		_mm256_storeu_si256(out + 24, c10_rslt_m256i);


		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 26);
		c10_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c10_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 32 - 26)), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 10) );
		_mm256_storeu_si256(out + 25, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 4), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 10) );
		_mm256_storeu_si256(out + 26, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 14), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 10) );
		_mm256_storeu_si256(out + 27, c10_rslt_m256i);


		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 24);
		c10_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c10_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 32 - 24)), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 10) );
		_mm256_storeu_si256(out + 28, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 2), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 10) );
		_mm256_storeu_si256(out + 29, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c10_load_rslt_m256i, 12), c10_and_mask_m256i );
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 10) );
		_mm256_storeu_si256(out + 30, c10_rslt_m256i);

		c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 22);
		c10_rslt_m256i = _mm256_or_si256( c10_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 10) );
		_mm256_storeu_si256(out + 31, c10_rslt_m256i);
	}
}


// 11-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c11(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c11_and_mask_m256i = _mm256_set1_epi32(0x07ff);
		__m256i c11_load_rslt_m256i, c11_rslt_m256i;


		c11_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c11_rslt_m256i = _mm256_and_si256( c11_load_rslt_m256i, c11_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 11), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 22);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 22)), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 1), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 12), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 23);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 23)), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 2), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 13), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 24);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 24)), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 3), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 14), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 25);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 25)), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 4), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 15), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 26);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 26)), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 5), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 15, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 16), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 27);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 27)), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 6), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 17), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 28);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 28)), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 7), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 18), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 29);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 29)), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 8), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 19), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 30);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 30)), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 9), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 20), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 31);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 31)), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 10), c11_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 21);
		_mm256_storeu_si256(out + 31, c11_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c11_and_mask_m256i = _mm256_set1_epi32(0x07ff);
		__m256i c11_load_rslt_m256i, c11_rslt_m256i;


		c11_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c11_rslt_m256i = _mm256_and_si256( c11_load_rslt_m256i, c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 11) );
		_mm256_storeu_si256(out + 0, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 11), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 11) );
		_mm256_storeu_si256(out + 1, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 22);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 22)), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 11) );
		_mm256_storeu_si256(out + 2, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 1), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 11) );
		_mm256_storeu_si256(out + 3, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 12), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 11) );
		_mm256_storeu_si256(out + 4, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 23);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 23)), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 11) );
		_mm256_storeu_si256(out + 5, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 2), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 11) );
		_mm256_storeu_si256(out + 6, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 13), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 11) );
		_mm256_storeu_si256(out + 7, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 24);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 24)), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 11) );
		_mm256_storeu_si256(out + 8, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 3), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 11) );
		_mm256_storeu_si256(out + 9, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 14), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 11) );
		_mm256_storeu_si256(out + 10, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 25);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 25)), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 11) );
		_mm256_storeu_si256(out + 11, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 4), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 11) );
		_mm256_storeu_si256(out + 12, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 15), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 11) );
		_mm256_storeu_si256(out + 13, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 26);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 26)), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 11) );
		_mm256_storeu_si256(out + 14, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 5), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 11) );
		_mm256_storeu_si256(out + 15, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 16), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 11) );
		_mm256_storeu_si256(out + 16, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 27);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 27)), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 11) );
		_mm256_storeu_si256(out + 17, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 6), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 11) );
		_mm256_storeu_si256(out + 18, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 17), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 11) );
		_mm256_storeu_si256(out + 19, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 28);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 28)), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 11) );
		_mm256_storeu_si256(out + 20, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 7), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 11) );
		_mm256_storeu_si256(out + 21, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 18), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 11) );
		_mm256_storeu_si256(out + 22, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 29);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 29)), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 11) );
		_mm256_storeu_si256(out + 23, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 8), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 11) );
		_mm256_storeu_si256(out + 24, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 19), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 11) );
		_mm256_storeu_si256(out + 25, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 30);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 30)), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 11) );
		_mm256_storeu_si256(out + 26, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 9), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 11) );
		_mm256_storeu_si256(out + 27, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 20), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 11) );
		_mm256_storeu_si256(out + 28, c11_rslt_m256i);


		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 31);
		c11_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c11_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 32 - 31)), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 11) );
		_mm256_storeu_si256(out + 29, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c11_load_rslt_m256i, 10), c11_and_mask_m256i );
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 11) );
		_mm256_storeu_si256(out + 30, c11_rslt_m256i);

		c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 21);
		c11_rslt_m256i = _mm256_or_si256( c11_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 11) );
		_mm256_storeu_si256(out + 31, c11_rslt_m256i);
	}
}


// 12-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c12(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c12_and_mask_m256i = _mm256_set1_epi32(0x0fff);
		__m256i c12_load_rslt_m256i, c12_rslt_m256i;


		c12_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c12_rslt_m256i = _mm256_and_si256( c12_load_rslt_m256i, c12_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 12), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c12_rslt_m256i);


		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 24);
		c12_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c12_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 32 - 24)), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 4), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 16), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c12_rslt_m256i);


		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 28);
		c12_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c12_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 32 - 28)), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 8), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 20);
		_mm256_storeu_si256(out + 7, c12_rslt_m256i);


		c12_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c12_rslt_m256i = _mm256_and_si256( c12_load_rslt_m256i, c12_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 12), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c12_rslt_m256i);


		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 24);
		c12_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c12_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 32 - 24)), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 4), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 16), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c12_rslt_m256i);


		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 28);
		c12_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c12_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 32 - 28)), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 8), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 20);
		_mm256_storeu_si256(out + 15, c12_rslt_m256i);


		c12_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c12_rslt_m256i = _mm256_and_si256( c12_load_rslt_m256i, c12_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 12), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c12_rslt_m256i);


		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 24);
		c12_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c12_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 32 - 24)), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 4), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 16), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c12_rslt_m256i);


		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 28);
		c12_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c12_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 32 - 28)), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 8), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 20);
		_mm256_storeu_si256(out + 23, c12_rslt_m256i);


		c12_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c12_rslt_m256i = _mm256_and_si256( c12_load_rslt_m256i, c12_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 12), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c12_rslt_m256i);


		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 24);
		c12_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c12_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 32 - 24)), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 4), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 16), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c12_rslt_m256i);


		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 28);
		c12_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c12_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 32 - 28)), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 8), c12_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 20);
		_mm256_storeu_si256(out + 31, c12_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c12_and_mask_m256i = _mm256_set1_epi32(0x0fff);
		__m256i c12_load_rslt_m256i, c12_rslt_m256i;


		c12_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c12_rslt_m256i = _mm256_and_si256( c12_load_rslt_m256i, c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 12) );
		_mm256_storeu_si256(out + 0, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 12), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 12) );
		_mm256_storeu_si256(out + 1, c12_rslt_m256i);


		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 24);
		c12_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c12_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 32 - 24)), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 12) );
		_mm256_storeu_si256(out + 2, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 4), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 12) );
		_mm256_storeu_si256(out + 3, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 16), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 12) );
		_mm256_storeu_si256(out + 4, c12_rslt_m256i);


		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 28);
		c12_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c12_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 32 - 28)), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 12) );
		_mm256_storeu_si256(out + 5, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 8), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 12) );
		_mm256_storeu_si256(out + 6, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 20);
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 12) );
		_mm256_storeu_si256(out + 7, c12_rslt_m256i);


		c12_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c12_rslt_m256i = _mm256_and_si256( c12_load_rslt_m256i, c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 12) );
		_mm256_storeu_si256(out + 8, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 12), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 12) );
		_mm256_storeu_si256(out + 9, c12_rslt_m256i);


		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 24);
		c12_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c12_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 32 - 24)), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 12) );
		_mm256_storeu_si256(out + 10, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 4), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 12) );
		_mm256_storeu_si256(out + 11, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 16), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 12) );
		_mm256_storeu_si256(out + 12, c12_rslt_m256i);


		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 28);
		c12_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c12_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 32 - 28)), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 12) );
		_mm256_storeu_si256(out + 13, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 8), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 12) );
		_mm256_storeu_si256(out + 14, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 20);
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 12) );
		_mm256_storeu_si256(out + 15, c12_rslt_m256i);


		c12_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c12_rslt_m256i = _mm256_and_si256( c12_load_rslt_m256i, c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 12) );
		_mm256_storeu_si256(out + 16, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 12), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 12) );
		_mm256_storeu_si256(out + 17, c12_rslt_m256i);


		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 24);
		c12_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c12_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 32 - 24)), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 12) );
		_mm256_storeu_si256(out + 18, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 4), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 12) );
		_mm256_storeu_si256(out + 19, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 16), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 12) );
		_mm256_storeu_si256(out + 20, c12_rslt_m256i);


		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 28);
		c12_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c12_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 32 - 28)), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 12) );
		_mm256_storeu_si256(out + 21, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 8), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 12) );
		_mm256_storeu_si256(out + 22, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 20);
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 12) );
		_mm256_storeu_si256(out + 23, c12_rslt_m256i);


		c12_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c12_rslt_m256i = _mm256_and_si256( c12_load_rslt_m256i, c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 12) );
		_mm256_storeu_si256(out + 24, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 12), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 12) );
		_mm256_storeu_si256(out + 25, c12_rslt_m256i);


		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 24);
		c12_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c12_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 32 - 24)), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 12) );
		_mm256_storeu_si256(out + 26, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 4), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 12) );
		_mm256_storeu_si256(out + 27, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 16), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 12) );
		_mm256_storeu_si256(out + 28, c12_rslt_m256i);


		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 28);
		c12_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c12_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 32 - 28)), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 12) );
		_mm256_storeu_si256(out + 29, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c12_load_rslt_m256i, 8), c12_and_mask_m256i );
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 12) );
		_mm256_storeu_si256(out + 30, c12_rslt_m256i);

		c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 20);
		c12_rslt_m256i = _mm256_or_si256( c12_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 12) );
		_mm256_storeu_si256(out + 31, c12_rslt_m256i);
	}
}


// 13-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c13(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c13_and_mask_m256i = _mm256_set1_epi32(0x1fff);
		__m256i c13_load_rslt_m256i, c13_rslt_m256i;


		c13_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c13_rslt_m256i = _mm256_and_si256( c13_load_rslt_m256i, c13_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 13), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 26);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 26)), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 7), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 20);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 20)), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 1), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 14), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 27);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 27)), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 8), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 21);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 21)), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 2), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 15), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 28);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 28)), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 9), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 22);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 22)), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 3), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 15, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 16), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 29);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 29)), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 10), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 23);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 23)), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 4), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 17), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 30);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 30)), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 11), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 24);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 24)), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 5), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 18), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 31);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 31)), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 12), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 25);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 25)), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 6), c13_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 19);
		_mm256_storeu_si256(out + 31, c13_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c13_and_mask_m256i = _mm256_set1_epi32(0x1fff);
		__m256i c13_load_rslt_m256i, c13_rslt_m256i;


		c13_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c13_rslt_m256i = _mm256_and_si256( c13_load_rslt_m256i, c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 13) );
		_mm256_storeu_si256(out + 0, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 13), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 13) );
		_mm256_storeu_si256(out + 1, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 26);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 26)), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 13) );
		_mm256_storeu_si256(out + 2, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 7), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 13) );
		_mm256_storeu_si256(out + 3, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 20);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 20)), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 13) );
		_mm256_storeu_si256(out + 4, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 1), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 13) );
		_mm256_storeu_si256(out + 5, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 14), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 13) );
		_mm256_storeu_si256(out + 6, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 27);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 27)), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 13) );
		_mm256_storeu_si256(out + 7, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 8), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 13) );
		_mm256_storeu_si256(out + 8, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 21);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 21)), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 13) );
		_mm256_storeu_si256(out + 9, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 2), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 13) );
		_mm256_storeu_si256(out + 10, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 15), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 13) );
		_mm256_storeu_si256(out + 11, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 28);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 28)), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 13) );
		_mm256_storeu_si256(out + 12, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 9), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 13) );
		_mm256_storeu_si256(out + 13, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 22);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 22)), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 13) );
		_mm256_storeu_si256(out + 14, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 3), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 13) );
		_mm256_storeu_si256(out + 15, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 16), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 13) );
		_mm256_storeu_si256(out + 16, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 29);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 29)), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 13) );
		_mm256_storeu_si256(out + 17, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 10), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 13) );
		_mm256_storeu_si256(out + 18, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 23);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 23)), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 13) );
		_mm256_storeu_si256(out + 19, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 4), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 13) );
		_mm256_storeu_si256(out + 20, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 17), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 13) );
		_mm256_storeu_si256(out + 21, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 30);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 30)), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 13) );
		_mm256_storeu_si256(out + 22, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 11), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 13) );
		_mm256_storeu_si256(out + 23, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 24);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 24)), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 13) );
		_mm256_storeu_si256(out + 24, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 5), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 13) );
		_mm256_storeu_si256(out + 25, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 18), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 13) );
		_mm256_storeu_si256(out + 26, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 31);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 31)), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 13) );
		_mm256_storeu_si256(out + 27, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 12), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 13) );
		_mm256_storeu_si256(out + 28, c13_rslt_m256i);


		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 25);
		c13_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c13_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 32 - 25)), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 13) );
		_mm256_storeu_si256(out + 29, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c13_load_rslt_m256i, 6), c13_and_mask_m256i );
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 13) );
		_mm256_storeu_si256(out + 30, c13_rslt_m256i);

		c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 19);
		c13_rslt_m256i = _mm256_or_si256( c13_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 13) );
		_mm256_storeu_si256(out + 31, c13_rslt_m256i);
	}
}


// 14-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c14(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c14_and_mask_m256i = _mm256_set1_epi32(0x3fff);
		__m256i c14_load_rslt_m256i, c14_rslt_m256i;


		c14_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c14_rslt_m256i = _mm256_and_si256( c14_load_rslt_m256i, c14_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 14), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 28);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 28)), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 10), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 24);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 24)), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 6), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 20);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 20)), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 2), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 16), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 30);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 30)), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 12), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 26);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 26)), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 8), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 22);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 22)), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 4), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 18);
		_mm256_storeu_si256(out + 15, c14_rslt_m256i);


		c14_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c14_rslt_m256i = _mm256_and_si256( c14_load_rslt_m256i, c14_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 14), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 28);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 28)), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 10), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 24);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 24)), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 6), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 20);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 20)), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 2), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 16), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 30);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 30)), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 12), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 26);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 26)), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 8), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 22);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 22)), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 4), c14_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 18);
		_mm256_storeu_si256(out + 31, c14_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c14_and_mask_m256i = _mm256_set1_epi32(0x3fff);
		__m256i c14_load_rslt_m256i, c14_rslt_m256i;


		c14_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c14_rslt_m256i = _mm256_and_si256( c14_load_rslt_m256i, c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 14) );
		_mm256_storeu_si256(out + 0, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 14), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 14) );
		_mm256_storeu_si256(out + 1, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 28);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 28)), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 14) );
		_mm256_storeu_si256(out + 2, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 10), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 14) );
		_mm256_storeu_si256(out + 3, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 24);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 24)), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 14) );
		_mm256_storeu_si256(out + 4, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 6), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 14) );
		_mm256_storeu_si256(out + 5, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 20);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 20)), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 14) );
		_mm256_storeu_si256(out + 6, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 2), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 14) );
		_mm256_storeu_si256(out + 7, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 16), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 14) );
		_mm256_storeu_si256(out + 8, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 30);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 30)), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 14) );
		_mm256_storeu_si256(out + 9, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 12), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 14) );
		_mm256_storeu_si256(out + 10, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 26);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 26)), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 14) );
		_mm256_storeu_si256(out + 11, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 8), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 14) );
		_mm256_storeu_si256(out + 12, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 22);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 22)), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 14) );
		_mm256_storeu_si256(out + 13, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 4), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 14) );
		_mm256_storeu_si256(out + 14, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 18);
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 14) );
		_mm256_storeu_si256(out + 15, c14_rslt_m256i);


		c14_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c14_rslt_m256i = _mm256_and_si256( c14_load_rslt_m256i, c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 14) );
		_mm256_storeu_si256(out + 16, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 14), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 14) );
		_mm256_storeu_si256(out + 17, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 28);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 28)), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 14) );
		_mm256_storeu_si256(out + 18, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 10), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 14) );
		_mm256_storeu_si256(out + 19, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 24);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 24)), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 14) );
		_mm256_storeu_si256(out + 20, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 6), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 14) );
		_mm256_storeu_si256(out + 21, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 20);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 20)), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 14) );
		_mm256_storeu_si256(out + 22, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 2), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 14) );
		_mm256_storeu_si256(out + 23, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 16), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 14) );
		_mm256_storeu_si256(out + 24, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 30);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 30)), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 14) );
		_mm256_storeu_si256(out + 25, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 12), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 14) );
		_mm256_storeu_si256(out + 26, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 26);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 26)), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 14) );
		_mm256_storeu_si256(out + 27, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 8), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 14) );
		_mm256_storeu_si256(out + 28, c14_rslt_m256i);


		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 22);
		c14_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c14_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 32 - 22)), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 14) );
		_mm256_storeu_si256(out + 29, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c14_load_rslt_m256i, 4), c14_and_mask_m256i );
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 14) );
		_mm256_storeu_si256(out + 30, c14_rslt_m256i);

		c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 18);
		c14_rslt_m256i = _mm256_or_si256( c14_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 14) );
		_mm256_storeu_si256(out + 31, c14_rslt_m256i);
	}
}


// 15-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c15(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c15_and_mask_m256i = _mm256_set1_epi32(0x7fff);
		__m256i c15_load_rslt_m256i, c15_rslt_m256i;


		c15_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c15_rslt_m256i = _mm256_and_si256( c15_load_rslt_m256i, c15_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 15), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 30);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 30)), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 13), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 28);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 28)), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 11), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 26);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 26)), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 9), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 24);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 24)), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 7), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 22);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 22)), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 5), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 20);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 20)), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 3), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 18);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 18)), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 1), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 15, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 16), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 31);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 31)), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 14), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 29);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 29)), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 12), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 27);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 27)), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 10), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 25);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 25)), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 8), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 23);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 23)), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 6), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 21);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 21)), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 4), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 19);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 19)), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 2), c15_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 17);
		_mm256_storeu_si256(out + 31, c15_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c15_and_mask_m256i = _mm256_set1_epi32(0x7fff);
		__m256i c15_load_rslt_m256i, c15_rslt_m256i;


		c15_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c15_rslt_m256i = _mm256_and_si256( c15_load_rslt_m256i, c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 15) );
		_mm256_storeu_si256(out + 0, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 15), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 15) );
		_mm256_storeu_si256(out + 1, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 30);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 30)), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 15) );
		_mm256_storeu_si256(out + 2, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 13), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 15) );
		_mm256_storeu_si256(out + 3, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 28);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 28)), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 15) );
		_mm256_storeu_si256(out + 4, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 11), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 15) );
		_mm256_storeu_si256(out + 5, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 26);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 26)), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 15) );
		_mm256_storeu_si256(out + 6, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 9), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 15) );
		_mm256_storeu_si256(out + 7, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 24);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 24)), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 15) );
		_mm256_storeu_si256(out + 8, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 7), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 15) );
		_mm256_storeu_si256(out + 9, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 22);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 22)), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 15) );
		_mm256_storeu_si256(out + 10, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 5), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 15) );
		_mm256_storeu_si256(out + 11, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 20);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 20)), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 15) );
		_mm256_storeu_si256(out + 12, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 3), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 15) );
		_mm256_storeu_si256(out + 13, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 18);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 18)), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 15) );
		_mm256_storeu_si256(out + 14, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 1), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 15) );
		_mm256_storeu_si256(out + 15, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 16), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 15) );
		_mm256_storeu_si256(out + 16, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 31);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 31)), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 15) );
		_mm256_storeu_si256(out + 17, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 14), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 15) );
		_mm256_storeu_si256(out + 18, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 29);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 29)), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 15) );
		_mm256_storeu_si256(out + 19, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 12), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 15) );
		_mm256_storeu_si256(out + 20, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 27);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 27)), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 15) );
		_mm256_storeu_si256(out + 21, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 10), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 15) );
		_mm256_storeu_si256(out + 22, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 25);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 25)), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 15) );
		_mm256_storeu_si256(out + 23, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 8), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 15) );
		_mm256_storeu_si256(out + 24, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 23);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 23)), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 15) );
		_mm256_storeu_si256(out + 25, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 6), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 15) );
		_mm256_storeu_si256(out + 26, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 21);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 21)), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 15) );
		_mm256_storeu_si256(out + 27, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 4), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 15) );
		_mm256_storeu_si256(out + 28, c15_rslt_m256i);


		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 19);
		c15_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c15_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 32 - 19)), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 15) );
		_mm256_storeu_si256(out + 29, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c15_load_rslt_m256i, 2), c15_and_mask_m256i );
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 15) );
		_mm256_storeu_si256(out + 30, c15_rslt_m256i);

		c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 17);
		c15_rslt_m256i = _mm256_or_si256( c15_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 15) );
		_mm256_storeu_si256(out + 31, c15_rslt_m256i);
	}
}


// 16-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c16(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c16_and_mask_m256i = _mm256_set1_epi32(0xffff);
		__m256i c16_load_rslt_m256i, c16_rslt_m256i;


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		_mm256_storeu_si256(out + 1, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		_mm256_storeu_si256(out + 3, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		_mm256_storeu_si256(out + 5, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		_mm256_storeu_si256(out + 7, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		_mm256_storeu_si256(out + 9, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		_mm256_storeu_si256(out + 11, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		_mm256_storeu_si256(out + 13, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		_mm256_storeu_si256(out + 15, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		_mm256_storeu_si256(out + 17, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		_mm256_storeu_si256(out + 19, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		_mm256_storeu_si256(out + 21, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		_mm256_storeu_si256(out + 23, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		_mm256_storeu_si256(out + 25, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		_mm256_storeu_si256(out + 27, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		_mm256_storeu_si256(out + 29, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		_mm256_storeu_si256(out + 31, c16_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c16_and_mask_m256i = _mm256_set1_epi32(0xffff);
		__m256i c16_load_rslt_m256i, c16_rslt_m256i;


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 16) );
		_mm256_storeu_si256(out + 0, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 16) );
		_mm256_storeu_si256(out + 1, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 16) );
		_mm256_storeu_si256(out + 2, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 16) );
		_mm256_storeu_si256(out + 3, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 16) );
		_mm256_storeu_si256(out + 4, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 16) );
		_mm256_storeu_si256(out + 5, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 16) );
		_mm256_storeu_si256(out + 6, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 16) );
		_mm256_storeu_si256(out + 7, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 16) );
		_mm256_storeu_si256(out + 8, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 16) );
		_mm256_storeu_si256(out + 9, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 16) );
		_mm256_storeu_si256(out + 10, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 16) );
		_mm256_storeu_si256(out + 11, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 16) );
		_mm256_storeu_si256(out + 12, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 16) );
		_mm256_storeu_si256(out + 13, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 16) );
		_mm256_storeu_si256(out + 14, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 16) );
		_mm256_storeu_si256(out + 15, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 16) );
		_mm256_storeu_si256(out + 16, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 16) );
		_mm256_storeu_si256(out + 17, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 16) );
		_mm256_storeu_si256(out + 18, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 16) );
		_mm256_storeu_si256(out + 19, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 16) );
		_mm256_storeu_si256(out + 20, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 16) );
		_mm256_storeu_si256(out + 21, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 16) );
		_mm256_storeu_si256(out + 22, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 16) );
		_mm256_storeu_si256(out + 23, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 16) );
		_mm256_storeu_si256(out + 24, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 16) );
		_mm256_storeu_si256(out + 25, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 16) );
		_mm256_storeu_si256(out + 26, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 16) );
		_mm256_storeu_si256(out + 27, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 16) );
		_mm256_storeu_si256(out + 28, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 16) );
		_mm256_storeu_si256(out + 29, c16_rslt_m256i);


		c16_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c16_rslt_m256i = _mm256_and_si256( c16_load_rslt_m256i, c16_and_mask_m256i );
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 16) );
		_mm256_storeu_si256(out + 30, c16_rslt_m256i);

		c16_rslt_m256i = _mm256_srli_epi32(c16_load_rslt_m256i, 16);
		c16_rslt_m256i = _mm256_or_si256( c16_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 16) );
		_mm256_storeu_si256(out + 31, c16_rslt_m256i);
	}
}


// 17-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c17(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c17_and_mask_m256i = _mm256_set1_epi32(0x01ffff);
		__m256i c17_load_rslt_m256i, c17_rslt_m256i;


		c17_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c17_rslt_m256i = _mm256_and_si256( c17_load_rslt_m256i, c17_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 17);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 17)), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 2), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 19);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 19)), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 4), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 21);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 21)), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 6), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 23);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 23)), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 8), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 25);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 25)), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 10), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 27);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 27)), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 12), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 29);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 29)), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 14), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 31);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 31)), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 15, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 16);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 16)), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 1), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 18);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 18)), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 3), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 20);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 20)), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 5), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 22);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 22)), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 7), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 24);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 24)), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 9), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 26);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 26)), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 11), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 28);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 28)), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 13), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 30);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 30)), c17_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 15);
		_mm256_storeu_si256(out + 31, c17_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c17_and_mask_m256i = _mm256_set1_epi32(0x01ffff);
		__m256i c17_load_rslt_m256i, c17_rslt_m256i;


		c17_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c17_rslt_m256i = _mm256_and_si256( c17_load_rslt_m256i, c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 17) );
		_mm256_storeu_si256(out + 0, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 17);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 17)), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 17) );
		_mm256_storeu_si256(out + 1, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 2), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 17) );
		_mm256_storeu_si256(out + 2, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 19);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 19)), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 17) );
		_mm256_storeu_si256(out + 3, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 4), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 17) );
		_mm256_storeu_si256(out + 4, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 21);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 21)), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 17) );
		_mm256_storeu_si256(out + 5, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 6), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 17) );
		_mm256_storeu_si256(out + 6, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 23);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 23)), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 17) );
		_mm256_storeu_si256(out + 7, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 8), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 17) );
		_mm256_storeu_si256(out + 8, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 25);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 25)), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 17) );
		_mm256_storeu_si256(out + 9, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 10), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 17) );
		_mm256_storeu_si256(out + 10, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 27);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 27)), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 17) );
		_mm256_storeu_si256(out + 11, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 12), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 17) );
		_mm256_storeu_si256(out + 12, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 29);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 29)), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 17) );
		_mm256_storeu_si256(out + 13, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 14), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 17) );
		_mm256_storeu_si256(out + 14, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 31);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 31)), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 17) );
		_mm256_storeu_si256(out + 15, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 16);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 16)), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 17) );
		_mm256_storeu_si256(out + 16, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 1), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 17) );
		_mm256_storeu_si256(out + 17, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 18);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 18)), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 17) );
		_mm256_storeu_si256(out + 18, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 3), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 17) );
		_mm256_storeu_si256(out + 19, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 20);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 20)), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 17) );
		_mm256_storeu_si256(out + 20, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 5), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 17) );
		_mm256_storeu_si256(out + 21, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 22);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 22)), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 17) );
		_mm256_storeu_si256(out + 22, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 7), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 17) );
		_mm256_storeu_si256(out + 23, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 24);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 24)), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 17) );
		_mm256_storeu_si256(out + 24, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 9), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 17) );
		_mm256_storeu_si256(out + 25, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 26);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 26)), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 17) );
		_mm256_storeu_si256(out + 26, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 11), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 17) );
		_mm256_storeu_si256(out + 27, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 28);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 28)), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 17) );
		_mm256_storeu_si256(out + 28, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c17_load_rslt_m256i, 13), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 17) );
		_mm256_storeu_si256(out + 29, c17_rslt_m256i);


		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 30);
		c17_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c17_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 32 - 30)), c17_and_mask_m256i );
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 17) );
		_mm256_storeu_si256(out + 30, c17_rslt_m256i);

		c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 15);
		c17_rslt_m256i = _mm256_or_si256( c17_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 17) );
		_mm256_storeu_si256(out + 31, c17_rslt_m256i);
	}
}


// 18-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c18(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c18_and_mask_m256i = _mm256_set1_epi32(0x03ffff);
		__m256i c18_load_rslt_m256i, c18_rslt_m256i;


		c18_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c18_rslt_m256i = _mm256_and_si256( c18_load_rslt_m256i, c18_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 18);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 18)), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 4), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 22);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 22)), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 8), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 26);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 26)), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 12), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 30);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 30)), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 16);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 16)), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 2), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 20);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 20)), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 6), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 24);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 24)), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 10), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 28);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 28)), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 14);
		_mm256_storeu_si256(out + 15, c18_rslt_m256i);


		c18_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c18_rslt_m256i = _mm256_and_si256( c18_load_rslt_m256i, c18_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 18);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 18)), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 4), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 22);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 22)), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 8), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 26);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 26)), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 12), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 30);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 30)), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 16);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 16)), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 2), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 20);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 20)), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 6), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 24);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 24)), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 10), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 28);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 28)), c18_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 14);
		_mm256_storeu_si256(out + 31, c18_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c18_and_mask_m256i = _mm256_set1_epi32(0x03ffff);
		__m256i c18_load_rslt_m256i, c18_rslt_m256i;


		c18_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c18_rslt_m256i = _mm256_and_si256( c18_load_rslt_m256i, c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 18) );
		_mm256_storeu_si256(out + 0, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 18);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 18)), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 18) );
		_mm256_storeu_si256(out + 1, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 4), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 18) );
		_mm256_storeu_si256(out + 2, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 22);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 22)), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 18) );
		_mm256_storeu_si256(out + 3, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 8), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 18) );
		_mm256_storeu_si256(out + 4, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 26);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 26)), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 18) );
		_mm256_storeu_si256(out + 5, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 12), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 18) );
		_mm256_storeu_si256(out + 6, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 30);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 30)), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 18) );
		_mm256_storeu_si256(out + 7, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 16);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 16)), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 18) );
		_mm256_storeu_si256(out + 8, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 2), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 18) );
		_mm256_storeu_si256(out + 9, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 20);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 20)), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 18) );
		_mm256_storeu_si256(out + 10, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 6), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 18) );
		_mm256_storeu_si256(out + 11, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 24);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 24)), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 18) );
		_mm256_storeu_si256(out + 12, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 10), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 18) );
		_mm256_storeu_si256(out + 13, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 28);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 28)), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 18) );
		_mm256_storeu_si256(out + 14, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 14);
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 18) );
		_mm256_storeu_si256(out + 15, c18_rslt_m256i);


		c18_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c18_rslt_m256i = _mm256_and_si256( c18_load_rslt_m256i, c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 18) );
		_mm256_storeu_si256(out + 16, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 18);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 18)), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 18) );
		_mm256_storeu_si256(out + 17, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 4), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 18) );
		_mm256_storeu_si256(out + 18, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 22);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 22)), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 18) );
		_mm256_storeu_si256(out + 19, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 8), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 18) );
		_mm256_storeu_si256(out + 20, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 26);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 26)), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 18) );
		_mm256_storeu_si256(out + 21, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 12), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 18) );
		_mm256_storeu_si256(out + 22, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 30);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 30)), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 18) );
		_mm256_storeu_si256(out + 23, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 16);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 16)), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 18) );
		_mm256_storeu_si256(out + 24, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 2), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 18) );
		_mm256_storeu_si256(out + 25, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 20);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 20)), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 18) );
		_mm256_storeu_si256(out + 26, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 6), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 18) );
		_mm256_storeu_si256(out + 27, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 24);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 24)), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 18) );
		_mm256_storeu_si256(out + 28, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c18_load_rslt_m256i, 10), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 18) );
		_mm256_storeu_si256(out + 29, c18_rslt_m256i);


		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 28);
		c18_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c18_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 32 - 28)), c18_and_mask_m256i );
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 18) );
		_mm256_storeu_si256(out + 30, c18_rslt_m256i);

		c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 14);
		c18_rslt_m256i = _mm256_or_si256( c18_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 18) );
		_mm256_storeu_si256(out + 31, c18_rslt_m256i);
	}
}


// 19-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c19(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c19_and_mask_m256i = _mm256_set1_epi32(0x07ffff);
		__m256i c19_load_rslt_m256i, c19_rslt_m256i;


		c19_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c19_rslt_m256i = _mm256_and_si256( c19_load_rslt_m256i, c19_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 19);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 19)), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 6), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 25);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 25)), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 12), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 31);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 31)), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 18);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 18)), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 5), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 24);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 24)), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 11), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 30);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 30)), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 17);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 17)), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 4), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 23);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 23)), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 10), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 29);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 29)), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 15, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 16);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 16)), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 3), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 22);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 22)), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 9), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 28);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 28)), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 15);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 15)), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 2), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 21);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 21)), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 8), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 27);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 27)), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 14);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 14)), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 1), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 20);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 20)), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 7), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 26);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 26)), c19_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 13);
		_mm256_storeu_si256(out + 31, c19_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c19_and_mask_m256i = _mm256_set1_epi32(0x07ffff);
		__m256i c19_load_rslt_m256i, c19_rslt_m256i;


		c19_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c19_rslt_m256i = _mm256_and_si256( c19_load_rslt_m256i, c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 19) );
		_mm256_storeu_si256(out + 0, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 19);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 19)), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 19) );
		_mm256_storeu_si256(out + 1, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 6), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 19) );
		_mm256_storeu_si256(out + 2, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 25);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 25)), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 19) );
		_mm256_storeu_si256(out + 3, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 12), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 19) );
		_mm256_storeu_si256(out + 4, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 31);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 31)), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 19) );
		_mm256_storeu_si256(out + 5, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 18);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 18)), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 19) );
		_mm256_storeu_si256(out + 6, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 5), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 19) );
		_mm256_storeu_si256(out + 7, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 24);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 24)), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 19) );
		_mm256_storeu_si256(out + 8, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 11), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 19) );
		_mm256_storeu_si256(out + 9, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 30);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 30)), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 19) );
		_mm256_storeu_si256(out + 10, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 17);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 17)), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 19) );
		_mm256_storeu_si256(out + 11, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 4), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 19) );
		_mm256_storeu_si256(out + 12, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 23);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 23)), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 19) );
		_mm256_storeu_si256(out + 13, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 10), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 19) );
		_mm256_storeu_si256(out + 14, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 29);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 29)), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 19) );
		_mm256_storeu_si256(out + 15, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 16);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 16)), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 19) );
		_mm256_storeu_si256(out + 16, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 3), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 19) );
		_mm256_storeu_si256(out + 17, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 22);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 22)), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 19) );
		_mm256_storeu_si256(out + 18, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 9), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 19) );
		_mm256_storeu_si256(out + 19, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 28);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 28)), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 19) );
		_mm256_storeu_si256(out + 20, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 15);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 15)), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 19) );
		_mm256_storeu_si256(out + 21, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 2), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 19) );
		_mm256_storeu_si256(out + 22, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 21);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 21)), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 19) );
		_mm256_storeu_si256(out + 23, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 8), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 19) );
		_mm256_storeu_si256(out + 24, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 27);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 27)), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 19) );
		_mm256_storeu_si256(out + 25, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 14);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 14)), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 19) );
		_mm256_storeu_si256(out + 26, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 1), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 19) );
		_mm256_storeu_si256(out + 27, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 20);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 20)), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 19) );
		_mm256_storeu_si256(out + 28, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c19_load_rslt_m256i, 7), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 19) );
		_mm256_storeu_si256(out + 29, c19_rslt_m256i);


		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 26);
		c19_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c19_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 32 - 26)), c19_and_mask_m256i );
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 19) );
		_mm256_storeu_si256(out + 30, c19_rslt_m256i);

		c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 13);
		c19_rslt_m256i = _mm256_or_si256( c19_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 19) );
		_mm256_storeu_si256(out + 31, c19_rslt_m256i);
	}
}


// 20-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c20(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c20_and_mask_m256i = _mm256_set1_epi32(0x0fffff);
		__m256i c20_load_rslt_m256i, c20_rslt_m256i;


		c20_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c20_rslt_m256i = _mm256_and_si256( c20_load_rslt_m256i, c20_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 20);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 20)), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c20_load_rslt_m256i, 8), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 28);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 28)), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 16);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 16)), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c20_load_rslt_m256i, 4), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 24);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 24)), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 12);
		_mm256_storeu_si256(out + 7, c20_rslt_m256i);


		c20_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c20_rslt_m256i = _mm256_and_si256( c20_load_rslt_m256i, c20_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 20);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 20)), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c20_load_rslt_m256i, 8), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 28);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 28)), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 16);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 16)), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c20_load_rslt_m256i, 4), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 24);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 24)), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 12);
		_mm256_storeu_si256(out + 15, c20_rslt_m256i);


		c20_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c20_rslt_m256i = _mm256_and_si256( c20_load_rslt_m256i, c20_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 20);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 20)), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c20_load_rslt_m256i, 8), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 28);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 28)), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 16);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 16)), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c20_load_rslt_m256i, 4), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 24);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 24)), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 12);
		_mm256_storeu_si256(out + 23, c20_rslt_m256i);


		c20_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c20_rslt_m256i = _mm256_and_si256( c20_load_rslt_m256i, c20_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 20);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 20)), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c20_load_rslt_m256i, 8), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 28);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 28)), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 16);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 16)), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c20_load_rslt_m256i, 4), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 24);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 24)), c20_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 12);
		_mm256_storeu_si256(out + 31, c20_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c20_and_mask_m256i = _mm256_set1_epi32(0x0fffff);
		__m256i c20_load_rslt_m256i, c20_rslt_m256i;


		c20_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c20_rslt_m256i = _mm256_and_si256( c20_load_rslt_m256i, c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 20) );
		_mm256_storeu_si256(out + 0, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 20);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 20)), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 20) );
		_mm256_storeu_si256(out + 1, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c20_load_rslt_m256i, 8), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 20) );
		_mm256_storeu_si256(out + 2, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 28);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 28)), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 20) );
		_mm256_storeu_si256(out + 3, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 16);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 16)), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 20) );
		_mm256_storeu_si256(out + 4, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c20_load_rslt_m256i, 4), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 20) );
		_mm256_storeu_si256(out + 5, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 24);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 24)), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 20) );
		_mm256_storeu_si256(out + 6, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 12);
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 20) );
		_mm256_storeu_si256(out + 7, c20_rslt_m256i);


		c20_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c20_rslt_m256i = _mm256_and_si256( c20_load_rslt_m256i, c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 20) );
		_mm256_storeu_si256(out + 8, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 20);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 20)), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 20) );
		_mm256_storeu_si256(out + 9, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c20_load_rslt_m256i, 8), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 20) );
		_mm256_storeu_si256(out + 10, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 28);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 28)), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 20) );
		_mm256_storeu_si256(out + 11, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 16);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 16)), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 20) );
		_mm256_storeu_si256(out + 12, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c20_load_rslt_m256i, 4), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 20) );
		_mm256_storeu_si256(out + 13, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 24);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 24)), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 20) );
		_mm256_storeu_si256(out + 14, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 12);
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 20) );
		_mm256_storeu_si256(out + 15, c20_rslt_m256i);


		c20_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c20_rslt_m256i = _mm256_and_si256( c20_load_rslt_m256i, c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 20) );
		_mm256_storeu_si256(out + 16, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 20);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 20)), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 20) );
		_mm256_storeu_si256(out + 17, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c20_load_rslt_m256i, 8), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 20) );
		_mm256_storeu_si256(out + 18, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 28);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 28)), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 20) );
		_mm256_storeu_si256(out + 19, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 16);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 16)), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 20) );
		_mm256_storeu_si256(out + 20, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c20_load_rslt_m256i, 4), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 20) );
		_mm256_storeu_si256(out + 21, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 24);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 24)), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 20) );
		_mm256_storeu_si256(out + 22, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 12);
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 20) );
		_mm256_storeu_si256(out + 23, c20_rslt_m256i);


		c20_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c20_rslt_m256i = _mm256_and_si256( c20_load_rslt_m256i, c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 20) );
		_mm256_storeu_si256(out + 24, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 20);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 20)), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 20) );
		_mm256_storeu_si256(out + 25, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c20_load_rslt_m256i, 8), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 20) );
		_mm256_storeu_si256(out + 26, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 28);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 28)), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 20) );
		_mm256_storeu_si256(out + 27, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 16);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 16)), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 20) );
		_mm256_storeu_si256(out + 28, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c20_load_rslt_m256i, 4), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 20) );
		_mm256_storeu_si256(out + 29, c20_rslt_m256i);


		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 24);
		c20_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c20_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 32 - 24)), c20_and_mask_m256i );
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 20) );
		_mm256_storeu_si256(out + 30, c20_rslt_m256i);

		c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 12);
		c20_rslt_m256i = _mm256_or_si256( c20_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 20) );
		_mm256_storeu_si256(out + 31, c20_rslt_m256i);
	}
}


// 21-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c21(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c21_and_mask_m256i = _mm256_set1_epi32(0x1fffff);
		__m256i c21_load_rslt_m256i, c21_rslt_m256i;


		c21_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c21_rslt_m256i = _mm256_and_si256( c21_load_rslt_m256i, c21_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 21);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 21)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 10), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 31);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 31)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 20);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 20)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 9), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 30);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 30)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 19);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 19)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 8), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 29);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 29)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 18);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 18)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 7), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 28);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 28)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 17);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 17)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 6), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 27);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 27)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 15, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 16);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 16)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 5), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 26);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 26)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 15);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 15)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 4), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 25);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 25)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 14);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 14)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 3), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 24);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 24)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 13);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 13)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 2), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 23);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 23)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 12);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 12)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 1), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 22);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 22)), c21_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 11);
		_mm256_storeu_si256(out + 31, c21_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c21_and_mask_m256i = _mm256_set1_epi32(0x1fffff);
		__m256i c21_load_rslt_m256i, c21_rslt_m256i;


		c21_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c21_rslt_m256i = _mm256_and_si256( c21_load_rslt_m256i, c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 21) );
		_mm256_storeu_si256(out + 0, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 21);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 21)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 21) );
		_mm256_storeu_si256(out + 1, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 10), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 21) );
		_mm256_storeu_si256(out + 2, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 31);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 31)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 21) );
		_mm256_storeu_si256(out + 3, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 20);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 20)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 21) );
		_mm256_storeu_si256(out + 4, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 9), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 21) );
		_mm256_storeu_si256(out + 5, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 30);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 30)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 21) );
		_mm256_storeu_si256(out + 6, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 19);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 19)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 21) );
		_mm256_storeu_si256(out + 7, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 8), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 21) );
		_mm256_storeu_si256(out + 8, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 29);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 29)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 21) );
		_mm256_storeu_si256(out + 9, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 18);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 18)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 21) );
		_mm256_storeu_si256(out + 10, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 7), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 21) );
		_mm256_storeu_si256(out + 11, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 28);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 28)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 21) );
		_mm256_storeu_si256(out + 12, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 17);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 17)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 21) );
		_mm256_storeu_si256(out + 13, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 6), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 21) );
		_mm256_storeu_si256(out + 14, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 27);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 27)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 21) );
		_mm256_storeu_si256(out + 15, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 16);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 16)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 21) );
		_mm256_storeu_si256(out + 16, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 5), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 21) );
		_mm256_storeu_si256(out + 17, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 26);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 26)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 21) );
		_mm256_storeu_si256(out + 18, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 15);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 15)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 21) );
		_mm256_storeu_si256(out + 19, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 4), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 21) );
		_mm256_storeu_si256(out + 20, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 25);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 25)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 21) );
		_mm256_storeu_si256(out + 21, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 14);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 14)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 21) );
		_mm256_storeu_si256(out + 22, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 3), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 21) );
		_mm256_storeu_si256(out + 23, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 24);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 24)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 21) );
		_mm256_storeu_si256(out + 24, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 13);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 13)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 21) );
		_mm256_storeu_si256(out + 25, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 2), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 21) );
		_mm256_storeu_si256(out + 26, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 23);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 23)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 21) );
		_mm256_storeu_si256(out + 27, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 12);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 12)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 21) );
		_mm256_storeu_si256(out + 28, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c21_load_rslt_m256i, 1), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 21) );
		_mm256_storeu_si256(out + 29, c21_rslt_m256i);


		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 22);
		c21_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c21_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 32 - 22)), c21_and_mask_m256i );
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 21) );
		_mm256_storeu_si256(out + 30, c21_rslt_m256i);

		c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 11);
		c21_rslt_m256i = _mm256_or_si256( c21_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 21) );
		_mm256_storeu_si256(out + 31, c21_rslt_m256i);
	}
}


// 22-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c22(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c22_and_mask_m256i = _mm256_set1_epi32(0x3fffff);
		__m256i c22_load_rslt_m256i, c22_rslt_m256i;


		c22_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c22_rslt_m256i = _mm256_and_si256( c22_load_rslt_m256i, c22_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 22);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 22)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 12);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 12)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c22_load_rslt_m256i, 2), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 24);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 24)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 14);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 14)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c22_load_rslt_m256i, 4), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 26);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 26)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 16);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 16)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c22_load_rslt_m256i, 6), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 28);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 28)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 18);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 18)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c22_load_rslt_m256i, 8), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 30);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 30)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 20);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 20)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 10);
		_mm256_storeu_si256(out + 15, c22_rslt_m256i);


		c22_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c22_rslt_m256i = _mm256_and_si256( c22_load_rslt_m256i, c22_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 22);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 22)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 12);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 12)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c22_load_rslt_m256i, 2), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 24);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 24)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 14);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 14)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c22_load_rslt_m256i, 4), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 26);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 26)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 16);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 16)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c22_load_rslt_m256i, 6), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 28);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 28)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 18);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 18)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c22_load_rslt_m256i, 8), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 30);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 30)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 20);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 20)), c22_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 10);
		_mm256_storeu_si256(out + 31, c22_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c22_and_mask_m256i = _mm256_set1_epi32(0x3fffff);
		__m256i c22_load_rslt_m256i, c22_rslt_m256i;


		c22_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c22_rslt_m256i = _mm256_and_si256( c22_load_rslt_m256i, c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 22) );
		_mm256_storeu_si256(out + 0, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 22);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 22)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 22) );
		_mm256_storeu_si256(out + 1, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 12);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 12)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 22) );
		_mm256_storeu_si256(out + 2, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c22_load_rslt_m256i, 2), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 22) );
		_mm256_storeu_si256(out + 3, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 24);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 24)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 22) );
		_mm256_storeu_si256(out + 4, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 14);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 14)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 22) );
		_mm256_storeu_si256(out + 5, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c22_load_rslt_m256i, 4), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 22) );
		_mm256_storeu_si256(out + 6, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 26);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 26)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 22) );
		_mm256_storeu_si256(out + 7, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 16);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 16)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 22) );
		_mm256_storeu_si256(out + 8, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c22_load_rslt_m256i, 6), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 22) );
		_mm256_storeu_si256(out + 9, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 28);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 28)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 22) );
		_mm256_storeu_si256(out + 10, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 18);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 18)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 22) );
		_mm256_storeu_si256(out + 11, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c22_load_rslt_m256i, 8), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 22) );
		_mm256_storeu_si256(out + 12, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 30);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 30)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 22) );
		_mm256_storeu_si256(out + 13, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 20);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 20)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 22) );
		_mm256_storeu_si256(out + 14, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 10);
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 22) );
		_mm256_storeu_si256(out + 15, c22_rslt_m256i);


		c22_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c22_rslt_m256i = _mm256_and_si256( c22_load_rslt_m256i, c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 22) );
		_mm256_storeu_si256(out + 16, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 22);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 22)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 22) );
		_mm256_storeu_si256(out + 17, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 12);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 12)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 22) );
		_mm256_storeu_si256(out + 18, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c22_load_rslt_m256i, 2), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 22) );
		_mm256_storeu_si256(out + 19, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 24);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 24)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 22) );
		_mm256_storeu_si256(out + 20, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 14);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 14)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 22) );
		_mm256_storeu_si256(out + 21, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c22_load_rslt_m256i, 4), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 22) );
		_mm256_storeu_si256(out + 22, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 26);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 26)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 22) );
		_mm256_storeu_si256(out + 23, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 16);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 16)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 22) );
		_mm256_storeu_si256(out + 24, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c22_load_rslt_m256i, 6), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 22) );
		_mm256_storeu_si256(out + 25, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 28);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 28)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 22) );
		_mm256_storeu_si256(out + 26, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 18);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 18)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 22) );
		_mm256_storeu_si256(out + 27, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c22_load_rslt_m256i, 8), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 22) );
		_mm256_storeu_si256(out + 28, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 30);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 30)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 22) );
		_mm256_storeu_si256(out + 29, c22_rslt_m256i);


		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 20);
		c22_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c22_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 32 - 20)), c22_and_mask_m256i );
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 22) );
		_mm256_storeu_si256(out + 30, c22_rslt_m256i);

		c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 10);
		c22_rslt_m256i = _mm256_or_si256( c22_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 22) );
		_mm256_storeu_si256(out + 31, c22_rslt_m256i);
	}
}


// 23-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c23(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c23_and_mask_m256i = _mm256_set1_epi32(0x7fffff);
		__m256i c23_load_rslt_m256i, c23_rslt_m256i;


		c23_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c23_rslt_m256i = _mm256_and_si256( c23_load_rslt_m256i, c23_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 23);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 23)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 14);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 14)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c23_rslt_m256i);

		c23_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c23_load_rslt_m256i, 5), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 28);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 28)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 19);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 19)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 10);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 10)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c23_rslt_m256i);

		c23_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c23_load_rslt_m256i, 1), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 24);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 24)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 15);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 15)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c23_rslt_m256i);

		c23_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c23_load_rslt_m256i, 6), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 29);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 29)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 20);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 20)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 11);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 11)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c23_rslt_m256i);

		c23_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c23_load_rslt_m256i, 2), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 25);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 25)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 15, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 16);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 16)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c23_rslt_m256i);

		c23_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c23_load_rslt_m256i, 7), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 30);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 30)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 21);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 21)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 12);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 12)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c23_rslt_m256i);

		c23_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c23_load_rslt_m256i, 3), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 26);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 26)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 17);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 17)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c23_rslt_m256i);

		c23_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c23_load_rslt_m256i, 8), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 31);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 31)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 22);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 22)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 13);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 13)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c23_rslt_m256i);

		c23_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c23_load_rslt_m256i, 4), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 27);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 27)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 18);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 22);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 18)), c23_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c23_rslt_m256i);

		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 9);
		_mm256_storeu_si256(out + 31, c23_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c23_and_mask_m256i = _mm256_set1_epi32(0x7fffff);
		__m256i c23_load_rslt_m256i, c23_rslt_m256i;


		c23_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c23_rslt_m256i = _mm256_and_si256( c23_load_rslt_m256i, c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 23) );
		_mm256_storeu_si256(out + 0, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 23);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 23)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 23) );
		_mm256_storeu_si256(out + 1, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 14);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 14)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 23) );
		_mm256_storeu_si256(out + 2, c23_rslt_m256i);

		c23_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c23_load_rslt_m256i, 5), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 23) );
		_mm256_storeu_si256(out + 3, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 28);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 28)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 23) );
		_mm256_storeu_si256(out + 4, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 19);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 19)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 23) );
		_mm256_storeu_si256(out + 5, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 10);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 10)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 23) );
		_mm256_storeu_si256(out + 6, c23_rslt_m256i);

		c23_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c23_load_rslt_m256i, 1), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 23) );
		_mm256_storeu_si256(out + 7, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 24);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 24)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 23) );
		_mm256_storeu_si256(out + 8, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 15);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 15)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 23) );
		_mm256_storeu_si256(out + 9, c23_rslt_m256i);

		c23_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c23_load_rslt_m256i, 6), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 23) );
		_mm256_storeu_si256(out + 10, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 29);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 29)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 23) );
		_mm256_storeu_si256(out + 11, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 20);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 20)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 23) );
		_mm256_storeu_si256(out + 12, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 11);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 11)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 23) );
		_mm256_storeu_si256(out + 13, c23_rslt_m256i);

		c23_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c23_load_rslt_m256i, 2), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 23) );
		_mm256_storeu_si256(out + 14, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 25);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 25)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 23) );
		_mm256_storeu_si256(out + 15, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 16);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 16)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 23) );
		_mm256_storeu_si256(out + 16, c23_rslt_m256i);

		c23_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c23_load_rslt_m256i, 7), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 23) );
		_mm256_storeu_si256(out + 17, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 30);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 30)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 23) );
		_mm256_storeu_si256(out + 18, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 21);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 21)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 23) );
		_mm256_storeu_si256(out + 19, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 12);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 12)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 23) );
		_mm256_storeu_si256(out + 20, c23_rslt_m256i);

		c23_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c23_load_rslt_m256i, 3), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 23) );
		_mm256_storeu_si256(out + 21, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 26);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 26)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 23) );
		_mm256_storeu_si256(out + 22, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 17);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 17)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 23) );
		_mm256_storeu_si256(out + 23, c23_rslt_m256i);

		c23_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c23_load_rslt_m256i, 8), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 23) );
		_mm256_storeu_si256(out + 24, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 31);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 31)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 23) );
		_mm256_storeu_si256(out + 25, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 22);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 22)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 23) );
		_mm256_storeu_si256(out + 26, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 13);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 13)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 23) );
		_mm256_storeu_si256(out + 27, c23_rslt_m256i);

		c23_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c23_load_rslt_m256i, 4), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 23) );
		_mm256_storeu_si256(out + 28, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 27);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 27)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 23) );
		_mm256_storeu_si256(out + 29, c23_rslt_m256i);


		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 18);
		c23_load_rslt_m256i = _mm256_loadu_si256(in + 22);

		c23_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 32 - 18)), c23_and_mask_m256i );
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 23) );
		_mm256_storeu_si256(out + 30, c23_rslt_m256i);

		c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 9);
		c23_rslt_m256i = _mm256_or_si256( c23_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 23) );
		_mm256_storeu_si256(out + 31, c23_rslt_m256i);
	}
}


// 24-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c24(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c24_and_mask_m256i = _mm256_set1_epi32(0xffffff);
		__m256i c24_load_rslt_m256i, c24_rslt_m256i;


		c24_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c24_rslt_m256i = _mm256_and_si256( c24_load_rslt_m256i, c24_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 24);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 24)), c24_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 16);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 16)), c24_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c24_rslt_m256i);

		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 8);
		_mm256_storeu_si256(out + 3, c24_rslt_m256i);


		c24_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c24_rslt_m256i = _mm256_and_si256( c24_load_rslt_m256i, c24_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 24);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 24)), c24_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 16);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 16)), c24_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c24_rslt_m256i);

		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 8);
		_mm256_storeu_si256(out + 7, c24_rslt_m256i);


		c24_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c24_rslt_m256i = _mm256_and_si256( c24_load_rslt_m256i, c24_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 24);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 24)), c24_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 16);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 16)), c24_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c24_rslt_m256i);

		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 8);
		_mm256_storeu_si256(out + 11, c24_rslt_m256i);


		c24_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c24_rslt_m256i = _mm256_and_si256( c24_load_rslt_m256i, c24_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 24);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 24)), c24_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 16);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 16)), c24_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c24_rslt_m256i);

		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 8);
		_mm256_storeu_si256(out + 15, c24_rslt_m256i);


		c24_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c24_rslt_m256i = _mm256_and_si256( c24_load_rslt_m256i, c24_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 24);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 24)), c24_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 16);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 16)), c24_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c24_rslt_m256i);

		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 8);
		_mm256_storeu_si256(out + 19, c24_rslt_m256i);


		c24_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c24_rslt_m256i = _mm256_and_si256( c24_load_rslt_m256i, c24_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 24);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 24)), c24_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 16);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 16)), c24_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c24_rslt_m256i);

		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 8);
		_mm256_storeu_si256(out + 23, c24_rslt_m256i);


		c24_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c24_rslt_m256i = _mm256_and_si256( c24_load_rslt_m256i, c24_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 24);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 24)), c24_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 16);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 16)), c24_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c24_rslt_m256i);

		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 8);
		_mm256_storeu_si256(out + 27, c24_rslt_m256i);


		c24_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c24_rslt_m256i = _mm256_and_si256( c24_load_rslt_m256i, c24_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 24);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 22);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 24)), c24_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 16);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 23);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 16)), c24_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c24_rslt_m256i);

		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 8);
		_mm256_storeu_si256(out + 31, c24_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c24_and_mask_m256i = _mm256_set1_epi32(0xffffff);
		__m256i c24_load_rslt_m256i, c24_rslt_m256i;


		c24_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c24_rslt_m256i = _mm256_and_si256( c24_load_rslt_m256i, c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 24) );
		_mm256_storeu_si256(out + 0, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 24);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 24)), c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 24) );
		_mm256_storeu_si256(out + 1, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 16);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 16)), c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 24) );
		_mm256_storeu_si256(out + 2, c24_rslt_m256i);

		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 8);
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 24) );
		_mm256_storeu_si256(out + 3, c24_rslt_m256i);


		c24_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c24_rslt_m256i = _mm256_and_si256( c24_load_rslt_m256i, c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 24) );
		_mm256_storeu_si256(out + 4, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 24);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 24)), c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 24) );
		_mm256_storeu_si256(out + 5, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 16);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 16)), c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 24) );
		_mm256_storeu_si256(out + 6, c24_rslt_m256i);

		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 8);
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 24) );
		_mm256_storeu_si256(out + 7, c24_rslt_m256i);


		c24_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c24_rslt_m256i = _mm256_and_si256( c24_load_rslt_m256i, c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 24) );
		_mm256_storeu_si256(out + 8, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 24);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 24)), c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 24) );
		_mm256_storeu_si256(out + 9, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 16);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 16)), c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 24) );
		_mm256_storeu_si256(out + 10, c24_rslt_m256i);

		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 8);
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 24) );
		_mm256_storeu_si256(out + 11, c24_rslt_m256i);


		c24_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c24_rslt_m256i = _mm256_and_si256( c24_load_rslt_m256i, c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 24) );
		_mm256_storeu_si256(out + 12, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 24);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 24)), c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 24) );
		_mm256_storeu_si256(out + 13, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 16);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 16)), c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 24) );
		_mm256_storeu_si256(out + 14, c24_rslt_m256i);

		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 8);
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 24) );
		_mm256_storeu_si256(out + 15, c24_rslt_m256i);


		c24_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c24_rslt_m256i = _mm256_and_si256( c24_load_rslt_m256i, c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 24) );
		_mm256_storeu_si256(out + 16, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 24);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 24)), c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 24) );
		_mm256_storeu_si256(out + 17, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 16);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 16)), c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 24) );
		_mm256_storeu_si256(out + 18, c24_rslt_m256i);

		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 8);
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 24) );
		_mm256_storeu_si256(out + 19, c24_rslt_m256i);


		c24_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c24_rslt_m256i = _mm256_and_si256( c24_load_rslt_m256i, c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 24) );
		_mm256_storeu_si256(out + 20, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 24);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 24)), c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 24) );
		_mm256_storeu_si256(out + 21, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 16);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 16)), c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 24) );
		_mm256_storeu_si256(out + 22, c24_rslt_m256i);

		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 8);
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 24) );
		_mm256_storeu_si256(out + 23, c24_rslt_m256i);


		c24_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c24_rslt_m256i = _mm256_and_si256( c24_load_rslt_m256i, c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 24) );
		_mm256_storeu_si256(out + 24, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 24);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 24)), c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 24) );
		_mm256_storeu_si256(out + 25, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 16);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 16)), c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 24) );
		_mm256_storeu_si256(out + 26, c24_rslt_m256i);

		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 8);
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 24) );
		_mm256_storeu_si256(out + 27, c24_rslt_m256i);


		c24_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c24_rslt_m256i = _mm256_and_si256( c24_load_rslt_m256i, c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 24) );
		_mm256_storeu_si256(out + 28, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 24);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 22);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 24)), c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 24) );
		_mm256_storeu_si256(out + 29, c24_rslt_m256i);


		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 16);
		c24_load_rslt_m256i = _mm256_loadu_si256(in + 23);

		c24_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 32 - 16)), c24_and_mask_m256i );
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 24) );
		_mm256_storeu_si256(out + 30, c24_rslt_m256i);

		c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 8);
		c24_rslt_m256i = _mm256_or_si256( c24_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 24) );
		_mm256_storeu_si256(out + 31, c24_rslt_m256i);
	}
}


// 25-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c25(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c25_and_mask_m256i = _mm256_set1_epi32(0x01ffffff);
		__m256i c25_load_rslt_m256i, c25_rslt_m256i;


		c25_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c25_rslt_m256i = _mm256_and_si256( c25_load_rslt_m256i, c25_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 25);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 25)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 18);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 18)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 11);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 11)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c25_rslt_m256i);

		c25_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c25_load_rslt_m256i, 4), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 29);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 29)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 22);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 22)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 15);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 15)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 8);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 8)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c25_rslt_m256i);

		c25_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c25_load_rslt_m256i, 1), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 26);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 26)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 19);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 19)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 12);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 12)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c25_rslt_m256i);

		c25_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c25_load_rslt_m256i, 5), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 30);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 30)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 23);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 23)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 15, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 16);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 16)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 9);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 9)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c25_rslt_m256i);

		c25_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c25_load_rslt_m256i, 2), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 27);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 27)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 20);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 20)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 13);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 13)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c25_rslt_m256i);

		c25_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c25_load_rslt_m256i, 6), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 31);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 31)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 24);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 24)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 17);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 17)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 10);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 10)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c25_rslt_m256i);

		c25_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c25_load_rslt_m256i, 3), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 28);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 22);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 28)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 21);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 23);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 21)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 14);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 24);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 14)), c25_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c25_rslt_m256i);

		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 7);
		_mm256_storeu_si256(out + 31, c25_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c25_and_mask_m256i = _mm256_set1_epi32(0x01ffffff);
		__m256i c25_load_rslt_m256i, c25_rslt_m256i;


		c25_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c25_rslt_m256i = _mm256_and_si256( c25_load_rslt_m256i, c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 25) );
		_mm256_storeu_si256(out + 0, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 25);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 25)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 25) );
		_mm256_storeu_si256(out + 1, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 18);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 18)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 25) );
		_mm256_storeu_si256(out + 2, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 11);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 11)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 25) );
		_mm256_storeu_si256(out + 3, c25_rslt_m256i);

		c25_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c25_load_rslt_m256i, 4), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 25) );
		_mm256_storeu_si256(out + 4, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 29);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 29)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 25) );
		_mm256_storeu_si256(out + 5, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 22);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 22)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 25) );
		_mm256_storeu_si256(out + 6, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 15);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 15)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 25) );
		_mm256_storeu_si256(out + 7, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 8);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 8)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 25) );
		_mm256_storeu_si256(out + 8, c25_rslt_m256i);

		c25_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c25_load_rslt_m256i, 1), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 25) );
		_mm256_storeu_si256(out + 9, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 26);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 26)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 25) );
		_mm256_storeu_si256(out + 10, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 19);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 19)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 25) );
		_mm256_storeu_si256(out + 11, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 12);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 12)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 25) );
		_mm256_storeu_si256(out + 12, c25_rslt_m256i);

		c25_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c25_load_rslt_m256i, 5), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 25) );
		_mm256_storeu_si256(out + 13, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 30);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 30)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 25) );
		_mm256_storeu_si256(out + 14, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 23);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 23)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 25) );
		_mm256_storeu_si256(out + 15, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 16);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 16)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 25) );
		_mm256_storeu_si256(out + 16, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 9);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 9)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 25) );
		_mm256_storeu_si256(out + 17, c25_rslt_m256i);

		c25_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c25_load_rslt_m256i, 2), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 25) );
		_mm256_storeu_si256(out + 18, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 27);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 27)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 25) );
		_mm256_storeu_si256(out + 19, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 20);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 20)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 25) );
		_mm256_storeu_si256(out + 20, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 13);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 13)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 25) );
		_mm256_storeu_si256(out + 21, c25_rslt_m256i);

		c25_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c25_load_rslt_m256i, 6), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 25) );
		_mm256_storeu_si256(out + 22, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 31);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 31)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 25) );
		_mm256_storeu_si256(out + 23, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 24);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 24)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 25) );
		_mm256_storeu_si256(out + 24, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 17);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 17)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 25) );
		_mm256_storeu_si256(out + 25, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 10);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 10)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 25) );
		_mm256_storeu_si256(out + 26, c25_rslt_m256i);

		c25_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c25_load_rslt_m256i, 3), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 25) );
		_mm256_storeu_si256(out + 27, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 28);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 22);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 28)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 25) );
		_mm256_storeu_si256(out + 28, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 21);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 23);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 21)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 25) );
		_mm256_storeu_si256(out + 29, c25_rslt_m256i);


		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 14);
		c25_load_rslt_m256i = _mm256_loadu_si256(in + 24);

		c25_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 32 - 14)), c25_and_mask_m256i );
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 25) );
		_mm256_storeu_si256(out + 30, c25_rslt_m256i);

		c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 7);
		c25_rslt_m256i = _mm256_or_si256( c25_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 25) );
		_mm256_storeu_si256(out + 31, c25_rslt_m256i);
	}
}


// 26-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c26(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c26_and_mask_m256i = _mm256_set1_epi32(0x03ffffff);
		__m256i c26_load_rslt_m256i, c26_rslt_m256i;


		c26_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c26_rslt_m256i = _mm256_and_si256( c26_load_rslt_m256i, c26_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 26);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 26)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 20);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 20)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 14);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 14)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 8);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 8)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c26_rslt_m256i);

		c26_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c26_load_rslt_m256i, 2), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 28);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 28)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 22);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 22)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 16);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 16)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 10);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 10)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c26_rslt_m256i);

		c26_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c26_load_rslt_m256i, 4), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 30);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 30)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 24);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 24)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 18);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 18)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 12);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 12)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c26_rslt_m256i);

		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 6);
		_mm256_storeu_si256(out + 15, c26_rslt_m256i);


		c26_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c26_rslt_m256i = _mm256_and_si256( c26_load_rslt_m256i, c26_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 26);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 26)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 20);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 20)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 14);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 14)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 8);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 8)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c26_rslt_m256i);

		c26_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c26_load_rslt_m256i, 2), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 28);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 28)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 22);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 22)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 16);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 16)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 10);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 10)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c26_rslt_m256i);

		c26_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c26_load_rslt_m256i, 4), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 30);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 22);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 30)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 24);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 23);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 24)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 18);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 24);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 18)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 12);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 25);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 12)), c26_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c26_rslt_m256i);

		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 6);
		_mm256_storeu_si256(out + 31, c26_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c26_and_mask_m256i = _mm256_set1_epi32(0x03ffffff);
		__m256i c26_load_rslt_m256i, c26_rslt_m256i;


		c26_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c26_rslt_m256i = _mm256_and_si256( c26_load_rslt_m256i, c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 26) );
		_mm256_storeu_si256(out + 0, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 26);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 26)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 26) );
		_mm256_storeu_si256(out + 1, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 20);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 20)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 26) );
		_mm256_storeu_si256(out + 2, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 14);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 14)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 26) );
		_mm256_storeu_si256(out + 3, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 8);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 8)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 26) );
		_mm256_storeu_si256(out + 4, c26_rslt_m256i);

		c26_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c26_load_rslt_m256i, 2), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 26) );
		_mm256_storeu_si256(out + 5, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 28);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 28)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 26) );
		_mm256_storeu_si256(out + 6, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 22);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 22)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 26) );
		_mm256_storeu_si256(out + 7, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 16);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 16)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 26) );
		_mm256_storeu_si256(out + 8, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 10);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 10)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 26) );
		_mm256_storeu_si256(out + 9, c26_rslt_m256i);

		c26_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c26_load_rslt_m256i, 4), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 26) );
		_mm256_storeu_si256(out + 10, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 30);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 30)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 26) );
		_mm256_storeu_si256(out + 11, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 24);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 24)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 26) );
		_mm256_storeu_si256(out + 12, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 18);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 18)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 26) );
		_mm256_storeu_si256(out + 13, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 12);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 12)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 26) );
		_mm256_storeu_si256(out + 14, c26_rslt_m256i);

		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 6);
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 26) );
		_mm256_storeu_si256(out + 15, c26_rslt_m256i);


		c26_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c26_rslt_m256i = _mm256_and_si256( c26_load_rslt_m256i, c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 26) );
		_mm256_storeu_si256(out + 16, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 26);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 26)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 26) );
		_mm256_storeu_si256(out + 17, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 20);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 20)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 26) );
		_mm256_storeu_si256(out + 18, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 14);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 14)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 26) );
		_mm256_storeu_si256(out + 19, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 8);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 8)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 26) );
		_mm256_storeu_si256(out + 20, c26_rslt_m256i);

		c26_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c26_load_rslt_m256i, 2), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 26) );
		_mm256_storeu_si256(out + 21, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 28);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 28)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 26) );
		_mm256_storeu_si256(out + 22, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 22);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 22)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 26) );
		_mm256_storeu_si256(out + 23, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 16);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 16)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 26) );
		_mm256_storeu_si256(out + 24, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 10);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 10)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 26) );
		_mm256_storeu_si256(out + 25, c26_rslt_m256i);

		c26_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c26_load_rslt_m256i, 4), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 26) );
		_mm256_storeu_si256(out + 26, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 30);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 22);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 30)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 26) );
		_mm256_storeu_si256(out + 27, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 24);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 23);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 24)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 26) );
		_mm256_storeu_si256(out + 28, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 18);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 24);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 18)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 26) );
		_mm256_storeu_si256(out + 29, c26_rslt_m256i);


		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 12);
		c26_load_rslt_m256i = _mm256_loadu_si256(in + 25);

		c26_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 32 - 12)), c26_and_mask_m256i );
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 26) );
		_mm256_storeu_si256(out + 30, c26_rslt_m256i);

		c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 6);
		c26_rslt_m256i = _mm256_or_si256( c26_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 26) );
		_mm256_storeu_si256(out + 31, c26_rslt_m256i);
	}
}


// 27-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c27(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c27_and_mask_m256i = _mm256_set1_epi32(0x07ffffff);
		__m256i c27_load_rslt_m256i, c27_rslt_m256i;


		c27_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c27_rslt_m256i = _mm256_and_si256( c27_load_rslt_m256i, c27_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 27);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 27)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 22);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 22)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 17);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 17)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 12);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 12)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 7);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 7)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c27_rslt_m256i);

		c27_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c27_load_rslt_m256i, 2), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 29);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 29)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 24);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 24)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 19);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 19)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 14);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 14)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 9);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 9)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c27_rslt_m256i);

		c27_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c27_load_rslt_m256i, 4), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 31);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 31)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 26);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 26)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 21);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 21)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 15, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 16);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 16)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 11);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 11)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 6);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 6)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c27_rslt_m256i);

		c27_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c27_load_rslt_m256i, 1), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 28);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 28)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 23);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 23)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 18);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 18)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 13);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 13)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 8);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 8)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c27_rslt_m256i);

		c27_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c27_load_rslt_m256i, 3), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 30);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 22);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 30)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 25);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 23);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 25)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 20);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 24);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 20)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 15);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 25);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 15)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 10);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 26);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 10)), c27_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c27_rslt_m256i);

		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 5);
		_mm256_storeu_si256(out + 31, c27_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c27_and_mask_m256i = _mm256_set1_epi32(0x07ffffff);
		__m256i c27_load_rslt_m256i, c27_rslt_m256i;


		c27_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c27_rslt_m256i = _mm256_and_si256( c27_load_rslt_m256i, c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 27) );
		_mm256_storeu_si256(out + 0, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 27);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 27)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 27) );
		_mm256_storeu_si256(out + 1, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 22);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 22)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 27) );
		_mm256_storeu_si256(out + 2, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 17);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 17)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 27) );
		_mm256_storeu_si256(out + 3, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 12);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 12)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 27) );
		_mm256_storeu_si256(out + 4, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 7);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 7)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 27) );
		_mm256_storeu_si256(out + 5, c27_rslt_m256i);

		c27_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c27_load_rslt_m256i, 2), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 27) );
		_mm256_storeu_si256(out + 6, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 29);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 29)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 27) );
		_mm256_storeu_si256(out + 7, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 24);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 24)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 27) );
		_mm256_storeu_si256(out + 8, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 19);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 19)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 27) );
		_mm256_storeu_si256(out + 9, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 14);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 14)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 27) );
		_mm256_storeu_si256(out + 10, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 9);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 9)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 27) );
		_mm256_storeu_si256(out + 11, c27_rslt_m256i);

		c27_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c27_load_rslt_m256i, 4), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 27) );
		_mm256_storeu_si256(out + 12, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 31);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 31)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 27) );
		_mm256_storeu_si256(out + 13, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 26);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 26)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 27) );
		_mm256_storeu_si256(out + 14, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 21);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 21)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 27) );
		_mm256_storeu_si256(out + 15, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 16);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 16)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 27) );
		_mm256_storeu_si256(out + 16, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 11);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 11)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 27) );
		_mm256_storeu_si256(out + 17, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 6);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 6)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 27) );
		_mm256_storeu_si256(out + 18, c27_rslt_m256i);

		c27_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c27_load_rslt_m256i, 1), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 27) );
		_mm256_storeu_si256(out + 19, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 28);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 28)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 27) );
		_mm256_storeu_si256(out + 20, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 23);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 23)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 27) );
		_mm256_storeu_si256(out + 21, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 18);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 18)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 27) );
		_mm256_storeu_si256(out + 22, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 13);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 13)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 27) );
		_mm256_storeu_si256(out + 23, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 8);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 8)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 27) );
		_mm256_storeu_si256(out + 24, c27_rslt_m256i);

		c27_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c27_load_rslt_m256i, 3), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 27) );
		_mm256_storeu_si256(out + 25, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 30);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 22);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 30)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 27) );
		_mm256_storeu_si256(out + 26, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 25);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 23);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 25)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 27) );
		_mm256_storeu_si256(out + 27, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 20);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 24);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 20)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 27) );
		_mm256_storeu_si256(out + 28, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 15);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 25);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 15)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 27) );
		_mm256_storeu_si256(out + 29, c27_rslt_m256i);


		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 10);
		c27_load_rslt_m256i = _mm256_loadu_si256(in + 26);

		c27_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 32 - 10)), c27_and_mask_m256i );
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 27) );
		_mm256_storeu_si256(out + 30, c27_rslt_m256i);

		c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 5);
		c27_rslt_m256i = _mm256_or_si256( c27_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 27) );
		_mm256_storeu_si256(out + 31, c27_rslt_m256i);
	}
}


// 28-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c28(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c28_and_mask_m256i = _mm256_set1_epi32(0x0fffffff);
		__m256i c28_load_rslt_m256i, c28_rslt_m256i;


		c28_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c28_rslt_m256i = _mm256_and_si256( c28_load_rslt_m256i, c28_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 28);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 28)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 24);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 24)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 20);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 20)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 16);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 16)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 12);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 12)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 8);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 8)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c28_rslt_m256i);

		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 4);
		_mm256_storeu_si256(out + 7, c28_rslt_m256i);


		c28_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c28_rslt_m256i = _mm256_and_si256( c28_load_rslt_m256i, c28_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 28);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 28)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 24);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 24)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 20);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 20)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 16);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 16)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 12);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 12)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 8);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 8)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c28_rslt_m256i);

		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 4);
		_mm256_storeu_si256(out + 15, c28_rslt_m256i);


		c28_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c28_rslt_m256i = _mm256_and_si256( c28_load_rslt_m256i, c28_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 28);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 28)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 24);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 24)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 20);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 20)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 16);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 16)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 12);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 12)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 8);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 8)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c28_rslt_m256i);

		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 4);
		_mm256_storeu_si256(out + 23, c28_rslt_m256i);


		c28_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c28_rslt_m256i = _mm256_and_si256( c28_load_rslt_m256i, c28_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 28);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 22);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 28)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 24);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 23);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 24)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 20);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 24);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 20)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 16);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 25);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 16)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 12);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 26);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 12)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 8);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 27);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 8)), c28_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c28_rslt_m256i);

		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 4);
		_mm256_storeu_si256(out + 31, c28_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c28_and_mask_m256i = _mm256_set1_epi32(0x0fffffff);
		__m256i c28_load_rslt_m256i, c28_rslt_m256i;


		c28_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c28_rslt_m256i = _mm256_and_si256( c28_load_rslt_m256i, c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 28) );
		_mm256_storeu_si256(out + 0, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 28);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 28)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 28) );
		_mm256_storeu_si256(out + 1, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 24);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 24)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 28) );
		_mm256_storeu_si256(out + 2, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 20);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 20)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 28) );
		_mm256_storeu_si256(out + 3, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 16);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 16)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 28) );
		_mm256_storeu_si256(out + 4, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 12);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 12)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 28) );
		_mm256_storeu_si256(out + 5, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 8);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 8)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 28) );
		_mm256_storeu_si256(out + 6, c28_rslt_m256i);

		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 4);
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 28) );
		_mm256_storeu_si256(out + 7, c28_rslt_m256i);


		c28_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c28_rslt_m256i = _mm256_and_si256( c28_load_rslt_m256i, c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 28) );
		_mm256_storeu_si256(out + 8, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 28);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 28)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 28) );
		_mm256_storeu_si256(out + 9, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 24);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 24)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 28) );
		_mm256_storeu_si256(out + 10, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 20);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 20)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 28) );
		_mm256_storeu_si256(out + 11, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 16);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 16)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 28) );
		_mm256_storeu_si256(out + 12, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 12);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 12)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 28) );
		_mm256_storeu_si256(out + 13, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 8);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 8)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 28) );
		_mm256_storeu_si256(out + 14, c28_rslt_m256i);

		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 4);
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 28) );
		_mm256_storeu_si256(out + 15, c28_rslt_m256i);


		c28_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c28_rslt_m256i = _mm256_and_si256( c28_load_rslt_m256i, c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 28) );
		_mm256_storeu_si256(out + 16, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 28);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 28)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 28) );
		_mm256_storeu_si256(out + 17, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 24);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 24)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 28) );
		_mm256_storeu_si256(out + 18, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 20);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 20)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 28) );
		_mm256_storeu_si256(out + 19, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 16);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 16)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 28) );
		_mm256_storeu_si256(out + 20, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 12);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 12)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 28) );
		_mm256_storeu_si256(out + 21, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 8);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 8)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 28) );
		_mm256_storeu_si256(out + 22, c28_rslt_m256i);

		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 4);
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 28) );
		_mm256_storeu_si256(out + 23, c28_rslt_m256i);


		c28_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c28_rslt_m256i = _mm256_and_si256( c28_load_rslt_m256i, c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 28) );
		_mm256_storeu_si256(out + 24, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 28);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 22);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 28)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 28) );
		_mm256_storeu_si256(out + 25, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 24);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 23);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 24)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 28) );
		_mm256_storeu_si256(out + 26, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 20);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 24);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 20)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 28) );
		_mm256_storeu_si256(out + 27, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 16);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 25);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 16)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 28) );
		_mm256_storeu_si256(out + 28, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 12);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 26);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 12)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 28) );
		_mm256_storeu_si256(out + 29, c28_rslt_m256i);


		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 8);
		c28_load_rslt_m256i = _mm256_loadu_si256(in + 27);

		c28_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 32 - 8)), c28_and_mask_m256i );
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 28) );
		_mm256_storeu_si256(out + 30, c28_rslt_m256i);

		c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 4);
		c28_rslt_m256i = _mm256_or_si256( c28_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 28) );
		_mm256_storeu_si256(out + 31, c28_rslt_m256i);
	}
}


// 29-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c29(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c29_and_mask_m256i = _mm256_set1_epi32(0x1fffffff);
		__m256i c29_load_rslt_m256i, c29_rslt_m256i;


		c29_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c29_rslt_m256i = _mm256_and_si256( c29_load_rslt_m256i, c29_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 29);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 29)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 26);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 26)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 23);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 23)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 20);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 20)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 17);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 17)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 14);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 14)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 11);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 11)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 8);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 8)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 5);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 5)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c29_rslt_m256i);

		c29_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c29_load_rslt_m256i, 2), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 31);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 31)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 28);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 28)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 25);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 25)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 22);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 22)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 19);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 19)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 15, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 16);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 16)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 13);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 13)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 10);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 10)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 7);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 7)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 4);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 4)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c29_rslt_m256i);

		c29_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c29_load_rslt_m256i, 1), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 30);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 30)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 27);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 27)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 24);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 22);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 24)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 21);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 23);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 21)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 18);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 24);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 18)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 15);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 25);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 15)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 12);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 26);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 12)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 9);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 27);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 9)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 6);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 28);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 6)), c29_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c29_rslt_m256i);

		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 3);
		_mm256_storeu_si256(out + 31, c29_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c29_and_mask_m256i = _mm256_set1_epi32(0x1fffffff);
		__m256i c29_load_rslt_m256i, c29_rslt_m256i;


		c29_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c29_rslt_m256i = _mm256_and_si256( c29_load_rslt_m256i, c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 29) );
		_mm256_storeu_si256(out + 0, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 29);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 29)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 29) );
		_mm256_storeu_si256(out + 1, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 26);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 26)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 29) );
		_mm256_storeu_si256(out + 2, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 23);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 23)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 29) );
		_mm256_storeu_si256(out + 3, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 20);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 20)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 29) );
		_mm256_storeu_si256(out + 4, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 17);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 17)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 29) );
		_mm256_storeu_si256(out + 5, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 14);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 14)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 29) );
		_mm256_storeu_si256(out + 6, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 11);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 11)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 29) );
		_mm256_storeu_si256(out + 7, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 8);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 8)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 29) );
		_mm256_storeu_si256(out + 8, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 5);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 5)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 29) );
		_mm256_storeu_si256(out + 9, c29_rslt_m256i);

		c29_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c29_load_rslt_m256i, 2), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 29) );
		_mm256_storeu_si256(out + 10, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 31);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 31)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 29) );
		_mm256_storeu_si256(out + 11, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 28);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 28)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 29) );
		_mm256_storeu_si256(out + 12, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 25);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 25)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 29) );
		_mm256_storeu_si256(out + 13, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 22);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 22)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 29) );
		_mm256_storeu_si256(out + 14, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 19);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 19)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 29) );
		_mm256_storeu_si256(out + 15, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 16);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 16)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 29) );
		_mm256_storeu_si256(out + 16, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 13);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 13)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 29) );
		_mm256_storeu_si256(out + 17, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 10);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 10)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 29) );
		_mm256_storeu_si256(out + 18, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 7);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 7)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 29) );
		_mm256_storeu_si256(out + 19, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 4);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 4)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 29) );
		_mm256_storeu_si256(out + 20, c29_rslt_m256i);

		c29_rslt_m256i = _mm256_and_si256( _mm256_srli_epi32(c29_load_rslt_m256i, 1), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 29) );
		_mm256_storeu_si256(out + 21, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 30);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 30)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 29) );
		_mm256_storeu_si256(out + 22, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 27);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 27)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 29) );
		_mm256_storeu_si256(out + 23, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 24);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 22);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 24)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 29) );
		_mm256_storeu_si256(out + 24, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 21);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 23);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 21)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 29) );
		_mm256_storeu_si256(out + 25, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 18);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 24);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 18)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 29) );
		_mm256_storeu_si256(out + 26, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 15);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 25);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 15)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 29) );
		_mm256_storeu_si256(out + 27, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 12);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 26);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 12)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 29) );
		_mm256_storeu_si256(out + 28, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 9);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 27);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 9)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 29) );
		_mm256_storeu_si256(out + 29, c29_rslt_m256i);


		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 6);
		c29_load_rslt_m256i = _mm256_loadu_si256(in + 28);

		c29_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 32 - 6)), c29_and_mask_m256i );
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 29) );
		_mm256_storeu_si256(out + 30, c29_rslt_m256i);

		c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 3);
		c29_rslt_m256i = _mm256_or_si256( c29_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 29) );
		_mm256_storeu_si256(out + 31, c29_rslt_m256i);
	}
}


// 30-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c30(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c30_and_mask_m256i = _mm256_set1_epi32(0x3fffffff);
		__m256i c30_load_rslt_m256i, c30_rslt_m256i;


		c30_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c30_rslt_m256i = _mm256_and_si256( c30_load_rslt_m256i, c30_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 30);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 30)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 28);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 28)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 26);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 26)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 24);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 24)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 22);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 22)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 20);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 20)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 18);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 18)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 16);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 16)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 14);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 14)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 12);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 12)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 10);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 10)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 8);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 8)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 6);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 6)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 4);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 4)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c30_rslt_m256i);

		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 2);
		_mm256_storeu_si256(out + 15, c30_rslt_m256i);


		c30_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c30_rslt_m256i = _mm256_and_si256( c30_load_rslt_m256i, c30_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 30);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 30)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 28);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 28)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 26);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 26)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 24);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 24)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 22);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 22)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 20);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 20)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 18);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 22);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 18)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 16);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 23);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 16)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 14);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 24);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 14)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 12);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 25);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 12)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 10);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 26);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 10)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 8);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 27);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 8)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 6);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 28);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 6)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 4);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 29);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 4)), c30_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c30_rslt_m256i);

		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 2);
		_mm256_storeu_si256(out + 31, c30_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c30_and_mask_m256i = _mm256_set1_epi32(0x3fffffff);
		__m256i c30_load_rslt_m256i, c30_rslt_m256i;


		c30_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c30_rslt_m256i = _mm256_and_si256( c30_load_rslt_m256i, c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 30) );
		_mm256_storeu_si256(out + 0, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 30);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 30)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 30) );
		_mm256_storeu_si256(out + 1, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 28);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 28)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 30) );
		_mm256_storeu_si256(out + 2, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 26);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 26)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 30) );
		_mm256_storeu_si256(out + 3, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 24);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 24)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 30) );
		_mm256_storeu_si256(out + 4, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 22);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 22)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 30) );
		_mm256_storeu_si256(out + 5, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 20);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 20)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 30) );
		_mm256_storeu_si256(out + 6, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 18);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 18)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 30) );
		_mm256_storeu_si256(out + 7, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 16);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 16)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 30) );
		_mm256_storeu_si256(out + 8, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 14);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 14)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 30) );
		_mm256_storeu_si256(out + 9, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 12);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 12)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 30) );
		_mm256_storeu_si256(out + 10, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 10);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 10)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 30) );
		_mm256_storeu_si256(out + 11, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 8);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 8)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 30) );
		_mm256_storeu_si256(out + 12, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 6);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 6)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 30) );
		_mm256_storeu_si256(out + 13, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 4);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 4)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 30) );
		_mm256_storeu_si256(out + 14, c30_rslt_m256i);

		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 2);
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 30) );
		_mm256_storeu_si256(out + 15, c30_rslt_m256i);


		c30_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c30_rslt_m256i = _mm256_and_si256( c30_load_rslt_m256i, c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 30) );
		_mm256_storeu_si256(out + 16, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 30);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 30)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 30) );
		_mm256_storeu_si256(out + 17, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 28);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 28)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 30) );
		_mm256_storeu_si256(out + 18, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 26);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 26)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 30) );
		_mm256_storeu_si256(out + 19, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 24);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 24)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 30) );
		_mm256_storeu_si256(out + 20, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 22);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 22)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 30) );
		_mm256_storeu_si256(out + 21, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 20);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 20)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 30) );
		_mm256_storeu_si256(out + 22, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 18);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 22);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 18)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 30) );
		_mm256_storeu_si256(out + 23, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 16);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 23);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 16)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 30) );
		_mm256_storeu_si256(out + 24, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 14);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 24);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 14)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 30) );
		_mm256_storeu_si256(out + 25, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 12);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 25);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 12)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 30) );
		_mm256_storeu_si256(out + 26, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 10);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 26);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 10)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 30) );
		_mm256_storeu_si256(out + 27, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 8);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 27);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 8)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 30) );
		_mm256_storeu_si256(out + 28, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 6);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 28);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 6)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 30) );
		_mm256_storeu_si256(out + 29, c30_rslt_m256i);


		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 4);
		c30_load_rslt_m256i = _mm256_loadu_si256(in + 29);

		c30_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 32 - 4)), c30_and_mask_m256i );
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 30) );
		_mm256_storeu_si256(out + 30, c30_rslt_m256i);

		c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 2);
		c30_rslt_m256i = _mm256_or_si256( c30_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 30) );
		_mm256_storeu_si256(out + 31, c30_rslt_m256i);
	}
}


// 31-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c31(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		const __m256i c31_and_mask_m256i = _mm256_set1_epi32(0x7fffffff);
		__m256i c31_load_rslt_m256i, c31_rslt_m256i;


		c31_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c31_rslt_m256i = _mm256_and_si256( c31_load_rslt_m256i, c31_and_mask_m256i );
		_mm256_storeu_si256(out + 0, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 31);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 31)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 1, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 30);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 30)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 2, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 29);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 29)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 3, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 28);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 28)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 4, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 27);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 27)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 5, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 26);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 26)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 6, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 25);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 25)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 7, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 24);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 24)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 8, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 23);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 23)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 9, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 22);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 22)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 10, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 21);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 21)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 11, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 20);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 20)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 12, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 19);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 19)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 13, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 18);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 18)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 14, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 17);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 17)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 15, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 16);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 16)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 16, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 15);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 15)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 17, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 14);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 14)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 18, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 13);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 13)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 19, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 12);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 12)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 20, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 11);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 11)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 21, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 10);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 22);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 10)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 22, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 9);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 23);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 9)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 23, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 8);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 24);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 8)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 24, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 7);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 25);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 7)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 25, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 6);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 26);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 6)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 26, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 5);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 27);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 5)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 27, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 4);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 28);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 4)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 28, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 3);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 29);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 3)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 29, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 2);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 30);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 2)), c31_and_mask_m256i );
		_mm256_storeu_si256(out + 30, c31_rslt_m256i);

		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 1);
		_mm256_storeu_si256(out + 31, c31_rslt_m256i);
	}
	else { // Rice Coding
		const __m256i c31_and_mask_m256i = _mm256_set1_epi32(0x7fffffff);
		__m256i c31_load_rslt_m256i, c31_rslt_m256i;


		c31_load_rslt_m256i = _mm256_loadu_si256(in + 0);

		c31_rslt_m256i = _mm256_and_si256( c31_load_rslt_m256i, c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 0), 31) );
		_mm256_storeu_si256(out + 0, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 31);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 1);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 31)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 1), 31) );
		_mm256_storeu_si256(out + 1, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 30);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 2);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 30)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 2), 31) );
		_mm256_storeu_si256(out + 2, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 29);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 3);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 29)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 3), 31) );
		_mm256_storeu_si256(out + 3, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 28);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 4);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 28)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 4), 31) );
		_mm256_storeu_si256(out + 4, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 27);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 5);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 27)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 5), 31) );
		_mm256_storeu_si256(out + 5, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 26);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 6);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 26)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 6), 31) );
		_mm256_storeu_si256(out + 6, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 25);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 7);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 25)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 7), 31) );
		_mm256_storeu_si256(out + 7, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 24);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 8);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 24)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 8), 31) );
		_mm256_storeu_si256(out + 8, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 23);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 9);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 23)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 9), 31) );
		_mm256_storeu_si256(out + 9, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 22);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 10);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 22)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 10), 31) );
		_mm256_storeu_si256(out + 10, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 21);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 11);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 21)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 11), 31) );
		_mm256_storeu_si256(out + 11, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 20);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 12);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 20)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 12), 31) );
		_mm256_storeu_si256(out + 12, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 19);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 13);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 19)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 13), 31) );
		_mm256_storeu_si256(out + 13, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 18);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 14);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 18)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 14), 31) );
		_mm256_storeu_si256(out + 14, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 17);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 15);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 17)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 15), 31) );
		_mm256_storeu_si256(out + 15, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 16);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 16);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 16)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 16), 31) );
		_mm256_storeu_si256(out + 16, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 15);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 17);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 15)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 17), 31) );
		_mm256_storeu_si256(out + 17, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 14);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 18);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 14)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 18), 31) );
		_mm256_storeu_si256(out + 18, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 13);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 19);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 13)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 19), 31) );
		_mm256_storeu_si256(out + 19, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 12);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 20);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 12)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 20), 31) );
		_mm256_storeu_si256(out + 20, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 11);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 21);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 11)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 21), 31) );
		_mm256_storeu_si256(out + 21, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 10);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 22);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 10)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 22), 31) );
		_mm256_storeu_si256(out + 22, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 9);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 23);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 9)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 23), 31) );
		_mm256_storeu_si256(out + 23, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 8);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 24);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 8)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 24), 31) );
		_mm256_storeu_si256(out + 24, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 7);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 25);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 7)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 25), 31) );
		_mm256_storeu_si256(out + 25, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 6);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 26);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 6)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 26), 31) );
		_mm256_storeu_si256(out + 26, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 5);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 27);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 5)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 27), 31) );
		_mm256_storeu_si256(out + 27, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 4);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 28);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 4)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 28), 31) );
		_mm256_storeu_si256(out + 28, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 3);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 29);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 3)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 29), 31) );
		_mm256_storeu_si256(out + 29, c31_rslt_m256i);


		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 2);
		c31_load_rslt_m256i = _mm256_loadu_si256(in + 30);

		c31_rslt_m256i = _mm256_and_si256( _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 32 - 2)), c31_and_mask_m256i );
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 30), 31) );
		_mm256_storeu_si256(out + 30, c31_rslt_m256i);

		c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 1);
		c31_rslt_m256i = _mm256_or_si256( c31_rslt_m256i, _mm256_slli_epi32(_mm256_loadu_si256(quotient + 31), 31) );
		_mm256_storeu_si256(out + 31, c31_rslt_m256i);
	}
}


// 32-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256_c32(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const uint32_t *in32 = reinterpret_cast<const uint32_t *>(in);
	uint32_t *out32 = reinterpret_cast<uint32_t *>(out);
	for (uint32_t numberofValuesUnpacked = 0; numberofValuesUnpacked < 256; numberofValuesUnpacked += 32) {
		memcpy32(in32, out32);
		in32 += 32;
		out32 += 32;
	}
}


// 1-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c1(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c1_load_rslt_m256i, c1_rslt_m256i;


	c1_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c1_rslt_m256i = c1_load_rslt_m256i;

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 1));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 2));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 3));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 4));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 5));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 6));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 7));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 8));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 9));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 10));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 11));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 12));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 13));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 14));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 15));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 16));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 17));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 18));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 19));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 20));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 21));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 22));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 23));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 24));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 25));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 26));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 27));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 28));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 29));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 30));

	c1_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 0, c1_rslt_m256i);
}


// 2-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c2(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c2_load_rslt_m256i, c2_rslt_m256i;


	c2_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c2_rslt_m256i = c2_load_rslt_m256i;

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 2));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 4));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 6));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 8));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 10));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 12));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 14));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 16));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 18));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 20));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 22));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 24));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 26));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 28));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 0, c2_rslt_m256i);


	c2_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c2_rslt_m256i = c2_load_rslt_m256i;

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 2));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 4));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 6));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 8));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 10));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 12));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 14));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 16));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 18));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 20));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 22));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 24));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 26));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 28));

	c2_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 1, c2_rslt_m256i);
}


// 3-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c3(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c3_load_rslt_m256i, c3_rslt_m256i;


	c3_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c3_rslt_m256i = c3_load_rslt_m256i;

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 3));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 6));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 9));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 12));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 15));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 18));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 21));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 24));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 27));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 0, c3_rslt_m256i);


	c3_rslt_m256i = _mm256_srli_epi32(c3_load_rslt_m256i, 32 - 30);

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 1));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 4));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 7));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 10));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 13));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 16));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 19));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 22));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 25));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 28));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 1, c3_rslt_m256i);


	c3_rslt_m256i = _mm256_srli_epi32(c3_load_rslt_m256i, 32 - 31);

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 2));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 5));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 8));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 11));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 14));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 17));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 20));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 23));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 26));

	c3_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 2, c3_rslt_m256i);
}


// 4-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c4(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c4_load_rslt_m256i, c4_rslt_m256i;


	c4_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c4_rslt_m256i = c4_load_rslt_m256i;

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 4));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 8));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 12));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 16));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 20));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 24));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 0, c4_rslt_m256i);


	c4_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c4_rslt_m256i = c4_load_rslt_m256i;

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 4));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 8));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 12));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 16));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 20));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 24));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 1, c4_rslt_m256i);


	c4_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c4_rslt_m256i = c4_load_rslt_m256i;

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 4));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 8));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 12));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 16));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 20));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 24));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 2, c4_rslt_m256i);


	c4_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c4_rslt_m256i = c4_load_rslt_m256i;

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 4));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 8));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 12));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 16));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 20));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 24));

	c4_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 3, c4_rslt_m256i);
}


// 5-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c5(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c5_load_rslt_m256i, c5_rslt_m256i;


	c5_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c5_rslt_m256i = c5_load_rslt_m256i;

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 5));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 10));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 15));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 20));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 25));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 0, c5_rslt_m256i);


	c5_rslt_m256i = _mm256_srli_epi32(c5_load_rslt_m256i, 32 - 30);

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 3));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 8));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 13));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 18));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 23));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 1, c5_rslt_m256i);


	c5_rslt_m256i = _mm256_srli_epi32(c5_load_rslt_m256i, 32 - 28);

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 1));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 6));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 11));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 16));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 21));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 26));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 2, c5_rslt_m256i);


	c5_rslt_m256i = _mm256_srli_epi32(c5_load_rslt_m256i, 32 - 31);

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 4));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 9));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 14));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 19));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 24));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 3, c5_rslt_m256i);


	c5_rslt_m256i = _mm256_srli_epi32(c5_load_rslt_m256i, 32 - 29);

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 2));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 7));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 12));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 17));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 22));

	c5_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 4, c5_rslt_m256i);
}


// 6-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c6(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c6_load_rslt_m256i, c6_rslt_m256i;


	c6_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c6_rslt_m256i = c6_load_rslt_m256i;

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 6));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 12));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 18));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 24));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 0, c6_rslt_m256i);


	c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 32 - 30);

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 4));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 10));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 16));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 22));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 1, c6_rslt_m256i);


	c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 32 - 28);

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 2));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 8));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 14));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 20));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 2, c6_rslt_m256i);


	c6_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c6_rslt_m256i = c6_load_rslt_m256i;

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 6));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 12));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 18));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 24));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 3, c6_rslt_m256i);


	c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 32 - 30);

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 4));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 10));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 16));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 22));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 4, c6_rslt_m256i);


	c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 32 - 28);

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 2));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 8));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 14));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 20));

	c6_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 5, c6_rslt_m256i);
}


// 7-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c7(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c7_load_rslt_m256i, c7_rslt_m256i;


	c7_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c7_rslt_m256i = c7_load_rslt_m256i;

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 7));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 14));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 21));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 0, c7_rslt_m256i);


	c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 32 - 28);

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 3));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 10));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 17));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 24));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 1, c7_rslt_m256i);


	c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 32 - 31);

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 6));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 13));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 20));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 2, c7_rslt_m256i);


	c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 32 - 27);

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 2));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 9));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 16));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 23));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 3, c7_rslt_m256i);


	c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 32 - 30);

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 5));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 12));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 19));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 4, c7_rslt_m256i);


	c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 32 - 26);

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 1));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 8));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 15));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 22));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 5, c7_rslt_m256i);


	c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 32 - 29);

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 4));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 11));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 18));

	c7_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 6, c7_rslt_m256i);
}


// 8-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c8(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c8_load_rslt_m256i, c8_rslt_m256i;


	c8_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c8_rslt_m256i = c8_load_rslt_m256i;

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 8));

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 16));

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 0, c8_rslt_m256i);


	c8_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c8_rslt_m256i = c8_load_rslt_m256i;

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 8));

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 16));

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 1, c8_rslt_m256i);


	c8_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c8_rslt_m256i = c8_load_rslt_m256i;

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 8));

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 16));

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 2, c8_rslt_m256i);


	c8_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c8_rslt_m256i = c8_load_rslt_m256i;

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 8));

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 16));

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 3, c8_rslt_m256i);


	c8_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c8_rslt_m256i = c8_load_rslt_m256i;

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 8));

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 16));

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 4, c8_rslt_m256i);


	c8_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c8_rslt_m256i = c8_load_rslt_m256i;

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 8));

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 16));

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 5, c8_rslt_m256i);


	c8_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c8_rslt_m256i = c8_load_rslt_m256i;

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 8));

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 16));

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 6, c8_rslt_m256i);


	c8_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c8_rslt_m256i = c8_load_rslt_m256i;

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 8));

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 16));

	c8_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 7, c8_rslt_m256i);
}


// 9-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c9(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c9_load_rslt_m256i, c9_rslt_m256i;


	c9_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c9_rslt_m256i = c9_load_rslt_m256i;

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 9));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 18));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 0, c9_rslt_m256i);


	c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 32 - 27);

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 4));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 13));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 22));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 1, c9_rslt_m256i);


	c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 32 - 31);

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 8));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 17));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 2, c9_rslt_m256i);


	c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 32 - 26);

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 3));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 12));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 21));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 3, c9_rslt_m256i);


	c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 32 - 30);

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 7));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 16));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 4, c9_rslt_m256i);


	c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 32 - 25);

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 2));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 11));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 20));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 5, c9_rslt_m256i);


	c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 32 - 29);

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 6));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 15));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 6, c9_rslt_m256i);


	c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 32 - 24);

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 1));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 10));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 19));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 7, c9_rslt_m256i);


	c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 32 - 28);

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 5));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 14));

	c9_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 8, c9_rslt_m256i);
}


// 10-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c10(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c10_load_rslt_m256i, c10_rslt_m256i;


	c10_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c10_rslt_m256i = c10_load_rslt_m256i;

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 10));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 20));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 0, c10_rslt_m256i);


	c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 32 - 30);

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 8));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 18));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 1, c10_rslt_m256i);


	c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 32 - 28);

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 6));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 16));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 2, c10_rslt_m256i);


	c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 32 - 26);

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 4));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 14));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 3, c10_rslt_m256i);


	c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 32 - 24);

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 2));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 12));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 4, c10_rslt_m256i);


	c10_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c10_rslt_m256i = c10_load_rslt_m256i;

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 10));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 20));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 5, c10_rslt_m256i);


	c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 32 - 30);

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 8));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 18));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 6, c10_rslt_m256i);


	c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 32 - 28);

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 6));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 16));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 7, c10_rslt_m256i);


	c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 32 - 26);

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 4));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 14));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 8, c10_rslt_m256i);


	c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 32 - 24);

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 2));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 12));

	c10_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 9, c10_rslt_m256i);
}


// 11-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c11(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c11_load_rslt_m256i, c11_rslt_m256i;


	c11_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c11_rslt_m256i = c11_load_rslt_m256i;

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 11));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 0, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 22);

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 1));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 12));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 1, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 23);

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 2));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 13));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 2, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 24);

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 3));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 14));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 3, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 25);

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 4));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 15));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 4, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 26);

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 5));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 16));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 5, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 27);

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 6));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 17));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 6, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 28);

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 7));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 18));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 7, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 29);

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 8));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 19));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 8, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 30);

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 9));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 20));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 9, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 31);

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 10));

	c11_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 10, c11_rslt_m256i);
}


// 12-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c12(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c12_load_rslt_m256i, c12_rslt_m256i;


	c12_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c12_rslt_m256i = c12_load_rslt_m256i;

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 12));

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 0, c12_rslt_m256i);


	c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 32 - 24);

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 4));

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 16));

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 1, c12_rslt_m256i);


	c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 32 - 28);

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 8));

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 2, c12_rslt_m256i);


	c12_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c12_rslt_m256i = c12_load_rslt_m256i;

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 12));

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 3, c12_rslt_m256i);


	c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 32 - 24);

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 4));

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 16));

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 4, c12_rslt_m256i);


	c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 32 - 28);

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 8));

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 5, c12_rslt_m256i);


	c12_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c12_rslt_m256i = c12_load_rslt_m256i;

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 12));

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 6, c12_rslt_m256i);


	c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 32 - 24);

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 4));

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 16));

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 7, c12_rslt_m256i);


	c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 32 - 28);

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 8));

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 8, c12_rslt_m256i);


	c12_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c12_rslt_m256i = c12_load_rslt_m256i;

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 12));

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 9, c12_rslt_m256i);


	c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 32 - 24);

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 4));

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 16));

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 10, c12_rslt_m256i);


	c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 32 - 28);

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 8));

	c12_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 11, c12_rslt_m256i);
}


// 13-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c13(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c13_load_rslt_m256i, c13_rslt_m256i;


	c13_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c13_rslt_m256i = c13_load_rslt_m256i;

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 13));

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 0, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 26);

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 7));

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 1, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 20);

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 1));

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 14));

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 2, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 27);

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 8));

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 3, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 21);

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 2));

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 15));

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 4, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 28);

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 9));

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 5, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 22);

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 3));

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 16));

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 6, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 29);

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 10));

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 7, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 23);

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 4));

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 17));

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 8, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 30);

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 11));

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 9, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 24);

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 5));

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 18));

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 10, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 31);

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 12));

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 11, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 25);

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 6));

	c13_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 12, c13_rslt_m256i);
}


// 14-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c14(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c14_load_rslt_m256i, c14_rslt_m256i;


	c14_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c14_rslt_m256i = c14_load_rslt_m256i;

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 14));

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 0, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 28);

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 10));

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 1, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 24);

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 6));

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 2, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 20);

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 2));

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 16));

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 3, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 30);

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 12));

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 4, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 26);

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 8));

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 5, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 22);

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 4));

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 6, c14_rslt_m256i);


	c14_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c14_rslt_m256i = c14_load_rslt_m256i;

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 14));

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 7, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 28);

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 10));

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 8, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 24);

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 6));

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 9, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 20);

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 2));

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 16));

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 10, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 30);

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 12));

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 11, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 26);

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 8));

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 12, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 22);

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 4));

	c14_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 13, c14_rslt_m256i);
}


// 15-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c15(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c15_load_rslt_m256i, c15_rslt_m256i;


	c15_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c15_rslt_m256i = c15_load_rslt_m256i;

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 15));

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 0, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 30);

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 13));

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 1, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 28);

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 11));

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 2, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 26);

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 9));

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 3, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 24);

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 7));

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 4, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 22);

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 5));

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 5, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 20);

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 3));

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 6, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 18);

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 1));

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 16));

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 7, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 31);

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 14));

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 8, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 29);

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 12));

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 9, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 27);

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 10));

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 10, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 25);

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 8));

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 11, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 23);

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 6));

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 12, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 21);

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 4));

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 13, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 19);

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 2));

	c15_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 17));

	_mm256_storeu_si256(out + 14, c15_rslt_m256i);
}


// 16-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c16(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c16_load_rslt_m256i, c16_rslt_m256i;


	c16_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 0, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 1, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 2, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 3, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 4, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 5, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 6, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 7, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 8, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 9, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 10, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 11, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 12, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 13, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 14, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 15, c16_rslt_m256i);
}


// 17-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c17(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c17_load_rslt_m256i, c17_rslt_m256i;


	c17_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c17_rslt_m256i = c17_load_rslt_m256i;

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 17));

	_mm256_storeu_si256(out + 0, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 17);

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 2));

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 1, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 19);

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 4));

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 2, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 21);

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 6));

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 3, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 23);

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 8));

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 4, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 25);

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 10));

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 5, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 27);

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 12));

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 6, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 29);

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 14));

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 7, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 31);

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 8, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 16);

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 1));

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 9, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 18);

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 3));

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 10, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 20);

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 5));

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 11, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 22);

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 7));

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 12, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 24);

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 9));

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 13, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 26);

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 11));

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 14, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 28);

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 13));

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 15, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 30);

	c17_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 15));

	_mm256_storeu_si256(out + 16, c17_rslt_m256i);
}


// 18-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c18(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c18_load_rslt_m256i, c18_rslt_m256i;


	c18_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c18_rslt_m256i = c18_load_rslt_m256i;

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 0, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 18);

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 4));

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 1, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 22);

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 8));

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 2, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 26);

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 12));

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 3, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 30);

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 4, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 16);

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 2));

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 5, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 20);

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 6));

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 6, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 24);

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 10));

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 7, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 28);

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 8, c18_rslt_m256i);


	c18_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c18_rslt_m256i = c18_load_rslt_m256i;

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 9, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 18);

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 4));

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 10, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 22);

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 8));

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 11, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 26);

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 12));

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 12, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 30);

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 13, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 16);

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 2));

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 14, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 20);

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 6));

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 15, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 24);

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 10));

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 16, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 28);

	c18_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 17, c18_rslt_m256i);
}


// 19-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c19(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c19_load_rslt_m256i, c19_rslt_m256i;


	c19_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c19_rslt_m256i = c19_load_rslt_m256i;

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 0, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 19);

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 6));

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 1, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 25);

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 12));

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 2, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 31);

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 3, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 18);

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 5));

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 4, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 24);

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 11));

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 5, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 30);

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 17));

	_mm256_storeu_si256(out + 6, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 17);

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 4));

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 7, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 23);

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 10));

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 8, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 29);

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 9, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 16);

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 3));

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 10, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 22);

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 9));

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 11, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 28);

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 15));

	_mm256_storeu_si256(out + 12, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 15);

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 2));

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 13, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 21);

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 8));

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 14, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 27);

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 15, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 14);

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 1));

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 16, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 20);

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 7));

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 17, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 26);

	c19_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 13));

	_mm256_storeu_si256(out + 18, c19_rslt_m256i);
}


// 20-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c20(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c20_load_rslt_m256i, c20_rslt_m256i;


	c20_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c20_rslt_m256i = c20_load_rslt_m256i;

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 0, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 20);

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 8));

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 1, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 28);

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 2, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 16);

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 4));

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 3, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 24);

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 4, c20_rslt_m256i);


	c20_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c20_rslt_m256i = c20_load_rslt_m256i;

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 5, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 20);

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 8));

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 6, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 28);

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 7, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 16);

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 4));

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 8, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 24);

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 9, c20_rslt_m256i);


	c20_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c20_rslt_m256i = c20_load_rslt_m256i;

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 10, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 20);

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 8));

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 11, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 28);

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 12, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 16);

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 4));

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 13, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 24);

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 14, c20_rslt_m256i);


	c20_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c20_rslt_m256i = c20_load_rslt_m256i;

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 15, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 20);

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 8));

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 16, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 28);

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 17, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 16);

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 4));

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 18, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 24);

	c20_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 19, c20_rslt_m256i);
}


// 21-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c21(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c21_load_rslt_m256i, c21_rslt_m256i;


	c21_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c21_rslt_m256i = c21_load_rslt_m256i;

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 0, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 21);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 10));

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 1, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 31);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 2, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 20);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 9));

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 3, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 30);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 4, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 19);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 8));

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 5, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 29);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 6, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 18);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 7));

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 7, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 28);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 17));

	_mm256_storeu_si256(out + 8, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 17);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 6));

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 9, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 27);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 10, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 16);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 5));

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 11, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 26);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 15));

	_mm256_storeu_si256(out + 12, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 15);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 4));

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 13, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 25);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 14, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 14);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 3));

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 15, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 24);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 13));

	_mm256_storeu_si256(out + 16, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 13);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 2));

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 17, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 23);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 18, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 12);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 1));

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 19, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 22);

	c21_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 11));

	_mm256_storeu_si256(out + 20, c21_rslt_m256i);
}


// 22-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c22(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c22_load_rslt_m256i, c22_rslt_m256i;


	c22_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c22_rslt_m256i = c22_load_rslt_m256i;

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 0, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 22);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 1, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 12);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 2));

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 2, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 24);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 3, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 14);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 4));

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 4, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 26);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 5, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 16);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 6));

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 6, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 28);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 7, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 18);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 8));

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 8, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 30);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 9, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 20);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 10, c22_rslt_m256i);


	c22_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c22_rslt_m256i = c22_load_rslt_m256i;

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 11, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 22);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 12, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 12);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 2));

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 13, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 24);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 14, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 14);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 4));

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 15, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 26);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 16, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 16);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 6));

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 17, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 28);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 18, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 18);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 8));

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 19, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 30);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 20, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 20);

	c22_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 21, c22_rslt_m256i);
}


// 23-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c23(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c23_load_rslt_m256i, c23_rslt_m256i;


	c23_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c23_rslt_m256i = c23_load_rslt_m256i;

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 0, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 23);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 1, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 14);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 5));

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 2, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 28);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 3, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 19);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 4, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 10);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 1));

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 5, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 24);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 15));

	_mm256_storeu_si256(out + 6, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 15);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 6));

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 7, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 29);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 8, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 20);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 11));

	_mm256_storeu_si256(out + 9, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 11);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 2));

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 10, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 25);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 11, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 16);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 7));

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 12, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 30);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 13, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 21);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 14, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 12);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 3));

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 15, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 26);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 17));

	_mm256_storeu_si256(out + 16, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 17);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 8));

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 17, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 31);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 18, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 22);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 13));

	_mm256_storeu_si256(out + 19, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 13);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 4));

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 20, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 27);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 21, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 18);

	c23_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 9));

	_mm256_storeu_si256(out + 22, c23_rslt_m256i);
}


// 24-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c24(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c24_load_rslt_m256i, c24_rslt_m256i;


	c24_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c24_rslt_m256i = c24_load_rslt_m256i;

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 0, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 24);

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 1, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 16);

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 2, c24_rslt_m256i);


	c24_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c24_rslt_m256i = c24_load_rslt_m256i;

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 3, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 24);

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 4, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 16);

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 5, c24_rslt_m256i);


	c24_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c24_rslt_m256i = c24_load_rslt_m256i;

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 6, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 24);

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 7, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 16);

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 8, c24_rslt_m256i);


	c24_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c24_rslt_m256i = c24_load_rslt_m256i;

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 9, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 24);

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 10, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 16);

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 11, c24_rslt_m256i);


	c24_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c24_rslt_m256i = c24_load_rslt_m256i;

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 12, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 24);

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 13, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 16);

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 14, c24_rslt_m256i);


	c24_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c24_rslt_m256i = c24_load_rslt_m256i;

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 15, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 24);

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 16, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 16);

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 17, c24_rslt_m256i);


	c24_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c24_rslt_m256i = c24_load_rslt_m256i;

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 18, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 24);

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 19, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 16);

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 20, c24_rslt_m256i);


	c24_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c24_rslt_m256i = c24_load_rslt_m256i;

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 21, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 24);

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 22, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 16);

	c24_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 23, c24_rslt_m256i);
}


// 25-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c25(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c25_load_rslt_m256i, c25_rslt_m256i;


	c25_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c25_rslt_m256i = c25_load_rslt_m256i;

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 0, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 25);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 1, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 18);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 11));

	_mm256_storeu_si256(out + 2, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 11);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 4));

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 3, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 29);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 4, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 22);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 15));

	_mm256_storeu_si256(out + 5, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 15);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 6, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 8);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 1));

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 7, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 26);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 8, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 19);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 9, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 12);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 5));

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 10, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 30);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 11, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 23);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 12, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 16);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 9));

	_mm256_storeu_si256(out + 13, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 9);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 2));

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 14, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 27);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 15, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 20);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 13));

	_mm256_storeu_si256(out + 16, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 13);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 6));

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 17, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 31);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 18, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 24);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 17));

	_mm256_storeu_si256(out + 19, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 17);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 20, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 10);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 3));

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 21, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 28);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 22, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 21);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 23, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 14);

	c25_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 7));

	_mm256_storeu_si256(out + 24, c25_rslt_m256i);
}


// 26-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c26(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c26_load_rslt_m256i, c26_rslt_m256i;


	c26_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c26_rslt_m256i = c26_load_rslt_m256i;

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 0, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 26);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 1, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 20);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 2, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 14);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 3, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 8);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 2));

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 4, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 28);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 5, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 22);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 6, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 16);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 7, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 10);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 4));

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 8, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 30);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 9, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 24);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 10, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 18);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 11, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 12);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 6));

	_mm256_storeu_si256(out + 12, c26_rslt_m256i);


	c26_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c26_rslt_m256i = c26_load_rslt_m256i;

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 13, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 26);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 14, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 20);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 15, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 14);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 16, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 8);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 2));

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 17, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 28);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 18, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 22);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 19, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 16);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 20, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 10);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 4));

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 21, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 30);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 22, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 24);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 23, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 18);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 24, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 12);

	c26_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 6));

	_mm256_storeu_si256(out + 25, c26_rslt_m256i);
}


// 27-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c27(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c27_load_rslt_m256i, c27_rslt_m256i;


	c27_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c27_rslt_m256i = c27_load_rslt_m256i;

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 0, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 27);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 1, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 22);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 17));

	_mm256_storeu_si256(out + 2, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 17);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 3, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 12);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 7));

	_mm256_storeu_si256(out + 4, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 7);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 2));

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 5, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 29);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 6, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 24);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 7, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 19);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 8, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 14);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 9));

	_mm256_storeu_si256(out + 9, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 9);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 4));

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 10, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 31);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 11, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 26);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 12, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 21);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 13, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 16);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 11));

	_mm256_storeu_si256(out + 14, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 11);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 6));

	_mm256_storeu_si256(out + 15, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 6);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 1));

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 16, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 28);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 17, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 23);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 18, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 18);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 13));

	_mm256_storeu_si256(out + 19, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 13);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 20, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 8);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 3));

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 21, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 30);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 22, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 25);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 23, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 20);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 15));

	_mm256_storeu_si256(out + 24, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 15);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 25, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 10);

	c27_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 5));

	_mm256_storeu_si256(out + 26, c27_rslt_m256i);
}


// 28-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c28(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c28_load_rslt_m256i, c28_rslt_m256i;


	c28_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c28_rslt_m256i = c28_load_rslt_m256i;

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 0, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 28);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 1, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 24);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 2, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 20);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 3, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 16);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 4, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 12);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 5, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 8);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 4));

	_mm256_storeu_si256(out + 6, c28_rslt_m256i);


	c28_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c28_rslt_m256i = c28_load_rslt_m256i;

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 7, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 28);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 8, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 24);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 9, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 20);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 10, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 16);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 11, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 12);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 12, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 8);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 4));

	_mm256_storeu_si256(out + 13, c28_rslt_m256i);


	c28_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c28_rslt_m256i = c28_load_rslt_m256i;

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 14, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 28);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 15, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 24);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 16, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 20);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 17, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 16);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 18, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 12);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 19, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 8);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 4));

	_mm256_storeu_si256(out + 20, c28_rslt_m256i);


	c28_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c28_rslt_m256i = c28_load_rslt_m256i;

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 21, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 28);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 22, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 24);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 23, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 20);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 24, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 16);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 25, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 12);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 26, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 8);

	c28_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 4));

	_mm256_storeu_si256(out + 27, c28_rslt_m256i);
}


// 29-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c29(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c29_load_rslt_m256i, c29_rslt_m256i;


	c29_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c29_rslt_m256i = c29_load_rslt_m256i;

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 0, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 29);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 1, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 26);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 2, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 23);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 3, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 20);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 17));

	_mm256_storeu_si256(out + 4, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 17);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 5, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 14);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 11));

	_mm256_storeu_si256(out + 6, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 11);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 7, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 8);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 5));

	_mm256_storeu_si256(out + 8, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 5);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 2));

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 9, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 31);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 10, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 28);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 11, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 25);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 12, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 22);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 13, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 19);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 14, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 16);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 13));

	_mm256_storeu_si256(out + 15, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 13);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 16, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 10);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 7));

	_mm256_storeu_si256(out + 17, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 7);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 4));

	_mm256_storeu_si256(out + 18, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 4);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 1));

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 19, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 30);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 20, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 27);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 21, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 24);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 22, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 21);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 23, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 18);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 15));

	_mm256_storeu_si256(out + 24, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 15);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 25, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 12);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 9));

	_mm256_storeu_si256(out + 26, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 9);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 6));

	_mm256_storeu_si256(out + 27, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 6);

	c29_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 3));

	_mm256_storeu_si256(out + 28, c29_rslt_m256i);
}


// 30-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c30(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c30_load_rslt_m256i, c30_rslt_m256i;


	c30_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c30_rslt_m256i = c30_load_rslt_m256i;

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 0, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 30);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 1, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 28);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 2, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 26);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 3, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 24);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 4, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 22);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 5, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 20);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 6, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 18);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 7, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 16);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 8, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 14);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 9, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 12);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 10, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 10);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 11, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 8);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 6));

	_mm256_storeu_si256(out + 12, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 6);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 4));

	_mm256_storeu_si256(out + 13, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 4);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 2));

	_mm256_storeu_si256(out + 14, c30_rslt_m256i);


	c30_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c30_rslt_m256i = c30_load_rslt_m256i;

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 15, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 30);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 16, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 28);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 17, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 26);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 18, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 24);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 19, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 22);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 20, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 20);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 21, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 18);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 22, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 16);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 23, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 14);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 24, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 12);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 25, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 10);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 26, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 8);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 6));

	_mm256_storeu_si256(out + 27, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 6);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 4));

	_mm256_storeu_si256(out + 28, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 4);

	c30_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 2));

	_mm256_storeu_si256(out + 29, c30_rslt_m256i);
}


// 31-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c31(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	__m256i c31_load_rslt_m256i, c31_rslt_m256i;


	c31_load_rslt_m256i = _mm256_loadu_si256(in + 0);
	c31_rslt_m256i = c31_load_rslt_m256i;

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 1);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 0, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 31);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 2);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 1, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 30);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 3);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 2, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 29);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 4);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 3, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 28);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 5);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 4, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 27);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 6);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 5, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 26);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 7);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 6, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 25);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 8);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 7, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 24);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 9);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 8, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 23);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 10);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 9, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 22);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 11);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 10, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 21);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 12);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 11, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 20);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 13);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 12, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 19);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 14);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 13, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 18);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 15);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 17));

	_mm256_storeu_si256(out + 14, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 17);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 16);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 15, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 16);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 17);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 15));

	_mm256_storeu_si256(out + 16, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 15);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 18);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 17, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 14);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 19);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 13));

	_mm256_storeu_si256(out + 18, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 13);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 20);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 19, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 12);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 21);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 11));

	_mm256_storeu_si256(out + 20, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 11);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 22);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 21, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 10);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 23);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 9));

	_mm256_storeu_si256(out + 22, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 9);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 24);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 23, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 8);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 25);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 7));

	_mm256_storeu_si256(out + 24, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 7);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 26);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 6));

	_mm256_storeu_si256(out + 25, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 6);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 27);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 5));

	_mm256_storeu_si256(out + 26, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 5);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 28);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 4));

	_mm256_storeu_si256(out + 27, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 4);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 29);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 3));

	_mm256_storeu_si256(out + 28, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 3);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 30);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 2));

	_mm256_storeu_si256(out + 29, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 2);

	c31_load_rslt_m256i = _mm256_loadu_si256(in + 31);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 1));

	_mm256_storeu_si256(out + 30, c31_rslt_m256i);
}


// 32-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256_c32(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const uint32_t *in32 = reinterpret_cast<const uint32_t *>(in);
	uint32_t *out32 = reinterpret_cast<uint32_t *>(out);
	for (uint32_t numberofValuesUnpacked = 0; numberofValuesUnpacked < 256; numberofValuesUnpacked += 32) {
		memcpy32(in32, out32);
		in32 += 32;
		out32 += 32;
	}
}


// 1-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c1(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c1_and_mask_m256i = _mm256_set1_epi32(0x01);
	__m256i c1_load_rslt_m256i, c1_rslt_m256i;


	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c1_and_mask_m256i);
	c1_rslt_m256i = c1_load_rslt_m256i;

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 1));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 2));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 3));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 4));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 5));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 6));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 7));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 8));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 9));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 10));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 11));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 12));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 13));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 14));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 15));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 16));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 17));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 18));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 19));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 20));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 21));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 22));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 23));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 24));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 25));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 26));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 27));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 28));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 29));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 30));

	c1_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c1_and_mask_m256i);
	c1_rslt_m256i = _mm256_or_si256(c1_rslt_m256i, _mm256_slli_epi32(c1_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 0, c1_rslt_m256i);
}


// 2-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c2(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c2_and_mask_m256i = _mm256_set1_epi32(0x03);
	__m256i c2_load_rslt_m256i, c2_rslt_m256i;


	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c2_and_mask_m256i);
	c2_rslt_m256i = c2_load_rslt_m256i;

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 2));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 4));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 6));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 8));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 10));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 12));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 14));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 16));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 18));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 20));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 22));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 24));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 26));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 28));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 0, c2_rslt_m256i);


	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c2_and_mask_m256i);
	c2_rslt_m256i = c2_load_rslt_m256i;

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 2));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 4));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 6));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 8));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 10));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 12));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 14));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 16));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 18));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 20));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 22));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 24));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 26));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 28));

	c2_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c2_and_mask_m256i);
	c2_rslt_m256i = _mm256_or_si256(c2_rslt_m256i, _mm256_slli_epi32(c2_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 1, c2_rslt_m256i);
}


// 3-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c3(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c3_and_mask_m256i = _mm256_set1_epi32(0x07);
	__m256i c3_load_rslt_m256i, c3_rslt_m256i;


	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c3_and_mask_m256i);
	c3_rslt_m256i = c3_load_rslt_m256i;

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 3));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 6));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 9));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 12));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 15));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 18));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 21));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 24));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 27));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 0, c3_rslt_m256i);


	c3_rslt_m256i = _mm256_srli_epi32(c3_load_rslt_m256i, 32 - 30);

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 1));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 4));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 7));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 10));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 13));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 16));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 19));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 22));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 25));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 28));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 1, c3_rslt_m256i);


	c3_rslt_m256i = _mm256_srli_epi32(c3_load_rslt_m256i, 32 - 31);

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 2));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 5));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 8));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 11));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 14));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 17));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 20));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 23));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 26));

	c3_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c3_and_mask_m256i);
	c3_rslt_m256i = _mm256_or_si256(c3_rslt_m256i, _mm256_slli_epi32(c3_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 2, c3_rslt_m256i);
}


// 4-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c4(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c4_and_mask_m256i = _mm256_set1_epi32(0x0f);
	__m256i c4_load_rslt_m256i, c4_rslt_m256i;


	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c4_and_mask_m256i);
	c4_rslt_m256i = c4_load_rslt_m256i;

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 4));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 8));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 12));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 16));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 20));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 24));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 0, c4_rslt_m256i);


	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c4_and_mask_m256i);
	c4_rslt_m256i = c4_load_rslt_m256i;

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 4));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 8));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 12));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 16));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 20));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 24));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 1, c4_rslt_m256i);


	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c4_and_mask_m256i);
	c4_rslt_m256i = c4_load_rslt_m256i;

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 4));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 8));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 12));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 16));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 20));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 24));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 2, c4_rslt_m256i);


	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c4_and_mask_m256i);
	c4_rslt_m256i = c4_load_rslt_m256i;

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 4));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 8));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 12));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 16));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 20));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 24));

	c4_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c4_and_mask_m256i);
	c4_rslt_m256i = _mm256_or_si256(c4_rslt_m256i, _mm256_slli_epi32(c4_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 3, c4_rslt_m256i);
}


// 5-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c5(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c5_and_mask_m256i = _mm256_set1_epi32(0x1f);
	__m256i c5_load_rslt_m256i, c5_rslt_m256i;


	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c5_and_mask_m256i);
	c5_rslt_m256i = c5_load_rslt_m256i;

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 5));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 10));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 15));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 20));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 25));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 0, c5_rslt_m256i);


	c5_rslt_m256i = _mm256_srli_epi32(c5_load_rslt_m256i, 32 - 30);

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 3));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 8));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 13));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 18));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 23));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 1, c5_rslt_m256i);


	c5_rslt_m256i = _mm256_srli_epi32(c5_load_rslt_m256i, 32 - 28);

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 1));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 6));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 11));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 16));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 21));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 26));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 2, c5_rslt_m256i);


	c5_rslt_m256i = _mm256_srli_epi32(c5_load_rslt_m256i, 32 - 31);

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 4));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 9));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 14));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 19));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 24));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 3, c5_rslt_m256i);


	c5_rslt_m256i = _mm256_srli_epi32(c5_load_rslt_m256i, 32 - 29);

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 2));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 7));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 12));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 17));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 22));

	c5_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c5_and_mask_m256i);
	c5_rslt_m256i = _mm256_or_si256(c5_rslt_m256i, _mm256_slli_epi32(c5_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 4, c5_rslt_m256i);
}


// 6-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c6(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c6_and_mask_m256i = _mm256_set1_epi32(0x3f);
	__m256i c6_load_rslt_m256i, c6_rslt_m256i;


	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c6_and_mask_m256i);
	c6_rslt_m256i = c6_load_rslt_m256i;

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 6));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 12));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 18));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 24));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 0, c6_rslt_m256i);


	c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 32 - 30);

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 4));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 10));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 16));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 22));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 1, c6_rslt_m256i);


	c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 32 - 28);

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 2));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 8));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 14));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 20));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 2, c6_rslt_m256i);


	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c6_and_mask_m256i);
	c6_rslt_m256i = c6_load_rslt_m256i;

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 6));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 12));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 18));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 24));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 3, c6_rslt_m256i);


	c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 32 - 30);

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 4));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 10));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 16));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 22));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 4, c6_rslt_m256i);


	c6_rslt_m256i = _mm256_srli_epi32(c6_load_rslt_m256i, 32 - 28);

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 2));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 8));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 14));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 20));

	c6_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c6_and_mask_m256i);
	c6_rslt_m256i = _mm256_or_si256(c6_rslt_m256i, _mm256_slli_epi32(c6_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 5, c6_rslt_m256i);
}


// 7-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c7(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c7_and_mask_m256i = _mm256_set1_epi32(0x7f);
	__m256i c7_load_rslt_m256i, c7_rslt_m256i;


	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c7_and_mask_m256i);
	c7_rslt_m256i = c7_load_rslt_m256i;

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 7));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 14));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 21));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 0, c7_rslt_m256i);


	c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 32 - 28);

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 3));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 10));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 17));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 24));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 1, c7_rslt_m256i);


	c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 32 - 31);

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 6));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 13));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 20));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 2, c7_rslt_m256i);


	c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 32 - 27);

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 2));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 9));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 16));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 23));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 3, c7_rslt_m256i);


	c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 32 - 30);

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 5));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 12));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 19));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 4, c7_rslt_m256i);


	c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 32 - 26);

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 1));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 8));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 15));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 22));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 5, c7_rslt_m256i);


	c7_rslt_m256i = _mm256_srli_epi32(c7_load_rslt_m256i, 32 - 29);

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 4));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 11));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 18));

	c7_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c7_and_mask_m256i);
	c7_rslt_m256i = _mm256_or_si256(c7_rslt_m256i, _mm256_slli_epi32(c7_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 6, c7_rslt_m256i);
}


// 8-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c8(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c8_and_mask_m256i = _mm256_set1_epi32(0xff);
	__m256i c8_load_rslt_m256i, c8_rslt_m256i;


	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c8_and_mask_m256i);
	c8_rslt_m256i = c8_load_rslt_m256i;

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 8));

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 16));

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 0, c8_rslt_m256i);


	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c8_and_mask_m256i);
	c8_rslt_m256i = c8_load_rslt_m256i;

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 8));

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 16));

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 1, c8_rslt_m256i);


	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c8_and_mask_m256i);
	c8_rslt_m256i = c8_load_rslt_m256i;

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 8));

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 16));

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 2, c8_rslt_m256i);


	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c8_and_mask_m256i);
	c8_rslt_m256i = c8_load_rslt_m256i;

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 8));

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 16));

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 3, c8_rslt_m256i);


	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c8_and_mask_m256i);
	c8_rslt_m256i = c8_load_rslt_m256i;

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 8));

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 16));

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 4, c8_rslt_m256i);


	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c8_and_mask_m256i);
	c8_rslt_m256i = c8_load_rslt_m256i;

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 8));

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 16));

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 5, c8_rslt_m256i);


	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c8_and_mask_m256i);
	c8_rslt_m256i = c8_load_rslt_m256i;

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 8));

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 16));

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 6, c8_rslt_m256i);


	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c8_and_mask_m256i);
	c8_rslt_m256i = c8_load_rslt_m256i;

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 8));

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 16));

	c8_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c8_and_mask_m256i);
	c8_rslt_m256i = _mm256_or_si256(c8_rslt_m256i, _mm256_slli_epi32(c8_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 7, c8_rslt_m256i);
}


// 9-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c9(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c9_and_mask_m256i = _mm256_set1_epi32(0x01ff);
	__m256i c9_load_rslt_m256i, c9_rslt_m256i;


	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c9_and_mask_m256i);
	c9_rslt_m256i = c9_load_rslt_m256i;

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 9));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 18));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 0, c9_rslt_m256i);


	c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 32 - 27);

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 4));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 13));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 22));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 1, c9_rslt_m256i);


	c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 32 - 31);

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 8));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 17));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 2, c9_rslt_m256i);


	c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 32 - 26);

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 3));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 12));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 21));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 3, c9_rslt_m256i);


	c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 32 - 30);

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 7));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 16));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 4, c9_rslt_m256i);


	c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 32 - 25);

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 2));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 11));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 20));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 5, c9_rslt_m256i);


	c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 32 - 29);

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 6));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 15));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 6, c9_rslt_m256i);


	c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 32 - 24);

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 1));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 10));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 19));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 7, c9_rslt_m256i);


	c9_rslt_m256i = _mm256_srli_epi32(c9_load_rslt_m256i, 32 - 28);

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 5));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 14));

	c9_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c9_and_mask_m256i);
	c9_rslt_m256i = _mm256_or_si256(c9_rslt_m256i, _mm256_slli_epi32(c9_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 8, c9_rslt_m256i);
}


// 10-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c10(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c10_and_mask_m256i = _mm256_set1_epi32(0x03ff);
	__m256i c10_load_rslt_m256i, c10_rslt_m256i;


	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c10_and_mask_m256i);
	c10_rslt_m256i = c10_load_rslt_m256i;

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 10));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 20));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 0, c10_rslt_m256i);


	c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 32 - 30);

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 8));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 18));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 1, c10_rslt_m256i);


	c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 32 - 28);

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 6));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 16));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 2, c10_rslt_m256i);


	c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 32 - 26);

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 4));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 14));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 3, c10_rslt_m256i);


	c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 32 - 24);

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 2));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 12));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 4, c10_rslt_m256i);


	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c10_and_mask_m256i);
	c10_rslt_m256i = c10_load_rslt_m256i;

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 10));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 20));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 5, c10_rslt_m256i);


	c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 32 - 30);

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 8));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 18));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 6, c10_rslt_m256i);


	c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 32 - 28);

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 6));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 16));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 7, c10_rslt_m256i);


	c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 32 - 26);

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 4));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 14));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 8, c10_rslt_m256i);


	c10_rslt_m256i = _mm256_srli_epi32(c10_load_rslt_m256i, 32 - 24);

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 2));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 12));

	c10_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c10_and_mask_m256i);
	c10_rslt_m256i = _mm256_or_si256(c10_rslt_m256i, _mm256_slli_epi32(c10_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 9, c10_rslt_m256i);
}


// 11-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c11(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c11_and_mask_m256i = _mm256_set1_epi32(0x07ff);
	__m256i c11_load_rslt_m256i, c11_rslt_m256i;


	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c11_and_mask_m256i);
	c11_rslt_m256i = c11_load_rslt_m256i;

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 11));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 0, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 22);

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 1));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 12));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 1, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 23);

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 2));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 13));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 2, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 24);

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 3));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 14));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 3, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 25);

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 4));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 15));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 4, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 26);

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 5));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 16));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 5, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 27);

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 6));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 17));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 6, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 28);

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 7));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 18));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 7, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 29);

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 8));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 19));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 8, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 30);

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 9));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 20));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 9, c11_rslt_m256i);


	c11_rslt_m256i = _mm256_srli_epi32(c11_load_rslt_m256i, 32 - 31);

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 10));

	c11_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c11_and_mask_m256i);
	c11_rslt_m256i = _mm256_or_si256(c11_rslt_m256i, _mm256_slli_epi32(c11_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 10, c11_rslt_m256i);
}


// 12-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c12(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c12_and_mask_m256i = _mm256_set1_epi32(0x0fff);
	__m256i c12_load_rslt_m256i, c12_rslt_m256i;


	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c12_and_mask_m256i);
	c12_rslt_m256i = c12_load_rslt_m256i;

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 12));

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 0, c12_rslt_m256i);


	c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 32 - 24);

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 4));

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 16));

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 1, c12_rslt_m256i);


	c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 32 - 28);

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 8));

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 2, c12_rslt_m256i);


	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c12_and_mask_m256i);
	c12_rslt_m256i = c12_load_rslt_m256i;

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 12));

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 3, c12_rslt_m256i);


	c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 32 - 24);

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 4));

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 16));

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 4, c12_rslt_m256i);


	c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 32 - 28);

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 8));

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 5, c12_rslt_m256i);


	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c12_and_mask_m256i);
	c12_rslt_m256i = c12_load_rslt_m256i;

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 12));

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 6, c12_rslt_m256i);


	c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 32 - 24);

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 4));

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 16));

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 7, c12_rslt_m256i);


	c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 32 - 28);

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 8));

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 8, c12_rslt_m256i);


	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c12_and_mask_m256i);
	c12_rslt_m256i = c12_load_rslt_m256i;

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 12));

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 9, c12_rslt_m256i);


	c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 32 - 24);

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 4));

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 16));

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 10, c12_rslt_m256i);


	c12_rslt_m256i = _mm256_srli_epi32(c12_load_rslt_m256i, 32 - 28);

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 8));

	c12_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c12_and_mask_m256i);
	c12_rslt_m256i = _mm256_or_si256(c12_rslt_m256i, _mm256_slli_epi32(c12_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 11, c12_rslt_m256i);
}


// 13-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c13(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c13_and_mask_m256i = _mm256_set1_epi32(0x1fff);
	__m256i c13_load_rslt_m256i, c13_rslt_m256i;


	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c13_and_mask_m256i);
	c13_rslt_m256i = c13_load_rslt_m256i;

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 13));

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 0, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 26);

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 7));

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 1, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 20);

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 1));

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 14));

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 2, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 27);

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 8));

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 3, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 21);

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 2));

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 15));

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 4, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 28);

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 9));

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 5, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 22);

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 3));

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 16));

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 6, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 29);

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 10));

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 7, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 23);

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 4));

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 17));

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 8, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 30);

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 11));

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 9, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 24);

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 5));

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 18));

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 10, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 31);

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 12));

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 11, c13_rslt_m256i);


	c13_rslt_m256i = _mm256_srli_epi32(c13_load_rslt_m256i, 32 - 25);

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 6));

	c13_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c13_and_mask_m256i);
	c13_rslt_m256i = _mm256_or_si256(c13_rslt_m256i, _mm256_slli_epi32(c13_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 12, c13_rslt_m256i);
}


// 14-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c14(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c14_and_mask_m256i = _mm256_set1_epi32(0x3fff);
	__m256i c14_load_rslt_m256i, c14_rslt_m256i;


	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c14_and_mask_m256i);
	c14_rslt_m256i = c14_load_rslt_m256i;

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 14));

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 0, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 28);

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 10));

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 1, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 24);

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 6));

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 2, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 20);

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 2));

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 16));

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 3, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 30);

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 12));

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 4, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 26);

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 8));

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 5, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 22);

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 4));

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 6, c14_rslt_m256i);


	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c14_and_mask_m256i);
	c14_rslt_m256i = c14_load_rslt_m256i;

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 14));

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 7, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 28);

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 10));

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 8, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 24);

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 6));

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 9, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 20);

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 2));

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 16));

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 10, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 30);

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 12));

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 11, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 26);

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 8));

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 12, c14_rslt_m256i);


	c14_rslt_m256i = _mm256_srli_epi32(c14_load_rslt_m256i, 32 - 22);

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 4));

	c14_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c14_and_mask_m256i);
	c14_rslt_m256i = _mm256_or_si256(c14_rslt_m256i, _mm256_slli_epi32(c14_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 13, c14_rslt_m256i);
}


// 15-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c15(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c15_and_mask_m256i = _mm256_set1_epi32(0x7fff);
	__m256i c15_load_rslt_m256i, c15_rslt_m256i;


	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c15_and_mask_m256i);
	c15_rslt_m256i = c15_load_rslt_m256i;

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 15));

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 0, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 30);

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 13));

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 1, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 28);

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 11));

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 2, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 26);

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 9));

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 3, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 24);

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 7));

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 4, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 22);

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 5));

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 5, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 20);

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 3));

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 6, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 18);

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 1));

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 16));

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 7, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 31);

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 14));

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 8, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 29);

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 12));

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 9, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 27);

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 10));

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 10, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 25);

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 8));

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 11, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 23);

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 6));

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 12, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 21);

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 4));

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 13, c15_rslt_m256i);


	c15_rslt_m256i = _mm256_srli_epi32(c15_load_rslt_m256i, 32 - 19);

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 2));

	c15_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c15_and_mask_m256i);
	c15_rslt_m256i = _mm256_or_si256(c15_rslt_m256i, _mm256_slli_epi32(c15_load_rslt_m256i, 17));

	_mm256_storeu_si256(out + 14, c15_rslt_m256i);
}


// 16-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c16(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c16_and_mask_m256i = _mm256_set1_epi32(0xffff);
	__m256i c16_load_rslt_m256i, c16_rslt_m256i;


	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c16_and_mask_m256i);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c16_and_mask_m256i);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 0, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c16_and_mask_m256i);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c16_and_mask_m256i);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 1, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c16_and_mask_m256i);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c16_and_mask_m256i);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 2, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c16_and_mask_m256i);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c16_and_mask_m256i);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 3, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c16_and_mask_m256i);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c16_and_mask_m256i);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 4, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c16_and_mask_m256i);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c16_and_mask_m256i);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 5, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c16_and_mask_m256i);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c16_and_mask_m256i);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 6, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c16_and_mask_m256i);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c16_and_mask_m256i);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 7, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c16_and_mask_m256i);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c16_and_mask_m256i);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 8, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c16_and_mask_m256i);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c16_and_mask_m256i);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 9, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c16_and_mask_m256i);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c16_and_mask_m256i);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 10, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c16_and_mask_m256i);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c16_and_mask_m256i);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 11, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c16_and_mask_m256i);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c16_and_mask_m256i);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 12, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c16_and_mask_m256i);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c16_and_mask_m256i);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 13, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c16_and_mask_m256i);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c16_and_mask_m256i);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 14, c16_rslt_m256i);


	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c16_and_mask_m256i);
	c16_rslt_m256i = c16_load_rslt_m256i;

	c16_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c16_and_mask_m256i);
	c16_rslt_m256i = _mm256_or_si256(c16_rslt_m256i, _mm256_slli_epi32(c16_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 15, c16_rslt_m256i);
}


// 17-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c17(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c17_and_mask_m256i = _mm256_set1_epi32(0x01ffff);
	__m256i c17_load_rslt_m256i, c17_rslt_m256i;


	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c17_and_mask_m256i);
	c17_rslt_m256i = c17_load_rslt_m256i;

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 17));

	_mm256_storeu_si256(out + 0, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 17);

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 2));

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 1, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 19);

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 4));

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 2, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 21);

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 6));

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 3, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 23);

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 8));

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 4, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 25);

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 10));

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 5, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 27);

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 12));

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 6, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 29);

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 14));

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 7, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 31);

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 8, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 16);

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 1));

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 9, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 18);

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 3));

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 10, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 20);

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 5));

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 11, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 22);

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 7));

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 12, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 24);

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 9));

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 13, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 26);

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 11));

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 14, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 28);

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 13));

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 15, c17_rslt_m256i);


	c17_rslt_m256i = _mm256_srli_epi32(c17_load_rslt_m256i, 32 - 30);

	c17_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c17_and_mask_m256i);
	c17_rslt_m256i = _mm256_or_si256(c17_rslt_m256i, _mm256_slli_epi32(c17_load_rslt_m256i, 15));

	_mm256_storeu_si256(out + 16, c17_rslt_m256i);
}


// 18-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c18(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c18_and_mask_m256i = _mm256_set1_epi32(0x03ffff);
	__m256i c18_load_rslt_m256i, c18_rslt_m256i;


	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c18_and_mask_m256i);
	c18_rslt_m256i = c18_load_rslt_m256i;

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 0, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 18);

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 4));

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 1, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 22);

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 8));

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 2, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 26);

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 12));

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 3, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 30);

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 4, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 16);

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 2));

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 5, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 20);

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 6));

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 6, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 24);

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 10));

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 7, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 28);

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 8, c18_rslt_m256i);


	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c18_and_mask_m256i);
	c18_rslt_m256i = c18_load_rslt_m256i;

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 9, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 18);

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 4));

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 10, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 22);

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 8));

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 11, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 26);

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 12));

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 12, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 30);

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 13, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 16);

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 2));

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 14, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 20);

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 6));

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 15, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 24);

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 10));

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 16, c18_rslt_m256i);


	c18_rslt_m256i = _mm256_srli_epi32(c18_load_rslt_m256i, 32 - 28);

	c18_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c18_and_mask_m256i);
	c18_rslt_m256i = _mm256_or_si256(c18_rslt_m256i, _mm256_slli_epi32(c18_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 17, c18_rslt_m256i);
}


// 19-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c19(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c19_and_mask_m256i = _mm256_set1_epi32(0x07ffff);
	__m256i c19_load_rslt_m256i, c19_rslt_m256i;


	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c19_and_mask_m256i);
	c19_rslt_m256i = c19_load_rslt_m256i;

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 0, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 19);

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 6));

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 1, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 25);

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 12));

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 2, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 31);

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 3, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 18);

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 5));

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 4, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 24);

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 11));

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 5, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 30);

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 17));

	_mm256_storeu_si256(out + 6, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 17);

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 4));

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 7, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 23);

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 10));

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 8, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 29);

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 9, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 16);

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 3));

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 10, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 22);

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 9));

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 11, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 28);

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 15));

	_mm256_storeu_si256(out + 12, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 15);

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 2));

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 13, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 21);

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 8));

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 14, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 27);

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 15, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 14);

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 1));

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 16, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 20);

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 7));

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 17, c19_rslt_m256i);


	c19_rslt_m256i = _mm256_srli_epi32(c19_load_rslt_m256i, 32 - 26);

	c19_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c19_and_mask_m256i);
	c19_rslt_m256i = _mm256_or_si256(c19_rslt_m256i, _mm256_slli_epi32(c19_load_rslt_m256i, 13));

	_mm256_storeu_si256(out + 18, c19_rslt_m256i);
}


// 20-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c20(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c20_and_mask_m256i = _mm256_set1_epi32(0x0fffff);
	__m256i c20_load_rslt_m256i, c20_rslt_m256i;


	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c20_and_mask_m256i);
	c20_rslt_m256i = c20_load_rslt_m256i;

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 0, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 20);

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 8));

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 1, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 28);

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 2, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 16);

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 4));

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 3, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 24);

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 4, c20_rslt_m256i);


	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c20_and_mask_m256i);
	c20_rslt_m256i = c20_load_rslt_m256i;

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 5, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 20);

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 8));

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 6, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 28);

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 7, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 16);

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 4));

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 8, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 24);

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 9, c20_rslt_m256i);


	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c20_and_mask_m256i);
	c20_rslt_m256i = c20_load_rslt_m256i;

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 10, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 20);

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 8));

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 11, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 28);

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 12, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 16);

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 4));

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 13, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 24);

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 14, c20_rslt_m256i);


	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c20_and_mask_m256i);
	c20_rslt_m256i = c20_load_rslt_m256i;

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 15, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 20);

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 8));

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 16, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 28);

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 17, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 16);

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 4));

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 18, c20_rslt_m256i);


	c20_rslt_m256i = _mm256_srli_epi32(c20_load_rslt_m256i, 32 - 24);

	c20_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c20_and_mask_m256i);
	c20_rslt_m256i = _mm256_or_si256(c20_rslt_m256i, _mm256_slli_epi32(c20_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 19, c20_rslt_m256i);
}


// 21-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c21(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c21_and_mask_m256i = _mm256_set1_epi32(0x1fffff);
	__m256i c21_load_rslt_m256i, c21_rslt_m256i;


	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c21_and_mask_m256i);
	c21_rslt_m256i = c21_load_rslt_m256i;

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 0, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 21);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 10));

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 1, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 31);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 2, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 20);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 9));

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 3, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 30);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 4, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 19);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 8));

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 5, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 29);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 6, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 18);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 7));

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 7, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 28);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 17));

	_mm256_storeu_si256(out + 8, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 17);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 6));

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 9, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 27);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 10, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 16);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 5));

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 11, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 26);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 15));

	_mm256_storeu_si256(out + 12, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 15);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 4));

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 13, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 25);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 14, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 14);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 3));

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 15, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 24);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 13));

	_mm256_storeu_si256(out + 16, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 13);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 2));

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 17, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 23);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 18, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 12);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 1));

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 19, c21_rslt_m256i);


	c21_rslt_m256i = _mm256_srli_epi32(c21_load_rslt_m256i, 32 - 22);

	c21_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c21_and_mask_m256i);
	c21_rslt_m256i = _mm256_or_si256(c21_rslt_m256i, _mm256_slli_epi32(c21_load_rslt_m256i, 11));

	_mm256_storeu_si256(out + 20, c21_rslt_m256i);
}


// 22-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c22(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c22_and_mask_m256i = _mm256_set1_epi32(0x3fffff);
	__m256i c22_load_rslt_m256i, c22_rslt_m256i;


	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c22_and_mask_m256i);
	c22_rslt_m256i = c22_load_rslt_m256i;

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 0, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 22);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 1, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 12);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 2));

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 2, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 24);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 3, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 14);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 4));

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 4, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 26);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 5, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 16);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 6));

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 6, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 28);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 7, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 18);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 8));

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 8, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 30);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 9, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 20);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 10, c22_rslt_m256i);


	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c22_and_mask_m256i);
	c22_rslt_m256i = c22_load_rslt_m256i;

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 11, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 22);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 12, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 12);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 2));

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 13, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 24);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 14, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 14);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 4));

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 15, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 26);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 16, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 16);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 6));

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 17, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 28);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 18, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 18);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 8));

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 19, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 30);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 20, c22_rslt_m256i);


	c22_rslt_m256i = _mm256_srli_epi32(c22_load_rslt_m256i, 32 - 20);

	c22_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c22_and_mask_m256i);
	c22_rslt_m256i = _mm256_or_si256(c22_rslt_m256i, _mm256_slli_epi32(c22_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 21, c22_rslt_m256i);
}


// 23-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c23(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c23_and_mask_m256i = _mm256_set1_epi32(0x7fffff);
	__m256i c23_load_rslt_m256i, c23_rslt_m256i;


	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c23_and_mask_m256i);
	c23_rslt_m256i = c23_load_rslt_m256i;

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 0, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 23);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 1, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 14);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 5));

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 2, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 28);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 3, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 19);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 4, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 10);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 1));

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 5, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 24);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 15));

	_mm256_storeu_si256(out + 6, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 15);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 6));

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 7, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 29);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 8, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 20);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 11));

	_mm256_storeu_si256(out + 9, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 11);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 2));

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 10, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 25);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 11, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 16);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 7));

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 12, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 30);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 13, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 21);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 14, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 12);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 3));

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 15, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 26);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 17));

	_mm256_storeu_si256(out + 16, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 17);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 8));

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 17, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 31);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 18, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 22);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 13));

	_mm256_storeu_si256(out + 19, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 13);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 4));

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 20, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 27);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 21, c23_rslt_m256i);


	c23_rslt_m256i = _mm256_srli_epi32(c23_load_rslt_m256i, 32 - 18);

	c23_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c23_and_mask_m256i);
	c23_rslt_m256i = _mm256_or_si256(c23_rslt_m256i, _mm256_slli_epi32(c23_load_rslt_m256i, 9));

	_mm256_storeu_si256(out + 22, c23_rslt_m256i);
}


// 24-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c24(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c24_and_mask_m256i = _mm256_set1_epi32(0xffffff);
	__m256i c24_load_rslt_m256i, c24_rslt_m256i;


	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c24_and_mask_m256i);
	c24_rslt_m256i = c24_load_rslt_m256i;

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 0, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 24);

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 1, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 16);

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 2, c24_rslt_m256i);


	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c24_and_mask_m256i);
	c24_rslt_m256i = c24_load_rslt_m256i;

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 3, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 24);

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 4, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 16);

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 5, c24_rslt_m256i);


	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c24_and_mask_m256i);
	c24_rslt_m256i = c24_load_rslt_m256i;

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 6, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 24);

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 7, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 16);

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 8, c24_rslt_m256i);


	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c24_and_mask_m256i);
	c24_rslt_m256i = c24_load_rslt_m256i;

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 9, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 24);

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 10, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 16);

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 11, c24_rslt_m256i);


	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c24_and_mask_m256i);
	c24_rslt_m256i = c24_load_rslt_m256i;

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 12, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 24);

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 13, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 16);

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 14, c24_rslt_m256i);


	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c24_and_mask_m256i);
	c24_rslt_m256i = c24_load_rslt_m256i;

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 15, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 24);

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 16, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 16);

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 17, c24_rslt_m256i);


	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c24_and_mask_m256i);
	c24_rslt_m256i = c24_load_rslt_m256i;

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 18, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 24);

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 19, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 16);

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 20, c24_rslt_m256i);


	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c24_and_mask_m256i);
	c24_rslt_m256i = c24_load_rslt_m256i;

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 21, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 24);

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 22, c24_rslt_m256i);


	c24_rslt_m256i = _mm256_srli_epi32(c24_load_rslt_m256i, 32 - 16);

	c24_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c24_and_mask_m256i);
	c24_rslt_m256i = _mm256_or_si256(c24_rslt_m256i, _mm256_slli_epi32(c24_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 23, c24_rslt_m256i);
}


// 25-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c25(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c25_and_mask_m256i = _mm256_set1_epi32(0x01ffffff);
	__m256i c25_load_rslt_m256i, c25_rslt_m256i;


	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c25_and_mask_m256i);
	c25_rslt_m256i = c25_load_rslt_m256i;

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 0, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 25);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 1, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 18);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 11));

	_mm256_storeu_si256(out + 2, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 11);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 4));

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 3, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 29);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 4, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 22);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 15));

	_mm256_storeu_si256(out + 5, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 15);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 6, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 8);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 1));

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 7, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 26);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 8, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 19);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 9, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 12);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 5));

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 10, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 30);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 11, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 23);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 12, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 16);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 9));

	_mm256_storeu_si256(out + 13, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 9);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 2));

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 14, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 27);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 15, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 20);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 13));

	_mm256_storeu_si256(out + 16, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 13);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 6));

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 17, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 31);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 18, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 24);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 17));

	_mm256_storeu_si256(out + 19, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 17);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 20, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 10);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 3));

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 21, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 28);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 22, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 21);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 23, c25_rslt_m256i);


	c25_rslt_m256i = _mm256_srli_epi32(c25_load_rslt_m256i, 32 - 14);

	c25_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c25_and_mask_m256i);
	c25_rslt_m256i = _mm256_or_si256(c25_rslt_m256i, _mm256_slli_epi32(c25_load_rslt_m256i, 7));

	_mm256_storeu_si256(out + 24, c25_rslt_m256i);
}


// 26-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c26(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c26_and_mask_m256i = _mm256_set1_epi32(0x03ffffff);
	__m256i c26_load_rslt_m256i, c26_rslt_m256i;


	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c26_and_mask_m256i);
	c26_rslt_m256i = c26_load_rslt_m256i;

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 0, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 26);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 1, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 20);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 2, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 14);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 3, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 8);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 2));

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 4, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 28);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 5, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 22);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 6, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 16);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 7, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 10);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 4));

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 8, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 30);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 9, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 24);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 10, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 18);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 11, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 12);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 6));

	_mm256_storeu_si256(out + 12, c26_rslt_m256i);


	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c26_and_mask_m256i);
	c26_rslt_m256i = c26_load_rslt_m256i;

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 13, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 26);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 14, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 20);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 15, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 14);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 16, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 8);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 2));

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 17, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 28);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 18, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 22);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 19, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 16);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 20, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 10);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 4));

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 21, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 30);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 22, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 24);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 23, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 18);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 24, c26_rslt_m256i);


	c26_rslt_m256i = _mm256_srli_epi32(c26_load_rslt_m256i, 32 - 12);

	c26_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c26_and_mask_m256i);
	c26_rslt_m256i = _mm256_or_si256(c26_rslt_m256i, _mm256_slli_epi32(c26_load_rslt_m256i, 6));

	_mm256_storeu_si256(out + 25, c26_rslt_m256i);
}


// 27-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c27(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c27_and_mask_m256i = _mm256_set1_epi32(0x07ffffff);
	__m256i c27_load_rslt_m256i, c27_rslt_m256i;


	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c27_and_mask_m256i);
	c27_rslt_m256i = c27_load_rslt_m256i;

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 0, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 27);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 1, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 22);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 17));

	_mm256_storeu_si256(out + 2, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 17);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 3, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 12);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 7));

	_mm256_storeu_si256(out + 4, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 7);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 2));

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 5, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 29);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 6, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 24);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 7, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 19);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 8, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 14);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 9));

	_mm256_storeu_si256(out + 9, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 9);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 4));

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 10, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 31);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 11, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 26);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 12, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 21);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 13, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 16);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 11));

	_mm256_storeu_si256(out + 14, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 11);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 6));

	_mm256_storeu_si256(out + 15, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 6);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 1));

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 16, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 28);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 17, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 23);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 18, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 18);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 13));

	_mm256_storeu_si256(out + 19, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 13);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 20, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 8);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 3));

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 21, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 30);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 22, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 25);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 23, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 20);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 15));

	_mm256_storeu_si256(out + 24, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 15);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 25, c27_rslt_m256i);


	c27_rslt_m256i = _mm256_srli_epi32(c27_load_rslt_m256i, 32 - 10);

	c27_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c27_and_mask_m256i);
	c27_rslt_m256i = _mm256_or_si256(c27_rslt_m256i, _mm256_slli_epi32(c27_load_rslt_m256i, 5));

	_mm256_storeu_si256(out + 26, c27_rslt_m256i);
}


// 28-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c28(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c28_and_mask_m256i = _mm256_set1_epi32(0x0fffffff);
	__m256i c28_load_rslt_m256i, c28_rslt_m256i;


	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c28_and_mask_m256i);
	c28_rslt_m256i = c28_load_rslt_m256i;

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 0, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 28);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 1, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 24);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 2, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 20);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 3, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 16);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 4, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 12);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 5, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 8);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 4));

	_mm256_storeu_si256(out + 6, c28_rslt_m256i);


	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c28_and_mask_m256i);
	c28_rslt_m256i = c28_load_rslt_m256i;

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 7, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 28);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 8, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 24);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 9, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 20);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 10, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 16);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 11, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 12);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 12, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 8);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 4));

	_mm256_storeu_si256(out + 13, c28_rslt_m256i);


	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c28_and_mask_m256i);
	c28_rslt_m256i = c28_load_rslt_m256i;

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 14, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 28);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 15, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 24);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 16, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 20);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 17, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 16);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 18, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 12);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 19, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 8);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 4));

	_mm256_storeu_si256(out + 20, c28_rslt_m256i);


	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c28_and_mask_m256i);
	c28_rslt_m256i = c28_load_rslt_m256i;

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 21, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 28);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 22, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 24);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 23, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 20);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 24, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 16);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 25, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 12);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 26, c28_rslt_m256i);


	c28_rslt_m256i = _mm256_srli_epi32(c28_load_rslt_m256i, 32 - 8);

	c28_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c28_and_mask_m256i);
	c28_rslt_m256i = _mm256_or_si256(c28_rslt_m256i, _mm256_slli_epi32(c28_load_rslt_m256i, 4));

	_mm256_storeu_si256(out + 27, c28_rslt_m256i);
}


// 29-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c29(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c29_and_mask_m256i = _mm256_set1_epi32(0x1fffffff);
	__m256i c29_load_rslt_m256i, c29_rslt_m256i;


	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c29_and_mask_m256i);
	c29_rslt_m256i = c29_load_rslt_m256i;

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 0, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 29);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 1, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 26);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 2, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 23);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 3, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 20);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 17));

	_mm256_storeu_si256(out + 4, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 17);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 5, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 14);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 11));

	_mm256_storeu_si256(out + 6, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 11);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 7, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 8);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 5));

	_mm256_storeu_si256(out + 8, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 5);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 2));

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 9, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 31);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 10, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 28);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 11, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 25);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 12, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 22);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 13, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 19);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 14, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 16);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 13));

	_mm256_storeu_si256(out + 15, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 13);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 16, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 10);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 7));

	_mm256_storeu_si256(out + 17, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 7);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 4));

	_mm256_storeu_si256(out + 18, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 4);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 1));

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 19, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 30);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 20, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 27);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 21, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 24);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 22, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 21);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 23, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 18);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 15));

	_mm256_storeu_si256(out + 24, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 15);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 25, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 12);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 9));

	_mm256_storeu_si256(out + 26, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 9);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 6));

	_mm256_storeu_si256(out + 27, c29_rslt_m256i);


	c29_rslt_m256i = _mm256_srli_epi32(c29_load_rslt_m256i, 32 - 6);

	c29_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c29_and_mask_m256i);
	c29_rslt_m256i = _mm256_or_si256(c29_rslt_m256i, _mm256_slli_epi32(c29_load_rslt_m256i, 3));

	_mm256_storeu_si256(out + 28, c29_rslt_m256i);
}


// 30-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c30(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c30_and_mask_m256i = _mm256_set1_epi32(0x3fffffff);
	__m256i c30_load_rslt_m256i, c30_rslt_m256i;


	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c30_and_mask_m256i);
	c30_rslt_m256i = c30_load_rslt_m256i;

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 0, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 30);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 1, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 28);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 2, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 26);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 3, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 24);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 4, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 22);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 5, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 20);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 6, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 18);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 7, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 16);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 8, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 14);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 9, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 12);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 10, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 10);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 11, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 8);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 6));

	_mm256_storeu_si256(out + 12, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 6);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 4));

	_mm256_storeu_si256(out + 13, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 4);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 2));

	_mm256_storeu_si256(out + 14, c30_rslt_m256i);


	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c30_and_mask_m256i);
	c30_rslt_m256i = c30_load_rslt_m256i;

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 15, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 30);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 16, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 28);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 17, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 26);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 18, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 24);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 19, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 22);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 20, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 20);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 21, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 18);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 22, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 16);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 23, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 14);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 24, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 12);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 25, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 10);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 26, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 8);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 6));

	_mm256_storeu_si256(out + 27, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 6);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 4));

	_mm256_storeu_si256(out + 28, c30_rslt_m256i);


	c30_rslt_m256i = _mm256_srli_epi32(c30_load_rslt_m256i, 32 - 4);

	c30_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c30_and_mask_m256i);
	c30_rslt_m256i = _mm256_or_si256(c30_rslt_m256i, _mm256_slli_epi32(c30_load_rslt_m256i, 2));

	_mm256_storeu_si256(out + 29, c30_rslt_m256i);
}


// 31-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c31(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const __m256i c31_and_mask_m256i = _mm256_set1_epi32(0x7fffffff);
	__m256i c31_load_rslt_m256i, c31_rslt_m256i;


	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 0), c31_and_mask_m256i);
	c31_rslt_m256i = c31_load_rslt_m256i;

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 1), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 31));

	_mm256_storeu_si256(out + 0, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 31);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 2), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 30));

	_mm256_storeu_si256(out + 1, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 30);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 3), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 29));

	_mm256_storeu_si256(out + 2, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 29);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 4), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 28));

	_mm256_storeu_si256(out + 3, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 28);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 5), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 27));

	_mm256_storeu_si256(out + 4, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 27);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 6), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 26));

	_mm256_storeu_si256(out + 5, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 26);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 7), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 25));

	_mm256_storeu_si256(out + 6, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 25);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 8), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 24));

	_mm256_storeu_si256(out + 7, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 24);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 9), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 23));

	_mm256_storeu_si256(out + 8, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 23);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 10), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 22));

	_mm256_storeu_si256(out + 9, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 22);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 11), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 21));

	_mm256_storeu_si256(out + 10, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 21);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 12), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 20));

	_mm256_storeu_si256(out + 11, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 20);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 13), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 19));

	_mm256_storeu_si256(out + 12, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 19);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 14), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 18));

	_mm256_storeu_si256(out + 13, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 18);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 15), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 17));

	_mm256_storeu_si256(out + 14, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 17);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 16), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 16));

	_mm256_storeu_si256(out + 15, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 16);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 17), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 15));

	_mm256_storeu_si256(out + 16, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 15);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 18), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 14));

	_mm256_storeu_si256(out + 17, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 14);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 19), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 13));

	_mm256_storeu_si256(out + 18, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 13);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 20), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 12));

	_mm256_storeu_si256(out + 19, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 12);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 21), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 11));

	_mm256_storeu_si256(out + 20, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 11);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 22), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 10));

	_mm256_storeu_si256(out + 21, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 10);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 23), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 9));

	_mm256_storeu_si256(out + 22, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 9);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 24), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 8));

	_mm256_storeu_si256(out + 23, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 8);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 25), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 7));

	_mm256_storeu_si256(out + 24, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 7);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 26), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 6));

	_mm256_storeu_si256(out + 25, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 6);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 27), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 5));

	_mm256_storeu_si256(out + 26, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 5);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 28), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 4));

	_mm256_storeu_si256(out + 27, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 4);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 29), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 3));

	_mm256_storeu_si256(out + 28, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 3);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 30), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 2));

	_mm256_storeu_si256(out + 29, c31_rslt_m256i);


	c31_rslt_m256i = _mm256_srli_epi32(c31_load_rslt_m256i, 32 - 2);

	c31_load_rslt_m256i = _mm256_and_si256(_mm256_loadu_si256(in + 31), c31_and_mask_m256i);
	c31_rslt_m256i = _mm256_or_si256(c31_rslt_m256i, _mm256_slli_epi32(c31_load_rslt_m256i, 1));

	_mm256_storeu_si256(out + 30, c31_rslt_m256i);
}


// 32-bit
template <bool IsRiceCoding>
void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256_c32(const __m256i *  __restrict__  in,
		__m256i *  __restrict__  out) {
	const uint32_t *in32 = reinterpret_cast<const uint32_t *>(in);
	uint32_t *out32 = reinterpret_cast<uint32_t *>(out);
	for (uint32_t numberofValuesUnpacked = 0; numberofValuesUnpacked < 256; numberofValuesUnpacked += 32) {
		memcpy32(in32, out32);
		in32 += 32;
		out32 += 32;
	}
}


#endif /* VERTICALAVXUNPACKERIMP_H_ */
