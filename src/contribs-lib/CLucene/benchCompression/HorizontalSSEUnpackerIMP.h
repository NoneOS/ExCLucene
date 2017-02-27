/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef HORIZONTALSSEUNPACKERIMP_H_
#define HORIZONTALSSEUNPACKERIMP_H_

#include "util.h"

template <bool IsRiceCoding>
const __m128i HorizontalSSEUnpacker<IsRiceCoding>::Horizontal_SSE_and_msk_m128i[33] = {
		_mm_set1_epi32(0x00),    // 0-bit (unused)

		_mm_set1_epi32(0x01),    // 1-bit
		_mm_set1_epi32(0x03),    // 2-bit
		_mm_set1_epi32(0x07),    // 3-bit
		_mm_set1_epi32(0x0f),    // 4-bit
		_mm_set1_epi32(0x1f),    // 5-bit
		_mm_set1_epi32(0x3f),    // 6-bit
		_mm_set1_epi32(0x7f),    // 7-bit
		_mm_set1_epi32(0xff),    // 8-bit (unused)

		_mm_set1_epi32(0x01ff),  // 9-bit
		_mm_set1_epi32(0x03ff),  // 10-bit
		_mm_set1_epi32(0x07ff),  // 11-bit
		_mm_set1_epi32(0x0fff),  // 12-bit
		_mm_set1_epi32(0x1fff),  // 13-bit
		_mm_set1_epi32(0x3fff),  // 14-bit
		_mm_set1_epi32(0x7fff),  // 15-bit
		_mm_set1_epi32(0xffff),  // 16-bit (unused)

		_mm_set1_epi32(0x01ffff),    // 17-bit
		_mm_set1_epi32(0x03ffff),    // 18-bit
		_mm_set1_epi32(0x07ffff),    // 19-bit
		_mm_set1_epi32(0x0fffff),    // 20-bit
		_mm_set1_epi32(0x1fffff),    // 21-bit
		_mm_set1_epi32(0x3fffff),    // 22-bit
		_mm_set1_epi32(0x7fffff),    // 23-bit
		_mm_set1_epi32(0xffffff),    // 24-bit (unused)

		_mm_set1_epi32(0x01ffffff),  // 25-bit
		_mm_set1_epi32(0x03ffffff),  // 26-bit
		_mm_set1_epi32(0x07ffffff),  // 27-bit
		_mm_set1_epi32(0x0fffffff),  // 28-bit
		_mm_set1_epi32(0x1fffffff),  // 29-bit
		_mm_set1_epi32(0x3fffffff),  // 30-bit
		_mm_set1_epi32(0x7fffffff),  // 31-bit
		_mm_set1_epi32(0xffffffff)   // 32-bit (unused)
};

template <bool IsRiceCoding>
const __m128i HorizontalSSEUnpacker<IsRiceCoding>::Horizontal_SSE_mul_msk_m128i[33][2] = {
		{  _mm_set_epi32(0x00, 0x00, 0x00, 0x00), _mm_set_epi32(0x00, 0x00, 0x00, 0x00)  },    // 0-bit (unused)

		{  _mm_set_epi32(0x01, 0x02, 0x04, 0x08), _mm_set_epi32(0x01, 0x02, 0x04, 0x08)  },    // 1-bit
		{  _mm_set_epi32(0x01, 0x04, 0x10, 0x40), _mm_set_epi32(0x01, 0x04, 0x10, 0x40)  },    // 2-bit
		{  _mm_set_epi32(0x20, 0x01, 0x08, 0x40), _mm_set_epi32(0x04, 0x20, 0x01, 0x08)  },    // 3-bit
		{  _mm_set_epi32(0x01, 0x10, 0x01, 0x10), _mm_set_epi32(0x01, 0x10, 0x01, 0x10)  },    // 4-bit
		{  _mm_set_epi32(0x01, 0x20, 0x04, 0x80), _mm_set_epi32(0x08, 0x01, 0x20, 0x04)  },    // 5-bit
		{  _mm_set_epi32(0x10, 0x04, 0x01, 0x40), _mm_set_epi32(0x10, 0x04, 0x01, 0x40)  },    // 6-bit
		{  _mm_set_epi32(0x04, 0x02, 0x01, 0x80), _mm_set_epi32(0x08, 0x04, 0x02, 0x01)  },    // 7-bit
		{  _mm_set_epi32(0x01, 0x01, 0x01, 0x01), _mm_set_epi32(0x01, 0x01, 0x01, 0x01)  },    // 8-bit (unused)

		{  _mm_set_epi32(0x01, 0x02, 0x04, 0x08), _mm_set_epi32(0x01, 0x02, 0x04, 0x08)  },    // 9-bit
		{  _mm_set_epi32(0x01, 0x04, 0x10, 0x40), _mm_set_epi32(0x01, 0x04, 0x10, 0x40)  },    // 10-bit
		{  _mm_set_epi32(0x20, 0x01, 0x08, 0x40), _mm_set_epi32(0x04, 0x20, 0x01, 0x08)  },    // 11-bit
		{  _mm_set_epi32(0x01, 0x10, 0x01, 0x10), _mm_set_epi32(0x01, 0x10, 0x01, 0x10)  },    // 12-bit
		{  _mm_set_epi32(0x01, 0x20, 0x04, 0x80), _mm_set_epi32(0x08, 0x01, 0x20, 0x04)  },    // 13-bit
		{  _mm_set_epi32(0x10, 0x04, 0x01, 0x40), _mm_set_epi32(0x10, 0x04, 0x01, 0x40)  },    // 14-bit
		{  _mm_set_epi32(0x04, 0x02, 0x01, 0x80), _mm_set_epi32(0x08, 0x04, 0x02, 0x01)  },    // 15-bit
		{  _mm_set_epi32(0x01, 0x01, 0x01, 0x01), _mm_set_epi32(0x01, 0x01, 0x01, 0x01)  },    // 16-bit (unused)

		{  _mm_set_epi32(0x01, 0x02, 0x04, 0x08), _mm_set_epi32(0x01, 0x02, 0x04, 0x08)  },    // 17-bit
		{  _mm_set_epi32(0x01, 0x04, 0x10, 0x40), _mm_set_epi32(0x01, 0x04, 0x10, 0x40)  },    // 18-bit
		{  _mm_set_epi32(0x20, 0x01, 0x08, 0x40), _mm_set_epi32(0x04, 0x20, 0x01, 0x08)  },    // 19-bit
		{  _mm_set_epi32(0x01, 0x10, 0x01, 0x10), _mm_set_epi32(0x01, 0x10, 0x01, 0x10)  },    // 20-bit
		{  _mm_set_epi32(0x01, 0x20, 0x04, 0x80), _mm_set_epi32(0x08, 0x01, 0x20, 0x04)  },    // 21-bit
		{  _mm_set_epi32(0x10, 0x04, 0x01, 0x40), _mm_set_epi32(0x10, 0x04, 0x01, 0x40)  },    // 22-bit
		{  _mm_set_epi32(0x04, 0x02, 0x01, 0x80), _mm_set_epi32(0x08, 0x04, 0x02, 0x01)  },    // 23-bit
		{  _mm_set_epi32(0x01, 0x01, 0x01, 0x01), _mm_set_epi32(0x01, 0x01, 0x01, 0x01)  },    // 24-bit (unused)

		{  _mm_set_epi32(0x01, 0x02, 0x04, 0x08), _mm_set_epi32(0x01, 0x02, 0x04, 0x08)  },    // 25-bit
		{  _mm_set_epi32(0x01, 0x04, 0x10, 0x40), _mm_set_epi32(0x01, 0x04, 0x10, 0x40)  },    // 26-bit
		{  _mm_set_epi32(0x04, 0x08, 0x01, 0x08), _mm_set_epi32(0x01, 0x08, 0x20, 0x02)  },    // 27-bit (exception)
		{  _mm_set_epi32(0x01, 0x10, 0x01, 0x10), _mm_set_epi32(0x01, 0x10, 0x01, 0x10)  },    // 28-bit
		{  _mm_set_epi32(0x01, 0x01, 0x04, 0x04), _mm_set_epi32(0x01, 0x01, 0x04, 0x04)  },    // 29-bit (exception)
		{  _mm_set_epi32(0x01, 0x01, 0x04, 0x04), _mm_set_epi32(0x01, 0x01, 0x04, 0x04)  },    // 30-bit (exception)
		{  _mm_set_epi32(0x01, 0x01, 0x01, 0x01), _mm_set_epi32(0x01, 0x02, 0x02, 0x02)  },    // 31-bit (exception)
		{  _mm_set_epi32(0x01, 0x01, 0x01, 0x01), _mm_set_epi32(0x01, 0x01, 0x01, 0x01)  }     // 32-bit (unused)
};




// 0-bit
template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c0(const __m128i * __restrict__,
		__m128i *  __restrict__  out) {
	if (!IsRiceCoding) { // NewPFor etc.
		uint32_t *out32 = reinterpret_cast<uint32_t *>(out);
		for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 32) {
			memset32(out);
			out32 += 32;
		}
	}
	else { // Rice Coding
		const uint32_t *quotient32 = reinterpret_cast<const uint32_t *>(quotient);
		uint32_t *out32 = reinterpret_cast<uint32_t *>(out);
		for (uint32_t numberofValuesUnpacked = 0; numberofValuesUnpacked < 128; numberofValuesUnpacked += 32) {
			memcpy32(quotient32, out32);
			quotient32 += 32;
			out32 += 32;
		}
	}
}




// 1-bit
template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack8_c1(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor
		__m128i Horizontal_SSE_c1_shfl_msk_m128i = _mm_set1_epi32(byte);

		__m128i c1_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c1_shfl_msk_m128i);
		__m128i c1_mul_rslt_m128i = _mm_mullo_epi32(c1_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[1][0]);

		// unpacks 1st 4 values
		__m128i c1_srli_rslt_m128i = _mm_srli_epi32(c1_mul_rslt_m128i, 3);
		__m128i c1_rslt_m128i = _mm_and_si128(c1_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[1]);
		_mm_storeu_si128(out++, c1_rslt_m128i);

		// unpacks 2nd 4 values
		c1_srli_rslt_m128i = _mm_srli_epi32(c1_mul_rslt_m128i, 7);
		c1_rslt_m128i = _mm_and_si128(c1_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[1]);
		_mm_storeu_si128(out++, c1_rslt_m128i);
	}
	else { // Rice Coding
		__m128i Horizontal_SSE_c1_shfl_msk_m128i = _mm_set1_epi32(byte);

		__m128i c1_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c1_shfl_msk_m128i);
		__m128i c1_mul_rslt_m128i = _mm_mullo_epi32(c1_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[1][0]);

		// unpacks 1st 4 values
		__m128i c1_srli_rslt_m128i = _mm_srli_epi32(c1_mul_rslt_m128i, 3);
		__m128i c1_and_rslt_m128i = _mm_and_si128(c1_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[1]);
		__m128i c1_rslt_m128i = _mm_or_si128(c1_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 1));
		_mm_storeu_si128(out++, c1_rslt_m128i);

		// unpacks 2nd 4 values
		c1_srli_rslt_m128i = _mm_srli_epi32(c1_mul_rslt_m128i, 7);
		c1_and_rslt_m128i = _mm_and_si128(c1_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[1]);
		c1_rslt_m128i = _mm_or_si128(c1_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 1));
		_mm_storeu_si128(out++, c1_rslt_m128i);
	}
}


template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c1(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	__m128i c1_load_rslt_m128i = _mm_loadu_si128(in);        // 16 bytes; contains 128 values

	__horizontal_sse_unpack8_c1<0>(c1_load_rslt_m128i, out);  // unpacks 1st 8 values

	__horizontal_sse_unpack8_c1<1>(c1_load_rslt_m128i, out);  // unpacks 2nd 8 values

	__horizontal_sse_unpack8_c1<2>(c1_load_rslt_m128i, out);  // unpacks 3rd 8 values

	__horizontal_sse_unpack8_c1<3>(c1_load_rslt_m128i, out);  // unpacks 4th 8 values

	__horizontal_sse_unpack8_c1<4>(c1_load_rslt_m128i, out);  // unpacks 5th 8 values

	__horizontal_sse_unpack8_c1<5>(c1_load_rslt_m128i, out);  // unpacks 6th 8 values

	__horizontal_sse_unpack8_c1<6>(c1_load_rslt_m128i, out);  // unpacks 7th 8 values

	__horizontal_sse_unpack8_c1<7>(c1_load_rslt_m128i, out);  // unpacks 8th 8 values

	__horizontal_sse_unpack8_c1<8>(c1_load_rslt_m128i, out);  // unpacks 9th 8 values

	__horizontal_sse_unpack8_c1<9>(c1_load_rslt_m128i, out);  // unpacks 10th 8 values

	__horizontal_sse_unpack8_c1<10>(c1_load_rslt_m128i, out); // unpacks 11th 8 values

	__horizontal_sse_unpack8_c1<11>(c1_load_rslt_m128i, out); // unpacks 12th 8 values

	__horizontal_sse_unpack8_c1<12>(c1_load_rslt_m128i, out); // unpacks 13th 8 values

	__horizontal_sse_unpack8_c1<13>(c1_load_rslt_m128i, out); // unpacks 14th 8 values

	__horizontal_sse_unpack8_c1<14>(c1_load_rslt_m128i, out); // unpacks 15th 8 values

	__horizontal_sse_unpack8_c1<15>(c1_load_rslt_m128i, out); // unpacks 16th 8 values
}

//// alternative
//template <bool IsRiceCoding>
//void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c1(const __m128i * __restrict__ in,
//		__m128i *  __restrict__  out) {
//	if (!IsRiceCoding) { // NewPFor etc.
//		__m128i c1_load_rslt_m128i = _mm_loadu_si128(in);  // 16 bytes; contains 128 values
//		for (uint32_t byte = 0; byte < 16; ++byte) {       // loops 16 times; unpacks 8 values/loop
//			__m128i Horizontal_SSE_c1_shfl_msk_m128i = _mm_set1_epi32(byte);
//			__m128i c1_shfl_rslt_m128i = _mm_shuffle_epi8(c1_load_rslt_m128i, Horizontal_SSE_c1_shfl_msk_m128i);
//			__m128i c1_mul_rslt_m128i = _mm_mullo_epi32(c1_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[1][0]);
//
//			// unpacks 1st 4 values
//			__m128i c1_srli_rslt_m128i = _mm_srli_epi32(c1_mul_rslt_m128i, 3);
//			__m128i c1_rslt_m128i = _mm_and_si128(c1_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[1]);
//			_mm_storeu_si128(out++, c1_rslt_m128i);
//
//			// unpacks 2nd 4 values
//			c1_srli_rslt_m128i = _mm_srli_epi32(c1_mul_rslt_m128i, 7);
//			c1_rslt_m128i = _mm_and_si128(c1_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[1]);
//			_mm_storeu_si128(out++, c1_rslt_m128i);
//		}
//	}
//	else { // Rice Coding
//		__m128i c1_load_rslt_m128i = _mm_loadu_si128(in);  // 16 bytes; contains 128 values
//		for (uint32_t byte = 0; byte < 16; ++byte) {       // loops 16 times; unpacks 8 values/loop
//			__m128i Horizontal_SSE_c1_shfl_msk_m128i = _mm_set1_epi32(byte);
//			__m128i c1_shfl_rslt_m128i = _mm_shuffle_epi8(c1_load_rslt_m128i, Horizontal_SSE_c1_shfl_msk_m128i);
//			__m128i c1_mul_rslt_m128i = _mm_mullo_epi32(c1_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[1][0]);
//
//			// unpacks 1st 4 values
//			__m128i c1_srli_rslt_m128i = _mm_srli_epi32(c1_mul_rslt_m128i, 3);
//			__m128i c1_and_rslt_m128i = _mm_and_si128(c1_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[1]);
//			__m128i c1_rslt_m128i = _mm_or_si128(c1_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 1));
//			_mm_storeu_si128(out++, c1_rslt_m128i);
//
//			// unpacks 2nd 4 values
//			c1_srli_rslt_m128i = _mm_srli_epi32(c1_mul_rslt_m128i, 7);
//			c1_and_rslt_m128i = _mm_and_si128(c1_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[1]);
//			c1_rslt_m128i = _mm_or_si128(c1_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 1));
//			_mm_storeu_si128(out++, c1_rslt_m128i);
//		}
//	}
//}




// 2-bit
template <bool IsRiceCoding>
template <uint32_t byte>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c2(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c2_shfl_msk_m128i = _mm_set1_epi32(byte);

		__m128i c2_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c2_shfl_msk_m128i);
		__m128i c2_mul_rslt_m128i = _mm_mullo_epi32(c2_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[2][0]);
		__m128i c2_srli_rslt_m128i = _mm_srli_epi32(c2_mul_rslt_m128i, 6);
		__m128i c2_rslt_m128i = _mm_and_si128(c2_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[2]);
		_mm_storeu_si128(out++, c2_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c2_shfl_msk_m128i = _mm_set1_epi32(byte);

		__m128i c2_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c2_shfl_msk_m128i);
		__m128i c2_mul_rslt_m128i = _mm_mullo_epi32(c2_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[2][0]);
		__m128i c2_srli_rslt_m128i = _mm_srli_epi32(c2_mul_rslt_m128i, 6);
		__m128i c2_and_rslt_m128i = _mm_and_si128(c2_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[2]);
		__m128i c2_rslt_m128i = _mm_or_si128(c2_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 2));
		_mm_storeu_si128(out++, c2_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c2(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 64) {
		__m128i c2_load_rslt_m128i = _mm_loadu_si128(in++);        // 16 bytes; contains 64 values

		__horizontal_sse_unpack4_c2<0>(c2_load_rslt_m128i, out);    // unpacks 1st 4 values

		__horizontal_sse_unpack4_c2<1>(c2_load_rslt_m128i, out);    // unpacks 2nd 4 values

		__horizontal_sse_unpack4_c2<2>(c2_load_rslt_m128i, out);    // unpacks 3rd 4 values

		__horizontal_sse_unpack4_c2<3>(c2_load_rslt_m128i, out);    // unpacks 4th 4 values

		__horizontal_sse_unpack4_c2<4>(c2_load_rslt_m128i, out);    // unpacks 5th 4 values

		__horizontal_sse_unpack4_c2<5>(c2_load_rslt_m128i, out);    // unpacks 6th 4 values

		__horizontal_sse_unpack4_c2<6>(c2_load_rslt_m128i, out);    // unpacks 7th 4 values

		__horizontal_sse_unpack4_c2<7>(c2_load_rslt_m128i, out);    // unpacks 8th 4 values

		__horizontal_sse_unpack4_c2<8>(c2_load_rslt_m128i, out);    // unpacks 9th 4 values

		__horizontal_sse_unpack4_c2<9>(c2_load_rslt_m128i, out);    // unpacks 10th 4 values

		__horizontal_sse_unpack4_c2<10>(c2_load_rslt_m128i, out);   // unpacks 11th 4 values

		__horizontal_sse_unpack4_c2<11>(c2_load_rslt_m128i, out);   // unpacks 12th 4 values

		__horizontal_sse_unpack4_c2<12>(c2_load_rslt_m128i, out);   // unpacks 13th 4 values

		__horizontal_sse_unpack4_c2<13>(c2_load_rslt_m128i, out);   // unpacks 14th 4 values

		__horizontal_sse_unpack4_c2<14>(c2_load_rslt_m128i, out);   // unpacks 15th 4 values

		__horizontal_sse_unpack4_c2<15>(c2_load_rslt_m128i, out);   // unpacks 16th 4 values
	}
}

//// alternatives
//template <bool IsRiceCoding>
//template <uint32_t byte>
//forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack8_c2(const __m128i &InReg,
//		__m128i *  __restrict__  &out) {
//	if (!IsRiceCoding) { // NewPFor etc.
//		// unpacks 1st 4 values
//		__m128i Horizontal_SSE_c2_shfl_msk_m128i = _mm_set1_epi32(byte);
//
//		__m128i c2_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c2_shfl_msk_m128i);
//		__m128i c2_mul_rslt_m128i = _mm_mullo_epi32(c2_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[2][0]);
//		__m128i c2_srli_rslt_m128i = _mm_srli_epi32(c2_mul_rslt_m128i, 6);
//		__m128i c2_rslt_m128i = _mm_and_si128(c2_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[2]);
//		_mm_storeu_si128(out++, c2_rslt_m128i);
//
//		// unpacks 2nd 4 values
//		Horizontal_SSE_c2_shfl_msk_m128i = _mm_set1_epi32(byte + 1);
//
//		c2_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c2_shfl_msk_m128i);
//		c2_mul_rslt_m128i = _mm_mullo_epi32(c2_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[2][1]);
//		c2_srli_rslt_m128i = _mm_srli_epi32(c2_mul_rslt_m128i, 6);
//		c2_rslt_m128i = _mm_and_si128(c2_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[2]);
//		_mm_storeu_si128(out++, c2_rslt_m128i);
//	}
//	else {
//		// unpacks 1st 4 values
//		__m128i Horizontal_SSE_c2_shfl_msk_m128i = _mm_set1_epi32(byte);
//
//		__m128i c2_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c2_shfl_msk_m128i);
//		__m128i c2_mul_rslt_m128i = _mm_mullo_epi32(c2_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[2][0]);
//		__m128i c2_srli_rslt_m128i = _mm_srli_epi32(c2_mul_rslt_m128i, 6);
//		__m128i c2_and_rslt_m128i = _mm_and_si128(c2_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[2]);
//		__m128i c2_rslt_m128i = _mm_or_si128(c2_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 2));
//		_mm_storeu_si128(out++, c2_rslt_m128i);
//
//		// unpacks 2nd 4 values
//		Horizontal_SSE_c2_shfl_msk_m128i = _mm_set1_epi32(byte + 1);
//
//		c2_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c2_shfl_msk_m128i);
//		c2_mul_rslt_m128i = _mm_mullo_epi32(c2_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[2][1]);
//		c2_srli_rslt_m128i = _mm_srli_epi32(c2_mul_rslt_m128i, 6);
//		c2_and_rslt_m128i = _mm_and_si128(c2_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[2]);
//		c2_rslt_m128i = _mm_or_si128(c2_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 2));
//		_mm_storeu_si128(out++, c2_rslt_m128i);
//	}
//}
//
//template <bool IsRiceCoding>
//void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c2(const __m128i * __restrict__ in,
//		__m128i *  __restrict__  out) {
//	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 64) {
//		__m128i c2_load_rslt_m128i = _mm_loadu_si128(in++);       // 16 bytes; contains 64 values
//
//		__horizontal_sse_unpack8_c2<0>(c2_load_rslt_m128i, out);   // unpacks 1st 8 values
//
//		__horizontal_sse_unpack8_c2<2>(c2_load_rslt_m128i, out);   // unpacks 2nd 8 values
//
//		__horizontal_sse_unpack8_c2<4>(c2_load_rslt_m128i, out);   // unpacks 3rd 8 values
//
//		__horizontal_sse_unpack8_c2<6>(c2_load_rslt_m128i, out);   // unpacks 4th 8 values
//
//		__horizontal_sse_unpack8_c2<8>(c2_load_rslt_m128i, out);   // unpacks 5th 8 values
//
//		__horizontal_sse_unpack8_c2<10>(c2_load_rslt_m128i, out);  // unpacks 6th 8 values
//
//		__horizontal_sse_unpack8_c2<12>(c2_load_rslt_m128i, out);  // unpacks 7th 8 values
//
//		__horizontal_sse_unpack8_c2<14>(c2_load_rslt_m128i, out);  // unpacks 8th 8 values
//	}
//}
//
//template <bool IsRiceCoding>
//void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c2(const __m128i * __restrict__ in,
//		__m128i *  __restrict__  out) {
//	if (!IsRiceCoding) { // NewPFor etc.
//		for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 64) {
//			__m128i c2_load_rslt_m128i = _mm_loadu_si128(in++);  // 16 bytes; contains 64 values
//			for (uint32_t byte = 0; byte < 16; byte += 2) {      // loops 8 times; unpacks 8 values/loop
//				// unpacks 1st 4 values
//				__m128i Horizontal_SSE_c2_shfl_msk_m128i = _mm_set1_epi32(byte);
//
//				__m128i c2_shfl_rslt_m128i = _mm_shuffle_epi8(c2_load_rslt_m128i, Horizontal_SSE_c2_shfl_msk_m128i);
//				__m128i c2_mul_rslt_m128i = _mm_mullo_epi32(c2_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[2][0]);
//				__m128i c2_srli_rslt_m128i = _mm_srli_epi32(c2_mul_rslt_m128i, 6);
//				__m128i c2_rslt_m128i = _mm_and_si128(c2_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[2]);
//				_mm_storeu_si128(out++, c2_rslt_m128i);
//
//				// unpacks 2nd 4 values
//				Horizontal_SSE_c2_shfl_msk_m128i = _mm_set1_epi32(byte + 1);
//
//				c2_shfl_rslt_m128i = _mm_shuffle_epi8(c2_load_rslt_m128i, Horizontal_SSE_c2_shfl_msk_m128i);
//				c2_mul_rslt_m128i = _mm_mullo_epi32(c2_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[2][1]);
//				c2_srli_rslt_m128i = _mm_srli_epi32(c2_mul_rslt_m128i, 6);
//				c2_rslt_m128i = _mm_and_si128(c2_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[2]);
//				_mm_storeu_si128(out++, c2_rslt_m128i);
//			}
//		}
//	}
//	else { // Rice Coding
//		for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 64) {
//			__m128i c2_load_rslt_m128i = _mm_loadu_si128(in++);  // 16 bytes; contains 64 values
//			for (uint32_t byte = 0; byte < 16; byte += 2) {      // loops 8 times; unpacks 8 values/loop
//				// unpacks 1st 4 values
//				__m128i Horizontal_SSE_c2_shfl_msk_m128i = _mm_set1_epi32(byte);
//
//				__m128i c2_shfl_rslt_m128i = _mm_shuffle_epi8(c2_load_rslt_m128i, Horizontal_SSE_c2_shfl_msk_m128i);
//				__m128i c2_mul_rslt_m128i = _mm_mullo_epi32(c2_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[2][0]);
//				__m128i c2_srli_rslt_m128i = _mm_srli_epi32(c2_mul_rslt_m128i, 6);
//				__m128i c2_and_rslt_m128i = _mm_and_si128(c2_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[2]);
//				__m128i c2_rslt_m128i = _mm_or_si128(c2_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 2));
//				_mm_storeu_si128(out++, c2_rslt_m128i);
//
//				// unpacks 2nd 4 values
//				Horizontal_SSE_c2_shfl_msk_m128i = _mm_set1_epi32(byte + 1);
//
//				c2_shfl_rslt_m128i = _mm_shuffle_epi8(c2_load_rslt_m128i, Horizontal_SSE_c2_shfl_msk_m128i);
//				c2_mul_rslt_m128i = _mm_mullo_epi32(c2_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[2][1]);
//				c2_srli_rslt_m128i = _mm_srli_epi32(c2_mul_rslt_m128i, 6);
//				c2_and_rslt_m128i = _mm_and_si128(c2_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[2]);
//				c2_rslt_m128i = _mm_or_si128(c2_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 2));
//				_mm_storeu_si128(out++, c2_rslt_m128i);
//			}
//		}
//	}
//}




// 3-bit
template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack8_c3(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 1st 4 values
		__m128i Horizontal_SSE_c3_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, 0xff, byte + 1,
				0xff, 0xff, byte + 1, byte,
				0xff, 0xff, 0xff, byte + 0,
				0xff, 0xff, 0xff, byte + 0);

		__m128i c3_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c3_shfl_msk_m128i);
		__m128i c3_mul_rslt_m128i = _mm_mullo_epi32(c3_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[3][0]);
		__m128i c3_srli_rslt_m128i = _mm_srli_epi32(c3_mul_rslt_m128i, 6);
		__m128i c3_rslt_m128i = _mm_and_si128(c3_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[3]);
		_mm_storeu_si128(out++, c3_rslt_m128i);

		// unpacks 2nd 4 values
		Horizontal_SSE_c3_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, 0xff, byte + 2,
				0xff, 0xff, 0xff, byte + 2,
				0xff, 0xff, byte + 2, byte + 1,
				0xff, 0xff, 0xff, byte + 1);

		c3_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c3_shfl_msk_m128i);
		c3_mul_rslt_m128i = _mm_mullo_epi32(c3_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[3][1]);
		c3_srli_rslt_m128i = _mm_srli_epi32(c3_mul_rslt_m128i, 7);
		c3_rslt_m128i = _mm_and_si128(c3_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[3]);
		_mm_storeu_si128(out++, c3_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 1st 4 values
		__m128i Horizontal_SSE_c3_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, 0xff, byte + 1,
				0xff, 0xff, byte + 1, byte,
				0xff, 0xff, 0xff, byte + 0,
				0xff, 0xff, 0xff, byte + 0);

		__m128i c3_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c3_shfl_msk_m128i);
		__m128i c3_mul_rslt_m128i = _mm_mullo_epi32(c3_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[3][0]);
		__m128i c3_srli_rslt_m128i = _mm_srli_epi32(c3_mul_rslt_m128i, 6);
		__m128i c3_and_rslt_m128i = _mm_and_si128(c3_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[3]);
		__m128i c3_rslt_m128i = _mm_or_si128(c3_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 3));
		_mm_storeu_si128(out++, c3_rslt_m128i);

		// unpacks 2nd 4 values
		Horizontal_SSE_c3_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, 0xff, byte + 2,
				0xff, 0xff, 0xff, byte + 2,
				0xff, 0xff, byte + 2, byte + 1,
				0xff, 0xff, 0xff, byte + 1);

		c3_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c3_shfl_msk_m128i);
		c3_mul_rslt_m128i = _mm_mullo_epi32(c3_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[3][1]);
		c3_srli_rslt_m128i = _mm_srli_epi32(c3_mul_rslt_m128i, 7);
		c3_and_rslt_m128i = _mm_and_si128(c3_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[3]);
		c3_rslt_m128i = _mm_or_si128(c3_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 3));
		_mm_storeu_si128(out++, c3_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c3(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	__m128i c3_load_rslt1_m128i = _mm_loadu_si128(in + 0);       // 16 bytes; contains 42 values (43th value is incomplete)

	__horizontal_sse_unpack8_c3<0>(c3_load_rslt1_m128i, out);     // unpacks 1st 8 values

	__horizontal_sse_unpack8_c3<3>(c3_load_rslt1_m128i, out);     // unpacks 2nd 8 values

	__horizontal_sse_unpack8_c3<6>(c3_load_rslt1_m128i, out);     // unpacks 3rd 8 values

	__horizontal_sse_unpack8_c3<9>(c3_load_rslt1_m128i, out);     // unpacks 4th 8 values

	__horizontal_sse_unpack8_c3<12>(c3_load_rslt1_m128i, out);    // unpacks 5th 8 values


	__m128i c3_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c3_alignr_rslt1_m128i = _mm_alignr_epi8(c3_load_rslt2_m128i, c3_load_rslt1_m128i, 15); // 16 bytes; contains 42 values (43th value is incomplete)

	__horizontal_sse_unpack8_c3<0>(c3_alignr_rslt1_m128i, out);   // unpacks 6th 8 values

	__horizontal_sse_unpack8_c3<3>(c3_alignr_rslt1_m128i, out);   // unpacks 7th 8 values

	__horizontal_sse_unpack8_c3<6>(c3_alignr_rslt1_m128i, out);   // unpacks 8th 8 values

	__horizontal_sse_unpack8_c3<9>(c3_alignr_rslt1_m128i, out);   // unpacks 9th 8 values

	__horizontal_sse_unpack8_c3<12>(c3_alignr_rslt1_m128i, out);  // unpacks 10th 8 values


	__m128i c3_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c3_alignr_rslt2_m128i = _mm_alignr_epi8(c3_load_rslt3_m128i, c3_load_rslt2_m128i, 14); // 16 bytes; contains 42 values (43rd value is incomplete)

	__horizontal_sse_unpack8_c3<0>(c3_alignr_rslt2_m128i, out);  // unpacks 11th 8 values

	__horizontal_sse_unpack8_c3<3>(c3_alignr_rslt2_m128i, out);  // unpacks 12th 8 values

	__horizontal_sse_unpack8_c3<6>(c3_alignr_rslt2_m128i, out);  // unpacks 13th 8 values

	__horizontal_sse_unpack8_c3<9>(c3_alignr_rslt2_m128i, out);  // unpacks 14th 8 values

	__horizontal_sse_unpack8_c3<12>(c3_alignr_rslt2_m128i, out); // unpacks 15th 8 values


	__horizontal_sse_unpack8_c3<13>(c3_load_rslt3_m128i, out);   // unpacks 16th 8 values
}




// 4-bit
template <bool IsRiceCoding>
template <uint32_t byte>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c4(const __m128i &InReg,
		__m128i *   __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c4_shfl_msk_m128i = _mm_set_epi32(byte + 1, byte + 1, byte + 0, byte + 0);

		__m128i c4_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c4_shfl_msk_m128i);
		__m128i c4_mul_rslt_m128i = _mm_mullo_epi32(c4_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[4][0]);
		__m128i c4_srli_rslt_m128i = _mm_srli_epi32(c4_mul_rslt_m128i, 4);
		__m128i c4_rslt_m128i = _mm_and_si128(c4_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[4]);
		_mm_storeu_si128(out++, c4_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c4_shfl_msk_m128i = _mm_set_epi32(byte + 1, byte + 1, byte + 0, byte + 0);

		__m128i c4_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c4_shfl_msk_m128i);
		__m128i c4_mul_rslt_m128i = _mm_mullo_epi32(c4_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[4][0]);
		__m128i c4_srli_rslt_m128i = _mm_srli_epi32(c4_mul_rslt_m128i, 4);
		__m128i c4_and_rslt_m128i = _mm_and_si128(c4_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[4]);
		__m128i c4_rslt_m128i = _mm_or_si128(c4_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 4));
		_mm_storeu_si128(out++, c4_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c4(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 32) {
		__m128i c4_load_rslt_m128i = _mm_loadu_si128(in++);       // 16 bytes; contains 32 values

		__horizontal_sse_unpack4_c4<0>(c4_load_rslt_m128i, out);   // unpacks 1st 4 values

		__horizontal_sse_unpack4_c4<2>(c4_load_rslt_m128i, out);   // unpacks 2nd 4 values

		__horizontal_sse_unpack4_c4<4>(c4_load_rslt_m128i, out);   // unpacks 3rd 4 values

		__horizontal_sse_unpack4_c4<6>(c4_load_rslt_m128i, out);   // unpacks 4th 4 values

		__horizontal_sse_unpack4_c4<8>(c4_load_rslt_m128i, out);   // unpacks 5th 4 values

		__horizontal_sse_unpack4_c4<10>(c4_load_rslt_m128i, out);  // unpacks 6th 4 values

		__horizontal_sse_unpack4_c4<12>(c4_load_rslt_m128i, out);  // unpacks 7th 4 values

		__horizontal_sse_unpack4_c4<14>(c4_load_rslt_m128i, out);  // unpacks 8th 4 values
	}
}

 // alternatives
//template <bool IsRiceCoding>
//template <uint32_t byte>
//forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack8_c4(const __m128i &InReg,
//		__m128i *  __restrict__  &out) {
//	if (!IsRiceCoding) { // NewPFor etc.
//		// unpacks 1st 4 values
//		__m128i Horizontal_SSE_c4_shfl_msk_m128i = _mm_set_epi32(byte + 1, byte + 1, byte + 0, byte + 0);
//
//		__m128i c4_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c4_shfl_msk_m128i);
//		__m128i c4_mul_rslt_m128i = _mm_mullo_epi32(c4_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[4][0]);
//		__m128i c4_srli_rslt_m128i = _mm_srli_epi32(c4_mul_rslt_m128i, 4);
//		__m128i c4_rslt_m128i = _mm_and_si128(c4_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[4]);
//		_mm_storeu_si128(out++, c4_rslt_m128i);
//
//		// unpacks 2nd 4 values
//		Horizontal_SSE_c4_shfl_msk_m128i = _mm_set_epi32(byte + 3, byte + 3, byte + 2, byte + 2);
//
//		c4_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c4_shfl_msk_m128i);
//		c4_mul_rslt_m128i = _mm_mullo_epi32(c4_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[4][1]);
//		c4_srli_rslt_m128i = _mm_srli_epi32(c4_mul_rslt_m128i, 4);
//		c4_rslt_m128i = _mm_and_si128(c4_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[4]);
//		_mm_storeu_si128(out++, c4_rslt_m128i);
//	}
//	else { // Rice Coding
//		// unpacks 1st 4 values
//		__m128i Horizontal_SSE_c4_shfl_msk_m128i = _mm_set_epi32(byte + 1, byte + 1, byte + 0, byte + 0);
//
//		__m128i c4_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c4_shfl_msk_m128i);
//		__m128i c4_mul_rslt_m128i = _mm_mullo_epi32(c4_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[4][0]);
//		__m128i c4_srli_rslt_m128i = _mm_srli_epi32(c4_mul_rslt_m128i, 4);
//		__m128i c4_and_rslt_m128i = _mm_and_si128(c4_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[4]);
//		__m128i c4_rslt_m128i = _mm_or_si128(c4_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 4));
//		_mm_storeu_si128(out++, c4_rslt_m128i);
//
//		// unpacks 2nd 4 values
//		Horizontal_SSE_c4_shfl_msk_m128i = _mm_set_epi32(byte + 3, byte + 3, byte + 2, byte + 2);
//
//		c4_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c4_shfl_msk_m128i);
//		c4_mul_rslt_m128i = _mm_mullo_epi32(c4_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[4][1]);
//		c4_srli_rslt_m128i = _mm_srli_epi32(c4_mul_rslt_m128i, 4);
//		c4_and_rslt_m128i = _mm_and_si128(c4_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[4]);
//		c4_rslt_m128i = _mm_or_si128(c4_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 4));
//		_mm_storeu_si128(out++, c4_rslt_m128i);
//	}
//}
//
//template <bool IsRiceCoding>
//void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c4(const __m128i * __restrict__ in,
//		__m128i *  __restrict__  out) {
//	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 32) {
//		__m128i c4_load_rslt_m128i = _mm_loadu_si128(in++);      // 16 bytes; contains 32 values
//
//		__horizontal_sse_unpack8_c4<0>(c4_load_rslt_m128i, out);  // unpacks 1st 8 values
//
//		__horizontal_sse_unpack8_c4<4>(c4_load_rslt_m128i, out);  // unpacks 2nd 8 values
//
//		__horizontal_sse_unpack8_c4<8>(c4_load_rslt_m128i, out);  // unpacks 3rd 8 values
//
//		__horizontal_sse_unpack8_c4<12>(c4_load_rslt_m128i, out); // unpacks 4th 8 values
//	}
//}

//template <bool IsRiceCoding>
//void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c4(const __m128i * __restrict__ in,
//		__m128i *  __restrict__  out) {
//	if (!IsRiceCoding) { // NewPFor etc.
//		for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 32) {
//			__m128i c4_load_rslt_m128i = _mm_loadu_si128(in++);  // 16 bytes; contains 32 values
//			for (uint32_t byte = 0; byte < 16; byte += 4) {      // loops 4 times; unpacks 8 values/loop
//				// unpacks 1st 4 values
//				__m128i Horizontal_SSE_c4_shfl_msk_m128i = _mm_set_epi32(byte + 1, byte + 1, byte + 0, byte + 0);
//
//				__m128i c4_shfl_rslt_m128i = _mm_shuffle_epi8(c4_load_rslt_m128i, Horizontal_SSE_c4_shfl_msk_m128i);
//				__m128i c4_mul_rslt_m128i = _mm_mullo_epi32(c4_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[4][0]);
//				__m128i c4_srli_rslt_m128i = _mm_srli_epi32(c4_mul_rslt_m128i, 4);
//				__m128i c4_rslt_m128i = _mm_and_si128(c4_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[4]);
//				_mm_storeu_si128(out++, c4_rslt_m128i);
//
//				// unpacks 2nd 4 values
//				Horizontal_SSE_c4_shfl_msk_m128i = _mm_set_epi32(byte + 3, byte + 3, byte + 2, byte + 2);
//
//				c4_shfl_rslt_m128i = _mm_shuffle_epi8(c4_load_rslt_m128i, Horizontal_SSE_c4_shfl_msk_m128i);
//				c4_mul_rslt_m128i = _mm_mullo_epi32(c4_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[4][1]);
//				c4_srli_rslt_m128i = _mm_srli_epi32(c4_mul_rslt_m128i, 4);
//				c4_rslt_m128i = _mm_and_si128(c4_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[4]);
//				_mm_storeu_si128(out++, c4_rslt_m128i);
//			}
//		}
//	}
//	else { // Rice Coding
//		for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 32) {
//			__m128i c4_load_rslt_m128i = _mm_loadu_si128(in++);  // 16 bytes; contains 32 values
//			for (uint32_t byte = 0; byte < 16; byte += 4) {      // loops 4 times; unpacks 8 values/loop
//				// unpacks 1st 4 values
//				__m128i Horizontal_SSE_c4_shfl_msk_m128i = _mm_set_epi32(byte + 1, byte + 1, byte + 0, byte + 0);
//
//				__m128i c4_shfl_rslt_m128i = _mm_shuffle_epi8(c4_load_rslt_m128i, Horizontal_SSE_c4_shfl_msk_m128i);
//				__m128i c4_mul_rslt_m128i = _mm_mullo_epi32(c4_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[4][0]);
//				__m128i c4_srli_rslt_m128i = _mm_srli_epi32(c4_mul_rslt_m128i, 4);
//				__m128i c4_and_rslt_m128i = _mm_and_si128(c4_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[4]);
//				__m128i c4_rslt_m128i = _mm_or_si128(c4_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 4));
//				_mm_storeu_si128(out++, c4_rslt_m128i);
//
//				// unpacks 2nd 4 values
//				Horizontal_SSE_c4_shfl_msk_m128i = _mm_set_epi32(byte + 3, byte + 3, byte + 2, byte + 2);
//
//				c4_shfl_rslt_m128i = _mm_shuffle_epi8(c4_load_rslt_m128i, Horizontal_SSE_c4_shfl_msk_m128i);
//				c4_mul_rslt_m128i = _mm_mullo_epi32(c4_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[4][1]);
//				c4_srli_rslt_m128i = _mm_srli_epi32(c4_mul_rslt_m128i, 4);
//				c4_and_rslt_m128i = _mm_and_si128(c4_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[4]);
//				c4_rslt_m128i = _mm_or_si128(c4_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 4));
//				_mm_storeu_si128(out++, c4_rslt_m128i);
//			}
//		}
//	}
//}




// 5-bit
template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack8_c5(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 1st 4 values
		__m128i Horizontal_SSE_c5_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 2, byte + 1,
				0xff, 0xff, 0xff, byte + 1,
				0xff, 0xff, byte + 1, byte + 0,
				0xff, 0xff, 0xff, byte + 0);

		__m128i c5_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c5_shfl_msk_m128i);
		__m128i c5_mul_rslt_m128i = _mm_mullo_epi32(c5_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[5][0]);
		__m128i c5_srli_rslt_m128i = _mm_srli_epi32(c5_mul_rslt_m128i, 7);
		__m128i c5_rslt_m128i = _mm_and_si128(c5_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[5]);
		_mm_storeu_si128(out++, c5_rslt_m128i);

		// unpacks 2nd 4 values
		Horizontal_SSE_c5_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, 0xff, byte + 4,
				0xff, 0xff, byte + 4, byte + 3,
				0xff, 0xff, 0xff, byte + 3,
				0xff, 0xff, byte + 3, byte + 2);

		c5_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c5_shfl_msk_m128i);
		c5_mul_rslt_m128i = _mm_mullo_epi32(c5_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[5][1]);
		c5_srli_rslt_m128i = _mm_srli_epi32(c5_mul_rslt_m128i, 6);
		c5_rslt_m128i = _mm_and_si128(c5_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[5]);
		_mm_storeu_si128(out++, c5_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 1st 4 values
		__m128i Horizontal_SSE_c5_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 2, byte + 1,
				0xff, 0xff, 0xff, byte + 1,
				0xff, 0xff, byte + 1, byte + 0,
				0xff, 0xff, 0xff, byte + 0);

		__m128i c5_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c5_shfl_msk_m128i);
		__m128i c5_mul_rslt_m128i = _mm_mullo_epi32(c5_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[5][0]);
		__m128i c5_srli_rslt_m128i = _mm_srli_epi32(c5_mul_rslt_m128i, 7);
		__m128i c5_and_rslt_m128i = _mm_and_si128(c5_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[5]);
		__m128i c5_rslt_m128i = _mm_or_si128(c5_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 5));
		_mm_storeu_si128(out++, c5_rslt_m128i);

		// unpacks 2nd 4 values
		Horizontal_SSE_c5_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, 0xff, byte + 4,
				0xff, 0xff, byte + 4, byte + 3,
				0xff, 0xff, 0xff, byte + 3,
				0xff, 0xff, byte + 3, byte + 2);

		c5_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c5_shfl_msk_m128i);
		c5_mul_rslt_m128i = _mm_mullo_epi32(c5_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[5][1]);
		c5_srli_rslt_m128i = _mm_srli_epi32(c5_mul_rslt_m128i, 6);
		c5_and_rslt_m128i = _mm_and_si128(c5_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[5]);
		c5_rslt_m128i = _mm_or_si128(c5_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 5));
		_mm_storeu_si128(out++, c5_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c5(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	__m128i c5_load_rslt1_m128i = _mm_loadu_si128(in + 0);      // 16 bytes; contains 25 values (26th value is incomplete)

	__horizontal_sse_unpack8_c5<0>(c5_load_rslt1_m128i, out);    // unpacks 1st 8 values

	__horizontal_sse_unpack8_c5<5>(c5_load_rslt1_m128i, out);    // unpacks 2nd 8 values

	__horizontal_sse_unpack8_c5<10>(c5_load_rslt1_m128i, out);   // unpacks 3rd 8 values


	__m128i c5_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c5_alignr_rslt1_m128i = _mm_alignr_epi8(c5_load_rslt2_m128i, c5_load_rslt1_m128i, 15); // 16 bytes; contains 25 values (26th value is incomplete)

	__horizontal_sse_unpack8_c5<0>(c5_alignr_rslt1_m128i, out);  // unpacks 4th 8 values

	__horizontal_sse_unpack8_c5<5>(c5_alignr_rslt1_m128i, out);  // unpacks 5th 8 values

	__horizontal_sse_unpack8_c5<10>(c5_alignr_rslt1_m128i, out); // unpacks 6th 8 values


	__m128i c5_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c5_alignr_rslt2_m128i = _mm_alignr_epi8(c5_load_rslt3_m128i, c5_load_rslt2_m128i, 14); // 16 bytes; contains 25 values (26th value is incomplete)

	__horizontal_sse_unpack8_c5<0>(c5_alignr_rslt2_m128i, out);  // unpacks 7th 8 values

	__horizontal_sse_unpack8_c5<5>(c5_alignr_rslt2_m128i, out);  // unpacks 8th 8 values

	__horizontal_sse_unpack8_c5<10>(c5_alignr_rslt2_m128i, out); // unpacks 9th 8 values


	__m128i c5_load_rslt4_m128i = _mm_loadu_si128(in + 3);
	__m128i c5_alignr_rslt3_m128i = _mm_alignr_epi8(c5_load_rslt4_m128i, c5_load_rslt3_m128i, 13); // 16 bytes; contains 25 values (26th value is incomplete)

	__horizontal_sse_unpack8_c5<0>(c5_alignr_rslt3_m128i, out);  // unpacks 10th 8 values

	__horizontal_sse_unpack8_c5<5>(c5_alignr_rslt3_m128i, out);  // unpacks 11th 8 values

	__horizontal_sse_unpack8_c5<10>(c5_alignr_rslt3_m128i, out); // unpacks 12th 8 values


	__m128i c5_load_rslt5_m128i = _mm_loadu_si128(in + 4);
	__m128i c5_alignr_rslt4_m128i = _mm_alignr_epi8(c5_load_rslt5_m128i, c5_load_rslt4_m128i, 12); // 16 bytes; contains 25 values (26th value is incomplete)

	__horizontal_sse_unpack8_c5<0>(c5_alignr_rslt4_m128i, out);  // unpacks 13th 8 values

	__horizontal_sse_unpack8_c5<5>(c5_alignr_rslt4_m128i, out);  // unpacks 14th 8 values

	__horizontal_sse_unpack8_c5<10>(c5_alignr_rslt4_m128i, out); // unpacks 15th 8 values


	__horizontal_sse_unpack8_c5<11>(c5_load_rslt5_m128i, out);   // unpacks 16th 8 values
}




// 6-bit
template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c6(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c6_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, 0xff, byte + 2,
				0xff, 0xff, byte + 2, byte + 1,
				0xff, 0xff, byte + 1, byte + 0,
				0xff, 0xff, 0xff, byte + 0);

		__m128i c6_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c6_shfl_msk_m128i);
		__m128i c6_mul_rslt_m128i = _mm_mullo_epi32(c6_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[6][0]);
		__m128i c6_srli_rslt_m128i = _mm_srli_epi32(c6_mul_rslt_m128i, 6);
		__m128i c6_rslt_m128i = _mm_and_si128(c6_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[6]);
		_mm_storeu_si128(out++, c6_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c6_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, 0xff, byte + 2,
				0xff, 0xff, byte + 2, byte + 1,
				0xff, 0xff, byte + 1, byte + 0,
				0xff, 0xff, 0xff, byte + 0);

		__m128i c6_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c6_shfl_msk_m128i);
		__m128i c6_mul_rslt_m128i = _mm_mullo_epi32(c6_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[6][0]);
		__m128i c6_srli_rslt_m128i = _mm_srli_epi32(c6_mul_rslt_m128i, 6);
		__m128i c6_and_rslt_m128i = _mm_and_si128(c6_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[6]);
		__m128i c6_rslt_m128i = _mm_or_si128(c6_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 6));
		_mm_storeu_si128(out++, c6_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c6(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 64) {
		__m128i c6_load_rslt1_m128i = _mm_loadu_si128(in++);         // 16 bytes; contains 21 values (22nd value is incomplete)

		__horizontal_sse_unpack4_c6<0>(c6_load_rslt1_m128i, out);     // unpacks 1st 4 values

		__horizontal_sse_unpack4_c6<3>(c6_load_rslt1_m128i, out);     // unpacks 2nd 4 values

		__horizontal_sse_unpack4_c6<6>(c6_load_rslt1_m128i, out);     // unpacks 3rd 4 values

		__horizontal_sse_unpack4_c6<9>(c6_load_rslt1_m128i, out);     // unpacks 4th 4 values

		__horizontal_sse_unpack4_c6<12>(c6_load_rslt1_m128i, out);    // unpacks 5th 4 values


		__m128i c6_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c6_alignr_rslt1_m128i = _mm_alignr_epi8(c6_load_rslt2_m128i, c6_load_rslt1_m128i, 15); // 16 bytes; contains 21 values (22nd value is incomplete)

		__horizontal_sse_unpack4_c6<0>(c6_alignr_rslt1_m128i, out);    // unpacks 6th 4 values

		__horizontal_sse_unpack4_c6<3>(c6_alignr_rslt1_m128i, out);    // unpacks 7th 4 values

		__horizontal_sse_unpack4_c6<6>(c6_alignr_rslt1_m128i, out);    // unpacks 8th 4 values

		__horizontal_sse_unpack4_c6<9>(c6_alignr_rslt1_m128i, out);    // unpacks 9th 4 values

		__horizontal_sse_unpack4_c6<12>(c6_alignr_rslt1_m128i, out);   // unpacks 10th 4 values


		__m128i c6_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c6_alignr_rslt2_m128i = _mm_alignr_epi8(c6_load_rslt3_m128i, c6_load_rslt2_m128i, 14); // 16 bytes; contains 21 values (22nd value is incomplete)

		__horizontal_sse_unpack4_c6<0>(c6_alignr_rslt2_m128i, out);    // unpacks 11th 4 values

		__horizontal_sse_unpack4_c6<3>(c6_alignr_rslt2_m128i, out);    // unpacks 12th 4 values

		__horizontal_sse_unpack4_c6<6>(c6_alignr_rslt2_m128i, out);    // unpacks 13th 4 values

		__horizontal_sse_unpack4_c6<9>(c6_alignr_rslt2_m128i, out);    // unpacks 14th 4 values

		__horizontal_sse_unpack4_c6<12>(c6_alignr_rslt2_m128i, out);   // unpacks 15th 4 values


		__horizontal_sse_unpack4_c6<13>(c6_load_rslt3_m128i, out);     // unpacks 16th 4 values
	}
}




// 7-bit
template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack8_c7(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 1st 4 values
		__m128i Horizontal_SSE_c7_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 3, byte + 2,
				0xff, 0xff, byte + 2, byte + 1,
				0xff, 0xff, byte + 1, byte + 0,
				0xff, 0xff, 0xff, byte + 0);

		__m128i c7_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c7_shfl_msk_m128i);
		__m128i c7_mul_rslt_m128i = _mm_mullo_epi32(c7_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[7][0]);
		__m128i c7_srli_rslt_m128i = _mm_srli_epi32(c7_mul_rslt_m128i, 7);
		__m128i c7_rslt_m128i = _mm_and_si128(c7_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[7]);
		_mm_storeu_si128(out++, c7_rslt_m128i);

		// unpacks 2nd 4 values
		Horizontal_SSE_c7_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, 0xff, byte + 6,
				0xff, 0xff, byte + 6, byte + 5,
				0xff, 0xff, byte + 5, byte + 4,
				0xff, 0xff, byte + 4, byte + 3);

		c7_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c7_shfl_msk_m128i);
		c7_mul_rslt_m128i = _mm_mullo_epi32(c7_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[7][1]);
		c7_srli_rslt_m128i = _mm_srli_epi32(c7_mul_rslt_m128i, 4);
		c7_rslt_m128i = _mm_and_si128(c7_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[7]);
		_mm_storeu_si128(out++, c7_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 1st 4 values
		__m128i Horizontal_SSE_c7_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 3, byte + 2,
				0xff, 0xff, byte + 2, byte + 1,
				0xff, 0xff, byte + 1, byte + 0,
				0xff, 0xff, 0xff, byte + 0);

		__m128i c7_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c7_shfl_msk_m128i);
		__m128i c7_mul_rslt_m128i = _mm_mullo_epi32(c7_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[7][0]);
		__m128i c7_srli_rslt_m128i = _mm_srli_epi32(c7_mul_rslt_m128i, 7);
		__m128i c7_and_rslt_m128i = _mm_and_si128(c7_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[7]);
		__m128i c7_rslt_m128i = _mm_or_si128(c7_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 7));
		_mm_storeu_si128(out++, c7_rslt_m128i);

		// unpacks 2nd 4 values
		Horizontal_SSE_c7_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, 0xff, byte + 6,
				0xff, 0xff, byte + 6, byte + 5,
				0xff, 0xff, byte + 5, byte + 4,
				0xff, 0xff, byte + 4, byte + 3);

		c7_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c7_shfl_msk_m128i);
		c7_mul_rslt_m128i = _mm_mullo_epi32(c7_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[7][1]);
		c7_srli_rslt_m128i = _mm_srli_epi32(c7_mul_rslt_m128i, 4);
		c7_and_rslt_m128i = _mm_and_si128(c7_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[7]);
		c7_rslt_m128i = _mm_or_si128(c7_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 7));
		_mm_storeu_si128(out++, c7_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c7(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	__m128i c7_load_rslt1_m128i = _mm_loadu_si128(in + 0);      // 16 bytes; contains 18 values (19th value is incomplete)

	__horizontal_sse_unpack8_c7<0>(c7_load_rslt1_m128i, out);    // unpacks 1st 8 values

	__horizontal_sse_unpack8_c7<7>(c7_load_rslt1_m128i, out);    // unpacks 2nd 8 values


	__m128i c7_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c7_alignr_rslt1_m128i = _mm_alignr_epi8(c7_load_rslt2_m128i, c7_load_rslt1_m128i, 14);  // 16 bytes; contains 18 values (19th value is incomplete)

	__horizontal_sse_unpack8_c7<0>(c7_alignr_rslt1_m128i, out);  // unpacks 3rd 8 values

	__horizontal_sse_unpack8_c7<7>(c7_alignr_rslt1_m128i, out);  // unpacks 4th 8 values


	__m128i c7_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c7_alignr_rslt2_m128i = _mm_alignr_epi8(c7_load_rslt3_m128i, c7_load_rslt2_m128i, 12);  // 16 bytes; contains 18 values (19th value is incomplete)

	__horizontal_sse_unpack8_c7<0>(c7_alignr_rslt2_m128i, out);  // unpacks 5th 8 values

	__horizontal_sse_unpack8_c7<7>(c7_alignr_rslt2_m128i, out);  // unpacks 6th 8 values


	__m128i c7_load_rslt4_m128i = _mm_loadu_si128(in + 3);
	__m128i c7_alignr_rslt3_m128i = _mm_alignr_epi8(c7_load_rslt4_m128i, c7_load_rslt3_m128i, 10);  // 16 bytes; contains 18 values (19th value is incomplete)

	__horizontal_sse_unpack8_c7<0>(c7_alignr_rslt3_m128i, out);  // unpacks 7th 8 values

	__horizontal_sse_unpack8_c7<7>(c7_alignr_rslt3_m128i, out);  // unpacks 8th 8 values


	__m128i c7_load_rslt5_m128i = _mm_loadu_si128(in + 4);
	__m128i c7_alignr_rslt4_m128i = _mm_alignr_epi8(c7_load_rslt5_m128i, c7_load_rslt4_m128i, 8);  // 16 bytes; contains 18 values (19th value is incomplete)

	__horizontal_sse_unpack8_c7<0>(c7_alignr_rslt4_m128i, out);  // unpacks 9th 8 values

	__horizontal_sse_unpack8_c7<7>(c7_alignr_rslt4_m128i, out);  // unpacks 10th 8 values


	__m128i c7_load_rslt6_m128i = _mm_loadu_si128(in + 5);
	__m128i c7_alignr_rslt5_m128i = _mm_alignr_epi8(c7_load_rslt6_m128i, c7_load_rslt5_m128i, 6);  // 16 bytes; contains 18 values (19th value is incomplete)

	__horizontal_sse_unpack8_c7<0>(c7_alignr_rslt5_m128i, out);  // unpacks 11th 8 values

	__horizontal_sse_unpack8_c7<7>(c7_alignr_rslt5_m128i, out);  // unpacks 12th 8 values


	__m128i c7_load_rslt7_m128i = _mm_loadu_si128(in + 6);
	__m128i c7_alignr_rslt6_m128i = _mm_alignr_epi8(c7_load_rslt7_m128i, c7_load_rslt6_m128i, 4);  // 16 bytes; contains 18 values (19th value is incomplete)

	__horizontal_sse_unpack8_c7<0>(c7_alignr_rslt6_m128i, out);  // unpacks 13th 8 values

	__horizontal_sse_unpack8_c7<7>(c7_alignr_rslt6_m128i, out);  // unpacks 14th 8 values


	__horizontal_sse_unpack8_c7<2>(c7_load_rslt7_m128i, out);    // unpacks 15th 8 values

	__horizontal_sse_unpack8_c7<9>(c7_load_rslt7_m128i, out);    // unpacks 16th 8 values
}




// 8-bit
template <bool IsRiceCoding>
template <uint32_t byte>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c8(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c8_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, 0xff, byte + 3,
				0xff, 0xff, 0xff, byte + 2,
				0xff, 0xff, 0xff, byte + 1,
				0xff, 0xff, 0xff, byte + 0);

		__m128i c8_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c8_shfl_msk_m128i);
		_mm_storeu_si128(out++, c8_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c8_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, 0xff, byte + 3,
				0xff, 0xff, 0xff, byte + 2,
				0xff, 0xff, 0xff, byte + 1,
				0xff, 0xff, 0xff, byte + 0);

		__m128i c8_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c8_shfl_msk_m128i);
		__m128i c8_rslt_m128i = _mm_or_si128(c8_shfl_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 8));
		_mm_storeu_si128(out++, c8_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c8(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 16) {
		__m128i c8_load_rslt_m128i = _mm_loadu_si128(in++);      // 16 bytes; contains 16 values

		__horizontal_sse_unpack4_c8<0>(c8_load_rslt_m128i, out);  // unpacks 1st 4 values

		__horizontal_sse_unpack4_c8<4>(c8_load_rslt_m128i, out);  // unpacks 2nd 4 values

		__horizontal_sse_unpack4_c8<8>(c8_load_rslt_m128i, out);  // unpacks 3rd 4 values

		__horizontal_sse_unpack4_c8<12>(c8_load_rslt_m128i, out); // unpacks 4th 4 values
	}
}

//// alternatives
//template <bool IsRiceCoding>
//template <uint32_t byte>
//forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack8_c8(const __m128i &InReg,
//		__m128i *  __restrict__  &out) {
//	if (!IsRiceCoding) { // NewPFor etc.
//		// unpacks 1st 4 values
//		__m128i Horizontal_SSE_c8_shfl_msk_m128i = _mm_set_epi8(
//				0xff, 0xff, 0xff, byte + 3,
//				0xff, 0xff, 0xff, byte + 2,
//				0xff, 0xff, 0xff, byte + 1,
//				0xff, 0xff, 0xff, byte + 0);
//
//		__m128i c8_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c8_shfl_msk_m128i);
//		__m128i c8_rslt_m128i = _mm_or_si128(c8_shfl_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 8));
//		_mm_storeu_si128(out++, c8_rslt_m128i);
//
//		// unpacks 2nd 4 values
//		Horizontal_SSE_c8_shfl_msk_m128i = _mm_set_epi8(
//				0xff, 0xff, 0xff, byte + 7,
//				0xff, 0xff, 0xff, byte + 6,
//				0xff, 0xff, 0xff, byte + 5,
//				0xff, 0xff, 0xff, byte + 4);
//
//		c8_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c8_shfl_msk_m128i);
//		c8_rslt_m128i = _mm_or_si128(c8_shfl_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 8));
//		_mm_storeu_si128(out++, c8_rslt_m128i);
//	}
//	else { // Rice Coding
//		// unpacks 1st 4 values
//		__m128i Horizontal_SSE_c8_shfl_msk_m128i = _mm_set_epi8(
//				0xff, 0xff, 0xff, byte + 3,
//				0xff, 0xff, 0xff, byte + 2,
//				0xff, 0xff, 0xff, byte + 1,
//				0xff, 0xff, 0xff, byte + 0);
//
//		__m128i c8_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c8_shfl_msk_m128i);
//		_mm_storeu_si128(out++, c8_rslt_m128i);
//
//		// unpacks 2nd 4 values
//		Horizontal_SSE_c8_shfl_msk_m128i = _mm_set_epi8(
//				0xff, 0xff, 0xff, byte + 7,
//				0xff, 0xff, 0xff, byte + 6,
//				0xff, 0xff, 0xff, byte + 5,
//				0xff, 0xff, 0xff, byte + 4);
//
//		c8_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c8_shfl_msk_m128i);
//		_mm_storeu_si128(out++, c8_rslt_m128i);
//	}
//}
//
//template <bool IsRiceCoding>
//void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c8(const __m128i * __restrict__ in,
//		__m128i *  __restrict__  out) {
//	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 16) {
//		__m128i c8_load_rslt_m128i = _mm_loadu_si128(in++);     // 16 bytes; contains 16 values
//
//		__horizontal_sse_unpack8_c8<0>(c8_load_rslt_m128i, out); // unpacks 1st 8 values
//
//		__horizontal_sse_unpack8_c8<8>(c8_load_rslt_m128i, out); // unpacks 2nd 8 values
//	}
//}
//
//template <bool IsRiceCoding>
//void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c8(const __m128i * __restrict__ in,
//		__m128i *  __restrict__  out) {
//	if (!IsRiceCoding) { // NewPFor etc.
//		for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 16) {
//			__m128i c8_load_rslt_m128i = _mm_loadu_si128(in++);  // 16 bytes; contains 16 values
//			for (uint32_t byte = 0; byte < 16; byte += 8) {      // loops 2 times; unpacks 8 values/loop
//				// unpacks 1st 4 values
//				__m128i Horizontal_SSE_c8_shfl_msk_m128i = _mm_set_epi8(
//						0xff, 0xff, 0xff, byte + 3,
//						0xff, 0xff, 0xff, byte + 2,
//						0xff, 0xff, 0xff, byte + 1,
//						0xff, 0xff, 0xff, byte + 0);
//
//				__m128i c8_rslt_m128i = _mm_shuffle_epi8(c8_load_rslt_m128i, Horizontal_SSE_c8_shfl_msk_m128i);
//				_mm_storeu_si128(out++, c8_rslt_m128i);
//
//				// unpacks 2nd 4 values
//				Horizontal_SSE_c8_shfl_msk_m128i = _mm_set_epi8(
//						0xff, 0xff, 0xff, byte + 7,
//						0xff, 0xff, 0xff, byte + 6,
//						0xff, 0xff, 0xff, byte + 5,
//						0xff, 0xff, 0xff, byte + 4);
//
//				c8_rslt_m128i = _mm_shuffle_epi8(c8_load_rslt_m128i, Horizontal_SSE_c8_shfl_msk_m128i);
//				_mm_storeu_si128(out++, c8_rslt_m128i);
//			}
//		}
//	}
//	else { // Rice Coding
//		for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 16) {
//			__m128i c8_load_rslt_m128i = _mm_loadu_si128(in++);  // 16 bytes; contains 16 values
//			for (uint32_t byte = 0; byte < 16; byte += 8) {      // loops 2 times; unpacks 8 values/loop
//				// unpacks 1st 4 values
//				__m128i Horizontal_SSE_c8_shfl_msk_m128i = _mm_set_epi8(
//						0xff, 0xff, 0xff, byte + 3,
//						0xff, 0xff, 0xff, byte + 2,
//						0xff, 0xff, 0xff, byte + 1,
//						0xff, 0xff, 0xff, byte + 0);
//
//				__m128i c8_shfl_rslt_m128i = _mm_shuffle_epi8(c8_load_rslt_m128i, Horizontal_SSE_c8_shfl_msk_m128i);
//				__m128i c8_rslt_m128i = _mm_or_si128(c8_shfl_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 8));
//				_mm_storeu_si128(out++, c8_rslt_m128i);
//
//				// unpacks 2nd 4 values
//				Horizontal_SSE_c8_shfl_msk_m128i = _mm_set_epi8(
//						0xff, 0xff, 0xff, byte + 7,
//						0xff, 0xff, 0xff, byte + 6,
//						0xff, 0xff, 0xff, byte + 5,
//						0xff, 0xff, 0xff, byte + 4);
//
//				c8_shfl_rslt_m128i = _mm_shuffle_epi8(c8_load_rslt_m128i, Horizontal_SSE_c8_shfl_msk_m128i);
//				c8_rslt_m128i = _mm_or_si128(c8_shfl_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 8));
//				_mm_storeu_si128(out++, c8_rslt_m128i);
//			}
//		}
//	}
//}




// 9-bit
template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack8_c9(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 1st 4 values
		__m128i Horizontal_SSE_c9_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 4, byte + 3,
				0xff, 0xff, byte + 3, byte + 2,
				0xff, 0xff, byte + 2, byte + 1,
				0xff, 0xff, byte + 1, byte + 0);

		__m128i c9_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c9_shfl_msk_m128i);
		__m128i c9_mul_rslt_m128i = _mm_mullo_epi32(c9_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[9][0]);
		__m128i c9_srli_rslt_m128i = _mm_srli_epi32(c9_mul_rslt_m128i, 3);
		__m128i c9_rslt_m128i = _mm_and_si128(c9_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[9]);
		_mm_storeu_si128(out++, c9_rslt_m128i);

		// unpacks 2nd 4 values
		Horizontal_SSE_c9_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 8, byte + 7,
				0xff, 0xff, byte + 7, byte + 6,
				0xff, 0xff, byte + 6, byte + 5,
				0xff, 0xff, byte + 5, byte + 4);

		c9_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c9_shfl_msk_m128i);
		c9_mul_rslt_m128i = _mm_mullo_epi32(c9_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[9][1]);
		c9_srli_rslt_m128i = _mm_srli_epi32(c9_mul_rslt_m128i, 7);
		c9_rslt_m128i = _mm_and_si128(c9_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[9]);
		_mm_storeu_si128(out++, c9_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 1st 4 values
		__m128i Horizontal_SSE_c9_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 4, byte + 3,
				0xff, 0xff, byte + 3, byte + 2,
				0xff, 0xff, byte + 2, byte + 1,
				0xff, 0xff, byte + 1, byte + 0);

		__m128i c9_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c9_shfl_msk_m128i);
		__m128i c9_mul_rslt_m128i = _mm_mullo_epi32(c9_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[9][0]);
		__m128i c9_srli_rslt_m128i = _mm_srli_epi32(c9_mul_rslt_m128i, 3);
		__m128i c9_and_rslt_m128i = _mm_and_si128(c9_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[9]);
		__m128i c9_rslt_m128i = _mm_or_si128(c9_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 9));
		_mm_storeu_si128(out++, c9_rslt_m128i);

		// unpacks 2nd 4 values
		Horizontal_SSE_c9_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 8, byte + 7,
				0xff, 0xff, byte + 7, byte + 6,
				0xff, 0xff, byte + 6, byte + 5,
				0xff, 0xff, byte + 5, byte + 4);

		c9_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c9_shfl_msk_m128i);
		c9_mul_rslt_m128i = _mm_mullo_epi32(c9_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[9][1]);
		c9_srli_rslt_m128i = _mm_srli_epi32(c9_mul_rslt_m128i, 7);
		c9_and_rslt_m128i = _mm_and_si128(c9_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[9]);
		c9_rslt_m128i = _mm_or_si128(c9_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 9));
		_mm_storeu_si128(out++, c9_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c9(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	__m128i c9_load_rslt1_m128i = _mm_loadu_si128(in + 0);

	__horizontal_sse_unpack8_c9<0>(c9_load_rslt1_m128i, out);     // unpacks 1st 8 values


	__m128i c9_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c9_alignr_rslt1_m128i = _mm_alignr_epi8(c9_load_rslt2_m128i, c9_load_rslt1_m128i, 9);

	__horizontal_sse_unpack8_c9<0>(c9_alignr_rslt1_m128i, out);   // unpacks 2nd 8 values

	__horizontal_sse_unpack8_c9<2>(c9_load_rslt2_m128i, out);     // unpacks 3rd 8 values


	__m128i c9_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c9_alignr_rslt2_m128i = _mm_alignr_epi8(c9_load_rslt3_m128i, c9_load_rslt2_m128i, 11);

	__horizontal_sse_unpack8_c9<0>(c9_alignr_rslt2_m128i, out);   // unpacks 4th 8 values

	__horizontal_sse_unpack8_c9<4>(c9_load_rslt3_m128i, out);     // unpacks 5th 8 values


	__m128i c9_load_rslt4_m128i = _mm_loadu_si128(in + 3);
	__m128i c9_alignr_rslt3_m128i = _mm_alignr_epi8(c9_load_rslt4_m128i, c9_load_rslt3_m128i, 13);

	__horizontal_sse_unpack8_c9<0>(c9_alignr_rslt3_m128i, out);   // unpacks 6th 8 values

	__horizontal_sse_unpack8_c9<6>(c9_load_rslt4_m128i, out);     // unpacks 7th 8 values


	__m128i c9_load_rslt5_m128i = _mm_loadu_si128(in + 4);
	__m128i c9_alignr_rslt4_m128i = _mm_alignr_epi8(c9_load_rslt5_m128i, c9_load_rslt4_m128i, 15);

	__horizontal_sse_unpack8_c9<0>(c9_alignr_rslt4_m128i, out);   // unpacks 8th 8 values


	__m128i c9_load_rslt6_m128i = _mm_loadu_si128(in + 5);
	__m128i c9_alignr_rslt5_m128i = _mm_alignr_epi8(c9_load_rslt6_m128i, c9_load_rslt5_m128i, 8);

	__horizontal_sse_unpack8_c9<0>(c9_alignr_rslt5_m128i, out);   // unpacks 9th 8 values

	__horizontal_sse_unpack8_c9<1>(c9_load_rslt6_m128i, out);     // unpacks 10th 8 values


	__m128i c9_load_rslt7_m128i = _mm_loadu_si128(in + 6);
	__m128i c9_alignr_rslt6_m128i = _mm_alignr_epi8(c9_load_rslt7_m128i, c9_load_rslt6_m128i, 10);

	__horizontal_sse_unpack8_c9<0>(c9_alignr_rslt6_m128i, out);   // unpacks 11th 8 values

	__horizontal_sse_unpack8_c9<3>(c9_load_rslt7_m128i, out);     // unpacks 12th 8 values


	__m128i c9_load_rslt8_m128i = _mm_loadu_si128(in + 7);
	__m128i c9_alignr_rslt7_m128i = _mm_alignr_epi8(c9_load_rslt8_m128i, c9_load_rslt7_m128i, 12);

	__horizontal_sse_unpack8_c9<0>(c9_alignr_rslt7_m128i, out);   // unpacks 13th 8 values

	__horizontal_sse_unpack8_c9<5>(c9_load_rslt8_m128i, out);     // unpacks 14th 8 values


	__m128i c9_load_rslt9_m128i = _mm_loadu_si128(in + 8);
	__m128i c9_alignr_rslt8_m128i = _mm_alignr_epi8(c9_load_rslt9_m128i, c9_load_rslt8_m128i, 14);

	__horizontal_sse_unpack8_c9<0>(c9_alignr_rslt8_m128i, out);   // unpacks 15th 8 values

	__horizontal_sse_unpack8_c9<7>(c9_load_rslt9_m128i, out);     // unpacks 16th 8 values
}




// 10-bit
template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c10(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		__m128i Horizontal_SSE_c10_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 4, byte + 3,
				0xff, 0xff, byte + 3, byte + 2,
				0xff, 0xff, byte + 2, byte + 1,
				0xff, 0xff, byte + 1, byte + 0);

		__m128i c10_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c10_shfl_msk_m128i);
		__m128i c10_mul_rslt_m128i = _mm_mullo_epi32(c10_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[10][0]);
		__m128i c10_srli_rslt_m128i = _mm_srli_epi32(c10_mul_rslt_m128i, 6);
		__m128i c10_rslt_m128i = _mm_and_si128(c10_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[10]);
		_mm_storeu_si128(out++, c10_rslt_m128i);
	}
	else { // Rice Coding
		__m128i Horizontal_SSE_c10_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 4, byte + 3,
				0xff, 0xff, byte + 3, byte + 2,
				0xff, 0xff, byte + 2, byte + 1,
				0xff, 0xff, byte + 1, byte + 0);

		__m128i c10_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c10_shfl_msk_m128i);
		__m128i c10_mul_rslt_m128i = _mm_mullo_epi32(c10_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[10][0]);
		__m128i c10_srli_rslt_m128i = _mm_srli_epi32(c10_mul_rslt_m128i, 6);
		__m128i c10_and_rslt_m128i = _mm_and_si128(c10_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[10]);
		__m128i c10_rslt_m128i = _mm_or_si128(c10_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 10));
		_mm_storeu_si128(out++, c10_rslt_m128i);
	}
}


template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c10(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 64) {
		__m128i c10_load_rslt1_m128i = _mm_loadu_si128(in++);          // 16 bytes; contains 12 values (13th value is incomplete)

		__horizontal_sse_unpack4_c10<0>(c10_load_rslt1_m128i, out);     // unpacks 1st 4 values

		__horizontal_sse_unpack4_c10<5>(c10_load_rslt1_m128i, out);     // unpacks 2nd 4 values

		__horizontal_sse_unpack4_c10<10>(c10_load_rslt1_m128i, out);    // unpacks 3rd 4 values


		__m128i c10_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c10_alignr_rslt1_m128i = _mm_alignr_epi8(c10_load_rslt2_m128i, c10_load_rslt1_m128i, 15);  // 16 bytes; contains 12 values (13th value is incomplete)

		__horizontal_sse_unpack4_c10<0>(c10_alignr_rslt1_m128i, out);   // unpacks 4th 4 values

		__horizontal_sse_unpack4_c10<5>(c10_alignr_rslt1_m128i, out);   // unpacks 5th 4 values

		__horizontal_sse_unpack4_c10<10>(c10_alignr_rslt1_m128i, out);  // unpacks 6th 4 values


		__m128i c10_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c10_alignr_rslt2_m128i = _mm_alignr_epi8(c10_load_rslt3_m128i, c10_load_rslt2_m128i, 14);  // 16 bytes; contains 12 values (13th value is incomplete)

		__horizontal_sse_unpack4_c10<0>(c10_alignr_rslt2_m128i, out);   // unpacks 7th 4 values

		__horizontal_sse_unpack4_c10<5>(c10_alignr_rslt2_m128i, out);   // unpacks 8th 4 values

		__horizontal_sse_unpack4_c10<10>(c10_alignr_rslt2_m128i, out);  // unpacks 9th 4 values


		__m128i c10_load_rslt4_m128i = _mm_loadu_si128(in++);
		__m128i c10_alignr_rslt3_m128i = _mm_alignr_epi8(c10_load_rslt4_m128i, c10_load_rslt3_m128i, 13);  // 16 bytes; contains 12 values (13th value is incomplete)

		__horizontal_sse_unpack4_c10<0>(c10_alignr_rslt3_m128i, out);   // unpacks 10th 4 values

		__horizontal_sse_unpack4_c10<5>(c10_alignr_rslt3_m128i, out);   // unpacks 11th 4 values

		__horizontal_sse_unpack4_c10<10>(c10_alignr_rslt3_m128i, out);  // unpacks 12th 4 values


		__m128i c10_load_rslt5_m128i = _mm_loadu_si128(in++);
		__m128i c10_alignr_rslt4_m128i = _mm_alignr_epi8(c10_load_rslt5_m128i, c10_load_rslt4_m128i, 12);  // 16 bytes; contains 12 values (13th value is incomplete)

		__horizontal_sse_unpack4_c10<0>(c10_alignr_rslt4_m128i, out);   // unpacks 13th 4 values

		__horizontal_sse_unpack4_c10<5>(c10_alignr_rslt4_m128i, out);   // unpacks 14th 4 values

		__horizontal_sse_unpack4_c10<10>(c10_alignr_rslt4_m128i, out);  // unpacks 15th 4 values


		__horizontal_sse_unpack4_c10<11>(c10_load_rslt5_m128i, out);    // unpacks 16th 4 values
	}
}




// 11-bit
template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack8_c11(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 1st 4 values
		__m128i Horizontal_SSE_c11_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 5, byte + 4,
				0xff, byte + 4, byte + 3, byte + 2,
				0xff, 0xff, byte + 2, byte + 1,
				0xff, 0xff, byte + 1, byte + 0);

		__m128i c11_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c11_shfl_msk_m128i);
		__m128i c11_mul_rslt_m128i = _mm_mullo_epi32(c11_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[11][0]);
		__m128i c11_srli_rslt_m128i = _mm_srli_epi32(c11_mul_rslt_m128i, 6);
		__m128i c11_rslt_m128i = _mm_and_si128(c11_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[11]);
		_mm_storeu_si128(out++, c11_rslt_m128i);

		// unpacks 2nd 4 values
		Horizontal_SSE_c11_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 10, byte + 9,
				0xff, 0xff, byte + 9, byte + 8,
				0xff, byte + 8, byte + 7, byte + 6,
				0xff, 0xff, byte + 6, byte + 5);

		c11_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c11_shfl_msk_m128i);
		c11_mul_rslt_m128i = _mm_mullo_epi32(c11_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[11][1]);
		c11_srli_rslt_m128i = _mm_srli_epi32(c11_mul_rslt_m128i, 7);
		c11_rslt_m128i = _mm_and_si128(c11_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[11]);
		_mm_storeu_si128(out++, c11_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 1st 4 values
		__m128i Horizontal_SSE_c11_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 5, byte + 4,
				0xff, byte + 4, byte + 3, byte + 2,
				0xff, 0xff, byte + 2, byte + 1,
				0xff, 0xff, byte + 1, byte + 0);

		__m128i c11_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c11_shfl_msk_m128i);
		__m128i c11_mul_rslt_m128i = _mm_mullo_epi32(c11_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[11][0]);
		__m128i c11_srli_rslt_m128i = _mm_srli_epi32(c11_mul_rslt_m128i, 6);
		__m128i c11_and_rslt_m128i = _mm_and_si128(c11_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[11]);
		__m128i c11_rslt_m128i = _mm_or_si128(c11_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 11));
		_mm_storeu_si128(out++, c11_rslt_m128i);

		// unpacks 2nd 4 values
		Horizontal_SSE_c11_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 10, byte + 9,
				0xff, 0xff, byte + 9, byte + 8,
				0xff, byte + 8, byte + 7, byte + 6,
				0xff, 0xff, byte + 6, byte + 5);

		c11_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c11_shfl_msk_m128i);
		c11_mul_rslt_m128i = _mm_mullo_epi32(c11_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[11][1]);
		c11_srli_rslt_m128i = _mm_srli_epi32(c11_mul_rslt_m128i, 7);
		c11_and_rslt_m128i = _mm_and_si128(c11_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[11]);
		c11_rslt_m128i = _mm_or_si128(c11_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 11));
		_mm_storeu_si128(out++, c11_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c11(const __m128i *  __restrict__  in,
		__m128i *  __restrict__  out) {
	__m128i c11_load_rslt1_m128i = _mm_loadu_si128(in + 0);

	__horizontal_sse_unpack8_c11<0>(c11_load_rslt1_m128i, out);      // unpacks 1st 8 values


	__m128i c11_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c11_alignr_rslt1_m128i = _mm_alignr_epi8(c11_load_rslt2_m128i, c11_load_rslt1_m128i, 11);

	__horizontal_sse_unpack8_c11<0>(c11_alignr_rslt1_m128i, out);    // unpacks 2nd 8 values


	__m128i c11_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c11_alignr_rslt2_m128i = _mm_alignr_epi8(c11_load_rslt3_m128i, c11_load_rslt2_m128i, 6);

	__horizontal_sse_unpack8_c11<0>(c11_alignr_rslt2_m128i, out);    // unpacks 3rd 8 values

	__horizontal_sse_unpack8_c11<1>(c11_load_rslt3_m128i, out);      // unpacks 4th 8 values


	__m128i c11_load_rslt4_m128i = _mm_loadu_si128(in + 3);
	__m128i c11_alignr_rslt3_m128i = _mm_alignr_epi8(c11_load_rslt4_m128i, c11_load_rslt3_m128i, 12);

	__horizontal_sse_unpack8_c11<0>(c11_alignr_rslt3_m128i, out);    // unpacks 5th 8 values


	__m128i c11_load_rslt5_m128i = _mm_loadu_si128(in + 4);
	__m128i c11_alignr_rslt4_m128i = _mm_alignr_epi8(c11_load_rslt5_m128i, c11_load_rslt4_m128i, 7);

	__horizontal_sse_unpack8_c11<0>(c11_alignr_rslt4_m128i, out);    // unpacks 6th 8 values

	__horizontal_sse_unpack8_c11<2>(c11_load_rslt5_m128i, out);      // unpacks 7th 8 values


	__m128i c11_load_rslt6_m128i = _mm_loadu_si128(in + 5);
	__m128i c11_alignr_rslt5_m128i = _mm_alignr_epi8(c11_load_rslt6_m128i, c11_load_rslt5_m128i, 13);

	__horizontal_sse_unpack8_c11<0>(c11_alignr_rslt5_m128i, out);    // unpacks 8th 8 values


	__m128i c11_load_rslt7_m128i = _mm_loadu_si128(in + 6);
	__m128i c11_alignr_rslt6_m128i = _mm_alignr_epi8(c11_load_rslt7_m128i, c11_load_rslt6_m128i, 8);

	__horizontal_sse_unpack8_c11<0>(c11_alignr_rslt6_m128i, out);    // unpacks 9th 8 values

	__horizontal_sse_unpack8_c11<3>(c11_load_rslt7_m128i, out);      // unpacks 10th 8 values


	__m128i c11_load_rslt8_m128i = _mm_loadu_si128(in + 7);
	__m128i c11_alignr_rslt7_m128i = _mm_alignr_epi8(c11_load_rslt8_m128i, c11_load_rslt7_m128i, 14);

	__horizontal_sse_unpack8_c11<0>(c11_alignr_rslt7_m128i, out);    // unpacks 11th 8 values


	__m128i c11_load_rslt9_m128i = _mm_loadu_si128(in + 8);
	__m128i c11_alignr_rslt8_m128i = _mm_alignr_epi8(c11_load_rslt9_m128i, c11_load_rslt8_m128i, 9);

	__horizontal_sse_unpack8_c11<0>(c11_alignr_rslt8_m128i, out);    // unpacks 12th 8 values

	__horizontal_sse_unpack8_c11<4>(c11_load_rslt9_m128i, out);      // unpacks 13th 8 values


	__m128i c11_load_rslt10_m128i = _mm_loadu_si128(in + 9);
	__m128i c11_alignr_rslt9_m128i = _mm_alignr_epi8(c11_load_rslt10_m128i, c11_load_rslt9_m128i, 15);

	__horizontal_sse_unpack8_c11<0>(c11_alignr_rslt9_m128i, out);    // unpacks 14th 8 values


	__m128i c11_load_rslt11_m128i = _mm_loadu_si128(in + 10);
	__m128i c11_alignr_rslt10_m128i = _mm_alignr_epi8(c11_load_rslt11_m128i, c11_load_rslt10_m128i, 10);

	__horizontal_sse_unpack8_c11<0>(c11_alignr_rslt10_m128i, out);    // unpacks 15th 8 values

	__horizontal_sse_unpack8_c11<5>(c11_load_rslt11_m128i, out);      // unpacks 16th 8 values

}




// 12-bit
template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c12(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		__m128i Horizontal_SSE_c12_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 5, byte + 4,
				0xff, 0xff, byte + 4, byte + 3,
				0xff, 0xff, byte + 2, byte + 1,
				0xff, 0xff, byte + 1, byte + 0);

		__m128i c12_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c12_shfl_msk_m128i);
		__m128i c12_mul_rslt_m128i = _mm_mullo_epi32(c12_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[12][0]);
		__m128i c12_srli_rslt_m128i = _mm_srli_epi32(c12_mul_rslt_m128i, 4);
		__m128i c12_rslt_m128i = _mm_and_si128(c12_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[12]);
		_mm_storeu_si128(out++, c12_rslt_m128i);
	}
	else { // Rice Coding
		__m128i Horizontal_SSE_c12_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 5, byte + 4,
				0xff, 0xff, byte + 4, byte + 3,
				0xff, 0xff, byte + 2, byte + 1,
				0xff, 0xff, byte + 1, byte + 0);

		__m128i c12_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c12_shfl_msk_m128i);
		__m128i c12_mul_rslt_m128i = _mm_mullo_epi32(c12_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[12][0]);
		__m128i c12_srli_rslt_m128i = _mm_srli_epi32(c12_mul_rslt_m128i, 4);
		__m128i c12_and_rslt_m128i = _mm_and_si128(c12_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[12]);
		__m128i c12_rslt_m128i = _mm_or_si128(c12_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 12));
		_mm_storeu_si128(out++, c12_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c12(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 32) {
		__m128i c12_load_rslt1_m128i = _mm_loadu_si128(in++);

		__horizontal_sse_unpack4_c12<0>(c12_load_rslt1_m128i, out);      // unpacks 1st 4 values

		__horizontal_sse_unpack4_c12<6>(c12_load_rslt1_m128i, out);      // unpacks 2nd 4 values


		__m128i c12_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c12_alignr_rslt1_m128i = _mm_alignr_epi8(c12_load_rslt2_m128i, c12_load_rslt1_m128i, 12);

		__horizontal_sse_unpack4_c12<0>(c12_alignr_rslt1_m128i, out);    // unpacks 3rd 4 values

		__horizontal_sse_unpack4_c12<6>(c12_alignr_rslt1_m128i, out);    // unpacks 4th 4 values

		__horizontal_sse_unpack4_c12<8> (c12_load_rslt2_m128i, out);     // unpacks 5th 4 values


		__m128i c12_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c12_align_rslt2_m128i = _mm_alignr_epi8(c12_load_rslt3_m128i, c12_load_rslt2_m128i, 14);

		__horizontal_sse_unpack4_c12<0> (c12_align_rslt2_m128i, out);    // unpacks 6th 4 values

		__horizontal_sse_unpack4_c12<6> (c12_align_rslt2_m128i, out);    // unpacks 7th 4 values

		__horizontal_sse_unpack4_c12<10>(c12_load_rslt3_m128i, out);     // unpacks 8th 4 values
	}
}




// 13-bit
template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack8_c13(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 1st 4 values
		__m128i Horizontal_SSE_c13_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 6, byte + 5, byte + 4,
				0xff, 0xff, byte + 4, byte + 3,
				0xff, byte + 3, byte + 2, byte + 1,
				0xff, 0xff, byte + 1, byte + 0);

		__m128i c13_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c13_shfl_msk_m128i);
		__m128i c13_mul_rslt_m128i = _mm_mullo_epi32(c13_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[13][0]);
		__m128i c13_srli_rslt_m128i = _mm_srli_epi32(c13_mul_rslt_m128i, 7);
		__m128i c13_rslt_m128i = _mm_and_si128(c13_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[13]);
		_mm_storeu_si128(out++, c13_rslt_m128i);

		// unpacks 2nd 4 values
		Horizontal_SSE_c13_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 12, byte + 11,
				0xff, byte + 11, byte + 10, byte + 9,
				0xff, 0xff, byte + 9, byte + 8,
				0xff, byte + 8, byte + 7, byte + 6);

		c13_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c13_shfl_msk_m128i);
		c13_mul_rslt_m128i = _mm_mullo_epi32(c13_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[13][1]);
		c13_srli_rslt_m128i = _mm_srli_epi32(c13_mul_rslt_m128i, 6);
		c13_rslt_m128i = _mm_and_si128(c13_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[13]);
		_mm_storeu_si128(out++, c13_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 1st 4 values
		__m128i Horizontal_SSE_c13_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 6, byte + 5, byte + 4,
				0xff, 0xff, byte + 4, byte + 3,
				0xff, byte + 3, byte + 2, byte + 1,
				0xff, 0xff, byte + 1, byte + 0);

		__m128i c13_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c13_shfl_msk_m128i);
		__m128i c13_mul_rslt_m128i = _mm_mullo_epi32(c13_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[13][0]);
		__m128i c13_srli_rslt_m128i = _mm_srli_epi32(c13_mul_rslt_m128i, 7);
		__m128i c13_and_rslt_m128i = _mm_and_si128(c13_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[13]);
		__m128i c13_rslt_m128i = _mm_or_si128(c13_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 13));
		_mm_storeu_si128(out++, c13_rslt_m128i);

		// unpacks 2nd 4 values
		Horizontal_SSE_c13_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 12, byte + 11,
				0xff, byte + 11, byte + 10, byte + 9,
				0xff, 0xff, byte + 9, byte + 8,
				0xff, byte + 8, byte + 7, byte + 6);

		c13_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c13_shfl_msk_m128i);
		c13_mul_rslt_m128i = _mm_mullo_epi32(c13_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[13][1]);
		c13_srli_rslt_m128i = _mm_srli_epi32(c13_mul_rslt_m128i, 6);
		c13_and_rslt_m128i = _mm_and_si128(c13_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[13]);
		c13_rslt_m128i = _mm_or_si128(c13_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 13));
		_mm_storeu_si128(out++, c13_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c13(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	__m128i c13_load_rslt1_m128i = _mm_loadu_si128(in + 0);

	__horizontal_sse_unpack8_c13<0>(c13_load_rslt1_m128i, out);     // unpacks 1st 8 values


	__m128i c13_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c13_alignr_rslt1_m128i = _mm_alignr_epi8(c13_load_rslt2_m128i, c13_load_rslt1_m128i, 13);

	__horizontal_sse_unpack8_c13<0>(c13_alignr_rslt1_m128i, out);    // unpacks 2nd 8 values


	__m128i c13_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c13_alignr_rslt2_m128i = _mm_alignr_epi8(c13_load_rslt3_m128i, c13_load_rslt2_m128i, 10);

	__horizontal_sse_unpack8_c13<0>(c13_alignr_rslt2_m128i, out);    // unpacks 3rd 8 values


    __m128i c13_load_rslt4_m128i = _mm_loadu_si128(in + 3);
    __m128i c13_align_rslt3_m128i = _mm_alignr_epi8(c13_load_rslt4_m128i, c13_load_rslt3_m128i, 7);

    __horizontal_sse_unpack8_c13<0>(c13_align_rslt3_m128i, out);     // unpacks 4th 8 values


    __m128i c13_load_rslt5_m128i = _mm_loadu_si128(in + 4);
    __m128i c13_align_rslt4_m128i = _mm_alignr_epi8(c13_load_rslt5_m128i, c13_load_rslt4_m128i, 4);

    __horizontal_sse_unpack8_c13<0>(c13_align_rslt4_m128i, out);    // unpacks 5th 8 values

    __horizontal_sse_unpack8_c13<1>(c13_load_rslt5_m128i, out);     // unpacks 6th 8 values


    __m128i c13_load_rslt6_m128i = _mm_loadu_si128(in + 5);
    __m128i c13_align_rslt5_m128i = _mm_alignr_epi8(c13_load_rslt6_m128i, c13_load_rslt5_m128i, 14);

    __horizontal_sse_unpack8_c13<0>(c13_align_rslt5_m128i, out);    // unpacks 7th 8 values


    __m128i c13_load_rslt7_m128i = _mm_loadu_si128(in + 6);
    __m128i c13_align_rslt6_m128i = _mm_alignr_epi8(c13_load_rslt7_m128i, c13_load_rslt6_m128i, 11);

    __horizontal_sse_unpack8_c13<0>(c13_align_rslt6_m128i, out);    // unpacks 8th 8 values


    __m128i c13_load_rslt8_m128i = _mm_loadu_si128(in + 7);
    __m128i c13_align_rslt7_m128i = _mm_alignr_epi8(c13_load_rslt8_m128i, c13_load_rslt7_m128i, 8);

    __horizontal_sse_unpack8_c13<0>(c13_align_rslt7_m128i, out);    // unpacks 9th 8 values


    __m128i c13_load_rslt9_m128i = _mm_loadu_si128(in + 8);
    __m128i c13_align_rslt8_m128i = _mm_alignr_epi8(c13_load_rslt9_m128i, c13_load_rslt8_m128i, 5);

    __horizontal_sse_unpack8_c13<0>(c13_align_rslt8_m128i, out);    // unpacks 10th 8 values

    __horizontal_sse_unpack8_c13<2>(c13_load_rslt9_m128i, out);     // unpacks 11th 8 values


    __m128i c13_load_rslt10_m128i = _mm_loadu_si128(in + 9);
    __m128i c13_align_rslt9_m128i = _mm_alignr_epi8(c13_load_rslt10_m128i, c13_load_rslt9_m128i, 15);

    __horizontal_sse_unpack8_c13<0>(c13_align_rslt9_m128i, out);     // unpacks 12th 8 values


    __m128i c13_load_rslt11_m128i = _mm_loadu_si128(in + 10);
    __m128i c13_align_rslt10_m128i = _mm_alignr_epi8(c13_load_rslt11_m128i, c13_load_rslt10_m128i, 12);

    __horizontal_sse_unpack8_c13<0>(c13_align_rslt10_m128i, out);    // unpacks 13th 8 values


    __m128i c13_load_rslt12_m128i = _mm_loadu_si128(in + 11);
    __m128i c13_align_rslt11_m128i = _mm_alignr_epi8(c13_load_rslt12_m128i, c13_load_rslt11_m128i, 9);

    __horizontal_sse_unpack8_c13<0>(c13_align_rslt11_m128i, out);    // unpacks 14th 8 values


    __m128i c13_load_rslt13_m128i = _mm_loadu_si128(in + 12);
    __m128i c13_align_rslt12_m128i = _mm_alignr_epi8(c13_load_rslt13_m128i, c13_load_rslt12_m128i, 6);

    __horizontal_sse_unpack8_c13<0>(c13_align_rslt12_m128i, out);    // unpacks 15th 8 values

    __horizontal_sse_unpack8_c13<3>(c13_load_rslt13_m128i, out);     // unpacks 16th 8 values
}




// 14-bit
template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c14(const __m128i &InReg,
		__m128i *  __restrict__   &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 6, byte + 5,
				0xff, byte + 5, byte + 4, byte + 3,
				0xff, byte + 3, byte + 2, byte + 1,
				0xff, 0xff, byte + 1, byte + 0);

		__m128i c14_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_shfl_msk_m128i);
		__m128i c14_mul_rslt_m128i = _mm_mullo_epi32(c14_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[14][0]);
		__m128i c14_srli_rslt_m128i = _mm_srli_epi32(c14_mul_rslt_m128i, 6);
		__m128i c14_rslt_m128i = _mm_and_si128(c14_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[14]);
		_mm_storeu_si128(out++, c14_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 6, byte + 5,
				0xff, byte + 5, byte + 4, byte + 3,
				0xff, byte + 3, byte + 2, byte + 1,
				0xff, 0xff, byte + 1, byte + 0);

		__m128i c14_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_shfl_msk_m128i);
		__m128i c14_mul_rslt_m128i = _mm_mullo_epi32(c14_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[14][0]);
		__m128i c14_srli_rslt_m128i = _mm_srli_epi32(c14_mul_rslt_m128i, 6);
		__m128i c14_and_rslt_m128i = _mm_and_si128(c14_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[14]);
		__m128i c14_rslt_m128i = _mm_or_si128(c14_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 14));
		_mm_storeu_si128(out++, c14_rslt_m128i);

	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c14(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 64) {
	     __m128i c14_load_rslt1_m128i = _mm_loadu_si128(in++);

	     __horizontal_sse_unpack4_c14<0>(c14_load_rslt1_m128i, out);    // unpacks 1st 4 values

	     __horizontal_sse_unpack4_c14<7>(c14_load_rslt1_m128i, out);    // unpacks 2nd 4 values


	     __m128i c14_load_rslt2_m128i = _mm_loadu_si128(in++);
	     __m128i c14_align_rslt1_m128i = _mm_alignr_epi8(c14_load_rslt2_m128i, c14_load_rslt1_m128i, 14);

	     __horizontal_sse_unpack4_c14<0>(c14_align_rslt1_m128i, out);   // unpacks 3rd 4 values

	     __horizontal_sse_unpack4_c14<7>(c14_align_rslt1_m128i, out);   // unpacks 4th 4 values


	     __m128i c14_load_rslt3_m128i = _mm_loadu_si128(in++);
	     __m128i c14_align_rslt2_m128i = _mm_alignr_epi8(c14_load_rslt3_m128i, c14_load_rslt2_m128i, 12);

	     __horizontal_sse_unpack4_c14<0>(c14_align_rslt2_m128i, out);   // unpacks 5th 4 values

	     __horizontal_sse_unpack4_c14<7>(c14_align_rslt2_m128i, out);   // unpacks 6th 4 values


	     __m128i c14_load_rslt4_m128i = _mm_loadu_si128(in++);
	     __m128i c14_align_rslt3_m128i = _mm_alignr_epi8(c14_load_rslt4_m128i, c14_load_rslt3_m128i, 10);

	     __horizontal_sse_unpack4_c14<0>(c14_align_rslt3_m128i, out);   // unpacks 7th 4 values

	     __horizontal_sse_unpack4_c14<7>(c14_align_rslt3_m128i, out);   // unpacks 8th 4 values

	     __horizontal_sse_unpack4_c14<8>(c14_load_rslt4_m128i, out);    // unpacks 9th 4 values


	     __m128i c14_load_rslt5_m128i = _mm_loadu_si128(in++);
	     __m128i c14_align_rslt4_m128i = _mm_alignr_epi8(c14_load_rslt5_m128i, c14_load_rslt4_m128i, 15);

	     __horizontal_sse_unpack4_c14<0>(c14_align_rslt4_m128i, out);   // unpacks 10th 4 values

	     __horizontal_sse_unpack4_c14<7>(c14_align_rslt4_m128i, out);   // unpacks 11th 4 values


	     __m128i c14_load_rslt6_m128i = _mm_loadu_si128(in++);
	     __m128i c14_align_rslt5_m128i = _mm_alignr_epi8(c14_load_rslt6_m128i, c14_load_rslt5_m128i, 13);

	     __horizontal_sse_unpack4_c14<0>(c14_align_rslt5_m128i, out);   // unpacks 12th 4 values

	     __horizontal_sse_unpack4_c14<7>(c14_align_rslt5_m128i, out);   // unpacks 13th 4 values


	     __m128i c14_load_rslt7_m128i = _mm_loadu_si128(in++);
	     __m128i c14_align_rslt6_m128i = _mm_alignr_epi8(c14_load_rslt7_m128i, c14_load_rslt6_m128i, 11);

	     __horizontal_sse_unpack4_c14<0>(c14_align_rslt6_m128i, out);   // unpacks 13th 4 values

	     __horizontal_sse_unpack4_c14<7>(c14_align_rslt6_m128i, out);   // unpacks 14th 4 values

	     __horizontal_sse_unpack4_c14<9>(c14_load_rslt7_m128i, out);    // unpacks 16th 4 values
	}
}




// 15-bit
template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack8_c15(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 1st 4 values
		__m128i Horizontal_SSE_c15_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 7, byte + 6, byte + 5,
				0xff, byte + 5, byte + 4, byte + 3,
				0xff, byte + 3, byte + 2, byte + 1,
				0xff, 0xff, byte + 1, byte + 0);

		__m128i c15_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c15_shfl_msk_m128i);
		__m128i c15_mul_rslt_m128i = _mm_mullo_epi32(c15_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[15][0]);
		__m128i c15_srli_rslt_m128i = _mm_srli_epi32(c15_mul_rslt_m128i, 7);
		__m128i c15_rslt_m128i = _mm_and_si128(c15_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[15]);
		_mm_storeu_si128(out++, c15_rslt_m128i);

		// unpacks 2nd 4 values
		Horizontal_SSE_c15_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 14, byte + 13,
				0xff, byte + 13, byte + 12, byte + 11,
				0xff, byte + 11, byte + 10, byte + 9,
				0xff, byte + 9, byte + 8, byte + 7);

		c15_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c15_shfl_msk_m128i);
		c15_mul_rslt_m128i = _mm_mullo_epi32(c15_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[15][1]);
		c15_srli_rslt_m128i = _mm_srli_epi32(c15_mul_rslt_m128i, 4);
		c15_rslt_m128i = _mm_and_si128(c15_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[15]);
		_mm_storeu_si128(out++, c15_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 1st 4 values
		__m128i Horizontal_SSE_c15_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 7, byte + 6, byte + 5,
				0xff, byte + 5, byte + 4, byte + 3,
				0xff, byte + 3, byte + 2, byte + 1,
				0xff, 0xff, byte + 1, byte + 0);

		__m128i c15_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c15_shfl_msk_m128i);
		__m128i c15_mul_rslt_m128i = _mm_mullo_epi32(c15_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[15][0]);
		__m128i c15_srli_rslt_m128i = _mm_srli_epi32(c15_mul_rslt_m128i, 7);
		__m128i c15_and_rslt_m128i = _mm_and_si128(c15_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[15]);
		__m128i c15_rslt_m128i = _mm_or_si128(c15_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 15));
		_mm_storeu_si128(out++, c15_rslt_m128i);

		// unpacks 2nd 4 values
		Horizontal_SSE_c15_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 14, byte + 13,
				0xff, byte + 13, byte + 12, byte + 11,
				0xff, byte + 11, byte + 10, byte + 9,
				0xff, byte + 9, byte + 8, byte + 7);

		c15_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c15_shfl_msk_m128i);
		c15_mul_rslt_m128i = _mm_mullo_epi32(c15_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[15][1]);
		c15_srli_rslt_m128i = _mm_srli_epi32(c15_mul_rslt_m128i, 4);
		c15_and_rslt_m128i = _mm_and_si128(c15_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[15]);
		c15_rslt_m128i = _mm_or_si128(c15_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 15));
		_mm_storeu_si128(out++, c15_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c15(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
    __m128i c15_load_rslt1_m128i = _mm_loadu_si128(in + 0);

    __horizontal_sse_unpack8_c15<0>(c15_load_rslt1_m128i, out);      // unpacks 1st 8 values


    __m128i c15_load_rslt2_m128i = _mm_loadu_si128(in + 1);
    __m128i c15_align_rslt1_m128i = _mm_alignr_epi8(c15_load_rslt2_m128i, c15_load_rslt1_m128i, 15);

    __horizontal_sse_unpack8_c15<0>(c15_align_rslt1_m128i, out);     // unpacks 2nd 8 values


    __m128i c15_load_rslt3_m128i = _mm_loadu_si128(in + 2);
    __m128i c15_align_rslt2_m128i = _mm_alignr_epi8(c15_load_rslt3_m128i, c15_load_rslt2_m128i, 14);

    __horizontal_sse_unpack8_c15<0>(c15_align_rslt2_m128i, out);     // unpacks 3rd 8 values


    __m128i c15_load_rslt4_m128i = _mm_loadu_si128(in + 3);
    __m128i c15_align_rslt3_m128i = _mm_alignr_epi8(c15_load_rslt4_m128i, c15_load_rslt3_m128i, 13);

    __horizontal_sse_unpack8_c15<0>(c15_align_rslt3_m128i, out);     // unpacks 4th 8 values


    __m128i c15_load_rslt5_m128i = _mm_loadu_si128(in + 4);
    __m128i c15_align_rslt4_m128i = _mm_alignr_epi8(c15_load_rslt5_m128i, c15_load_rslt4_m128i, 12);

    __horizontal_sse_unpack8_c15<0>(c15_align_rslt4_m128i, out);     // unpacks 5th 8 values


    __m128i c15_load_rslt6_m128i = _mm_loadu_si128(in + 5);
    __m128i c15_align_rslt5_m128i = _mm_alignr_epi8(c15_load_rslt6_m128i, c15_load_rslt5_m128i, 11);

    __horizontal_sse_unpack8_c15<0>(c15_align_rslt5_m128i, out);     // unpacks 6th 8 values


    __m128i c15_load_rslt7_m128i = _mm_loadu_si128(in + 6);
    __m128i c15_align_rslt6_m128i = _mm_alignr_epi8(c15_load_rslt7_m128i, c15_load_rslt6_m128i, 10);

    __horizontal_sse_unpack8_c15<0>(c15_align_rslt6_m128i, out);     // unpacks 7th 8 values


    __m128i c15_load_rslt8_m128i = _mm_loadu_si128(in + 7);
    __m128i c15_align_rslt7_m128i = _mm_alignr_epi8(c15_load_rslt8_m128i, c15_load_rslt7_m128i, 9);

    __horizontal_sse_unpack8_c15<0>(c15_align_rslt7_m128i, out);     // unpacks 8th 8 values


    __m128i c15_load_rslt9_m128i = _mm_loadu_si128(in + 8);
    __m128i c15_align_rslt8_m128i = _mm_alignr_epi8(c15_load_rslt9_m128i, c15_load_rslt8_m128i, 8);

    __horizontal_sse_unpack8_c15<0>(c15_align_rslt8_m128i, out);     // unpacks 9th 8 values

    __m128i c15_load_rslt10_m128i = _mm_loadu_si128(in + 9);
    __m128i c15_align_rslt9_m128i = _mm_alignr_epi8(c15_load_rslt10_m128i, c15_load_rslt9_m128i, 7);

    __horizontal_sse_unpack8_c15<0>(c15_align_rslt9_m128i, out);     // unpacks 10th 8 values


    __m128i c15_load_rslt11_m128i = _mm_loadu_si128(in + 10);
    __m128i c15_align_rslt10_m128i = _mm_alignr_epi8(c15_load_rslt11_m128i, c15_load_rslt10_m128i, 6);

    __horizontal_sse_unpack8_c15<0>(c15_align_rslt10_m128i, out);    // unpacks 11th 8 values


    __m128i c15_load_rslt12_m128i = _mm_loadu_si128(in + 11);
    __m128i c15_align_rslt11_m128i = _mm_alignr_epi8(c15_load_rslt12_m128i, c15_load_rslt11_m128i, 5);

    __horizontal_sse_unpack8_c15<0>(c15_align_rslt11_m128i, out);    // unpacks 12th 8 values


    __m128i c15_load_rslt13_m128i = _mm_loadu_si128(in + 12);
    __m128i c15_align_rslt12_m128i = _mm_alignr_epi8(c15_load_rslt13_m128i, c15_load_rslt12_m128i, 4);

    __horizontal_sse_unpack8_c15<0>(c15_align_rslt12_m128i, out);    // unpacks 13th 8 values

    __m128i c15_load_rslt14_m128i = _mm_loadu_si128(in + 13);
    __m128i c15_align_rslt13_m128i = _mm_alignr_epi8(c15_load_rslt14_m128i, c15_load_rslt13_m128i, 3);

    __horizontal_sse_unpack8_c15<0>(c15_align_rslt13_m128i, out);    // unpacks 14th 8 values

    __m128i c15_load_rslt15_m128i = _mm_loadu_si128(in + 14);
    __m128i c15_align_rslt14_m128i = _mm_alignr_epi8(c15_load_rslt15_m128i, c15_load_rslt14_m128i, 2);

    __horizontal_sse_unpack8_c15<0>(c15_align_rslt14_m128i, out);    // unpacks 15th 8 values

    __horizontal_sse_unpack8_c15<1>(c15_load_rslt15_m128i, out);     // unpacks 16th 8 values
}




// 16-bit
template <bool IsRiceCoding>
template <uint32_t byte>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c16(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c16_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 7, byte + 6,
				0xff, 0xff, byte + 5, byte + 4,
				0xff, 0xff, byte + 3, byte + 2,
				0xff, 0xff, byte + 1, byte + 0);

		__m128i c16_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c16_shfl_msk_m128i);
		_mm_storeu_si128(out++, c16_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c16_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, byte + 7, byte + 6,
				0xff, 0xff, byte + 5, byte + 4,
				0xff, 0xff, byte + 3, byte + 2,
				0xff, 0xff, byte + 1, byte + 0);

		__m128i c16_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c16_shfl_msk_m128i);
		__m128i c16_rslt_m128i = _mm_or_si128(c16_shfl_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 16));
		_mm_storeu_si128(out++, c16_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c16(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 8) {
		__m128i c16_load_rslt_m128i = _mm_loadu_si128(in++);      // 16 bytes; contains 8 values

		__horizontal_sse_unpack4_c16<0>(c16_load_rslt_m128i, out); // unpacks 1st 4 values

		__horizontal_sse_unpack4_c16<8>(c16_load_rslt_m128i, out); // unpacks 2nd 4 values
	}
}

// // alternatives
//template <bool IsRiceCoding>
//template <uint32_t byte>
//forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack8_c16(const __m128i &InReg,
//		__m128i *  __restrict__  &out) {
//	if (!IsRiceCoding) { // NewPFor etc.
//		// unpacks 1st 4 values
//		__m128i Horizontal_SSE_c16_shfl_msk_m128i = _mm_set_epi8(
//				0xff, 0xff, 7, 6,
//				0xff, 0xff, 5, 4,
//				0xff, 0xff, 3, 2,
//				0xff, 0xff, 1, 0);
//
//		__m128i c16_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c16_shfl_msk_m128i);
//		_mm_storeu_si128(out++, c16_rslt_m128i);
//
//		// unpacks 2nd 4 values
//		Horizontal_SSE_c16_shfl_msk_m128i = _mm_set_epi8(
//				0xff, 0xff, 15, 14,
//				0xff, 0xff, 13, 12,
//				0xff, 0xff, 11, 10,
//				0xff, 0xff, 9, 8);
//
//		c16_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c16_shfl_msk_m128i);
//		_mm_storeu_si128(out++, c16_rslt_m128i);
//	}
//	else { // Rice Coding
//		// unpacks 1st 4 values
//		__m128i Horizontal_SSE_c16_shfl_msk_m128i = _mm_set_epi8(
//				0xff, 0xff, 7, 6,
//				0xff, 0xff, 5, 4,
//				0xff, 0xff, 3, 2,
//				0xff, 0xff, 1, 0);
//
//		__m128i c16_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c16_shfl_msk_m128i);
//		__m128i c16_rslt_m128i = _mm_or_si128(c16_shfl_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 16));
//		_mm_storeu_si128(out++, c16_rslt_m128i);
//
//		// unpacks 2nd 4 values
//		Horizontal_SSE_c16_shfl_msk_m128i = _mm_set_epi8(
//				0xff, 0xff, 15, 14,
//				0xff, 0xff, 13, 12,
//				0xff, 0xff, 11, 10,
//				0xff, 0xff, 9, 8);
//
//		c16_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c16_shfl_msk_m128i);
//		c16_rslt_m128i = _mm_or_si128(c16_shfl_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 16));
//		_mm_storeu_si128(out++, c16_rslt_m128i);
//	}
//}
//
//template <bool IsRiceCoding>
//void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c16(const __m128i * __restrict__ in,
//		__m128i *  __restrict__  out) {
//	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 8) {
//		__m128i c16_load_rslt_m128i = _mm_loadu_si128(in++);      // 16 bytes; contains 8 values
//
//		__horizontal_sse_unpack8_c16<0>(c16_load_rslt_m128i, out); // unpacks 8 values
//	}
//}
//
//template <bool IsRiceCoding>
//void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c16(const __m128i * __restrict__ in,
//		__m128i *  __restrict__  out) {
//	if (!IsRiceCoding) { // NewPFor etc.
//		for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 8) {
//			__m128i c16_load_rslt_m128i = _mm_loadu_si128(in++);  // 16 bytes; contains 8 values
//
//			// unpacks 1st 4 values
//			__m128i Horizontal_SSE_c16_shfl_msk_m128i = _mm_set_epi8(
//					0xff, 0xff, 7, 6,
//					0xff, 0xff, 5, 4,
//					0xff, 0xff, 3, 2,
//					0xff, 0xff, 1, 0);
//
//			__m128i c16_rslt_m128i = _mm_shuffle_epi8(c16_load_rslt_m128i, Horizontal_SSE_c16_shfl_msk_m128i);
//			_mm_storeu_si128(out++, c16_rslt_m128i);
//
//			// unpacks 2nd 4 values
//			Horizontal_SSE_c16_shfl_msk_m128i = _mm_set_epi8(
//					0xff, 0xff, 15, 14,
//					0xff, 0xff, 13, 12,
//					0xff, 0xff, 11, 10,
//					0xff, 0xff, 9, 8);
//
//			c16_rslt_m128i = _mm_shuffle_epi8(c16_load_rslt_m128i, Horizontal_SSE_c16_shfl_msk_m128i);
//			_mm_storeu_si128(out++, c16_rslt_m128i);
//		}
//	}
//	else { // Rice Coding
//		for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 8) {
//			__m128i c16_load_rslt_m128i = _mm_loadu_si128(in++);  // 16 bytes; contains 8 values
//
//			// unpacks 1st 4 values
//			__m128i Horizontal_SSE_c16_shfl_msk_m128i = _mm_set_epi8(
//					0xff, 0xff, 7, 6,
//					0xff, 0xff, 5, 4,
//					0xff, 0xff, 3, 2,
//					0xff, 0xff, 1, 0);
//
//			__m128i c16_shfl_rslt_m128i = _mm_shuffle_epi8(c16_load_rslt_m128i, Horizontal_SSE_c16_shfl_msk_m128i);
//			__m128i c16_rslt_m128i = _mm_or_si128(c16_shfl_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 16));
//			_mm_storeu_si128(out++, c16_rslt_m128i);
//
//			// unpacks 2nd 4 values
//			Horizontal_SSE_c16_shfl_msk_m128i = _mm_set_epi8(
//					0xff, 0xff, 15, 14,
//					0xff, 0xff, 13, 12,
//					0xff, 0xff, 11, 10,
//					0xff, 0xff, 9, 8);
//
//			c16_shfl_rslt_m128i = _mm_shuffle_epi8(c16_load_rslt_m128i, Horizontal_SSE_c16_shfl_msk_m128i);
//			c16_rslt_m128i = _mm_or_si128(c16_shfl_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 16));
//			_mm_storeu_si128(out++, c16_rslt_m128i);
//		}
//	}
//}




// 17-bit
template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c17_f1(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c17_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 8, byte + 7, byte + 6,
				0xff, byte + 6, byte + 5, byte + 4,
				0xff, byte + 4, byte + 3, byte + 2,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c17_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c17_shfl_msk_m128i);
		__m128i c17_mul_rslt_m128i = _mm_mullo_epi32(c17_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[17][0]);
		__m128i c17_srli_rslt_m128i = _mm_srli_epi32(c17_mul_rslt_m128i, 3);
		__m128i c17_rslt_m128i = _mm_and_si128(c17_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[17]);
		_mm_storeu_si128(out++, c17_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c17_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 8, byte + 7, byte + 6,
				0xff, byte + 6, byte + 5, byte + 4,
				0xff, byte + 4, byte + 3, byte + 2,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c17_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c17_shfl_msk_m128i);
		__m128i c17_mul_rslt_m128i = _mm_mullo_epi32(c17_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[17][0]);
		__m128i c17_srli_rslt_m128i = _mm_srli_epi32(c17_mul_rslt_m128i, 3);
		__m128i c17_and_rslt_m128i = _mm_and_si128(c17_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[17]);
		__m128i c17_rslt_m128i = _mm_or_si128(c17_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 17));
		_mm_storeu_si128(out++, c17_rslt_m128i);
	}
}

template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c17_f2(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c17_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 8, byte + 7, byte + 6,
				0xff, byte + 6, byte + 5, byte + 4,
				0xff, byte + 4, byte + 3, byte + 2,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c17_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c17_shfl_msk_m128i);
		__m128i c17_mul_rslt_m128i = _mm_mullo_epi32(c17_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[17][1]);
		__m128i c17_srli_rslt_m128i = _mm_srli_epi32(c17_mul_rslt_m128i, 7);
		__m128i c17_rslt_m128i = _mm_and_si128(c17_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[17]);
		_mm_storeu_si128(out++, c17_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c17_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 8, byte + 7, byte + 6,
				0xff, byte + 6, byte + 5, byte + 4,
				0xff, byte + 4, byte + 3, byte + 2,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c17_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c17_shfl_msk_m128i);
		__m128i c17_mul_rslt_m128i = _mm_mullo_epi32(c17_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[17][1]);
		__m128i c17_srli_rslt_m128i = _mm_srli_epi32(c17_mul_rslt_m128i, 7);
		__m128i c17_and_rslt_m128i = _mm_and_si128(c17_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[17]);
		__m128i c17_rslt_m128i = _mm_or_si128(c17_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 17));
		_mm_storeu_si128(out++, c17_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c17(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	__m128i c17_load_rslt1_m128i = _mm_loadu_si128(in + 0);

	__horizontal_sse_unpack4_c17_f1<0>(c17_load_rslt1_m128i, out);   // unpacks 1st 4 values


	__m128i c17_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c17_align_rslt1_m128i = _mm_alignr_epi8(c17_load_rslt2_m128i, c17_load_rslt1_m128i, 8);

	__horizontal_sse_unpack4_c17_f2<0>(c17_align_rslt1_m128i, out);  // unpacks 2nd 4 values

	__horizontal_sse_unpack4_c17_f1<1>(c17_load_rslt2_m128i, out);   // unpacks 3rd 4 values


	__m128i c17_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c17_align_rslt2_m128i = _mm_alignr_epi8(c17_load_rslt3_m128i, c17_load_rslt2_m128i, 9);

	__horizontal_sse_unpack4_c17_f2<0>(c17_align_rslt2_m128i, out);  // unpacks 4th 4 values

	__horizontal_sse_unpack4_c17_f1<2>(c17_load_rslt3_m128i, out);   // unpacks 5th 4 values


	__m128i c17_load_rslt4_m128i = _mm_loadu_si128(in + 3);
	__m128i c17_align_rslt3_m128i = _mm_alignr_epi8(c17_load_rslt4_m128i, c17_load_rslt3_m128i, 10);

	__horizontal_sse_unpack4_c17_f2<0>(c17_align_rslt3_m128i, out);  // unpacks 6th 4 values

	__horizontal_sse_unpack4_c17_f1<3>(c17_load_rslt4_m128i, out);   // unpacks 7th 4 values


	__m128i c17_load_rslt5_m128i = _mm_loadu_si128(in + 4);
	__m128i c17_align_rslt4_m128i = _mm_alignr_epi8(c17_load_rslt5_m128i, c17_load_rslt4_m128i, 11);

	__horizontal_sse_unpack4_c17_f2<0>(c17_align_rslt4_m128i, out);  // unpacks 8th 4 values

	__horizontal_sse_unpack4_c17_f1<4>(c17_load_rslt5_m128i, out);   // unpacks 9th 4 values


	__m128i c17_load_rslt6_m128i = _mm_loadu_si128(in + 5);
	__m128i c17_align_rslt5_m128i = _mm_alignr_epi8(c17_load_rslt6_m128i, c17_load_rslt5_m128i, 12);

	__horizontal_sse_unpack4_c17_f2<0>(c17_align_rslt5_m128i, out);  // unpacks 10th 4 values

	__horizontal_sse_unpack4_c17_f1<5>(c17_load_rslt6_m128i, out);   // unpacks 11th 4 values


	__m128i c17_load_rslt7_m128i = _mm_loadu_si128(in + 6);
	__m128i c17_align_rslt6_m128i = _mm_alignr_epi8(c17_load_rslt7_m128i, c17_load_rslt6_m128i, 13);

	__horizontal_sse_unpack4_c17_f2<0>(c17_align_rslt6_m128i, out);  // unpacks 12th 4 values

	__horizontal_sse_unpack4_c17_f1<6>(c17_load_rslt7_m128i, out);   // unpacks 13th 4 values


	__m128i c17_load_rslt8_m128i = _mm_loadu_si128(in + 7);
	__m128i c17_align_rslt7_m128i = _mm_alignr_epi8(c17_load_rslt8_m128i, c17_load_rslt7_m128i, 14);

	__horizontal_sse_unpack4_c17_f2<0>(c17_align_rslt7_m128i, out);  // unpacks 14th 4 values

	__horizontal_sse_unpack4_c17_f1<7>(c17_load_rslt8_m128i, out);   // unpacks 15th 4 values


	__m128i c17_load_rslt9_m128i = _mm_loadu_si128(in + 8);
	__m128i c17_align_rslt8_m128i = _mm_alignr_epi8(c17_load_rslt9_m128i, c17_load_rslt8_m128i, 15);

	__horizontal_sse_unpack4_c17_f2<0>(c17_align_rslt8_m128i, out);  // unpacks 16th 4 values


	__m128i c17_load_rslt10_m128i = _mm_loadu_si128(in + 9);
	__m128i c17_align_rslt9_m128i = _mm_alignr_epi8(c17_load_rslt10_m128i, c17_load_rslt9_m128i, 8);

	__horizontal_sse_unpack4_c17_f1<0>(c17_align_rslt9_m128i, out);  // unpacks 17th 4 values

	__horizontal_sse_unpack4_c17_f2<0>(c17_load_rslt10_m128i, out);  // unpacks 18th 4 values


	__m128i c17_load_rslt11_m128i = _mm_loadu_si128(in + 10);
	__m128i c17_align_rslt10_m128i = _mm_alignr_epi8(c17_load_rslt11_m128i, c17_load_rslt10_m128i, 9);

	__horizontal_sse_unpack4_c17_f1<0>(c17_align_rslt10_m128i, out); // unpacks 19th 4 values

	__horizontal_sse_unpack4_c17_f2<1>(c17_load_rslt11_m128i, out);  // unpacks 20th 4 values


	__m128i c17_load_rslt12_m128i = _mm_loadu_si128(in + 11);
	__m128i c17_align_rslt11_m128i = _mm_alignr_epi8(c17_load_rslt12_m128i, c17_load_rslt11_m128i, 10);

	__horizontal_sse_unpack4_c17_f1<0>(c17_align_rslt11_m128i, out); // unpacks 21st 4 values

	__horizontal_sse_unpack4_c17_f2<2>(c17_load_rslt12_m128i, out);  // unpacks 22nd 4 values


	__m128i c17_load_rslt13_m128i = _mm_loadu_si128(in + 12);
	__m128i c17_align_rslt12_m128i = _mm_alignr_epi8(c17_load_rslt13_m128i, c17_load_rslt12_m128i, 11);

	__horizontal_sse_unpack4_c17_f1<0>(c17_align_rslt12_m128i, out); // unpacks 23rd 4 values

	__horizontal_sse_unpack4_c17_f2<3>(c17_load_rslt13_m128i, out);  // unpacks 24th 4 values


	__m128i c17_load_rslt14_m128i = _mm_loadu_si128(in + 13);
	__m128i c17_align_rslt13_m128i = _mm_alignr_epi8(c17_load_rslt14_m128i, c17_load_rslt13_m128i, 12);

	__horizontal_sse_unpack4_c17_f1<0>(c17_align_rslt13_m128i, out); // unpacks 25th 4 values

	__horizontal_sse_unpack4_c17_f2<4>(c17_load_rslt14_m128i, out);  // unpacks 26th 4 values


	__m128i c17_load_rslt15_m128i = _mm_loadu_si128(in + 14);
	__m128i c17_align_rslt14_m128i = _mm_alignr_epi8(c17_load_rslt15_m128i, c17_load_rslt14_m128i, 13);

	__horizontal_sse_unpack4_c17_f1<0>(c17_align_rslt14_m128i, out); // unpacks 27th 4 values

	__horizontal_sse_unpack4_c17_f2<5>(c17_load_rslt15_m128i, out);  // unpacks 28th 4 values


	__m128i c17_load_rslt16_m128i = _mm_loadu_si128(in + 15);
	__m128i c17_align_rslt15_m128i = _mm_alignr_epi8(c17_load_rslt16_m128i, c17_load_rslt15_m128i, 14);

	__horizontal_sse_unpack4_c17_f1<0>(c17_align_rslt15_m128i, out); // unpacks 29th 4 values

	__horizontal_sse_unpack4_c17_f2<6>(c17_load_rslt16_m128i, out);  // unpacks 30th 4 values


	__m128i c17_load_rslt17_m128i = _mm_loadu_si128(in + 16);
	__m128i c17_align_rslt16_m128i = _mm_alignr_epi8(c17_load_rslt17_m128i, c17_load_rslt16_m128i, 15);

	__horizontal_sse_unpack4_c17_f1<0>(c17_align_rslt16_m128i, out); // unpacks 31st 4 values

	__horizontal_sse_unpack4_c17_f2<7>(c17_load_rslt17_m128i, out);  // unpacks 32nd 4 values
}




// 18-bit
template <bool IsRiceCoding>
template <uint32_t byte>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c18(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c18_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 8, byte + 7, byte + 6,
				0xff, byte + 6, byte + 5, byte + 4,
				0xff, byte + 4, byte + 3, byte + 2,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c18_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c18_shfl_msk_m128i);
		__m128i c18_mul_rslt_m128i = _mm_mullo_epi32(c18_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[18][0]);
		__m128i c18_srli_rslt_m128i = _mm_srli_epi32(c18_mul_rslt_m128i, 6);
		__m128i c18_rslt_m128i = _mm_and_si128(c18_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[18]);
		_mm_storeu_si128(out++, c18_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c18_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 8, byte + 7, byte + 6,
				0xff, byte + 6, byte + 5, byte + 4,
				0xff, byte + 4, byte + 3, byte + 2,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c18_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c18_shfl_msk_m128i);
		__m128i c18_mul_rslt_m128i = _mm_mullo_epi32(c18_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[18][0]);
		__m128i c18_srli_rslt_m128i = _mm_srli_epi32(c18_mul_rslt_m128i, 6);
		__m128i c18_and_rslt_m128i = _mm_and_si128(c18_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[18]);
		__m128i c18_rslt_m128i = _mm_or_si128(c18_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 18));
		_mm_storeu_si128(out++, c18_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c18(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 64) {
		__m128i c18_load_rslt1_m128i = _mm_loadu_si128(in++);

		__horizontal_sse_unpack4_c18<0>(c18_load_rslt1_m128i, out);      // unpacks 1st 4 values


		__m128i c18_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt1_m128i = _mm_alignr_epi8(c18_load_rslt2_m128i, c18_load_rslt1_m128i, 9);

		__horizontal_sse_unpack4_c18<0>(c18_alignr_rslt1_m128i, out);    // unpacks 2nd 4 values

		__horizontal_sse_unpack4_c18<2>(c18_load_rslt2_m128i, out);      // unpacks 3rd 4 values


		__m128i c18_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt2_m128i = _mm_alignr_epi8(c18_load_rslt3_m128i, c18_load_rslt2_m128i, 11);

		__horizontal_sse_unpack4_c18<0>(c18_alignr_rslt2_m128i, out);    // unpacks 4th 4 values

		__horizontal_sse_unpack4_c18<4>(c18_load_rslt3_m128i, out);      // unpacks 5th 4 values


		__m128i c18_load_rslt4_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt3_m128i = _mm_alignr_epi8(c18_load_rslt4_m128i, c18_load_rslt3_m128i, 13);

		__horizontal_sse_unpack4_c18<0>(c18_alignr_rslt3_m128i, out);    // unpacks 6th 4 values

		__horizontal_sse_unpack4_c18<6>(c18_load_rslt4_m128i, out);      // unpacks 7th 4 values


		__m128i c18_load_rslt5_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt4_m128i = _mm_alignr_epi8(c18_load_rslt5_m128i, c18_load_rslt4_m128i, 15);

		__horizontal_sse_unpack4_c18<0>(c18_alignr_rslt4_m128i, out);    // unpacks 8th 4 values


		__m128i c18_load_rslt6_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt5_m128i = _mm_alignr_epi8(c18_load_rslt6_m128i, c18_load_rslt5_m128i, 8);

		__horizontal_sse_unpack4_c18<0>(c18_alignr_rslt5_m128i, out);    // unpacks 9th 4 values

		__horizontal_sse_unpack4_c18<1>(c18_load_rslt6_m128i, out);      // unpacks 10th 4 values


		__m128i c18_load_rslt7_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt6_m128i = _mm_alignr_epi8(c18_load_rslt7_m128i, c18_load_rslt6_m128i, 10);

		__horizontal_sse_unpack4_c18<0>(c18_alignr_rslt6_m128i, out);    // unpacks 11th 4 values

		__horizontal_sse_unpack4_c18<3>(c18_load_rslt7_m128i, out);      // unpacks 12th 4 values


		__m128i c18_load_rslt8_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt7_m128i = _mm_alignr_epi8(c18_load_rslt8_m128i, c18_load_rslt7_m128i, 12);

		__horizontal_sse_unpack4_c18<0>(c18_alignr_rslt7_m128i, out);    // unpacks 13th 4 values

		__horizontal_sse_unpack4_c18<5>(c18_load_rslt8_m128i, out);      // unpacks 14th 4 values


		__m128i c18_load_rslt9_m128i = _mm_loadu_si128(in++);
		__m128i c18_alignr_rslt8_m128i = _mm_alignr_epi8(c18_load_rslt9_m128i, c18_load_rslt8_m128i, 14);

		__horizontal_sse_unpack4_c18<0>(c18_alignr_rslt8_m128i, out);    // unpacks 15th 4 values

		__horizontal_sse_unpack4_c18<7>(c18_load_rslt9_m128i, out);      // unpacks 16th 4 values
	}
}




// 19-bit
template <bool IsRiceCoding>
template <uint32_t byte>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c19_f1(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c19_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				0xff, byte + 4, byte + 3, byte + 2,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c19_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c19_shfl_msk_m128i);
		__m128i c19_mul_rslt_m128i = _mm_mullo_epi32(c19_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[19][0]);
		__m128i c19_srli_rslt_m128i = _mm_srli_epi32(c19_mul_rslt_m128i, 6);
		__m128i c19_rslt_m128i = _mm_and_si128(c19_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[19]);
		_mm_storeu_si128(out++, c19_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c19_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				0xff, byte + 4, byte + 3, byte + 2,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c19_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c19_shfl_msk_m128i);
		__m128i c19_mul_rslt_m128i = _mm_mullo_epi32(c19_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[19][0]);
		__m128i c19_srli_rslt_m128i = _mm_srli_epi32(c19_mul_rslt_m128i, 6);
		__m128i c19_and_rslt_m128i = _mm_and_si128(c19_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[19]);
		__m128i c19_rslt_m128i = _mm_or_si128(c19_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 19));
		_mm_storeu_si128(out++, c19_rslt_m128i);
	}
}

template <bool IsRiceCoding>
template <uint32_t byte>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c19_f2(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c19_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 9, byte + 8, byte + 7,
				0xff, byte + 7, byte + 6, byte + 5,
				byte + 5, byte + 4, byte + 3, byte + 2,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c19_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c19_shfl_msk_m128i);
		__m128i c19_mul_rslt_m128i = _mm_mullo_epi32(c19_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[19][1]);
		__m128i c19_srli_rslt_m128i = _mm_srli_epi32(c19_mul_rslt_m128i, 7);
		__m128i c19_rslt_m128i = _mm_and_si128(c19_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[19]);
		_mm_storeu_si128(out++, c19_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c19_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 9, byte + 8, byte + 7,
				0xff, byte + 7, byte + 6, byte + 5,
				byte + 5, byte + 4, byte + 3, byte + 2,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c19_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c19_shfl_msk_m128i);
		__m128i c19_mul_rslt_m128i = _mm_mullo_epi32(c19_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[19][1]);
		__m128i c19_srli_rslt_m128i = _mm_srli_epi32(c19_mul_rslt_m128i, 7);
		__m128i c19_and_rslt_m128i = _mm_and_si128(c19_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[19]);
		__m128i c19_rslt_m128i = _mm_or_si128(c19_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 19));
		_mm_storeu_si128(out++, c19_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c19(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
    __m128i c19_load_rslt1_m128i = _mm_loadu_si128(in + 0);

    __horizontal_sse_unpack4_c19_f1<0>(c19_load_rslt1_m128i, out);     // unpacks 1st 4 values


    __m128i c19_load_rslt2_m128i = _mm_loadu_si128(in + 1);
    __m128i c19_align_rslt1_m128i = _mm_alignr_epi8(c19_load_rslt2_m128i, c19_load_rslt1_m128i, 9);

    __horizontal_sse_unpack4_c19_f2<0>(c19_align_rslt1_m128i, out);    // unpacks 2nd 4 values

    __horizontal_sse_unpack4_c19_f1<3>(c19_load_rslt2_m128i, out);     // unpacks 3rd 4 values


    __m128i c19_load_rslt3_m128i = _mm_loadu_si128(in + 2);
    __m128i c19_align_rslt2_m128i = _mm_alignr_epi8(c19_load_rslt3_m128i, c19_load_rslt2_m128i, 12);

    __horizontal_sse_unpack4_c19_f2<0>(c19_align_rslt2_m128i, out);    // unpacks 4th 4 values

    __horizontal_sse_unpack4_c19_f1<6>(c19_load_rslt3_m128i, out);     // unpacks 5th 4 values


    __m128i c19_load_rslt4_m128i = _mm_loadu_si128(in + 3);
    __m128i c19_align_rslt3_m128i = _mm_alignr_epi8(c19_load_rslt4_m128i, c19_load_rslt3_m128i, 15);

    __horizontal_sse_unpack4_c19_f2<0>(c19_align_rslt3_m128i, out);    // unpacks 6th 4 values


    __m128i c19_load_rslt5_m128i = _mm_loadu_si128(in + 4);
    __m128i c19_align_rslt4_m128i = _mm_alignr_epi8(c19_load_rslt5_m128i, c19_load_rslt4_m128i, 9);

    __horizontal_sse_unpack4_c19_f1<0>(c19_align_rslt4_m128i, out);    // unpacks 7th 4 values

    __horizontal_sse_unpack4_c19_f2<2>(c19_load_rslt5_m128i, out);     // unpacks 8th 4 values


    __m128i c19_load_rslt6_m128i = _mm_loadu_si128(in + 5);
    __m128i c19_align_rslt5_m128i = _mm_alignr_epi8(c19_load_rslt6_m128i, c19_load_rslt5_m128i, 12);

    __horizontal_sse_unpack4_c19_f1<0>(c19_align_rslt5_m128i, out);    // unpacks 9th 4 values

    __horizontal_sse_unpack4_c19_f2<5>(c19_load_rslt6_m128i, out);     // unpacks 10th 4 values


    __m128i c19_load_rslt7_m128i = _mm_loadu_si128(in + 6);
    __m128i c19_align_rslt6_m128i = _mm_alignr_epi8(c19_load_rslt7_m128i, c19_load_rslt6_m128i, 15);

    __horizontal_sse_unpack4_c19_f1<0>(c19_align_rslt6_m128i, out);    // unpacks 11th 4 values


    __m128i c19_load_rslt8_m128i = _mm_loadu_si128(in + 7);
    __m128i c19_align_rslt7_m128i = _mm_alignr_epi8(c19_load_rslt8_m128i, c19_load_rslt7_m128i, 8);

    __horizontal_sse_unpack4_c19_f2<0>(c19_align_rslt7_m128i, out);    // unpacks 12th 4 values

    __horizontal_sse_unpack4_c19_f1<2>(c19_load_rslt8_m128i, out);     // unpacks 13th 4 values


    __m128i c19_load_rslt9_m128i = _mm_loadu_si128(in + 8);
    __m128i c19_align_rslt8_m128i = _mm_alignr_epi8(c19_load_rslt9_m128i, c19_load_rslt8_m128i, 11);

    __horizontal_sse_unpack4_c19_f2<0>(c19_align_rslt8_m128i, out);    // unpacks 14th 4 values

    __horizontal_sse_unpack4_c19_f1<5>(c19_load_rslt9_m128i, out);     // unpacks 15th 4 values


    __m128i c19_load_rslt10_m128i = _mm_loadu_si128(in + 9);
    __m128i c19_align_rslt9_m128i = _mm_alignr_epi8(c19_load_rslt10_m128i, c19_load_rslt9_m128i, 14);

    __horizontal_sse_unpack4_c19_f2<0>(c19_align_rslt9_m128i, out);    // unpacks 16th 4 values


    __m128i c19_load_rslt11_m128i = _mm_loadu_si128(in + 10);
    __m128i c19_align_rslt10_m128i = _mm_alignr_epi8(c19_load_rslt11_m128i, c19_load_rslt10_m128i, 8);

    __horizontal_sse_unpack4_c19_f1<0>(c19_align_rslt10_m128i, out);   // unpacks 17th 4 values

    __horizontal_sse_unpack4_c19_f2<1>(c19_load_rslt11_m128i, out);    // unpacks 18th 4 values


    __m128i c19_load_rslt12_m128i = _mm_loadu_si128(in + 11);
    __m128i c19_align_rslt11_m128i = _mm_alignr_epi8(c19_load_rslt12_m128i, c19_load_rslt11_m128i, 11);

    __horizontal_sse_unpack4_c19_f1<0>(c19_align_rslt11_m128i, out);   // unpacks 19th 4 values

    __horizontal_sse_unpack4_c19_f2<4>(c19_load_rslt12_m128i, out);    // unpacks 20th 4 values


    __m128i c19_load_rslt13_m128i = _mm_loadu_si128(in + 12);
    __m128i c19_align_rslt12_m128i = _mm_alignr_epi8(c19_load_rslt13_m128i, c19_load_rslt12_m128i, 14);

    __horizontal_sse_unpack4_c19_f1<0>(c19_align_rslt12_m128i, out);   // unpacks 21st 4 values


    __m128i c19_load_rslt14_m128i = _mm_loadu_si128(in + 13);
    __m128i c19_align_rslt13_m128i = _mm_alignr_epi8(c19_load_rslt14_m128i, c19_load_rslt13_m128i, 7);

    __horizontal_sse_unpack4_c19_f2<0>(c19_align_rslt13_m128i, out);   // unpacks 22nd 4 values

    __horizontal_sse_unpack4_c19_f1<1>(c19_load_rslt14_m128i, out);    // unpacks 23rd 4 values


    __m128i c19_load_rslt15_m128i = _mm_loadu_si128(in + 14);
    __m128i c19_align_rslt14_m128i = _mm_alignr_epi8(c19_load_rslt15_m128i, c19_load_rslt14_m128i, 10);

    __horizontal_sse_unpack4_c19_f2<0>(c19_align_rslt14_m128i, out);   // unpacks 24th 4 values

    __horizontal_sse_unpack4_c19_f1<4>(c19_load_rslt15_m128i, out);    // unpacks 25th 4 values


    __m128i c19_load_rslt16_m128i = _mm_loadu_si128(in + 15);
    __m128i c19_align_rslt15_m128i = _mm_alignr_epi8(c19_load_rslt16_m128i, c19_load_rslt15_m128i, 13);

    __horizontal_sse_unpack4_c19_f2<0>(c19_align_rslt15_m128i, out);   // unpacks 26th 4 values


    __m128i c19_load_rslt17_m128i = _mm_loadu_si128(in + 16);
    __m128i c19_align_rslt16_m128i = _mm_alignr_epi8(c19_load_rslt17_m128i, c19_load_rslt16_m128i, 7);

    __horizontal_sse_unpack4_c19_f1<0>(c19_align_rslt16_m128i, out);   // unpacks 27th 4 values

    __horizontal_sse_unpack4_c19_f2<0>(c19_load_rslt17_m128i, out);    // unpacks 28th 4 values


    __m128i c19_load_rslt18_m128i = _mm_loadu_si128(in + 17);
    __m128i c19_align_rslt17_m128i = _mm_alignr_epi8(c19_load_rslt18_m128i, c19_load_rslt17_m128i, 10);

    __horizontal_sse_unpack4_c19_f1<0>(c19_align_rslt17_m128i, out);   // unpacks 29th 4 values

    __horizontal_sse_unpack4_c19_f2<3>(c19_load_rslt18_m128i, out);    // unpacks 30th 4 values


    __m128i c19_load_rslt19_m128i = _mm_loadu_si128(in + 18);
    __m128i c19_align_rslt18_m128i = _mm_alignr_epi8(c19_load_rslt19_m128i, c19_load_rslt18_m128i, 13);

    __horizontal_sse_unpack4_c19_f1<0>(c19_align_rslt18_m128i, out);   // unpacks 31th 4 values

    __horizontal_sse_unpack4_c19_f2<6>(c19_load_rslt19_m128i, out);    // unpacks 32th 4 values
}




// 20-bit
template <bool IsRiceCoding>
template <uint32_t byte>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c20(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
	// unpacks 4 values
	__m128i Horizontal_SSE_c20_shfl_msk_m128i = _mm_set_epi8(
			0xff, byte + 9, byte + 8, byte + 7,
			0xff, byte + 7, byte + 6, byte + 5,
			0xff, byte + 4, byte + 3, byte + 2,
			0xff, byte + 2, byte + 1, byte + 0);

	__m128i c20_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c20_shfl_msk_m128i);
	__m128i c20_mul_rslt_m128i = _mm_mullo_epi32(c20_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[20][0]);
	__m128i c20_srli_rslt_m128i = _mm_srli_epi32(c20_mul_rslt_m128i, 4);
	__m128i c20_rslt_m128i = _mm_and_si128(c20_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[20]);
	_mm_storeu_si128(out++, c20_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c20_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 9, byte + 8, byte + 7,
				0xff, byte + 7, byte + 6, byte + 5,
				0xff, byte + 4, byte + 3, byte + 2,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c20_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c20_shfl_msk_m128i);
		__m128i c20_mul_rslt_m128i = _mm_mullo_epi32(c20_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[20][0]);
		__m128i c20_srli_rslt_m128i = _mm_srli_epi32(c20_mul_rslt_m128i, 4);
		__m128i c20_and_rslt_m128i = _mm_and_si128(c20_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[20]);
		__m128i c20_rslt_m128i = _mm_or_si128(c20_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 20));
		_mm_storeu_si128(out++, c20_rslt_m128i);
	}
}


template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c20(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 32) {
		__m128i c20_load_rslt1_m128i = _mm_loadu_si128(in++);

		__horizontal_sse_unpack4_c20<0>(c20_load_rslt1_m128i, out);    // unpacks 1st 4 values


		__m128i c20_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c20_alignr_rslt1_m128i = _mm_alignr_epi8(c20_load_rslt2_m128i, c20_load_rslt1_m128i, 10);

		__horizontal_sse_unpack4_c20<0>(c20_alignr_rslt1_m128i, out);  // unpacks 2nd 4 values

		__horizontal_sse_unpack4_c20<4>(c20_load_rslt2_m128i, out);    // unpacks 3rd 4 values


		__m128i c20_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c20_alignr_rslt2_m128i = _mm_alignr_epi8(c20_load_rslt3_m128i, c20_load_rslt2_m128i, 14);

		__horizontal_sse_unpack4_c20<0>(c20_alignr_rslt2_m128i, out);  // unpacks 4th 4 values


		__m128i c20_load_rslt4_m128i = _mm_loadu_si128(in++);
		__m128i c20_alignr_rslt3_m128i = _mm_alignr_epi8(c20_load_rslt4_m128i, c20_load_rslt3_m128i, 8);

		__horizontal_sse_unpack4_c20<0>(c20_alignr_rslt3_m128i, out);  // unpacks 5th 4 values

		__horizontal_sse_unpack4_c20<2>(c20_load_rslt4_m128i, out);    // unpacks 6th 4 values


		__m128i c20_load_rslt5_m128i = _mm_loadu_si128(in++);
		__m128i c20_alignr_rslt4_m128i = _mm_alignr_epi8(c20_load_rslt5_m128i, c20_load_rslt4_m128i, 12);

		__horizontal_sse_unpack4_c20<0>(c20_alignr_rslt4_m128i, out);  // unpacks 7th 4 values

		__horizontal_sse_unpack4_c20<6>(c20_load_rslt5_m128i, out);    // unpacks 8th 4 values
	}
}




// 21-bit
template <bool IsRiceCoding>
template <uint32_t byte>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c21_f1(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c21_shfl_msk_m128i = _mm_set_epi8(
				byte + 10, byte + 9, byte + 8, byte + 7,
				0xff, byte + 7, byte + 6, byte + 5,
				byte + 5, byte + 4, byte + 3, byte + 2,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c21_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c21_shfl_msk_m128i);
		__m128i c21_mul_rslt_m128i = _mm_mullo_epi32(c21_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[21][0]);
		__m128i c21_srli_rslt_m128i = _mm_srli_epi32(c21_mul_rslt_m128i, 7);
		__m128i c21_rslt_m128i = _mm_and_si128(c21_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[21]);
		_mm_storeu_si128(out++, c21_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c21_shfl_msk_m128i = _mm_set_epi8(
				byte + 10, byte + 9, byte + 8, byte + 7,
				0xff, byte + 7, byte + 6, byte + 5,
				byte + 5, byte + 4, byte + 3, byte + 2,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c21_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c21_shfl_msk_m128i);
		__m128i c21_mul_rslt_m128i = _mm_mullo_epi32(c21_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[21][0]);
		__m128i c21_srli_rslt_m128i = _mm_srli_epi32(c21_mul_rslt_m128i, 7);
		__m128i c21_and_rslt_m128i = _mm_and_si128(c21_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[21]);
		__m128i c21_rslt_m128i = _mm_or_si128(c21_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 21));
		_mm_storeu_si128(out++, c21_rslt_m128i);
	}
}

template <bool IsRiceCoding>
template <uint32_t byte>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c21_f2(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c21_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 10, byte + 9, byte + 8,
				byte + 8, byte + 7, byte + 6, byte + 5,
				0xff, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c21_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c21_shfl_msk_m128i);
		__m128i c21_mul_rslt_m128i = _mm_mullo_epi32(c21_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[21][1]);
		__m128i c21_srli_rslt_m128i = _mm_srli_epi32(c21_mul_rslt_m128i, 6);
		__m128i c21_rslt_m128i = _mm_and_si128(c21_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[21]);
		_mm_storeu_si128(out++, c21_rslt_m128i);
	}
	else { // RiceCoding
		// unpacks 4 values
		__m128i Horizontal_SSE_c21_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 10, byte + 9, byte + 8,
				byte + 8, byte + 7, byte + 6, byte + 5,
				0xff, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c21_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c21_shfl_msk_m128i);
		__m128i c21_mul_rslt_m128i = _mm_mullo_epi32(c21_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[21][1]);
		__m128i c21_srli_rslt_m128i = _mm_srli_epi32(c21_mul_rslt_m128i, 6);
		__m128i c21_and_rslt_m128i = _mm_and_si128(c21_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[21]);
		__m128i c21_rslt_m128i = _mm_or_si128(c21_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 21));
		_mm_storeu_si128(out++, c21_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c21(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
    __m128i c21_load_rslt1_m128i = _mm_loadu_si128(in + 0);

    __horizontal_sse_unpack4_c21_f1<0>(c21_load_rslt1_m128i, out);     // unpacks 1st 4 values


    __m128i c21_load_rslt2_m128i = _mm_loadu_si128(in + 1);
    __m128i c21_align_rslt1_m128i = _mm_alignr_epi8(c21_load_rslt2_m128i, c21_load_rslt1_m128i, 10);

    __horizontal_sse_unpack4_c21_f2<0>(c21_align_rslt1_m128i, out);    // unpacks 2nd 4 values

    __horizontal_sse_unpack4_c21_f1<5>(c21_load_rslt2_m128i, out);     // unpacks 3rd 4 values


    __m128i c21_load_rslt3_m128i = _mm_loadu_si128(in + 2);
    __m128i c21_align_rslt2_m128i = _mm_alignr_epi8(c21_load_rslt3_m128i, c21_load_rslt2_m128i, 15);

    __horizontal_sse_unpack4_c21_f2<0>(c21_align_rslt2_m128i, out);    // unpacks 4th 4 values


    __m128i c21_load_rslt4_m128i = _mm_loadu_si128(in + 3);
    __m128i c21_align_rslt3_m128i = _mm_alignr_epi8(c21_load_rslt4_m128i, c21_load_rslt3_m128i, 10);

    __horizontal_sse_unpack4_c21_f1<0>(c21_align_rslt3_m128i, out);    // unpacks 5th 4 values

    __horizontal_sse_unpack4_c21_f2<4>(c21_load_rslt4_m128i, out);     // unpacks 6th 4 values


    __m128i c21_load_rslt5_m128i = _mm_loadu_si128(in + 4);
    __m128i c21_align_rslt4_m128i = _mm_alignr_epi8(c21_load_rslt5_m128i, c21_load_rslt4_m128i, 15);

    __horizontal_sse_unpack4_c21_f1<0>(c21_align_rslt4_m128i, out);    // unpacks 7th 4 values


    __m128i c21_load_rslt6_m128i = _mm_loadu_si128(in + 5);
    __m128i c21_align_rslt5_m128i = _mm_alignr_epi8(c21_load_rslt6_m128i, c21_load_rslt5_m128i, 9);

    __horizontal_sse_unpack4_c21_f2<0>(c21_align_rslt5_m128i, out);    // unpacks 8th 4 values

    __horizontal_sse_unpack4_c21_f1<4>(c21_load_rslt6_m128i, out);     // unpacks 9th 4 values


    __m128i c21_load_rslt7_m128i = _mm_loadu_si128(in + 6);
    __m128i c21_align_rslt6_m128i = _mm_alignr_epi8(c21_load_rslt7_m128i, c21_load_rslt6_m128i, 14);

    __horizontal_sse_unpack4_c21_f2<0>(c21_align_rslt6_m128i, out);    // unpacks 10th 4 values


    __m128i c21_load_rslt8_m128i = _mm_loadu_si128(in + 7);
    __m128i c21_align_rslt7_m128i = _mm_alignr_epi8(c21_load_rslt8_m128i, c21_load_rslt7_m128i, 9);

    __horizontal_sse_unpack4_c21_f1<0>(c21_align_rslt7_m128i, out);    // unpacks 11th 4 values

    __horizontal_sse_unpack4_c21_f2<3>(c21_load_rslt8_m128i, out);     // unpacks 12th 4 values


    __m128i c21_load_rslt9_m128i = _mm_loadu_si128(in + 8);
    __m128i c21_align_rslt8_m128i = _mm_alignr_epi8(c21_load_rslt9_m128i, c21_load_rslt8_m128i, 14);

    __horizontal_sse_unpack4_c21_f1<0>(c21_align_rslt8_m128i, out);    // unpacks 13th 4 values


    __m128i c21_load_rslt10_m128i = _mm_loadu_si128(in + 9);
    __m128i c21_align_rslt9_m128i = _mm_alignr_epi8(c21_load_rslt10_m128i, c21_load_rslt9_m128i, 8);

    __horizontal_sse_unpack4_c21_f2<0>(c21_align_rslt9_m128i, out);    // unpacks 14th 4 values

    __horizontal_sse_unpack4_c21_f1<3>(c21_load_rslt10_m128i, out);    // unpacks 15th 4 values


    __m128i c21_load_rslt11_m128i = _mm_loadu_si128(in + 10);
    __m128i c21_align_rslt10_m128i = _mm_alignr_epi8(c21_load_rslt11_m128i, c21_load_rslt10_m128i, 13);

    __horizontal_sse_unpack4_c21_f2<0>(c21_align_rslt10_m128i, out);   // unpacks 16th 4 values


    __m128i c21_load_rslt12_m128i = _mm_loadu_si128(in + 11);
    __m128i c21_align_rslt11_m128i = _mm_alignr_epi8(c21_load_rslt12_m128i, c21_load_rslt11_m128i, 8);

    __horizontal_sse_unpack4_c21_f1<0>(c21_align_rslt11_m128i, out);   // unpacks 17th 4 values

    __horizontal_sse_unpack4_c21_f2<2>(c21_load_rslt12_m128i, out);    // unpacks 18th 4 values


    __m128i c21_load_rslt13_m128i = _mm_loadu_si128(in + 12);
    __m128i c21_align_rslt12_m128i = _mm_alignr_epi8(c21_load_rslt13_m128i, c21_load_rslt12_m128i, 13);

    __horizontal_sse_unpack4_c21_f1<0>(c21_align_rslt12_m128i, out);   // unpacks 19th 4 values


    __m128i c21_load_rslt14_m128i = _mm_loadu_si128(in + 13);
    __m128i c21_align_rslt13_m128i = _mm_alignr_epi8(c21_load_rslt14_m128i, c21_load_rslt13_m128i, 7);

    __horizontal_sse_unpack4_c21_f2<0>(c21_align_rslt13_m128i, out);   // unpacks 20th 4 values

    __horizontal_sse_unpack4_c21_f1<2>(c21_load_rslt14_m128i, out);    // unpacks 21st 4 values


    __m128i c21_load_rslt15_m128i = _mm_loadu_si128(in + 14);
    __m128i c21_align_rslt14_m128i = _mm_alignr_epi8(c21_load_rslt15_m128i, c21_load_rslt14_m128i, 12);

    __horizontal_sse_unpack4_c21_f2<0>(c21_align_rslt14_m128i, out);   // unpacks 22nd 4 values


    __m128i c21_load_rslt16_m128i = _mm_loadu_si128(in + 15);
    __m128i c21_align_rslt15_m128i = _mm_alignr_epi8(c21_load_rslt16_m128i, c21_load_rslt15_m128i, 7);

    __horizontal_sse_unpack4_c21_f1<0>(c21_align_rslt15_m128i, out);   // unpacks 23rd 4 values

    __horizontal_sse_unpack4_c21_f2<1>(c21_load_rslt16_m128i, out);    // unpacks 24th 4 values


    __m128i c21_load_rslt17_m128i = _mm_loadu_si128(in + 16);
    __m128i c21_align_rslt16_m128i = _mm_alignr_epi8(c21_load_rslt17_m128i, c21_load_rslt16_m128i, 12);

    __horizontal_sse_unpack4_c21_f1<0>(c21_align_rslt16_m128i, out);   // unpacks 25th 4 values


    __m128i c21_load_rslt18_m128i = _mm_loadu_si128(in + 17);
    __m128i c21_align_rslt17_m128i = _mm_alignr_epi8(c21_load_rslt18_m128i, c21_load_rslt17_m128i, 6);

    __horizontal_sse_unpack4_c21_f2<0>(c21_align_rslt17_m128i, out);   // unpacks 26th 4 values

    __horizontal_sse_unpack4_c21_f1<1>(c21_load_rslt18_m128i, out);    // unpacks 27th 4 values


    __m128i c21_load_rslt19_m128i = _mm_loadu_si128(in + 18);
    __m128i c21_align_rslt18_m128i = _mm_alignr_epi8(c21_load_rslt19_m128i, c21_load_rslt18_m128i, 11);

    __horizontal_sse_unpack4_c21_f2<0>(c21_align_rslt18_m128i, out);   // unpacks 28th 4 values


    __m128i c21_load_rslt20_m128i = _mm_loadu_si128(in + 19);
    __m128i c21_align_rslt19_m128i = _mm_alignr_epi8(c21_load_rslt20_m128i, c21_load_rslt19_m128i, 6);

    __horizontal_sse_unpack4_c21_f1<0>(c21_align_rslt19_m128i, out);   // unpacks 29th 4 values

    __horizontal_sse_unpack4_c21_f2<0>(c21_load_rslt20_m128i, out);    // unpacks 30th 4 values


    __m128i c21_load_rslt21_m128i = _mm_loadu_si128(in + 20);
    __m128i c21_align_rslt20_m128i = _mm_alignr_epi8(c21_load_rslt21_m128i, c21_load_rslt20_m128i, 11);

    __horizontal_sse_unpack4_c21_f1<0>(c21_align_rslt20_m128i, out);   // unpacks 31st 4 values

    __horizontal_sse_unpack4_c21_f2<5>(c21_load_rslt21_m128i, out);    // unpacks 32nd 4 values
}




// 22-bit
template <bool IsRiceCoding>
template <uint32_t byte>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c22(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c22_shfl_msk_m128i = _mm_set_epi8(
				byte + 11, byte + 10, byte + 9, byte + 8,
				byte + 8, byte + 7, byte + 6, byte + 5,
				byte + 5, byte + 4, byte + 3, byte + 2,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c22_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c22_shfl_msk_m128i);
		__m128i c22_mul_rslt_m128i = _mm_mullo_epi32(c22_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[22][0]);
		__m128i c22_srli_rslt_m128i = _mm_srli_epi32(c22_mul_rslt_m128i, 6);
		__m128i c22_rslt_m128i = _mm_and_si128(c22_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[22]);
		_mm_storeu_si128(out++, c22_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c22_shfl_msk_m128i = _mm_set_epi8(
				byte + 11, byte + 10, byte + 9, byte + 8,
				byte + 8, byte + 7, byte + 6, byte + 5,
				byte + 5, byte + 4, byte + 3, byte + 2,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c22_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c22_shfl_msk_m128i);
		__m128i c22_mul_rslt_m128i = _mm_mullo_epi32(c22_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[22][0]);
		__m128i c22_srli_rslt_m128i = _mm_srli_epi32(c22_mul_rslt_m128i, 6);
		__m128i c22_and_rslt_m128i = _mm_and_si128(c22_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[22]);
		__m128i c22_rslt_m128i = _mm_or_si128(c22_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 22));
		_mm_storeu_si128(out++, c22_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c22(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 64) {
	     __m128i c22_load_rslt1_m128i = _mm_loadu_si128(in++);

	     __horizontal_sse_unpack4_c22<0>(c22_load_rslt1_m128i, out);     // unpacks 1st 4 values


	     __m128i c22_load_rslt2_m128i = _mm_loadu_si128(in++);
	     __m128i c22_align_rslt1_m128i = _mm_alignr_epi8(c22_load_rslt2_m128i, c22_load_rslt1_m128i, 11);

	     __horizontal_sse_unpack4_c22<0>(c22_align_rslt1_m128i, out);    // unpacks 2nd 4 values


	     __m128i c22_load_rslt3_m128i = _mm_loadu_si128(in++);
	     __m128i c22_align_rslt2_m128i = _mm_alignr_epi8(c22_load_rslt3_m128i, c22_load_rslt2_m128i, 6);

	     __horizontal_sse_unpack4_c22<0>(c22_align_rslt2_m128i, out);    // unpacks 3rd 4 values

	     __horizontal_sse_unpack4_c22<1>(c22_load_rslt3_m128i, out);     // unpacks 4th 4 values


	     __m128i c22_load_rslt4_m128i = _mm_loadu_si128(in++);
	     __m128i c22_align_rslt3_m128i = _mm_alignr_epi8(c22_load_rslt4_m128i, c22_load_rslt3_m128i, 12);

	     __horizontal_sse_unpack4_c22<0>(c22_align_rslt3_m128i, out);    // unpacks 5th 4 values


	     __m128i c22_load_rslt5_m128i = _mm_loadu_si128(in++);
	     __m128i c22_align_rslt4_m128i = _mm_alignr_epi8(c22_load_rslt5_m128i, c22_load_rslt4_m128i, 7);

	     __horizontal_sse_unpack4_c22<0>(c22_align_rslt4_m128i, out);    // unpacks 6th 4 values

	     __horizontal_sse_unpack4_c22<2>(c22_load_rslt5_m128i, out);     // unpacks 7th 4 values


	     __m128i c22_load_rslt6_m128i = _mm_loadu_si128(in++);
	     __m128i c22_align_rslt5_m128i = _mm_alignr_epi8(c22_load_rslt6_m128i, c22_load_rslt5_m128i, 13);

	     __horizontal_sse_unpack4_c22<0>(c22_align_rslt5_m128i, out);    // unpacks 8th 4 values


	     __m128i c22_load_rslt7_m128i = _mm_loadu_si128(in++);
	     __m128i c22_align_rslt6_m128i = _mm_alignr_epi8(c22_load_rslt7_m128i, c22_load_rslt6_m128i, 8);

	     __horizontal_sse_unpack4_c22<0>(c22_align_rslt6_m128i, out);    // unpacks 9th 4 values

	     __horizontal_sse_unpack4_c22<3>(c22_load_rslt7_m128i, out);     // unpacks 10th 4 values


	     __m128i c22_load_rslt8_m128i = _mm_loadu_si128(in++);
	     __m128i c22_align_rslt7_m128i = _mm_alignr_epi8(c22_load_rslt8_m128i, c22_load_rslt7_m128i, 14);

	     __horizontal_sse_unpack4_c22<0>(c22_align_rslt7_m128i, out);    // unpacks 11th 4 values


	     __m128i c22_load_rslt9_m128i = _mm_loadu_si128(in++);
	     __m128i c22_align_rslt8_m128i = _mm_alignr_epi8(c22_load_rslt9_m128i, c22_load_rslt8_m128i, 9);

	     __horizontal_sse_unpack4_c22<0>(c22_align_rslt8_m128i, out);    // unpacks 12th 4 values

	     __horizontal_sse_unpack4_c22<4>(c22_load_rslt9_m128i, out);     // unpacks 13th 4 values


	     __m128i c22_load_rslt10_m128i = _mm_loadu_si128(in++);
	     __m128i c22_align_rslt9_m128i = _mm_alignr_epi8(c22_load_rslt10_m128i, c22_load_rslt9_m128i, 15);

	     __horizontal_sse_unpack4_c22<0>(c22_align_rslt9_m128i, out);    // unpacks 14th 4 values


	     __m128i c22_load_rslt11_m128i = _mm_loadu_si128(in++);
	     __m128i c22_align_rslt10_m128i = _mm_alignr_epi8(c22_load_rslt11_m128i, c22_load_rslt10_m128i, 10);

	     __horizontal_sse_unpack4_c22<0>(c22_align_rslt10_m128i, out);

	     __horizontal_sse_unpack4_c22<5>(c22_load_rslt11_m128i, out);
	}
}




// 23-bit
template <bool IsRiceCoding>
template <uint32_t byte>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c23_f1(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c23_shfl_msk_m128i = _mm_set_epi8(
				byte + 11, byte + 10, byte + 9, byte + 8,
				byte + 8, byte + 7, byte + 6, byte + 5,
				byte + 5, byte + 4, byte + 3, byte + 2,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c23_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c23_shfl_msk_m128i);
		__m128i c23_mul_rslt_m128i = _mm_mullo_epi32(c23_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[23][0]);
		__m128i c23_srli_rslt_m128i = _mm_srli_epi32(c23_mul_rslt_m128i, 7);
		__m128i c23_rslt_m128i = _mm_and_si128(c23_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[23]);
		_mm_storeu_si128(out++, c23_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c23_shfl_msk_m128i = _mm_set_epi8(
				byte + 11, byte + 10, byte + 9, byte + 8,
				byte + 8, byte + 7, byte + 6, byte + 5,
				byte + 5, byte + 4, byte + 3, byte + 2,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c23_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c23_shfl_msk_m128i);
		__m128i c23_mul_rslt_m128i = _mm_mullo_epi32(c23_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[23][0]);
		__m128i c23_srli_rslt_m128i = _mm_srli_epi32(c23_mul_rslt_m128i, 7);
		__m128i c23_and_rslt_m128i = _mm_and_si128(c23_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[23]);
		__m128i c23_rslt_m128i = _mm_or_si128(c23_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 23));
		_mm_storeu_si128(out++, c23_rslt_m128i);
	}
}

template <bool IsRiceCoding>
template <uint32_t byte>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c23_f2(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c23_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 11, byte + 10, byte + 9,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c23_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c23_shfl_msk_m128i);
		__m128i c23_mul_rslt_m128i = _mm_mullo_epi32(c23_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[23][1]);
		__m128i c23_srli_rslt_m128i = _mm_srli_epi32(c23_mul_rslt_m128i, 4);
		__m128i c23_rslt_m128i = _mm_and_si128(c23_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[23]);
		_mm_storeu_si128(out++, c23_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c23_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 11, byte + 10, byte + 9,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c23_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c23_shfl_msk_m128i);
		__m128i c23_mul_rslt_m128i = _mm_mullo_epi32(c23_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[23][1]);
		__m128i c23_srli_rslt_m128i = _mm_srli_epi32(c23_mul_rslt_m128i, 4);
		__m128i c23_and_rslt_m128i = _mm_and_si128(c23_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[23]);
		__m128i c23_rslt_m128i = _mm_or_si128(c23_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 23));
		_mm_storeu_si128(out++, c23_rslt_m128i);
	}
}


template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c23(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
    __m128i c23_load_rslt1_m128i = _mm_loadu_si128(in + 0);

     __horizontal_sse_unpack4_c23_f1<0>(c23_load_rslt1_m128i, out);     // unpacks 1st 4 values


     __m128i c23_load_rslt2_m128i = _mm_loadu_si128(in + 1);
     __m128i c23_align_rslt1_m128i = _mm_alignr_epi8(c23_load_rslt2_m128i, c23_load_rslt1_m128i, 11);

     __horizontal_sse_unpack4_c23_f2<0>(c23_align_rslt1_m128i, out);    // unpacks 2nd 4 values


     __m128i c23_load_rslt3_m128i = _mm_loadu_si128(in + 2);
     __m128i c23_align_rslt2_m128i = _mm_alignr_epi8(c23_load_rslt3_m128i, c23_load_rslt2_m128i, 7);

     __horizontal_sse_unpack4_c23_f1<0>(c23_align_rslt2_m128i, out);    // unpacks 3rd 4 values

     __horizontal_sse_unpack4_c23_f2<2>(c23_load_rslt3_m128i, out);     // unpacks 4th 4 values


     __m128i c23_load_rslt4_m128i = _mm_loadu_si128(in + 3);
     __m128i c23_align_rslt3_m128i = _mm_alignr_epi8(c23_load_rslt4_m128i, c23_load_rslt3_m128i, 14);

     __horizontal_sse_unpack4_c23_f1<0>(c23_align_rslt3_m128i, out);    // unpacks 5th 4 values


     __m128i c23_load_rslt5_m128i = _mm_loadu_si128(in + 4);
     __m128i c23_align_rslt4_m128i = _mm_alignr_epi8(c23_load_rslt5_m128i, c23_load_rslt4_m128i, 9);

     __horizontal_sse_unpack4_c23_f2<0>(c23_align_rslt4_m128i, out);    // unpacks 6th 4 values


     __m128i c23_load_rslt6_m128i = _mm_loadu_si128(in + 5);
     __m128i c23_align_rslt5_m128i = _mm_alignr_epi8(c23_load_rslt6_m128i, c23_load_rslt5_m128i, 5);

     __horizontal_sse_unpack4_c23_f1<0>(c23_align_rslt5_m128i, out);    // unpacks 7th 4 values

     __horizontal_sse_unpack4_c23_f2<0>(c23_load_rslt6_m128i, out);     // unpacks 8th 4 values


     __m128i c23_load_rslt7_m128i = _mm_loadu_si128(in + 6);
     __m128i c23_align_rslt6_m128i = _mm_alignr_epi8(c23_load_rslt7_m128i, c23_load_rslt6_m128i, 12);

     __horizontal_sse_unpack4_c23_f1<0>(c23_align_rslt6_m128i, out);    // unpacks 9th 4 values


     __m128i c23_load_rslt8_m128i = _mm_loadu_si128(in + 7);
     __m128i c23_align_rslt7_m128i = _mm_alignr_epi8(c23_load_rslt8_m128i, c23_load_rslt7_m128i, 7);

     __horizontal_sse_unpack4_c23_f2<0>(c23_align_rslt7_m128i, out);    // unpacks 10th 4 values

     __horizontal_sse_unpack4_c23_f1<3>(c23_load_rslt8_m128i, out);     // unpacks 11th 4 values


     __m128i c23_load_rslt9_m128i = _mm_loadu_si128(in + 8);
     __m128i c23_align_rslt8_m128i = _mm_alignr_epi8(c23_load_rslt9_m128i, c23_load_rslt8_m128i, 14);

     __horizontal_sse_unpack4_c23_f2<0>(c23_align_rslt8_m128i, out);    // unpacks 12th 4 values


     __m128i c23_load_rslt10_m128i = _mm_loadu_si128(in + 9);
     __m128i c23_align_rslt9_m128i = _mm_alignr_epi8(c23_load_rslt10_m128i, c23_load_rslt9_m128i, 10);

     __horizontal_sse_unpack4_c23_f1<0>(c23_align_rslt9_m128i, out);    // unpacks 13th 4 values


     __m128i c23_load_rslt11_m128i = _mm_loadu_si128(in + 10);
     __m128i c23_align_rslt10_m128i = _mm_alignr_epi8(c23_load_rslt11_m128i , c23_load_rslt10_m128i, 5);

     __horizontal_sse_unpack4_c23_f2<0>(c23_align_rslt10_m128i, out);   // unpacks 14th 4 values

     __horizontal_sse_unpack4_c23_f1<1>(c23_load_rslt11_m128i, out);    // unpacks 15th 4 values


     __m128i c23_load_rslt12_m128i = _mm_loadu_si128(in + 11);
     __m128i c23_align_rslt11_m128i = _mm_alignr_epi8(c23_load_rslt12_m128i, c23_load_rslt11_m128i, 12);

     __horizontal_sse_unpack4_c23_f2<0>(c23_align_rslt11_m128i, out);   // unpacks 16th 4 values


     __m128i c23_load_rslt13_m128i = _mm_loadu_si128(in + 12);
     __m128i c23_align_rslt12_m128i = _mm_alignr_epi8(c23_load_rslt13_m128i, c23_load_rslt12_m128i, 8);

     __horizontal_sse_unpack4_c23_f1<0>(c23_align_rslt12_m128i, out);   // unpacks 17th 4 values

     __horizontal_sse_unpack4_c23_f2<3>(c23_load_rslt13_m128i, out);    // unpacks 18th 4 values


     __m128i c23_load_rslt14_m128i = _mm_loadu_si128(in + 13);
     __m128i c23_align_rslt13_m128i = _mm_alignr_epi8(c23_load_rslt14_m128i, c23_load_rslt13_m128i, 15);

     __horizontal_sse_unpack4_c23_f1<0>(c23_align_rslt13_m128i, out);   // unpacks 19th 4 values


     __m128i c23_load_rslt15_m128i = _mm_loadu_si128(in + 14);
     __m128i c23_align_rslt14_m128i = _mm_alignr_epi8(c23_load_rslt15_m128i, c23_load_rslt14_m128i, 10);

     __horizontal_sse_unpack4_c23_f2<0>(c23_align_rslt14_m128i, out);   // unpacks 20th 4 values


     __m128i c23_load_rslt16_m128i = _mm_loadu_si128(in + 15);
     __m128i c23_align_rslt15_m128i = _mm_alignr_epi8(c23_load_rslt16_m128i, c23_load_rslt15_m128i, 6);

     __horizontal_sse_unpack4_c23_f1<0>(c23_align_rslt15_m128i, out);   // unpacks 21st 4 values

     __horizontal_sse_unpack4_c23_f2<1>(c23_load_rslt16_m128i, out);    // unpacks 22nd 4 values


     __m128i c23_load_rslt17_m128i = _mm_loadu_si128(in + 16);
     __m128i c23_align_rslt16_m128i = _mm_alignr_epi8(c23_load_rslt17_m128i, c23_load_rslt16_m128i, 13);

     __horizontal_sse_unpack4_c23_f1<0>(c23_align_rslt16_m128i, out);   // unpacks 23rd 4 values


     __m128i c23_load_rslt18_m128i = _mm_loadu_si128(in + 17);
     __m128i c23_align_rslt17_m128i = _mm_alignr_epi8(c23_load_rslt18_m128i, c23_load_rslt17_m128i, 8);

     __horizontal_sse_unpack4_c23_f2<0>(c23_align_rslt17_m128i, out);   // unpacks 24th 4 values

     __horizontal_sse_unpack4_c23_f1<4>(c23_load_rslt18_m128i, out);    // unpacks 25th 4 values


     __m128i c23_load_rslt19_m128i = _mm_loadu_si128(in + 18);
     __m128i c23_align_rslt18_m128i = _mm_alignr_epi8(c23_load_rslt19_m128i, c23_load_rslt18_m128i, 15);

     __horizontal_sse_unpack4_c23_f2<0>(c23_align_rslt18_m128i, out);   // unpacks 26th 4 values


     __m128i c23_load_rslt20_m128i = _mm_loadu_si128(in + 19);
     __m128i c23_align_rslt19_m128i = _mm_alignr_epi8(c23_load_rslt20_m128i, c23_load_rslt19_m128i, 11);

     __horizontal_sse_unpack4_c23_f1<0>(c23_align_rslt19_m128i, out);   // unpacks 27th 4 values


     __m128i c23_load_rslt21_m128i = _mm_loadu_si128(in + 20);
     __m128i c23_align_rslt20_m128i = _mm_alignr_epi8(c23_load_rslt21_m128i, c23_load_rslt20_m128i, 6);

     __horizontal_sse_unpack4_c23_f2<0>(c23_align_rslt20_m128i, out);   // unpacks 28th 4 values

     __horizontal_sse_unpack4_c23_f1<2>(c23_load_rslt21_m128i, out);    // unpacks 29th 4 values


     __m128i c23_load_rslt22_m128i = _mm_loadu_si128(in + 21);
     __m128i c23_align_rslt21_m128i = _mm_alignr_epi8(c23_load_rslt22_m128i, c23_load_rslt21_m128i, 13);

     __horizontal_sse_unpack4_c23_f2<0>(c23_align_rslt21_m128i, out);   // unpacks 30th 4 values


     __m128i c23_load_rslt23_m128i = _mm_loadu_si128(in + 22);
     __m128i c23_align_rslt22_m128i = _mm_alignr_epi8(c23_load_rslt23_m128i, c23_load_rslt22_m128i, 9);

     __horizontal_sse_unpack4_c23_f1<0>(c23_align_rslt22_m128i, out);   // unpacks 31st 4 values

     __horizontal_sse_unpack4_c23_f2<4>(c23_load_rslt23_m128i, out);    // unpacks 32nd 4 values
}




// 24-bit
template <bool IsRiceCoding>
template <uint32_t byte>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c24(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c24_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 11, byte + 10, byte + 9,
				0xff, byte + 8, byte + 7, byte + 6,
				0xff, byte + 5, byte + 4, byte + 3,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c24_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c24_shfl_msk_m128i);
		_mm_storeu_si128(out++, c24_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c24_shfl_msk_m128i = _mm_set_epi8(
				0xff, byte + 11, byte + 10, byte + 9,
				0xff, byte + 8, byte + 7, byte + 6,
				0xff, byte + 5, byte + 4, byte + 3,
				0xff, byte + 2, byte + 1, byte + 0);

		__m128i c24_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c24_shfl_msk_m128i);
		__m128i c24_rslt_m128i = _mm_or_si128(c24_shfl_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 24));
		_mm_storeu_si128(out++, c24_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c24(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 16) {
		__m128i c24_load_rslt1_m128i = _mm_loadu_si128(in++);

		__horizontal_sse_unpack4_c24<0>(c24_load_rslt1_m128i, out);    // unpacks 1st 4 values


		__m128i c24_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c24_alignr_rslt1_m128i = _mm_alignr_epi8(c24_load_rslt2_m128i, c24_load_rslt1_m128i, 12);

		__horizontal_sse_unpack4_c24<0>(c24_alignr_rslt1_m128i, out);  // unpacks 2nd 4 values


		__m128i c24_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c24_alignr_rslt2_m128i = _mm_alignr_epi8(c24_load_rslt3_m128i, c24_load_rslt2_m128i, 8);

		__horizontal_sse_unpack4_c24<0>(c24_alignr_rslt2_m128i, out);  // unpacks 3rd 4 values


		__horizontal_sse_unpack4_c24<4>(c24_load_rslt3_m128i, out);    // unpacks 4th 4 values
	}
}




// 25-bit
template <bool IsRiceCoding>
template <uint32_t byte>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c25_f1(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c25_shfl_msk_m128i = _mm_set_epi8(
				byte + 12, byte + 11, byte + 10, byte + 9,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c25_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c25_shfl_msk_m128i);
		__m128i c25_mul_rslt_m128i = _mm_mullo_epi32(c25_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[25][0]);
		__m128i c25_srli_rslt_m128i = _mm_srli_epi32(c25_mul_rslt_m128i, 3);
		__m128i c25_rslt_m128i = _mm_and_si128(c25_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[25]);
		_mm_storeu_si128(out++, c25_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c25_shfl_msk_m128i = _mm_set_epi8(
				byte + 12, byte + 11, byte + 10, byte + 9,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c25_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c25_shfl_msk_m128i);
		__m128i c25_mul_rslt_m128i = _mm_mullo_epi32(c25_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[25][0]);
		__m128i c25_srli_rslt_m128i = _mm_srli_epi32(c25_mul_rslt_m128i, 3);
		__m128i c25_and_rslt_m128i = _mm_and_si128(c25_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[25]);
		__m128i c25_rslt_m128i = _mm_or_si128(c25_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 25));
		_mm_storeu_si128(out++, c25_rslt_m128i);
	}
}

template <bool IsRiceCoding>
template <uint32_t byte>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c25_f2(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c25_shfl_msk_m128i = _mm_set_epi8(
				byte + 12, byte + 11, byte + 10, byte + 9,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c25_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c25_shfl_msk_m128i);
		__m128i c25_mul_rslt_m128i = _mm_mullo_epi32(c25_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[25][1]);
		__m128i c25_srli_rslt_m128i = _mm_srli_epi32(c25_mul_rslt_m128i, 7);
		__m128i c25_rslt_m128i = _mm_and_si128(c25_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[25]);
		_mm_storeu_si128(out++, c25_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c25_shfl_msk_m128i = _mm_set_epi8(
				byte + 12, byte + 11, byte + 10, byte + 9,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c25_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c25_shfl_msk_m128i);
		__m128i c25_mul_rslt_m128i = _mm_mullo_epi32(c25_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[25][1]);
		__m128i c25_srli_rslt_m128i = _mm_srli_epi32(c25_mul_rslt_m128i, 7);
		__m128i c25_and_rslt_m128i = _mm_and_si128(c25_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[25]);
		__m128i c25_rslt_m128i = _mm_or_si128(c25_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 25));
		_mm_storeu_si128(out++, c25_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c25(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
    __m128i c25_load_rslt1_m128i = _mm_loadu_si128(in + 0);

    __horizontal_sse_unpack4_c25_f1<0>(c25_load_rslt1_m128i, out);      // unpacks 1st 4 values


    __m128i c25_load_rslt2_m128i = _mm_loadu_si128(in + 1);
    __m128i c25_align_rslt1_m128i = _mm_alignr_epi8(c25_load_rslt2_m128i, c25_load_rslt1_m128i, 12);

    __horizontal_sse_unpack4_c25_f2<0>(c25_align_rslt1_m128i, out);     // unpacks 2nd 4 values


    __m128i c25_load_rslt3_m128i = _mm_loadu_si128(in + 2);
    __m128i c25_align_rslt2_m128i = _mm_alignr_epi8(c25_load_rslt3_m128i, c25_load_rslt2_m128i, 9);

    __horizontal_sse_unpack4_c25_f1<0>(c25_align_rslt2_m128i, out);     // unpacks 3rd 4 values

    __m128i c25_load_rslt4_m128i = _mm_loadu_si128(in + 3);
    __m128i c25_align_rslt3_m128i = _mm_alignr_epi8(c25_load_rslt4_m128i, c25_load_rslt3_m128i, 5);

    __horizontal_sse_unpack4_c25_f2<0>(c25_align_rslt3_m128i, out);     // unpacks 4th 4 values

    __horizontal_sse_unpack4_c25_f1<2>(c25_load_rslt4_m128i, out);      // unpacks 5th 4 values


    __m128i c25_load_rslt5_m128i = _mm_loadu_si128(in + 4);
    __m128i c25_align_rslt4_m128i = _mm_alignr_epi8(c25_load_rslt5_m128i, c25_load_rslt4_m128i, 14);

    __horizontal_sse_unpack4_c25_f2<0>(c25_align_rslt4_m128i, out);     // unpacks 6th 4 values


    __m128i c25_load_rslt6_m128i = _mm_loadu_si128(in + 5);
    __m128i c25_align_rslt5_m128i = _mm_alignr_epi8(c25_load_rslt6_m128i, c25_load_rslt5_m128i, 11);

    __horizontal_sse_unpack4_c25_f1<0>(c25_align_rslt5_m128i, out);     // unpacks 7th 4 values


    __m128i c25_load_rslt7_m128i = _mm_loadu_si128(in + 6);
    __m128i c25_align_rslt6_m128i = _mm_alignr_epi8(c25_load_rslt7_m128i, c25_load_rslt6_m128i, 7);

    __horizontal_sse_unpack4_c25_f2<0>(c25_align_rslt6_m128i, out);     // unpacks 8th 4 values


    __m128i c25_load_rslt8_m128i = _mm_loadu_si128(in + 7);
    __m128i c25_align_rslt7_m128i = _mm_alignr_epi8(c25_load_rslt8_m128i, c25_load_rslt7_m128i, 4);

    __horizontal_sse_unpack4_c25_f1<0>(c25_align_rslt7_m128i, out);     // unpacks 9th 4 values

    __horizontal_sse_unpack4_c25_f2<0>(c25_load_rslt8_m128i, out);      // unpacks 10th 4 values


    __m128i c25_load_rslt9_m128i = _mm_loadu_si128(in + 8);
    __m128i c25_align_rslt8_m128i = _mm_alignr_epi8(c25_load_rslt9_m128i, c25_load_rslt8_m128i, 13);

    __horizontal_sse_unpack4_c25_f1<0>(c25_align_rslt8_m128i, out);     // unpacks 11th 4 values


    __m128i c25_load_rslt10_m128i = _mm_loadu_si128(in + 9);
    __m128i c25_align_rslt9_m128i = _mm_alignr_epi8(c25_load_rslt10_m128i, c25_load_rslt9_m128i, 9);

    __horizontal_sse_unpack4_c25_f2<0>(c25_align_rslt9_m128i, out);     // unpacks 12th 4 values


    __m128i c25_load_rslt11_m128i = _mm_loadu_si128(in + 10);
    __m128i c25_align_rslt10_m128i = _mm_alignr_epi8(c25_load_rslt11_m128i, c25_load_rslt10_m128i, 6);

    __horizontal_sse_unpack4_c25_f1<0>(c25_align_rslt10_m128i, out);    // unpacks 13th 4 values

    __horizontal_sse_unpack4_c25_f2<2>(c25_load_rslt11_m128i, out);     // unpacks 14th 4 values


    __m128i c25_load_rslt12_m128i = _mm_loadu_si128(in + 11);
    __m128i c25_align_rslt11_m128i = _mm_alignr_epi8(c25_load_rslt12_m128i, c25_load_rslt11_m128i, 15);

    __horizontal_sse_unpack4_c25_f1<0>(c25_align_rslt11_m128i, out);    // unpacks 15th 4 values


    __m128i c25_load_rslt13_m128i = _mm_loadu_si128(in + 12);
    __m128i c25_align_rslt12_m128i = _mm_alignr_epi8(c25_load_rslt13_m128i, c25_load_rslt12_m128i, 11);

    __horizontal_sse_unpack4_c25_f2<0>(c25_align_rslt12_m128i, out);    // unpacks 16th 4 values


    __m128i c25_load_rslt14_m128i = _mm_loadu_si128(in + 13);
    __m128i c25_align_rslt13_m128i = _mm_alignr_epi8(c25_load_rslt14_m128i, c25_load_rslt13_m128i, 8);

    __horizontal_sse_unpack4_c25_f1<0>(c25_align_rslt13_m128i, out);    // unpacks 17th 4 values


    __m128i c25_load_rslt15_m128i = _mm_loadu_si128(in + 14);
    __m128i c25_align_rslt14_m128i = _mm_alignr_epi8(c25_load_rslt15_m128i, c25_load_rslt14_m128i, 4);

    __horizontal_sse_unpack4_c25_f2<0>(c25_align_rslt14_m128i, out);    // unpacks 18th 4 values

    __horizontal_sse_unpack4_c25_f1<1>(c25_load_rslt15_m128i, out);     // unpacks 19th 4 values


    __m128i c25_load_rslt16_m128i = _mm_loadu_si128(in + 15);
    __m128i c25_align_rslt15_m128i = _mm_alignr_epi8(c25_load_rslt16_m128i, c25_load_rslt15_m128i, 13);

    __horizontal_sse_unpack4_c25_f2<0>(c25_align_rslt15_m128i, out);    // unpacks 20th 4 values


    __m128i c25_load_rslt17_m128i = _mm_loadu_si128(in + 16);
    __m128i c25_align_rslt16_m128i = _mm_alignr_epi8(c25_load_rslt17_m128i, c25_load_rslt16_m128i, 10);

    __horizontal_sse_unpack4_c25_f1<0>(c25_align_rslt16_m128i, out);    // unpacks 21st 4 values


    __m128i c25_load_rslt18_m128i = _mm_loadu_si128(in + 17);
    __m128i c25_align_rslt17_m128i = _mm_alignr_epi8(c25_load_rslt18_m128i, c25_load_rslt17_m128i, 6);

    __horizontal_sse_unpack4_c25_f2<0>(c25_align_rslt17_m128i, out);    // unpacks 22nd 4 values

    __horizontal_sse_unpack4_c25_f1<3>(c25_load_rslt18_m128i, out);     // unpacks 23rd 4 values


    __m128i c25_load_rslt19_m128i = _mm_loadu_si128(in + 18);
    __m128i c25_align_rslt18_m128i = _mm_alignr_epi8(c25_load_rslt19_m128i, c25_load_rslt18_m128i, 15);

    __horizontal_sse_unpack4_c25_f2<0>(c25_align_rslt18_m128i, out);    // unpacks 24th 4 values


    __m128i c25_load_rslt20_m128i = _mm_loadu_si128(in + 19);
    __m128i c25_align_rslt19_m128i = _mm_alignr_epi8(c25_load_rslt20_m128i, c25_load_rslt19_m128i, 12);

    __horizontal_sse_unpack4_c25_f1<0>(c25_align_rslt19_m128i, out);    // unpacks 25th 4 values


    __m128i c25_load_rslt21_m128i = _mm_loadu_si128(in + 20);
    __m128i c25_align_rslt20_m128i = _mm_alignr_epi8(c25_load_rslt21_m128i, c25_load_rslt20_m128i, 8);

    __horizontal_sse_unpack4_c25_f2<0>(c25_align_rslt20_m128i, out);    // unpacks 26th 4 values


    __m128i c25_load_rslt22_m128i = _mm_loadu_si128(in + 21);
    __m128i c25_align_rslt21_m128i = _mm_alignr_epi8(c25_load_rslt22_m128i, c25_load_rslt21_m128i, 5);

    __horizontal_sse_unpack4_c25_f1<0>(c25_align_rslt21_m128i, out);    // unpacks 27th 4 values

    __horizontal_sse_unpack4_c25_f2<1>(c25_load_rslt22_m128i, out);     // unpacks 28th 4 values


    __m128i c25_load_rslt23_m128i = _mm_loadu_si128(in + 22);
    __m128i c25_align_rslt22_m128i = _mm_alignr_epi8(c25_load_rslt23_m128i, c25_load_rslt22_m128i, 14);

    __horizontal_sse_unpack4_c25_f1<0>(c25_align_rslt22_m128i, out);    // unpacks 29th 4 values


    __m128i c25_load_rslt24_m128i = _mm_loadu_si128(in + 23);
    __m128i c25_align_rslt23_m128i = _mm_alignr_epi8(c25_load_rslt24_m128i, c25_load_rslt23_m128i, 10);

    __horizontal_sse_unpack4_c25_f2<0>(c25_align_rslt23_m128i, out);    // unpacks 30th 4 values


    __m128i c25_load_rslt25_m128i = _mm_loadu_si128(in + 24);
    __m128i c25_align_rslt24_m128i = _mm_alignr_epi8(c25_load_rslt25_m128i, c25_load_rslt24_m128i, 7);

    __horizontal_sse_unpack4_c25_f1<0>(c25_align_rslt24_m128i, out);    // unpacks 31st 4 values

    __horizontal_sse_unpack4_c25_f2<3>(c25_load_rslt25_m128i, out);     // unpacks 32nd 4 values
}




// 26-bit
template <bool IsRiceCoding>
template <uint32_t byte>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c26(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c26_shfl_msk_m128i = _mm_set_epi8(
				byte + 12, byte + 11, byte + 10, byte + 9,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c26_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c26_shfl_msk_m128i);
		__m128i c26_mul_rslt_m128i = _mm_mullo_epi32(c26_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[26][0]);
		__m128i c26_srli_rslt_m128i = _mm_srli_epi32(c26_mul_rslt_m128i, 6);
		__m128i c26_rslt_m128i = _mm_and_si128(c26_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[26]);
		_mm_storeu_si128(out++, c26_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c26_shfl_msk_m128i = _mm_set_epi8(
				byte + 12, byte + 11, byte + 10, byte + 9,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c26_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c26_shfl_msk_m128i);
		__m128i c26_mul_rslt_m128i = _mm_mullo_epi32(c26_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[26][0]);
		__m128i c26_srli_rslt_m128i = _mm_srli_epi32(c26_mul_rslt_m128i, 6);
		__m128i c26_and_rslt_m128i = _mm_and_si128(c26_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[26]);
		__m128i c26_rslt_m128i = _mm_or_si128(c26_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 26));
		_mm_storeu_si128(out++, c26_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c26(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 64) {
		__m128i c26_load_rslt1_m128i = _mm_loadu_si128(in++);

		__horizontal_sse_unpack4_c26<0>(c26_load_rslt1_m128i, out);     // unpacks 1st 4 values


		__m128i c26_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt1_m128i = _mm_alignr_epi8(c26_load_rslt2_m128i, c26_load_rslt1_m128i, 13);

		__horizontal_sse_unpack4_c26<0>(c26_alignr_rslt1_m128i, out);   // unpacks 2nd 4 values


		__m128i c26_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt2_m128i = _mm_alignr_epi8(c26_load_rslt3_m128i, c26_load_rslt2_m128i, 10);

		__horizontal_sse_unpack4_c26<0>(c26_alignr_rslt2_m128i, out);   // unpacks 3rd 4 values


		__m128i c26_load_rslt4_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt3_m128i = _mm_alignr_epi8(c26_load_rslt4_m128i, c26_load_rslt3_m128i, 7);

		__horizontal_sse_unpack4_c26<0>(c26_alignr_rslt3_m128i, out);   // unpacks 4th 4 values


		__m128i c26_load_rslt5_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt4_m128i = _mm_alignr_epi8(c26_load_rslt5_m128i, c26_load_rslt4_m128i, 4);

		__horizontal_sse_unpack4_c26<0>(c26_alignr_rslt4_m128i, out);   // unpacks 5th 4 values

		__horizontal_sse_unpack4_c26<1>(c26_load_rslt5_m128i, out);     // unpacks 6th 4 values


		__m128i c26_load_rslt6_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt5_m128i = _mm_alignr_epi8(c26_load_rslt6_m128i, c26_load_rslt5_m128i, 14);

		__horizontal_sse_unpack4_c26<0>(c26_alignr_rslt5_m128i, out);   // unpacks 7th 4 values


		__m128i c26_load_rslt7_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt6_m128i = _mm_alignr_epi8(c26_load_rslt7_m128i, c26_load_rslt6_m128i, 11);

		__horizontal_sse_unpack4_c26<0>(c26_alignr_rslt6_m128i, out);   // unpacks 8th 4 values


		__m128i c26_load_rslt8_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt7_m128i = _mm_alignr_epi8(c26_load_rslt8_m128i, c26_load_rslt7_m128i, 8);

		__horizontal_sse_unpack4_c26<0>(c26_alignr_rslt7_m128i, out);   // unpacks 9th 4 values


		__m128i c26_load_rslt9_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt8_m128i = _mm_alignr_epi8(c26_load_rslt9_m128i, c26_load_rslt8_m128i, 5);

		__horizontal_sse_unpack4_c26<0>(c26_alignr_rslt8_m128i, out);   // unpacks 10th 4 values

		__horizontal_sse_unpack4_c26<2>(c26_load_rslt9_m128i, out);     // unpacks 11th 4 values


		__m128i c26_load_rslt10_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt9_m128i = _mm_alignr_epi8(c26_load_rslt10_m128i, c26_load_rslt9_m128i, 15);

		__horizontal_sse_unpack4_c26<0>(c26_alignr_rslt9_m128i, out);   // unpacks 12th 4 values


		__m128i c26_load_rslt11_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt10_m128i = _mm_alignr_epi8(c26_load_rslt11_m128i, c26_load_rslt10_m128i, 12);

		__horizontal_sse_unpack4_c26<0>(c26_alignr_rslt10_m128i, out);  // unpacks 13th 4 values


		__m128i c26_load_rslt12_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt11_m128i = _mm_alignr_epi8(c26_load_rslt12_m128i, c26_load_rslt11_m128i, 9);

		__horizontal_sse_unpack4_c26<0>(c26_alignr_rslt11_m128i, out);  // unpacks 14th 4 values


		__m128i c26_load_rslt13_m128i = _mm_loadu_si128(in++);
		__m128i c26_alignr_rslt12_m128i = _mm_alignr_epi8(c26_load_rslt13_m128i, c26_load_rslt12_m128i, 6);

		__horizontal_sse_unpack4_c26<0>(c26_alignr_rslt12_m128i, out);  // unpacks 15th 4 values

		__horizontal_sse_unpack4_c26<3>(c26_load_rslt13_m128i, out);    // unpacks 16th 4 values
	}
}




// 27-bit
template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c27_f1(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		__m128i Horizontal_SSE_c27_shfl_msk_m128i = _mm_set_epi8(
				byte + 13, byte + 12, byte + 11, byte + 10,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c27_shfl_rslt_m128i  = _mm_shuffle_epi8(InReg, Horizontal_SSE_c27_shfl_msk_m128i);
		__m128i c27_srli_rslt1_m128i  =  _mm_srli_epi64(c27_shfl_rslt_m128i, 6);

		const int Horizontal_SSE_c27_blend_msk_imm = 0x30;
		__m128i c27_blend_rslt_m128i = _mm_blend_epi16 (c27_shfl_rslt_m128i, c27_srli_rslt1_m128i, Horizontal_SSE_c27_blend_msk_imm);

		__m128i c27_mul_rslt_m128i   = _mm_mullo_epi32(c27_blend_rslt_m128i, Horizontal_SSE_mul_msk_m128i[27][0]);
		__m128i c27_srli_rslt2_m128i = _mm_srli_epi32(c27_mul_rslt_m128i, 3);
		__m128i c27_rslt_m128i = _mm_and_si128(c27_srli_rslt2_m128i, Horizontal_SSE_and_msk_m128i[27]);
		_mm_storeu_si128(out++, c27_rslt_m128i);
	}
	else { // Rice Coding
		__m128i Horizontal_SSE_c27_shfl_msk_m128i = _mm_set_epi8(
				byte + 13, byte + 12, byte + 11, byte + 10,
				byte + 9, byte + 8, byte + 7, byte + 6,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c27_shfl_rslt_m128i  = _mm_shuffle_epi8(InReg, Horizontal_SSE_c27_shfl_msk_m128i);
		__m128i c27_srli_rslt1_m128i  =  _mm_srli_epi64(c27_shfl_rslt_m128i, 6);

		const int Horizontal_SSE_c27_blend_msk_imm = 0x30;
		__m128i c27_blend_rslt_m128i = _mm_blend_epi16 (c27_shfl_rslt_m128i, c27_srli_rslt1_m128i, Horizontal_SSE_c27_blend_msk_imm);

		__m128i c27_mul_rslt_m128i   = _mm_mullo_epi32(c27_blend_rslt_m128i, Horizontal_SSE_mul_msk_m128i[27][0]);
		__m128i c27_srli_rslt2_m128i = _mm_srli_epi32(c27_mul_rslt_m128i, 3);
		__m128i c27_and_rslt_m128i = _mm_and_si128(c27_srli_rslt2_m128i, Horizontal_SSE_and_msk_m128i[27]);
		__m128i c27_rslt_m128i = _mm_or_si128(c27_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 27));
		_mm_storeu_si128(out++, c27_rslt_m128i);
	}
}

template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c27_f2(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		__m128i Horizontal_SSE_c27_shfl_msk_m128i = _mm_set_epi8(
				byte + 13, byte + 12, byte + 11, byte + 10,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c27_shfl_rslt_m128i  = _mm_shuffle_epi8(InReg, Horizontal_SSE_c27_shfl_msk_m128i);
		__m128i c27_slli_rslt_m128i  =  _mm_slli_epi64(c27_shfl_rslt_m128i, 1);

		const int Horizontal_SSE_c27_blend_msk_imm = 0x0C;
		__m128i c27_blend_rslt_m128i = _mm_blend_epi16 (c27_shfl_rslt_m128i, c27_slli_rslt_m128i, Horizontal_SSE_c27_blend_msk_imm);

		__m128i c27_mul_rslt_m128i   = _mm_mullo_epi32(c27_blend_rslt_m128i, Horizontal_SSE_mul_msk_m128i[27][1]);
		__m128i c27_srli_rslt_m128i = _mm_srli_epi32(c27_mul_rslt_m128i, 5);
		__m128i c27_rslt_m128i = _mm_and_si128(c27_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[27]);
		_mm_storeu_si128(out++, c27_rslt_m128i);
	}
	else {
		__m128i Horizontal_SSE_c27_shfl_msk_m128i = _mm_set_epi8(
				byte + 13, byte + 12, byte + 11, byte + 10,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c27_shfl_rslt_m128i  = _mm_shuffle_epi8(InReg, Horizontal_SSE_c27_shfl_msk_m128i);
		__m128i c27_slli_rslt_m128i  =  _mm_slli_epi64(c27_shfl_rslt_m128i, 1);

		const int Horizontal_SSE_c27_blend_msk_imm = 0x0C;
		__m128i c27_blend_rslt_m128i = _mm_blend_epi16 (c27_shfl_rslt_m128i, c27_slli_rslt_m128i, Horizontal_SSE_c27_blend_msk_imm);

		__m128i c27_mul_rslt_m128i   = _mm_mullo_epi32(c27_blend_rslt_m128i, Horizontal_SSE_mul_msk_m128i[27][1]);
		__m128i c27_srli_rslt_m128i = _mm_srli_epi32(c27_mul_rslt_m128i, 5);
		__m128i c27_and_rslt_m128i = _mm_and_si128(c27_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[27]);
		__m128i c27_rslt_m128i = _mm_or_si128(c27_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 27));
		_mm_storeu_si128(out++, c27_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c27(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	__m128i c27_load_rslt1_m128i = _mm_loadu_si128(in + 0);

    __horizontal_sse_unpack4_c27_f1<0>(c27_load_rslt1_m128i, out);      // 1st 4 values


    __m128i c27_load_rslt2_m128i = _mm_loadu_si128(in + 1);
    __m128i c27_align_rslt1_m128i = _mm_alignr_epi8(c27_load_rslt2_m128i, c27_load_rslt1_m128i, 13);

    __horizontal_sse_unpack4_c27_f2<0>(c27_align_rslt1_m128i, out);     // 2nd 4 values


    __m128i c27_load_rslt3_m128i = _mm_loadu_si128(in + 2);
    __m128i c27_align_rslt2_m128i = _mm_alignr_epi8(c27_load_rslt3_m128i, c27_load_rslt2_m128i, 11);

    __horizontal_sse_unpack4_c27_f1<0>(c27_align_rslt2_m128i, out);     // 3rd 4 values


    __m128i c27_load_rslt4_m128i = _mm_loadu_si128(in + 3);
    __m128i c27_align_rslt3_m128i = _mm_alignr_epi8(c27_load_rslt4_m128i, c27_load_rslt3_m128i, 8);

    __horizontal_sse_unpack4_c27_f2<0>(c27_align_rslt3_m128i, out);     // 4th 4 values


    __m128i c27_load_rslt5_m128i = _mm_loadu_si128(in + 4);
    __m128i c27_align_rslt4_m128i = _mm_alignr_epi8(c27_load_rslt5_m128i, c27_load_rslt4_m128i, 6);

    __horizontal_sse_unpack4_c27_f1<0>(c27_align_rslt4_m128i, out);     // 5th 4 values


    __m128i c27_load_rslt6_m128i = _mm_loadu_si128(in + 5);
    __m128i c27_align_rslt5_m128i = _mm_alignr_epi8(c27_load_rslt6_m128i, c27_load_rslt5_m128i, 3);

    __horizontal_sse_unpack4_c27_f2<0>(c27_align_rslt5_m128i, out);     // 6th 4 values

    __horizontal_sse_unpack4_c27_f1<1>(c27_load_rslt6_m128i, out);      // 7th 4 values


    __m128i c27_load_rslt7_m128i = _mm_loadu_si128(in + 6);
    __m128i c27_align_rslt6_m128i = _mm_alignr_epi8(c27_load_rslt7_m128i, c27_load_rslt6_m128i, 14);

    __horizontal_sse_unpack4_c27_f2<0>(c27_align_rslt6_m128i, out);     // 8th 4 values


    __m128i c27_load_rslt8_m128i = _mm_loadu_si128(in + 7);
    __m128i c27_align_rslt7_m128i = _mm_alignr_epi8(c27_load_rslt8_m128i, c27_load_rslt7_m128i, 12);

    __horizontal_sse_unpack4_c27_f1<0>(c27_align_rslt7_m128i, out);     // 9th 4 values


    __m128i c27_load_rslt9_m128i = _mm_loadu_si128(in + 8);
    __m128i c27_align_rslt8_m128i = _mm_alignr_epi8(c27_load_rslt9_m128i, c27_load_rslt8_m128i, 9);

    __horizontal_sse_unpack4_c27_f2<0>(c27_align_rslt8_m128i, out);     // 10th 4 values


    __m128i c27_load_rslt10_m128i = _mm_loadu_si128(in + 9);
    __m128i c27_align_rslt9_m128i = _mm_alignr_epi8(c27_load_rslt10_m128i, c27_load_rslt9_m128i, 7);

    __horizontal_sse_unpack4_c27_f1<0>(c27_align_rslt9_m128i, out);     // 11th 4 values


    __m128i c27_load_rslt11_m128i = _mm_loadu_si128(in + 10);
    __m128i c27_align_rslt10_m128i = _mm_alignr_epi8(c27_load_rslt11_m128i, c27_load_rslt10_m128i, 4);

    __horizontal_sse_unpack4_c27_f2<0>(c27_align_rslt10_m128i, out);    // 12th 4 values

    __horizontal_sse_unpack4_c27_f1<2>(c27_load_rslt11_m128i, out);     // 13th 4 values


    __m128i c27_load_rslt12_m128i = _mm_loadu_si128(in + 11);
    __m128i c27_align_rslt11_m128i = _mm_alignr_epi8(c27_load_rslt12_m128i, c27_load_rslt11_m128i, 15);

    __horizontal_sse_unpack4_c27_f2<0>(c27_align_rslt11_m128i, out);    // 14th 4 values


    __m128i c27_load_rslt13_m128i = _mm_loadu_si128(in + 12);
    __m128i c27_align_rslt12_m128i = _mm_alignr_epi8(c27_load_rslt13_m128i, c27_load_rslt12_m128i, 13);

    __horizontal_sse_unpack4_c27_f1<0>(c27_align_rslt12_m128i, out);    // 15th 4 values


    __m128i c27_load_rslt14_m128i = _mm_loadu_si128(in + 13);
    __m128i c27_align_rslt13_m128i = _mm_alignr_epi8(c27_load_rslt14_m128i, c27_load_rslt13_m128i, 10);

    __horizontal_sse_unpack4_c27_f2<0>(c27_align_rslt13_m128i, out);    // 16th 4 values


    __m128i c27_load_rslt15_m128i = _mm_loadu_si128(in + 14);
    __m128i c27_align_rslt14_m128i = _mm_alignr_epi8(c27_load_rslt15_m128i, c27_load_rslt14_m128i, 8);

    __horizontal_sse_unpack4_c27_f1<0>(c27_align_rslt14_m128i, out);    // 17th 4 values


    __m128i c27_load_rslt16_m128i = _mm_loadu_si128(in + 15);
    __m128i c27_align_rslt15_m128i = _mm_alignr_epi8(c27_load_rslt16_m128i, c27_load_rslt15_m128i, 5);

    __horizontal_sse_unpack4_c27_f2<0>(c27_align_rslt15_m128i, out);    // 18th 4 values


    __m128i c27_load_rslt17_m128i = _mm_loadu_si128(in + 16);
    __m128i c27_align_rslt16_m128i = _mm_alignr_epi8(c27_load_rslt17_m128i, c27_load_rslt16_m128i, 3);

    __horizontal_sse_unpack4_c27_f1<0>(c27_align_rslt16_m128i, out);    // 19th 4 values

    __horizontal_sse_unpack4_c27_f2<0>(c27_load_rslt17_m128i, out);     // 20th 4 values


    __m128i c27_load_rslt18_m128i = _mm_loadu_si128(in + 17);
    __m128i c27_align_rslt17_m128i = _mm_alignr_epi8(c27_load_rslt18_m128i, c27_load_rslt17_m128i, 14);

    __horizontal_sse_unpack4_c27_f1<0>(c27_align_rslt17_m128i, out);    // 21st 4 values


    __m128i c27_load_rslt19_m128i = _mm_loadu_si128(in + 18);
    __m128i c27_align_rslt18_m128i = _mm_alignr_epi8(c27_load_rslt19_m128i, c27_load_rslt18_m128i, 11);

    __horizontal_sse_unpack4_c27_f2<0>(c27_align_rslt18_m128i, out);   // 22nd 4 values


    __m128i c27_load_rslt20_m128i = _mm_loadu_si128(in + 19);
    __m128i c27_align_rslt19_m128i = _mm_alignr_epi8(c27_load_rslt20_m128i, c27_load_rslt19_m128i, 9);

    __horizontal_sse_unpack4_c27_f1<0>(c27_align_rslt19_m128i, out);    // 23th 4 values


    __m128i c27_load_rslt21_m128i = _mm_loadu_si128(in + 20);
    __m128i c27_align_rslt20_m128i = _mm_alignr_epi8(c27_load_rslt21_m128i, c27_load_rslt20_m128i, 6);

    __horizontal_sse_unpack4_c27_f2<0>(c27_align_rslt20_m128i, out);    // 24th 4 values


    __m128i c27_load_rslt22_m128i = _mm_loadu_si128(in + 21);
    __m128i c27_align_rslt21_m128i = _mm_alignr_epi8(c27_load_rslt22_m128i, c27_load_rslt21_m128i, 4);

    __horizontal_sse_unpack4_c27_f1<0>(c27_align_rslt21_m128i, out);    // 25th 4 values

    __horizontal_sse_unpack4_c27_f2<1>(c27_load_rslt22_m128i, out);     // 26th 4 values


    __m128i c27_load_rslt23_m128i = _mm_loadu_si128(in + 22);
    __m128i c27_align_rslt22_m128i = _mm_alignr_epi8(c27_load_rslt23_m128i, c27_load_rslt22_m128i, 15);

    __horizontal_sse_unpack4_c27_f1<0>(c27_align_rslt22_m128i, out);    // 27th 4 values


    __m128i c27_load_rslt24_m128i = _mm_loadu_si128(in + 23);
    __m128i c27_align_rslt23_m128i = _mm_alignr_epi8(c27_load_rslt24_m128i, c27_load_rslt23_m128i, 12);

    __horizontal_sse_unpack4_c27_f2<0>(c27_align_rslt23_m128i, out);    // 28th 4 values


    __m128i c27_load_rslt25_m128i = _mm_loadu_si128(in + 24);
    __m128i c27_align_rslt24_m128i = _mm_alignr_epi8(c27_load_rslt25_m128i, c27_load_rslt24_m128i, 10);

    __horizontal_sse_unpack4_c27_f1<0>(c27_align_rslt24_m128i, out);    // 29th 4 values


    __m128i c27_load_rslt26_m128i = _mm_loadu_si128(in + 25);
    __m128i c27_align_rslt25_m128i = _mm_alignr_epi8(c27_load_rslt26_m128i, c27_load_rslt25_m128i, 7);

    __horizontal_sse_unpack4_c27_f2<0>(c27_align_rslt25_m128i, out);    // 30th 4 values


    __m128i c27_load_rslt27_m128i = _mm_loadu_si128(in + 26);
    __m128i c27_align_rslt26_m128i = _mm_alignr_epi8(c27_load_rslt27_m128i, c27_load_rslt26_m128i, 5);

    __horizontal_sse_unpack4_c27_f1<0>(c27_align_rslt26_m128i, out);    // 31st 4 values

    __horizontal_sse_unpack4_c27_f2<2>(c27_load_rslt27_m128i, out);     // 32nd 4 values
}




// 28-bit
template <bool IsRiceCoding>
template <uint32_t byte>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c28(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c28_shfl_msk_m128i = _mm_set_epi8(
				byte + 13, byte + 12, byte + 11, byte + 10,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c28_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c28_shfl_msk_m128i);
		__m128i c28_mul_rslt_m128i = _mm_mullo_epi32(c28_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[28][0]);
		__m128i c28_srli_rslt_m128i = _mm_srli_epi32(c28_mul_rslt_m128i, 4);
		__m128i c28_rslt_m128i = _mm_and_si128(c28_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[28]);
		_mm_storeu_si128(out++, c28_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c28_shfl_msk_m128i = _mm_set_epi8(
				byte + 13, byte + 12, byte + 11, byte + 10,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 6, byte + 5, byte + 4, byte + 3,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c28_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c28_shfl_msk_m128i);
		__m128i c28_mul_rslt_m128i = _mm_mullo_epi32(c28_shfl_rslt_m128i, Horizontal_SSE_mul_msk_m128i[28][0]);
		__m128i c28_srli_rslt_m128i = _mm_srli_epi32(c28_mul_rslt_m128i, 4);
		__m128i c28_and_rslt_m128i = _mm_and_si128(c28_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[28]);
		__m128i c28_rslt_m128i = _mm_or_si128(c28_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 28));
		_mm_storeu_si128(out++, c28_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c28(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 32) {
		__m128i c28_load_rslt1_m128i = _mm_loadu_si128(in++);

		__horizontal_sse_unpack4_c28<0>(c28_load_rslt1_m128i, out);     // unpacks 1st 4 values


		__m128i c28_load_rslt2_m128i = _mm_loadu_si128(in++);
		__m128i c28_alignr_rslt1_m128i = _mm_alignr_epi8(c28_load_rslt2_m128i, c28_load_rslt1_m128i, 14);

		__horizontal_sse_unpack4_c28<0>(c28_alignr_rslt1_m128i, out);   // unpacks 2nd 4 values


		__m128i c28_load_rslt3_m128i = _mm_loadu_si128(in++);
		__m128i c28_alignr_rslt2_m128i = _mm_alignr_epi8(c28_load_rslt3_m128i, c28_load_rslt2_m128i, 12);

		__horizontal_sse_unpack4_c28<0>(c28_alignr_rslt2_m128i, out);   // unpacks 3rd 4 values


		__m128i c28_load_rslt4_m128i = _mm_loadu_si128(in++);
		__m128i c28_alignr_rslt3_m128i = _mm_alignr_epi8(c28_load_rslt4_m128i, c28_load_rslt3_m128i, 10);

		__horizontal_sse_unpack4_c28<0>(c28_alignr_rslt3_m128i, out);   // unpacks 4th 4 values


		__m128i c28_load_rslt5_m128i = _mm_loadu_si128(in++);
		__m128i c28_alignr_rslt4_m128i = _mm_alignr_epi8(c28_load_rslt5_m128i, c28_load_rslt4_m128i, 8);

		__horizontal_sse_unpack4_c28<0>(c28_alignr_rslt4_m128i, out);   // unpacks 5th 4 values


		__m128i c28_load_rslt6_m128i = _mm_loadu_si128(in++);
		__m128i c28_alignr_rslt5_m128i = _mm_alignr_epi8(c28_load_rslt6_m128i, c28_load_rslt5_m128i, 6);

		__horizontal_sse_unpack4_c28<0>(c28_alignr_rslt5_m128i, out);   // unpacks 6th 4 values


		__m128i c28_load_rslt7_m128i = _mm_loadu_si128(in++);
		__m128i c28_alignr_rslt6_m128i = _mm_alignr_epi8(c28_load_rslt7_m128i, c28_load_rslt6_m128i, 4);

		__horizontal_sse_unpack4_c28<0>(c28_alignr_rslt6_m128i, out);   // unpacks 7th 4 values

		__horizontal_sse_unpack4_c28<2>(c28_load_rslt7_m128i, out);     // unpacks 8th 4 values
	}
}




// 29-bit
template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c29_f1(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c29_shfl_msk_m128i = _mm_set_epi8(
				byte + 14, byte + 13, byte + 12, byte + 11,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c29_shfl_rslt_m128i  = _mm_shuffle_epi8(InReg, Horizontal_SSE_c29_shfl_msk_m128i);
		__m128i c29_slli_rslt_m128i  =  _mm_slli_epi64(c29_shfl_rslt_m128i, 3);

		const int Horizontal_SSE_c29_blend_msk_imm = 0xcc;
		__m128i c29_blend_rslt_m128i = _mm_blend_epi16 (c29_shfl_rslt_m128i, c29_slli_rslt_m128i, Horizontal_SSE_c29_blend_msk_imm);

		__m128i c29_mul_rslt_m128i   = _mm_mullo_epi32(c29_blend_rslt_m128i, Horizontal_SSE_mul_msk_m128i[29][0]);
		__m128i c29_srli_rslt_m128i = _mm_srli_epi32(c29_mul_rslt_m128i, 2);
		__m128i c29_rslt_m128i = _mm_and_si128(c29_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[29]);
		_mm_storeu_si128(out++, c29_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c29_shfl_msk_m128i = _mm_set_epi8(
				byte + 14, byte + 13, byte + 12, byte + 11,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c29_shfl_rslt_m128i  = _mm_shuffle_epi8(InReg, Horizontal_SSE_c29_shfl_msk_m128i);
		__m128i c29_slli_rslt_m128i  =  _mm_slli_epi64(c29_shfl_rslt_m128i, 3);

		const int Horizontal_SSE_c29_blend_msk_imm = 0xcc;
		__m128i c29_blend_rslt_m128i = _mm_blend_epi16 (c29_shfl_rslt_m128i, c29_slli_rslt_m128i, Horizontal_SSE_c29_blend_msk_imm);

		__m128i c29_mul_rslt_m128i   = _mm_mullo_epi32(c29_blend_rslt_m128i, Horizontal_SSE_mul_msk_m128i[29][0]);
		__m128i c29_srli_rslt_m128i = _mm_srli_epi32(c29_mul_rslt_m128i, 2);
		__m128i c29_and_rslt_m128i = _mm_and_si128(c29_srli_rslt_m128i, Horizontal_SSE_and_msk_m128i[29]);
		__m128i c29_rslt_m128i = _mm_or_si128(c29_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 29));
		_mm_storeu_si128(out++, c29_rslt_m128i);
	}
}

template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c29_f2(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// unpacks 4 values
		__m128i Horizontal_SSE_c29_shfl_msk_m128i = _mm_set_epi8(
				byte + 14, byte + 13, byte + 12, byte + 11,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c29_shfl_rslt_m128i  = _mm_shuffle_epi8(InReg, Horizontal_SSE_c29_shfl_msk_m128i);
		__m128i c29_srli_rslt1_m128i  =  _mm_srli_epi64(c29_shfl_rslt_m128i, 3);

		const int Horizontal_SSE_c29_blend_msk_imm = 0x33;
		__m128i c29_blend_rslt_m128i = _mm_blend_epi16(c29_shfl_rslt_m128i, c29_srli_rslt1_m128i, Horizontal_SSE_c29_blend_msk_imm);

		__m128i c29_mul_rslt_m128i   = _mm_mullo_epi32(c29_blend_rslt_m128i, Horizontal_SSE_mul_msk_m128i[29][1]);
		__m128i c29_srli_rslt2_m128i = _mm_srli_epi32(c29_mul_rslt_m128i, 3);
		__m128i c29_rslt_m128i = _mm_and_si128(c29_srli_rslt2_m128i, Horizontal_SSE_and_msk_m128i[29]);
		_mm_storeu_si128(out++, c29_rslt_m128i);
	}
	else { // Rice Coding
		// unpacks 4 values
		__m128i Horizontal_SSE_c29_shfl_msk_m128i = _mm_set_epi8(
				byte + 14, byte + 13, byte + 12, byte + 11,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c29_shfl_rslt_m128i  = _mm_shuffle_epi8(InReg, Horizontal_SSE_c29_shfl_msk_m128i);
		__m128i c29_srli_rslt1_m128i  =  _mm_srli_epi64(c29_shfl_rslt_m128i, 3);

		const int Horizontal_SSE_c29_blend_msk_imm = 0x33;
		__m128i c29_blend_rslt_m128i = _mm_blend_epi16(c29_shfl_rslt_m128i, c29_srli_rslt1_m128i, Horizontal_SSE_c29_blend_msk_imm);

		__m128i c29_mul_rslt_m128i   = _mm_mullo_epi32(c29_blend_rslt_m128i, Horizontal_SSE_mul_msk_m128i[29][1]);
		__m128i c29_srli_rslt2_m128i = _mm_srli_epi32(c29_mul_rslt_m128i, 3);
		__m128i c29_and_rslt_m128i = _mm_and_si128(c29_srli_rslt2_m128i, Horizontal_SSE_and_msk_m128i[29]);
		__m128i c29_rslt_m128i = _mm_or_si128(c29_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 29));
		_mm_storeu_si128(out++, c29_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c29(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
     __m128i c29_load_rslt1_m128i = _mm_loadu_si128(in + 0);

     __horizontal_sse_unpack4_c29_f1<0>(c29_load_rslt1_m128i, out);     // 1st 4 values


     __m128i c29_load_rslt2_m128i = _mm_loadu_si128(in + 1);
     __m128i c29_align_rslt1_m128i = _mm_alignr_epi8(c29_load_rslt2_m128i, c29_load_rslt1_m128i, 14);

     __horizontal_sse_unpack4_c29_f2<0>(c29_align_rslt1_m128i, out);    // 2nd 4 values


     __m128i c29_load_rslt3_m128i = _mm_loadu_si128(in + 2);
     __m128i c29_align_rslt2_m128i = _mm_alignr_epi8(c29_load_rslt3_m128i, c29_load_rslt2_m128i, 13);

     __horizontal_sse_unpack4_c29_f1<0>(c29_align_rslt2_m128i, out);    // 3rd 4 values


     __m128i c29_load_rslt4_m128i = _mm_loadu_si128(in + 3);
     __m128i c29_align_rslt3_m128i = _mm_alignr_epi8(c29_load_rslt4_m128i, c29_load_rslt3_m128i, 11);

     __horizontal_sse_unpack4_c29_f2<0>(c29_align_rslt3_m128i, out);    // 4th 4 values


     __m128i c29_load_rslt5_m128i = _mm_loadu_si128(in + 4);
     __m128i c29_align_rslt4_m128i = _mm_alignr_epi8(c29_load_rslt5_m128i, c29_load_rslt4_m128i, 10);

     __horizontal_sse_unpack4_c29_f1<0>(c29_align_rslt4_m128i, out);    // 5th 4 values


     __m128i c29_load_rslt6_m128i = _mm_loadu_si128(in + 5);
     __m128i c29_align_rslt5_m128i = _mm_alignr_epi8(c29_load_rslt6_m128i, c29_load_rslt5_m128i, 8);

     __horizontal_sse_unpack4_c29_f2<0>(c29_align_rslt5_m128i, out);    // 6th 4 values


     __m128i c29_load_rslt7_m128i = _mm_loadu_si128(in + 6);
     __m128i c29_align_rslt6_m128i = _mm_alignr_epi8(c29_load_rslt7_m128i, c29_load_rslt6_m128i, 7);

     __horizontal_sse_unpack4_c29_f1<0>(c29_align_rslt6_m128i, out);    // 7th 4 values


     __m128i c29_load_rslt8_m128i = _mm_loadu_si128(in + 7);
     __m128i c29_align_rslt7_m128i = _mm_alignr_epi8(c29_load_rslt8_m128i, c29_load_rslt7_m128i, 5);

     __horizontal_sse_unpack4_c29_f2<0>(c29_align_rslt7_m128i, out);    // 8th 4 values


     __m128i c29_load_rslt9_m128i = _mm_loadu_si128(in + 8);
     __m128i c29_align_rslt8_m128i = _mm_alignr_epi8(c29_load_rslt9_m128i, c29_load_rslt8_m128i, 4);

     __horizontal_sse_unpack4_c29_f1<0>(c29_align_rslt8_m128i, out);    // 9th 4 values


     __m128i c29_load_rslt10_m128i = _mm_loadu_si128(in + 9);
     __m128i c29_align_rslt9_m128i = _mm_alignr_epi8(c29_load_rslt10_m128i, c29_load_rslt9_m128i, 2);

     __horizontal_sse_unpack4_c29_f2<0>(c29_align_rslt9_m128i, out);    // 10th 4 values

     __horizontal_sse_unpack4_c29_f1<1>(c29_load_rslt10_m128i, out);    // 11th 4 values


     __m128i c29_load_rslt11_m128i = _mm_loadu_si128(in + 10);
     __m128i c29_align_rslt10_m128i = _mm_alignr_epi8(c29_load_rslt11_m128i, c29_load_rslt10_m128i, 15);

     __horizontal_sse_unpack4_c29_f2<0>(c29_align_rslt10_m128i, out);   // 12th 4 values


     __m128i c29_load_rslt12_m128i = _mm_loadu_si128(in + 11);
     __m128i c29_align_rslt11_m128i = _mm_alignr_epi8(c29_load_rslt12_m128i, c29_load_rslt11_m128i, 14);

     __horizontal_sse_unpack4_c29_f1<0>(c29_align_rslt11_m128i, out);   // 13th 4 values


     __m128i c29_load_rslt13_m128i = _mm_loadu_si128(in + 12);
     __m128i c29_align_rslt12_m128i = _mm_alignr_epi8(c29_load_rslt13_m128i, c29_load_rslt12_m128i, 12);

     __horizontal_sse_unpack4_c29_f2<0>(c29_align_rslt12_m128i, out);   // 14th 4 values


     __m128i c29_load_rslt14_m128i = _mm_loadu_si128(in + 13);
     __m128i c29_align_rslt13_m128i = _mm_alignr_epi8(c29_load_rslt14_m128i, c29_load_rslt13_m128i, 11);

     __horizontal_sse_unpack4_c29_f1<0>(c29_align_rslt13_m128i, out);   // 15th 4 values


     __m128i c29_load_rslt15_m128i = _mm_loadu_si128(in + 14);
     __m128i c29_align_rslt14_m128i = _mm_alignr_epi8(c29_load_rslt15_m128i, c29_load_rslt14_m128i, 9);

     __horizontal_sse_unpack4_c29_f2<0>(c29_align_rslt14_m128i, out);   // 16th 4 values


     __m128i c29_load_rslt16_m128i = _mm_loadu_si128(in + 15);
     __m128i c29_align_rslt15_m128i = _mm_alignr_epi8(c29_load_rslt16_m128i, c29_load_rslt15_m128i, 8);

     __horizontal_sse_unpack4_c29_f1<0>(c29_align_rslt15_m128i, out);   // 17th 4 values


     __m128i c29_load_rslt17_m128i = _mm_loadu_si128(in + 16);
     __m128i c29_align_rslt16_m128i = _mm_alignr_epi8(c29_load_rslt17_m128i, c29_load_rslt16_m128i, 6);

     __horizontal_sse_unpack4_c29_f2<0>(c29_align_rslt16_m128i, out);   // 18th 4 values


     __m128i c29_load_rslt18_m128i = _mm_loadu_si128(in + 17);
     __m128i c29_align_rslt17_m128i = _mm_alignr_epi8(c29_load_rslt18_m128i, c29_load_rslt17_m128i, 5);

     __horizontal_sse_unpack4_c29_f1<0>(c29_align_rslt17_m128i, out);   // 19th 4 values


     __m128i c29_load_rslt19_m128i = _mm_loadu_si128(in + 18);
     __m128i c29_align_rslt18_m128i = _mm_alignr_epi8(c29_load_rslt19_m128i, c29_load_rslt18_m128i, 3);

     __horizontal_sse_unpack4_c29_f2<0>(c29_align_rslt18_m128i, out);   // 20th 4 values


     __m128i c29_load_rslt20_m128i = _mm_loadu_si128(in + 19);
     __m128i c29_align_rslt19_m128i = _mm_alignr_epi8(c29_load_rslt20_m128i, c29_load_rslt19_m128i, 2);

     __horizontal_sse_unpack4_c29_f1<0>(c29_align_rslt19_m128i, out);   // 21st 4 values

     __horizontal_sse_unpack4_c29_f2<0>(c29_load_rslt20_m128i, out);    // 22nd 4 values


     __m128i c29_load_rslt21_m128i = _mm_loadu_si128(in + 20);
     __m128i c29_align_rslt20_m128i = _mm_alignr_epi8(c29_load_rslt21_m128i, c29_load_rslt20_m128i, 15);

     __horizontal_sse_unpack4_c29_f1<0>(c29_align_rslt20_m128i, out);   // 23rd 4 values


     __m128i c29_load_rslt22_m128i = _mm_loadu_si128(in + 21);
     __m128i c29_align_rslt21_m128i = _mm_alignr_epi8(c29_load_rslt22_m128i, c29_load_rslt21_m128i, 13);

     __horizontal_sse_unpack4_c29_f2<0>(c29_align_rslt21_m128i, out);   // 24th 4 values


     __m128i c29_load_rslt23_m128i = _mm_loadu_si128(in + 22);
     __m128i c29_align_rslt22_m128i = _mm_alignr_epi8(c29_load_rslt23_m128i, c29_load_rslt22_m128i, 12);

     __horizontal_sse_unpack4_c29_f1<0>(c29_align_rslt22_m128i, out);   // 25th 4 values


     __m128i c29_load_rslt24_m128i = _mm_loadu_si128(in + 23);
     __m128i c29_align_rslt23_m128i = _mm_alignr_epi8(c29_load_rslt24_m128i, c29_load_rslt23_m128i, 10);

     __horizontal_sse_unpack4_c29_f2<0>(c29_align_rslt23_m128i, out);   // 26th 4 values


     __m128i c29_load_rslt25_m128i = _mm_loadu_si128(in + 24);
     __m128i c29_align_rslt24_m128i = _mm_alignr_epi8(c29_load_rslt25_m128i, c29_load_rslt24_m128i, 9);

     __horizontal_sse_unpack4_c29_f1<0>(c29_align_rslt24_m128i, out);   // 27th 4 values


     __m128i c29_load_rslt26_m128i = _mm_loadu_si128(in + 25);
     __m128i c29_align_rslt25_m128i = _mm_alignr_epi8(c29_load_rslt26_m128i, c29_load_rslt25_m128i, 7);

     __horizontal_sse_unpack4_c29_f2<0>(c29_align_rslt25_m128i, out);   // 28th 4 values


     __m128i c29_load_rslt27_m128i = _mm_loadu_si128(in + 26);
     __m128i c29_align_rslt26_m128i = _mm_alignr_epi8(c29_load_rslt27_m128i, c29_load_rslt26_m128i, 6);

     __horizontal_sse_unpack4_c29_f1<0>(c29_align_rslt26_m128i, out);   // 29th 4 values


     __m128i c29_load_rslt28_m128i = _mm_loadu_si128(in + 27);
     __m128i c29_align_rslt27_m128i = _mm_alignr_epi8(c29_load_rslt28_m128i, c29_load_rslt27_m128i, 4);

     __horizontal_sse_unpack4_c29_f2<0>(c29_align_rslt27_m128i, out);   // 30th 4 values


     __m128i c29_load_rslt29_m128i = _mm_loadu_si128(in + 28);
     __m128i c29_align_rslt28_m128i = _mm_alignr_epi8(c29_load_rslt29_m128i, c29_load_rslt28_m128i, 3);

     __horizontal_sse_unpack4_c29_f1<0>(c29_align_rslt28_m128i, out);   // 31st 4 values

     __horizontal_sse_unpack4_c29_f2<1>(c29_load_rslt29_m128i, out);    // 32nd 4 values
}




// 30-bit
template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c30(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// note that the 1st value's DW is already in place (aligned)
		//           the 4th value's DW is already in place (unaligned, 2 bits to the left)
		__m128i Horizontal_SSE_c30_shfl_msk_m128i = _mm_set_epi8(
				byte + 14, byte + 13, byte + 12, byte + 11,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c30_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c30_shfl_msk_m128i);

		__m128i c30_slli_rslt_m128i = _mm_slli_epi64(c30_shfl_rslt_m128i, 2);   // shift the 2nd value's DW in place (aligned)
		__m128i c30_srli_rslt1_m128i = _mm_srli_epi64(c30_shfl_rslt_m128i, 2);  // shift the 3rd values's DW in place (unaligned, 2 bits to the left)

		// concatenating the 4 values's DWs
		const int Horizontal_SSE_c30_blend_msk1_imm = 0x0c;
		__m128i c30_blend_rslt1_m128i = _mm_blend_epi16(c30_shfl_rslt_m128i, c30_slli_rslt_m128i, Horizontal_SSE_c30_blend_msk1_imm);

		const int Horizontal_SSE_c30_blend_msk2_imm = 0x30;
		__m128i c30_blend_rslt2_m128i = _mm_blend_epi16(c30_blend_rslt1_m128i, c30_srli_rslt1_m128i, Horizontal_SSE_c30_blend_msk2_imm);

		// note that at this point the 4 values's DWs are already in place (unaligned)
		__m128i c30_mul_rslt_m128i = _mm_mullo_epi32(c30_blend_rslt2_m128i, Horizontal_SSE_mul_msk_m128i[30][0]);
		__m128i c30_srli_rslt2_m128i = _mm_srli_epi32(c30_mul_rslt_m128i, 2);
		__m128i c30_rslt_m128i = _mm_and_si128(c30_srli_rslt2_m128i, Horizontal_SSE_and_msk_m128i[30]);
		_mm_storeu_si128(out++, c30_rslt_m128i);
	}
	else { // Rice Coding
		// note that the 1st value's DW is already in place (aligned)
		//           the 4th value's DW is already in place (unaligned, 2 bits to the left)
		__m128i Horizontal_SSE_c30_shfl_msk_m128i = _mm_set_epi8(
				byte + 14, byte + 13, byte + 12, byte + 11,
				byte + 10, byte + 9, byte + 8, byte + 7,
				byte + 7, byte + 6, byte + 5, byte + 4,
				byte + 3, byte + 2, byte + 1, byte + 0);

		__m128i c30_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c30_shfl_msk_m128i);

		__m128i c30_slli_rslt_m128i = _mm_slli_epi64(c30_shfl_rslt_m128i, 2);   // shift the 2nd value's DW in place (aligned)
		__m128i c30_srli_rslt1_m128i = _mm_srli_epi64(c30_shfl_rslt_m128i, 2);  // shift the 3rd values's DW in place (unaligned, 2 bits to the left)

		// concatenating the 4 values's DWs
		const int Horizontal_SSE_c30_blend_msk1_imm = 0x0c;
		__m128i c30_blend_rslt1_m128i = _mm_blend_epi16(c30_shfl_rslt_m128i, c30_slli_rslt_m128i, Horizontal_SSE_c30_blend_msk1_imm);

		const int Horizontal_SSE_c30_blend_msk2_imm = 0x30;
		__m128i c30_blend_rslt2_m128i = _mm_blend_epi16(c30_blend_rslt1_m128i, c30_srli_rslt1_m128i, Horizontal_SSE_c30_blend_msk2_imm);

		// note that at this point the 4 values's DWs are already in place (unaligned)
		__m128i c30_mul_rslt_m128i = _mm_mullo_epi32(c30_blend_rslt2_m128i, Horizontal_SSE_mul_msk_m128i[30][0]);
		__m128i c30_srli_rslt2_m128i = _mm_srli_epi32(c30_mul_rslt_m128i, 2);
		__m128i c30_and_rslt_m128i = _mm_and_si128(c30_srli_rslt2_m128i, Horizontal_SSE_and_msk_m128i[30]);
		__m128i c30_rslt_m128i = _mm_or_si128(c30_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 30));
		_mm_storeu_si128(out++, c30_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c30(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 64) {
	     __m128i c30_load_rslt1_m128i = _mm_loadu_si128(in++);

	     __horizontal_sse_unpack4_c30<0>(c30_load_rslt1_m128i, out);    // unpacks 1st 4 values


	     __m128i c30_load_rslt2_m128i = _mm_loadu_si128(in++);
	     __m128i c30_align_rslt1_m128i = _mm_alignr_epi8(c30_load_rslt2_m128i, c30_load_rslt1_m128i, 15);

	     __horizontal_sse_unpack4_c30<0>(c30_align_rslt1_m128i, out);   // unpacks 2nd 4 values


	     __m128i c30_load_rslt3_m128i = _mm_loadu_si128(in++);
	     __m128i c30_align_rslt2_m128i = _mm_alignr_epi8(c30_load_rslt3_m128i, c30_load_rslt2_m128i, 14);

	     __horizontal_sse_unpack4_c30<0>(c30_align_rslt2_m128i, out);   // unpacks 3rd 4 values


	     __m128i c30_load_rslt4_m128i = _mm_loadu_si128(in++);
	     __m128i c30_align_rslt3_m128i = _mm_alignr_epi8(c30_load_rslt4_m128i, c30_load_rslt3_m128i, 13);

	     __horizontal_sse_unpack4_c30<0>(c30_align_rslt3_m128i, out);   // unpacks 4th 4 values


	     __m128i c30_load_rslt5_m128i = _mm_loadu_si128(in++);
	     __m128i c30_align_rslt4_m128i = _mm_alignr_epi8(c30_load_rslt5_m128i, c30_load_rslt4_m128i, 12);

	     __horizontal_sse_unpack4_c30<0>(c30_align_rslt4_m128i, out);   // unpacks 5th 4 values


	     __m128i c30_load_rslt6_m128i = _mm_loadu_si128(in++);
	     __m128i c30_align_rslt5_m128i = _mm_alignr_epi8(c30_load_rslt6_m128i, c30_load_rslt5_m128i, 11);

	     __horizontal_sse_unpack4_c30<0>(c30_align_rslt5_m128i, out);   // unpacks 6th 4 values


	     __m128i c30_load_rslt7_m128i = _mm_loadu_si128(in++);
	     __m128i c30_align_rslt6_m128i = _mm_alignr_epi8(c30_load_rslt7_m128i, c30_load_rslt6_m128i, 10);

	     __horizontal_sse_unpack4_c30<0>(c30_align_rslt6_m128i, out);   // unpacks 7th 4 values


	     __m128i c30_load_rslt8_m128i = _mm_loadu_si128(in++);
	     __m128i c30_align_rslt7_m128i = _mm_alignr_epi8(c30_load_rslt8_m128i, c30_load_rslt7_m128i, 9);

	     __horizontal_sse_unpack4_c30<0>(c30_align_rslt7_m128i, out);   // unpacks 8th 4 values


	     __m128i c30_load_rslt9_m128i = _mm_loadu_si128(in++);
	     __m128i c30_align_rslt8_m128i = _mm_alignr_epi8(c30_load_rslt9_m128i, c30_load_rslt8_m128i, 8);

	     __horizontal_sse_unpack4_c30<0>(c30_align_rslt8_m128i, out);   // unpacks 9th 4 values


	     __m128i c30_load_rslt10_m128i = _mm_loadu_si128(in++);
	     __m128i c30_align_rslt9_m128i = _mm_alignr_epi8(c30_load_rslt10_m128i, c30_load_rslt9_m128i, 7);

	     __horizontal_sse_unpack4_c30<0>(c30_align_rslt9_m128i, out);   // unpacks 10th 4 values


	     __m128i c30_load_rslt11_m128i = _mm_loadu_si128(in++);
	     __m128i c30_align_rslt10_m128i = _mm_alignr_epi8(c30_load_rslt11_m128i, c30_load_rslt10_m128i, 6);

	     __horizontal_sse_unpack4_c30<0>(c30_align_rslt10_m128i, out);  // unpacks 11th 4 values


	     __m128i c30_load_rslt12_m128i = _mm_loadu_si128(in++);
	     __m128i c30_align_rslt11_m128i = _mm_alignr_epi8(c30_load_rslt12_m128i, c30_load_rslt11_m128i, 5);

	     __horizontal_sse_unpack4_c30<0>(c30_align_rslt11_m128i, out);  // unpacks 12th 4 values


	     __m128i c30_load_rslt13_m128i = _mm_loadu_si128(in++);
	     __m128i c30_align_rslt12_m128i = _mm_alignr_epi8(c30_load_rslt13_m128i, c30_load_rslt12_m128i, 4);

	     __horizontal_sse_unpack4_c30<0>(c30_align_rslt12_m128i, out);  // unpacks 13th 4 values


	     __m128i c30_load_rslt14_m128i = _mm_loadu_si128(in++);
	     __m128i c30_align_rslt13_m128i = _mm_alignr_epi8(c30_load_rslt14_m128i, c30_load_rslt13_m128i, 3);

	     __horizontal_sse_unpack4_c30<0>(c30_align_rslt13_m128i, out);  // unpacks 14th 4 values


	     __m128i c30_load_rslt15_m128i = _mm_loadu_si128(in++);
	     __m128i c30_align_rslt14_m128i = _mm_alignr_epi8(c30_load_rslt15_m128i, c30_load_rslt14_m128i, 2);

	     __horizontal_sse_unpack4_c30<0>(c30_align_rslt14_m128i, out);  // unpacks 15th 4 values

	     __horizontal_sse_unpack4_c30<1>(c30_load_rslt15_m128i, out);   // unpacks 16th 4 values
	}
}




// 31-bit
template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c31_f1(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// note that the 1st value's DW is already in place (aligned)
		__m128i Horizontal_SSE_c31_shfl_msk_m128i = _mm_set_epi8(
				byte + 14, byte + 13, byte + 12, byte + 11,
				byte + 10, byte + 9, byte + 8, byte + 7,
				0xff, 0xff, 0xff, 0xff,
				0xff, 0xff, 0xff, 0xff);

		__m128i c31_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c31_shfl_msk_m128i);

		__m128i c31_srli_rslt_m128i = _mm_srli_epi64(c31_shfl_rslt_m128i, 6);  // shift the 3rd value's DW in place (aligned)
		__m128i c31_slli_rslt1_m128i = _mm_slli_epi64(InReg, 1);               // shift the 2nd value's DW in place (aligned)
		__m128i c31_slli_rslt2_m128i = _mm_slli_epi64(InReg, 3);               // shift the 4th value's DW in place (aligned)

		// concatenating the 4 above DWs
		const int Horizontal_SSE_c31_blend_msk1_imm = 0x30;
		__m128i c31_blend_rslt1_m128i = _mm_blend_epi16(InReg, c31_srli_rslt_m128i, Horizontal_SSE_c31_blend_msk1_imm);

		const int Horizontal_SSE_c31_blend_msk2_imm = 0x0c;
		__m128i c31_blend_rslt2_m128i = _mm_blend_epi16(c31_blend_rslt1_m128i, c31_slli_rslt1_m128i, Horizontal_SSE_c31_blend_msk2_imm);

		const int Horizontal_SSE_c31_blend_msk3_imm = 0xc0;
		__m128i c31_blend_rslt3_m128i = _mm_blend_epi16(c31_blend_rslt2_m128i, c31_slli_rslt2_m128i, Horizontal_SSE_c31_blend_msk3_imm);

		// note that at this point the 4 values's DWs are in place and aligned
		__m128i c31_rslt_m128i = _mm_and_si128(c31_blend_rslt3_m128i, Horizontal_SSE_and_msk_m128i[31]);
		_mm_storeu_si128(out++, c31_rslt_m128i);
	}
	else { // Rice Coding
		// note that the 1st value's DW is already in place (aligned)
		__m128i Horizontal_SSE_c31_shfl_msk_m128i = _mm_set_epi8(
				byte + 14, byte + 13, byte + 12, byte + 11,
				byte + 10, byte + 9, byte + 8, byte + 7,
				0xff, 0xff, 0xff, 0xff,
				0xff, 0xff, 0xff, 0xff);

		__m128i c31_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c31_shfl_msk_m128i);

		__m128i c31_srli_rslt_m128i = _mm_srli_epi64(c31_shfl_rslt_m128i, 6);  // shift the 3rd value's DW in place (aligned)
		__m128i c31_slli_rslt1_m128i = _mm_slli_epi64(InReg, 1);               // shift the 2nd value's DW in place (aligned)
		__m128i c31_slli_rslt2_m128i = _mm_slli_epi64(InReg, 3);               // shift the 4th value's DW in place (aligned)

		// concatenating the 4 above DWs
		const int Horizontal_SSE_c31_blend_msk1_imm = 0x30;
		__m128i c31_blend_rslt1_m128i = _mm_blend_epi16(InReg, c31_srli_rslt_m128i, Horizontal_SSE_c31_blend_msk1_imm);

		const int Horizontal_SSE_c31_blend_msk2_imm = 0x0c;
		__m128i c31_blend_rslt2_m128i = _mm_blend_epi16(c31_blend_rslt1_m128i, c31_slli_rslt1_m128i, Horizontal_SSE_c31_blend_msk2_imm);

		const int Horizontal_SSE_c31_blend_msk3_imm = 0xc0;
		__m128i c31_blend_rslt3_m128i = _mm_blend_epi16(c31_blend_rslt2_m128i, c31_slli_rslt2_m128i, Horizontal_SSE_c31_blend_msk3_imm);

		// note that at this point the 4 values's DWs are in place and aligned
		__m128i c31_and_rslt_m128i = _mm_and_si128(c31_blend_rslt3_m128i, Horizontal_SSE_and_msk_m128i[31]);
		__m128i c31_rslt_m128i = _mm_or_si128(c31_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 31));
		_mm_storeu_si128(out++, c31_rslt_m128i);
	}
}

template <bool IsRiceCoding>
template <uint32_t byte>
forceinline void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack4_c31_f2(const __m128i &InReg,
		__m128i *  __restrict__  &out) {
	if (!IsRiceCoding) { // NewPFor etc.
		// Note that the 4th values's DW is already in place (unaligned, 1 bit to the left)
		__m128i Horizontal_SSE_c31_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, 0xff, 0xff,
				0xff, 0xff, 0xff, 0xff,
				byte + 8, byte + 7, byte + 6, byte + 5,
				byte + 4, byte + 3, byte + 2, byte + 1);

		__m128i c31_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c31_shfl_msk_m128i);

		__m128i c31_slli_rslt_m128i = _mm_slli_epi64(c31_shfl_rslt_m128i, 6); // shift the 2nd value's DW in place (unaligned, 1 bit to the left)
		__m128i c31_srli_rslt1_m128i = _mm_srli_epi64(InReg, 3);              // shift the 1st value's DW in place (unaligned, 1 bit to the left)
		__m128i c31_srli_rslt2_m128i = _mm_srli_epi64(InReg, 1);              // shift the 3rd value's DW in place (unaligned, 1 bit to the left)

		// concatenating the 4 above DWs
		const int Horizontal_SSE_c31_blend_msk1_imm = 0x0c;
		__m128i c31_blend_rslt1_m128i = _mm_blend_epi16(InReg, c31_slli_rslt_m128i, Horizontal_SSE_c31_blend_msk1_imm);

		const int Horizontal_SSE_c31_blend_msk2_imm = 0x03;
		__m128i c31_blend_rslt2_m128i = _mm_blend_epi16(c31_blend_rslt1_m128i, c31_srli_rslt1_m128i, Horizontal_SSE_c31_blend_msk2_imm);

		const int Horizontal_SSE_c31_blend_msk3_imm = 0x30;
		__m128i c31_blend_rslt3_m128i = _mm_blend_epi16(c31_blend_rslt2_m128i, c31_srli_rslt2_m128i, Horizontal_SSE_c31_blend_msk3_imm);

		// Note that at this point the 4 values's DWs are in place but not aligned (1 bit to the left)
		__m128i c31_srli_rslt3_m128i = _mm_srli_epi32(c31_blend_rslt3_m128i, 1);
		__m128i c31_rslt_m128i = _mm_and_si128(c31_srli_rslt3_m128i, Horizontal_SSE_and_msk_m128i[31]);
		_mm_storeu_si128(out++, c31_rslt_m128i);
	}
	else { // Rice Coding
		// Note that the 4th values's DW is already in place (unaligned, 1 bit to the left)
		__m128i Horizontal_SSE_c31_shfl_msk_m128i = _mm_set_epi8(
				0xff, 0xff, 0xff, 0xff,
				0xff, 0xff, 0xff, 0xff,
				byte + 8, byte + 7, byte + 6, byte + 5,
				byte + 4, byte + 3, byte + 2, byte + 1);

		__m128i c31_shfl_rslt_m128i = _mm_shuffle_epi8(InReg, Horizontal_SSE_c31_shfl_msk_m128i);

		__m128i c31_slli_rslt_m128i = _mm_slli_epi64(c31_shfl_rslt_m128i, 6); // shift the 2nd value's DW in place (unaligned, 1 bit to the left)
		__m128i c31_srli_rslt1_m128i = _mm_srli_epi64(InReg, 3);              // shift the 1st value's DW in place (unaligned, 1 bit to the left)
		__m128i c31_srli_rslt2_m128i = _mm_srli_epi64(InReg, 1);              // shift the 3rd value's DW in place (unaligned, 1 bit to the left)

		// concatenating the 4 above DWs
		const int Horizontal_SSE_c31_blend_msk1_imm = 0x0c;
		__m128i c31_blend_rslt1_m128i = _mm_blend_epi16(InReg, c31_slli_rslt_m128i, Horizontal_SSE_c31_blend_msk1_imm);

		const int Horizontal_SSE_c31_blend_msk2_imm = 0x03;
		__m128i c31_blend_rslt2_m128i = _mm_blend_epi16(c31_blend_rslt1_m128i, c31_srli_rslt1_m128i, Horizontal_SSE_c31_blend_msk2_imm);

		const int Horizontal_SSE_c31_blend_msk3_imm = 0x30;
		__m128i c31_blend_rslt3_m128i = _mm_blend_epi16(c31_blend_rslt2_m128i, c31_srli_rslt2_m128i, Horizontal_SSE_c31_blend_msk3_imm);

		// Note that at this point the 4 values's DWs are in place but not aligned (1 bit to the left)
		__m128i c31_srli_rslt3_m128i = _mm_srli_epi32(c31_blend_rslt3_m128i, 1);
		__m128i c31_and_rslt_m128i = _mm_and_si128(c31_srli_rslt3_m128i, Horizontal_SSE_and_msk_m128i[31]);
		__m128i c31_rslt_m128i = _mm_or_si128(c31_and_rslt_m128i, _mm_slli_epi32(_mm_loadu_si128(quotient++), 31));
		_mm_storeu_si128(out++, c31_rslt_m128i);
	}
}

template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c31(const __m128i * __restrict__ in,
		__m128i *  __restrict__  out) {
	__m128i c31_load_rslt1_m128i = _mm_loadu_si128(in + 0);

	__horizontal_sse_unpack4_c31_f1<0>(c31_load_rslt1_m128i, out);      // unpacks 1st 4 values


	__m128i c31_load_rslt2_m128i = _mm_loadu_si128(in + 1);
	__m128i c31_alignr_rslt1_m128i = _mm_alignr_epi8(c31_load_rslt2_m128i, c31_load_rslt1_m128i, 15);

	__horizontal_sse_unpack4_c31_f2<0>(c31_alignr_rslt1_m128i, out);    // unpacks 2nd 4 values


	__m128i c31_load_rslt3_m128i = _mm_loadu_si128(in + 2);
	__m128i c31_alignr_rslt2_m128i = _mm_alignr_epi8(c31_load_rslt3_m128i, c31_load_rslt2_m128i, 15);

	__horizontal_sse_unpack4_c31_f1<0>(c31_alignr_rslt2_m128i, out);    // unpacks 3rd 4 values


	__m128i c31_load_rslt4_m128i = _mm_loadu_si128(in + 3);
	__m128i c31_alignr_rslt3_m128i = _mm_alignr_epi8(c31_load_rslt4_m128i, c31_load_rslt3_m128i, 14);

	__horizontal_sse_unpack4_c31_f2<0>(c31_alignr_rslt3_m128i, out);    // unpacks 4th 4 values


	__m128i c31_load_rslt5_m128i = _mm_loadu_si128(in + 4);
	__m128i c31_alignr_rslt4_m128i = _mm_alignr_epi8(c31_load_rslt5_m128i, c31_load_rslt4_m128i, 14);

	__horizontal_sse_unpack4_c31_f1<0>(c31_alignr_rslt4_m128i, out);    // unpacks 5th 4 values


	__m128i c31_load_rslt6_m128i = _mm_loadu_si128(in + 5);
	__m128i c31_alignr_rslt5_m128i = _mm_alignr_epi8(c31_load_rslt6_m128i, c31_load_rslt5_m128i, 13);

	__horizontal_sse_unpack4_c31_f2<0>(c31_alignr_rslt5_m128i, out);    // unpacks 6th 4 values


	__m128i c31_load_rslt7_m128i = _mm_loadu_si128(in + 6);
	__m128i c31_alignr_rslt6_m128i = _mm_alignr_epi8(c31_load_rslt7_m128i, c31_load_rslt6_m128i, 13);

	__horizontal_sse_unpack4_c31_f1<0>(c31_alignr_rslt6_m128i, out);    // unpacks 7th 4 values


	__m128i c31_load_rslt8_m128i = _mm_loadu_si128(in + 7);
	__m128i c31_alignr_rslt7_m128i = _mm_alignr_epi8(c31_load_rslt8_m128i, c31_load_rslt7_m128i, 12);

	__horizontal_sse_unpack4_c31_f2<0>(c31_alignr_rslt7_m128i, out);    // unpacks 8th 4 values


	__m128i c31_load_rslt9_m128i = _mm_loadu_si128(in + 8);
	__m128i c31_alignr_rslt8_m128i = _mm_alignr_epi8(c31_load_rslt9_m128i, c31_load_rslt8_m128i, 12);

	__horizontal_sse_unpack4_c31_f1<0>(c31_alignr_rslt8_m128i, out);    // unpacks 9th 4 values


	__m128i c31_load_rslt10_m128i = _mm_loadu_si128(in + 9);
	__m128i c31_alignr_rslt9_m128i = _mm_alignr_epi8(c31_load_rslt10_m128i, c31_load_rslt9_m128i, 11);

	__horizontal_sse_unpack4_c31_f2<0>(c31_alignr_rslt9_m128i, out);    // unpacks 10th 4 values


	__m128i c31_load_rslt11_m128i = _mm_loadu_si128(in + 10);
	__m128i c31_alignr_rslt10_m128i = _mm_alignr_epi8(c31_load_rslt11_m128i, c31_load_rslt10_m128i, 11);

	__horizontal_sse_unpack4_c31_f1<0>(c31_alignr_rslt10_m128i, out);   // unpacks 11th 4 values


	__m128i c31_load_rslt12_m128i = _mm_loadu_si128(in + 11);
	__m128i c31_alignr_rslt11_m128i = _mm_alignr_epi8(c31_load_rslt12_m128i, c31_load_rslt11_m128i, 10);

	__horizontal_sse_unpack4_c31_f2<0>(c31_alignr_rslt11_m128i, out);   // unpacks 12th 4 values


	__m128i c31_load_rslt13_m128i = _mm_loadu_si128(in + 12);
	__m128i c31_alignr_rslt12_m128i = _mm_alignr_epi8(c31_load_rslt13_m128i, c31_load_rslt12_m128i, 10);

	__horizontal_sse_unpack4_c31_f1<0>(c31_alignr_rslt12_m128i, out);   // unpacks 13th 4 values


	__m128i c31_load_rslt14_m128i = _mm_loadu_si128(in + 13);
	__m128i c31_alignr_rslt13_m128i = _mm_alignr_epi8(c31_load_rslt14_m128i, c31_load_rslt13_m128i, 9);

	__horizontal_sse_unpack4_c31_f2<0>(c31_alignr_rslt13_m128i, out);   // unpacks 14th 4 values


	__m128i c31_load_rslt15_m128i = _mm_loadu_si128(in + 14);
	__m128i c31_alignr_rslt14_m128i = _mm_alignr_epi8(c31_load_rslt15_m128i, c31_load_rslt14_m128i, 9);

	__horizontal_sse_unpack4_c31_f1<0>(c31_alignr_rslt14_m128i, out);   // unpacks 15th 4 values


	__m128i c31_load_rslt16_m128i = _mm_loadu_si128(in + 15);
	__m128i c31_alignr_rslt15_m128i = _mm_alignr_epi8(c31_load_rslt16_m128i, c31_load_rslt15_m128i, 8);

	__horizontal_sse_unpack4_c31_f2<0>(c31_alignr_rslt15_m128i, out);   // unpacks 16th 4 values


	__m128i c31_load_rslt17_m128i = _mm_loadu_si128(in + 16);
	__m128i c31_alignr_rslt16_m128i = _mm_alignr_epi8(c31_load_rslt17_m128i, c31_load_rslt16_m128i, 8);

	__horizontal_sse_unpack4_c31_f1<0>(c31_alignr_rslt16_m128i, out);   // unpacks 17th 4 values


	__m128i c31_load_rslt18_m128i = _mm_loadu_si128(in + 17);
	__m128i c31_alignr_rslt17_m128i = _mm_alignr_epi8(c31_load_rslt18_m128i, c31_load_rslt17_m128i, 7);

	__horizontal_sse_unpack4_c31_f2<0>(c31_alignr_rslt17_m128i, out);   // unpacks 18th 4 values


	__m128i c31_load_rslt19_m128i = _mm_loadu_si128(in + 18);
	__m128i c31_alignr_rslt18_m128i = _mm_alignr_epi8(c31_load_rslt19_m128i, c31_load_rslt18_m128i, 7);

	__horizontal_sse_unpack4_c31_f1<0>(c31_alignr_rslt18_m128i, out);   // unpacks 19th 4 values


	__m128i c31_load_rslt20_m128i = _mm_loadu_si128(in + 19);
	__m128i c31_alignr_rslt19_m128i = _mm_alignr_epi8(c31_load_rslt20_m128i, c31_load_rslt19_m128i, 6);

	__horizontal_sse_unpack4_c31_f2<0>(c31_alignr_rslt19_m128i, out);   // unpacks 20th 4 values


	__m128i c31_load_rslt21_m128i = _mm_loadu_si128(in + 20);
	__m128i c31_alignr_rslt20_m128i = _mm_alignr_epi8(c31_load_rslt21_m128i, c31_load_rslt20_m128i, 6);

	__horizontal_sse_unpack4_c31_f1<0>(c31_alignr_rslt20_m128i, out);   // unpacks 21st 4 values


	__m128i c31_load_rslt22_m128i = _mm_loadu_si128(in + 21);
	__m128i c31_alignr_rslt21_m128i = _mm_alignr_epi8(c31_load_rslt22_m128i, c31_load_rslt21_m128i, 5);

	__horizontal_sse_unpack4_c31_f2<0>(c31_alignr_rslt21_m128i, out);   // unpacks 22nd 4 values


	__m128i c31_load_rslt23_m128i = _mm_loadu_si128(in + 22);
	__m128i c31_alignr_rslt22_m128i = _mm_alignr_epi8(c31_load_rslt23_m128i, c31_load_rslt22_m128i, 5);

	__horizontal_sse_unpack4_c31_f1<0>(c31_alignr_rslt22_m128i, out);   // unpacks 23rd 4 values


	__m128i c31_load_rslt24_m128i = _mm_loadu_si128(in + 23);
	__m128i c31_alignr_rslt23_m128i = _mm_alignr_epi8(c31_load_rslt24_m128i, c31_load_rslt23_m128i, 4);

	__horizontal_sse_unpack4_c31_f2<0>(c31_alignr_rslt23_m128i, out);   // unpacks 24th 4 values


	__m128i c31_load_rslt25_m128i = _mm_loadu_si128(in + 24);
	__m128i c31_alignr_rslt24_m128i = _mm_alignr_epi8(c31_load_rslt25_m128i, c31_load_rslt24_m128i, 4);

	__horizontal_sse_unpack4_c31_f1<0>(c31_alignr_rslt24_m128i, out);   // unpacks 25th 4 values


	__m128i c31_load_rslt26_m128i = _mm_loadu_si128(in + 25);
	__m128i c31_alignr_rslt25_m128i = _mm_alignr_epi8(c31_load_rslt26_m128i, c31_load_rslt25_m128i, 3);

	__horizontal_sse_unpack4_c31_f2<0>(c31_alignr_rslt25_m128i, out);   // unpacks 26th 4 values


	__m128i c31_load_rslt27_m128i = _mm_loadu_si128(in + 26);
	__m128i c31_alignr_rslt26_m128i = _mm_alignr_epi8(c31_load_rslt27_m128i, c31_load_rslt26_m128i, 3);

	__horizontal_sse_unpack4_c31_f1<0>(c31_alignr_rslt26_m128i, out);   // unpacks 27th 4 values


	__m128i c31_load_rslt28_m128i = _mm_loadu_si128(in + 27);
	__m128i c31_alignr_rslt27_m128i = _mm_alignr_epi8(c31_load_rslt28_m128i, c31_load_rslt27_m128i, 2);

	__horizontal_sse_unpack4_c31_f2<0>(c31_alignr_rslt27_m128i, out);   // unpacks 28th 4 values


	__m128i c31_load_rslt29_m128i = _mm_loadu_si128(in + 28);
	__m128i c31_alignr_rslt28_m128i = _mm_alignr_epi8(c31_load_rslt29_m128i, c31_load_rslt28_m128i, 2);

	__horizontal_sse_unpack4_c31_f1<0>(c31_alignr_rslt28_m128i, out);   // unpacks 29th 4 values


	__m128i c31_load_rslt30_m128i = _mm_loadu_si128(in + 29);
	__m128i c31_alignr_rslt29_m128i = _mm_alignr_epi8(c31_load_rslt30_m128i, c31_load_rslt29_m128i, 1);

	__horizontal_sse_unpack4_c31_f2<0>(c31_alignr_rslt29_m128i, out);   // unpacks 30th 4 values


	__m128i c31_load_rslt31_m128i = _mm_loadu_si128(in + 30);
	__m128i c31_alignr_rslt30_m128i = _mm_alignr_epi8(c31_load_rslt31_m128i, c31_load_rslt30_m128i, 1);

	__horizontal_sse_unpack4_c31_f1<0>(c31_alignr_rslt30_m128i, out);   // unpacks 30th 4 values

	__horizontal_sse_unpack4_c31_f2<0>(c31_load_rslt31_m128i, out);     // unpacks 31st 4 values
}




// 32-bit
template <bool IsRiceCoding>
void HorizontalSSEUnpacker<IsRiceCoding>::__horizontal_sse_unpack128_c32(const __m128i * __restrict__ in_m128i,
		__m128i *  __restrict__  out_m128i) {
	const uint32_t *in = reinterpret_cast<const uint32_t *>(in_m128i);
	uint32_t *out = reinterpret_cast<uint32_t *>(out_m128i);
	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < 128; numberOfValuesUnpacked += 16) {
		memcpy16(in, out);
		in += 16;
		out += 16;
	}
}

#endif /* HORIZONTALSSEUNPACKERIMP_H_ */
