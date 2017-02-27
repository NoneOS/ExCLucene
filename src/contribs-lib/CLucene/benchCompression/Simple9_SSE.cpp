/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#include "Simple9_SSE.h"


const __m128i Simple9_SSE::Simple9_SSE_mul_msk_m128i[SIMPLE9_LEN] = {
		_mm_set_epi32(0x08, 0x04, 0x02, 0x01),       // 1-bit
		_mm_set_epi32(0x40, 0x10, 0x04, 0x01),       // 2-bit
		_mm_set_epi32(0x0200, 0x40, 0x08, 0x01),     // 3-bit
		_mm_set_epi32(0x1000, 0x0100, 0x10, 0x01),   // 4-bit
		_mm_set_epi32(0x8000, 0x0400, 0x20, 0x01),   // 5-bit
		_mm_set_epi32(0x200000, 0x4000, 0x80, 0x01), // 7-bit
		_mm_set_epi32(0x00, 0x040000, 0x0200, 0x01), // 9-bit
		_mm_set_epi32(0x00, 0x00, 0x4000, 0x01),     // 14-bit
		_mm_set_epi32(0x00, 0x00, 0x00, 0x01),       // 28-bit
};


const __m128i Simple9_SSE::Simple9_SSE_and_msk_m128i[SIMPLE9_LEN] = {
		_mm_set1_epi32(0x01),      // 1-bit
		_mm_set1_epi32(0x03),      // 2-bit
		_mm_set1_epi32(0x07),      // 3-bit
		_mm_set1_epi32(0x0F),      // 4-bit
		_mm_set1_epi32(0x1F),      // 5-bit
		_mm_set1_epi32(0x7F),      // 7-bit
		_mm_set1_epi32(0x01FF),    // 9-bit
		_mm_set1_epi32(0x3FFF),    // 14-bit
		_mm_set1_epi32(0xFFFFFFF)  // 28-bit
};


const uint32_t * Simple9_SSE::decodeArray(const uint32_t *in, size_t csize,
		uint32_t *out, size_t nvalue) {
	__m128i set1_rslt_m128i;
	__m128i mul_rslt_m128i;
	__m128i srli_rslt_m128i;
	__m128i rslt_m128i;

	const uint32_t *const endout(out + nvalue);
	while (endout > out) {
		const uint32_t codeword = in[0];
		++in;
		set1_rslt_m128i = _mm_set1_epi32(codeword);
		const uint32_t descriptor = codeword >> (32 - SIMPLE9_LOGDESC);
		switch (descriptor) {
		case 0: // 28 * 1-bit
			mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple9_SSE_mul_msk_m128i[0]);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 27);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 23);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)(out + 4 * 1), rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 19);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)(out + 4 * 2), rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 15);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)(out + 4 * 3), rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 11);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)(out + 4 * 4), rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 7);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)(out + 4 * 5), rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 3);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)(out + 4 * 6), rslt_m128i);

			out += 28;

			break;
		case 1: // 14 * 2-bit
			mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple9_SSE_mul_msk_m128i[1]);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 26);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[1]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 18);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[1]);
			_mm_storeu_si128((__m128i *)(out + 4 * 1), rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 10);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[1]);
			_mm_storeu_si128((__m128i *)(out + 4 * 2), rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 2);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[1]);
			_mm_storeu_si128((__m128i *)(out + 4 * 3), rslt_m128i);

			out += 14;

			break;
		case 2: // 9 * 3-bit
			mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple9_SSE_mul_msk_m128i[2]);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 25);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[2]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 13);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[2]);
			_mm_storeu_si128((__m128i *)(out + 4 * 1), rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 1);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[2]);
			_mm_storeu_si128((__m128i *)(out + 4 * 2), rslt_m128i);

			out += 9;

			break;
		case 3: // 7 * 4-bit
			mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple9_SSE_mul_msk_m128i[3]);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 24);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[3]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 8);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[3]);
			_mm_storeu_si128((__m128i *)(out + 4 * 1), rslt_m128i);

			out += 7;

			break;
		case 4: // 5 * 5-bit
			mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple9_SSE_mul_msk_m128i[4]);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 23);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[4]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 3);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[4]);
			_mm_storeu_si128((__m128i *)(out + 4 * 1), rslt_m128i);

			out += 5;

			break;
		case 5: // 4 * 7-bit
			mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple9_SSE_mul_msk_m128i[5]);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 21);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[5]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

			out += 4;

			break;
		case 6: // 3 * 9-bit
			mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple9_SSE_mul_msk_m128i[6]);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 19);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[6]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

			out += 3;

			break;
		case 7: // 2 * 14-bit
			mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple9_SSE_mul_msk_m128i[7]);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 14);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple9_SSE_and_msk_m128i[7]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

			out += 2;

			break;
		case 8: // 1 * 28-bit
			rslt_m128i = _mm_and_si128(set1_rslt_m128i, Simple9_SSE_and_msk_m128i[8]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

			++out;

			break;
		default: // invalid descriptor
			std::cerr << "Invalid descriptor: " << descriptor << std::endl;
			throw std::runtime_error("Invalid descriptor for " + name() + ".");
		}
	}

	return in;
}
