/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#include "Simple16_SSE.h"

const __m128i Simple16_SSE::Simple16_SSE_mul_msk_m128i[SIMPLE16_LEN] = {
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

// 3 * 6-bit + 1 * 5-bit
const __m128i Simple16_SSE::Simple16_SSE_c10_mul_m128i =
		_mm_set_epi32(0x020000, 0x1000, 0x40, 0x01);

// 2 * 5-bit + 2 * 6-bit
const __m128i Simple16_SSE::Simple16_SSE_c11_mul_m128i =
		_mm_set_epi32(0x020000, 0x0800, 0x20, 0x01);


const __m128i Simple16_SSE::Simple16_SSE_and_msk_m128i[SIMPLE16_LEN] = {
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

const __m128i Simple16_SSE::Simple16_SSE_c6_and_msk_m128i[2] = {
		_mm_set_epi32(0x0F, 0x0F, 0x0F, 0x07), // 1 * 3-bit + 3 * 4-bit
		_mm_set_epi32(0x07, 0x07, 0x07, 0x0F)  // 1 * 4-bit + 3 * 3-bit
};

// case 10: 3 * 6-bit + 2 * 5-bit
const __m128i Simple16_SSE::Simple16_SSE_c10_and_msk_m128i[2] = {
		_mm_set_epi32(0x1F, 0x3F, 0x3F, 0x3F), // 3 * 6-bit + 1 * 5-bit
		_mm_set_epi32(0x00, 0x00, 0x00, 0x1F)  // 1 * 5-bit
};

// case 11: 2 * 5-bit + 3 * 6-bit
const __m128i Simple16_SSE::Simple16_SSE_c11_and_msk_m128i[2] = {
		_mm_set_epi32(0x3F, 0x3F, 0x1F, 0x1F), // 2 * 5-bit + 2 * 6-bit
		_mm_set_epi32(0x00, 0x00, 0x00, 0x3F)  // 1 * 6-bit
};

// case 13: 1 * 10-bit + 2 * 9-bit
const __m128i Simple16_SSE::Simple16_SSE_c13_and_msk_m128i =
		_mm_set_epi32(0x00, 0x01FF, 0x01FF, 0x03FF);


const uint32_t * Simple16_SSE::decodeArray(const uint32_t *in, size_t csize,
		uint32_t *out, size_t nvalue) {
	__m128i set1_rslt_m128i;
	__m128i mul_rslt_m128i, mul_rslt2_m128i;
	__m128i srli_rslt_m128i;
	__m128i rslt_m128i;

	const uint32_t *const endout(out + nvalue);
	while (endout > out) {
		const uint32_t codeword = in[0];
		++in;
		set1_rslt_m128i = _mm_set1_epi32(codeword);
		const uint32_t descriptor = codeword >> (32 - SIMPLE16_LOGDESC);
		switch (descriptor) {
		case 0: // 28 * 1-bit
			mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[0]);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 27);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 23);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)(out + 4 * 1), rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 19);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)(out + 4 * 2), rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 15);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)(out + 4 * 3), rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 11);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)(out + 4 * 4), rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 7);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)(out + 4 * 5), rslt_m128i);

			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 3);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)(out + 4 * 6), rslt_m128i);

			out += 28;

			break;
		case 1: // 7 * 2-bit + 14 * 1-bit
		    mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[1]);

		    // 4 * 2-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 26);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[1]);
		    _mm_storeu_si128((__m128i *)out, rslt_m128i);

		    // 3 * 2-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 18);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[1]);
		    _mm_storeu_si128((__m128i *)(out + 4 * 1), rslt_m128i);


		    mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[0]);

		    // 4 * 1-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 13);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
		    _mm_storeu_si128((__m128i *)(out + 7), rslt_m128i);

		    // 4 * 1-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 9);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
		    _mm_storeu_si128((__m128i *)(out + 7 + 4 * 1), rslt_m128i);

		    // 4 * 1-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 5);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
		    _mm_storeu_si128((__m128i *)(out + 7 + 4 * 2), rslt_m128i);

		    // 2 * 1-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 1);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
		    _mm_storeu_si128((__m128i *)(out + 7 + 4 * 3), rslt_m128i);

			out += 21;

			break;
		case 2: // 7 * 1-bit + 7 * 2-bit + 7 * 1-bit
			mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[0]);

			// 4 * 1-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 27);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

			// 3 * 1-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 23);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)(out + 4 * 1), rslt_m128i);


		    mul_rslt2_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[1]);

		    // 4 * 2-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt2_m128i, 19);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[1]);
		    _mm_storeu_si128((__m128i *)(out + 7), rslt_m128i);

		    // 3 * 2-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt2_m128i, 11);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[1]);
		    _mm_storeu_si128((__m128i *)(out + 7 + 4 * 1), rslt_m128i);


		    // 4 * 1-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 6);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)(out + 7 + 7), rslt_m128i);

			// 3 * 1-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 2);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)(out + 7 + 7 + 4 * 1), rslt_m128i);

			out += 21;

			break;
		case 3: // 14 * 1-bit + 7 * 2-bit
			mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[0]);

			// 4 * 1-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 27);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)out, rslt_m128i);

			// 4 * 1-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 23);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)(out + 4 * 1), rslt_m128i);

			// 4 * 1-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 19);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)(out + 4 * 2), rslt_m128i);

			// 2 * 1-bit
			srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 15);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[0]);
			_mm_storeu_si128((__m128i *)(out + 4 * 3), rslt_m128i);


		    mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[1]);

		    // 4 * 2-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 12);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[1]);
		    _mm_storeu_si128((__m128i *)(out + 14), rslt_m128i);

		    // 3 * 2-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 4);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[1]);
		    _mm_storeu_si128((__m128i *)(out + 14 + 4 * 1), rslt_m128i);

			out += 21;

			break;
		case 4: // 14 * 2-bit
		    mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[1]);

		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 26);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[1]);
		    _mm_storeu_si128((__m128i *)out, rslt_m128i);

		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 18);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[1]);
		    _mm_storeu_si128((__m128i *)(out + 4 * 1), rslt_m128i);

		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 10);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[1]);
		    _mm_storeu_si128((__m128i *)(out + 4 * 2), rslt_m128i);

		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 2);
			rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[1]);
			_mm_storeu_si128((__m128i *)(out + 4 * 3), rslt_m128i);

			out += 14;

			break;
		case 5: // 1 * 4-bit + 8 * 3-bit
		    mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[3]);

		    // 1 * 4-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 24);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[3]);
		    _mm_storeu_si128((__m128i *)out, rslt_m128i);


		    mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[2]);

		    // 4 * 3-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 21);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[2]);
		    _mm_storeu_si128((__m128i *)(out + 1), rslt_m128i);

		    // 4 * 3-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 9);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[2]);
		    _mm_storeu_si128((__m128i *)(out + 1 + 4 * 1), rslt_m128i);

			out += 9;

			break;
		case 6: // 1 * 3-bit + 4 * 4-bit + 3 * 3-bit
		    mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[3]);

		    // 1 * 3-bit + 3 * 4-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 25);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_c6_and_msk_m128i[0]);
		    _mm_storeu_si128((__m128i *)out, rslt_m128i);


		    mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[2]);

		    // 1 * 4 + 3 * 3-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 9);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_c6_and_msk_m128i[1]);
		    _mm_storeu_si128((__m128i *)(out + 4 * 1), rslt_m128i);

			out += 8;

			break;
		case 7: // 7 * 4-bit
		    mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[3]);

		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 24);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[3]);
		    _mm_storeu_si128((__m128i *)out, rslt_m128i);

		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 8);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[3]);
		    _mm_storeu_si128((__m128i *)(out + 4 * 1), rslt_m128i);

			out += 7;

			break;
		case 8: // 4 * 5-bit + 2 * 4-bit
		    mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[4]);

		    // 4 * 5-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 23);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[4]);
		    _mm_storeu_si128((__m128i *)out, rslt_m128i);


		    mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[3]);

		    // 2 * 4-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 4);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[3]);
		    _mm_storeu_si128((__m128i *)(out + 4 * 1), rslt_m128i);

			out += 6;

			break;
		case 9: // 2 * 4-bit + 4 * 5-bit
		    mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[3]);

		    // 2 * 4-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 24);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[3]);
		    _mm_storeu_si128((__m128i *)out, rslt_m128i);


		    mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[4]);

		    // 4 * 5-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 15);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[4]);
		    _mm_storeu_si128((__m128i *)(out + 2), rslt_m128i);

			out += 6;

			break;
		case 10: // 3 * 6-bit + 2 * 5-bit
		    mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_c10_mul_m128i);

		    // 3 * 6-bit + 1 * 5-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 22);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_c10_and_msk_m128i[0]);
		    _mm_storeu_si128((__m128i *)out, rslt_m128i);

		    // 1 * 5-bit
		    rslt_m128i = _mm_and_si128(set1_rslt_m128i, Simple16_SSE_c10_and_msk_m128i[1]);
		    _mm_storeu_si128((__m128i *)(out + 4), rslt_m128i);

			out += 5;

			break;
		case 11: // 2 * 5-bit + 3 * 6-bit
		    mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_c11_mul_m128i);

		    // 2 * 5-bit + 2 * 6-bit
		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 23);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_c11_and_msk_m128i[0]);
		    _mm_storeu_si128((__m128i *)out, rslt_m128i);

		    // 1 * 6-bit
		    rslt_m128i = _mm_and_si128(set1_rslt_m128i, Simple16_SSE_c11_and_msk_m128i[1]);
		    _mm_storeu_si128((__m128i *)(out + 4), rslt_m128i);

			out += 5;

			break;
		case 12: // 4 * 7-bit
		    mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[5]);

		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 21);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[5]);
		    _mm_storeu_si128((__m128i *)out, rslt_m128i);

			out += 4;

			break;
		case 13: // 1 * 10-bit + 2 * 9-bit
			set1_rslt_m128i = _mm_set1_epi32(codeword);

		    mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[6]);

		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 18);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_c13_and_msk_m128i);
		    _mm_storeu_si128((__m128i *)out, rslt_m128i);

			out += 3;

			break;
		case 14: // 2 * 14-bit
		    mul_rslt_m128i = _mm_mullo_epi32(set1_rslt_m128i, Simple16_SSE_mul_msk_m128i[7]);

		    srli_rslt_m128i = _mm_srli_epi32(mul_rslt_m128i, 14);
		    rslt_m128i = _mm_and_si128(srli_rslt_m128i, Simple16_SSE_and_msk_m128i[7]);
		    _mm_storeu_si128((__m128i *)out, rslt_m128i);

			out += 2;

			break;
		case 15: // 1 * 28-bit
		    rslt_m128i = _mm_and_si128(set1_rslt_m128i, Simple16_SSE_and_msk_m128i[8]);
		    _mm_storeu_si128((__m128i *)out, rslt_m128i);

			++out;

			break;
		default:
			break;
		}
	}

	return in;
}


