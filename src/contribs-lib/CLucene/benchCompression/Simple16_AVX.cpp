/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#include "Simple16_AVX.h"


/************************************ shift mask *************************************/

// case 0: 28 * 1-bit
const __m256i Simple16_AVX::Simple16_AVX_c0_srlv_msk_m256i[4] = {
		_mm256_set_epi32(20, 21, 22, 23, 24, 25, 26, 27),
		_mm256_set_epi32(12, 13, 14, 15, 16, 17, 18, 19),
		_mm256_set_epi32(4, 5, 6, 7, 8, 9, 10, 11),
		_mm256_set_epi32(0, 0, 0, 0, 0, 1, 2, 3)
};

// case 1: 7 * 2-bit + 14 * 1-bit
const __m256i Simple16_AVX::Simple16_AVX_c1_srlv_msk_m256i[3] = {
		_mm256_set_epi32(13, 14, 16, 18, 20, 22, 24, 26),
		_mm256_set_epi32(5, 6, 7, 8, 9, 10, 11, 12),
		_mm256_set_epi32(0, 0, 0, 0, 1, 2, 3, 4)
};

// case 2 : 7 * 1-bit + 7 * 2-bit + 7 * 1-bit
const __m256i Simple16_AVX::Simple16_AVX_c2_srlv_msk_m256i[3] = {
		_mm256_set_epi32(19, 21, 22, 23, 24, 25, 26, 27),
		_mm256_set_epi32(5, 6, 7, 9, 11, 13, 15, 17),
		_mm256_set_epi32(0, 0, 0, 0, 1, 2, 3, 4)
};

// case 3: 14 * 1-bit + 7 * 2-bit
const __m256i Simple16_AVX::Simple16_AVX_c3_srlv_msk_m256i[3] = {
		_mm256_set_epi32(20, 21, 22, 23, 24, 25, 26, 27),
		_mm256_set_epi32(10, 12, 14, 15, 16, 17, 18, 19),
		_mm256_set_epi32(0, 0, 0, 0, 2, 4, 6, 8)
};

// case 4 : 14 * 2-bit
const __m256i Simple16_AVX::Simple16_AVX_c4_srlv_msk_m256i[2] = {
		_mm256_set_epi32(12, 14, 16, 18, 20, 22, 24, 26),
		_mm256_set_epi32(0, 0, 0, 2, 4, 6, 8, 10)
};

// case 5: 1 * 4-bit + 8 * 3-bit
const __m256i Simple16_AVX::Simple16_AVX_c5_srlv_msk_m256i =
		_mm256_set_epi32(3, 6, 9, 12, 15, 18, 21, 24);

// case 6: 1 * 3-bit + 4 * 4-bit + 3 * 3-bit
const __m256i Simple16_AVX::Simple16_AVX_c6_srlv_msk_m256i =
		_mm256_set_epi32(0, 3, 6, 9, 13, 17, 21, 25);

// case 7: 7 * 4-bit
const __m256i Simple16_AVX::Simple16_AVX_c7_srlv_msk_m256i =
		_mm256_set_epi32(0, 0, 4, 8, 12, 16, 20, 24);

// case 8: 4 * 5-bit + 2 * 4-bit
const __m256i Simple16_AVX::Simple16_AVX_c8_srlv_msk_m256i =
		_mm256_set_epi32(0, 0, 0, 4, 8, 13, 18, 23);

// case 9: 2 * 4-bit + 4 * 5-bit
const __m256i Simple16_AVX::Simple16_AVX_c9_srlv_msk_m256i =
		_mm256_set_epi32(0, 0, 0, 5, 10, 15, 20, 24);

// case 10: 3 * 6-bit + 2 * 5-bit
const __m256i Simple16_AVX::Simple16_AVX_c10_srlv_msk_m256i =
		_mm256_set_epi32(0, 0, 0, 0, 5, 10, 16, 22);

// case 11: 2 * 5-bit + 3 * 6-bit
const __m256i Simple16_AVX::Simple16_AVX_c11_srlv_msk_m256i =
		_mm256_set_epi32(0, 0, 0, 0, 6, 12, 18, 23);

// case 12: 4 * 7-bit
const __m256i Simple16_AVX::Simple16_AVX_c12_srlv_msk_m256i =
		_mm256_set_epi32(0, 0, 0, 0, 0, 7, 14, 21);

// case 13: 1 * 10-bit + 2 * 9-bit
const __m256i Simple16_AVX::Simple16_AVX_c13_srlv_msk_m256i =
		_mm256_set_epi32(0, 0, 0, 0, 0, 0, 9, 18);

// case 14: 2 * 14-bit
const __m256i Simple16_AVX::Simple16_AVX_c14_srlv_msk_m256i =
		_mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 14);

/************************************ shift mask *************************************/


/************************************* and mask **************************************/

// case 0: 28 * 1-bit
const __m256i Simple16_AVX::Simple16_AVX_c0_and_msk_m256i =
		_mm256_set1_epi32(0x01);

// case 1: 7 * 2-bit + 14 * 1-bit
const __m256i Simple16_AVX::Simple16_AVX_c1_and_msk_m256i[3] = {
		_mm256_set_epi32(0x01, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03),
		_mm256_set1_epi32(0x01),
		_mm256_set1_epi32(0x01)
};

// case 2 : 7 * 1-bit + 7 * 2-bit + 7 * 1-bit
const __m256i Simple16_AVX::Simple16_AVX_c2_and_msk_m256i[3] = {
		_mm256_set_epi32(0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01),
		_mm256_set_epi32(0x01, 0x01, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03),
		_mm256_set1_epi32(0x01)
};

// case 3: 14 * 1-bit + 7 * 2-bit
const __m256i Simple16_AVX::Simple16_AVX_c3_and_msk_m256i[3] = {
		_mm256_set1_epi32(0x01),
		_mm256_set_epi32(0x03, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01),
		_mm256_set1_epi32(0x03)
};

// case 4: 14 * 2-bit
const __m256i Simple16_AVX::Simple16_AVX_c4_and_msk_m256i =
		_mm256_set1_epi32(0x03);

// case 5: 1 * 4-bit + 8 * 3-bit
const __m256i Simple16_AVX::Simple16_AVX_c5_and_msk_m256i[2] = {
		_mm256_set_epi32(0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x0F),
		_mm256_set1_epi32(0x07)
};

// case 6: 1 * 3-bit + 4 * 4-bit + 3 * 3-bit
const __m256i Simple16_AVX::Simple16_AVX_c6_and_msk_m256i =
		_mm256_set_epi32(0x07, 0x07, 0x07, 0x0F, 0x0F, 0x0F, 0x0F, 0x07);

// case 7: 7 * 4-bit
const __m256i Simple16_AVX::Simple16_AVX_c7_and_msk_m256i =
		_mm256_set1_epi32(0x0F);

// case 8: 4 * 5-bit + 2 * 4-bit
const __m256i Simple16_AVX::Simple16_AVX_c8_and_msk_m256i =
		_mm256_set_epi32(0x00, 0x00, 0x0F, 0x0F, 0x1F, 0x1F, 0x1F, 0x1F);

// case 9: 2 * 4-bit + 4 * 5-bit
const __m256i Simple16_AVX::Simple16_AVX_c9_and_msk_m256i =
		_mm256_set_epi32(0x00, 0x00, 0x1F, 0x1F, 0x1F, 0x1F, 0x0F, 0x0F);

// case 10: 3 * 6-bit + 2 * 5-bit
const __m256i Simple16_AVX::Simple16_AVX_c10_and_msk_m256i =
		_mm256_set_epi32(0x00, 0x00, 0x00, 0x1F, 0x1F, 0x3F, 0x3F, 0x3F);

// case 11: 2 * 5-bit + 3 * 6-bit
const __m256i Simple16_AVX::Simple16_AVX_c11_and_msk_m256i =
		_mm256_set_epi32(0x00, 0x00, 0x00, 0x3F, 0x3F, 0x3F, 0x1F, 0x1F);

// case 12: 4 * 7-bit
const __m256i Simple16_AVX::Simple16_AVX_c12_and_msk_m256i =
		_mm256_set1_epi32(0x7F);

// case 13: 1 * 10-bit + 2 * 9-bit
const __m256i Simple16_AVX::Simple16_AVX_c13_and_msk_m256i =
		_mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x01FF, 0x01FF, 0x03FF);

// case 14: 2 * 14-bit
const __m256i Simple16_AVX::Simple16_AVX_c14_and_msk_m256i =
		_mm256_set1_epi32(0x03FFF);

// case 15: 1 * 28-bit
const __m256i Simple16_AVX::Simple16_AVX_c15_and_msk_m256i =
		_mm256_set1_epi32(0xFFFFFFF);

/************************************* and mask **************************************/


const uint32_t * Simple16_AVX::decodeArray(const uint32_t *in, size_t csize,
		uint32_t *out, size_t nvalue) {
	__m256i srlv_rslt_m256i;
	__m256i rslt_m256i;

	const uint32_t *const endout(out + nvalue);
	while (endout > out) {
		const uint32_t codeword = in[0];
		++in;
		__m256i set1_rslt_m256i = _mm256_set1_epi32(codeword);
		const uint32_t descriptor = codeword >> (32 - SIMPLE16_LOGDESC);
		switch (descriptor) {
		case 0: // 28 * 1-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c0_srlv_msk_m256i[0]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c0_and_msk_m256i);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c0_srlv_msk_m256i[1]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c0_and_msk_m256i);
		    _mm256_storeu_si256((__m256i *)(out + 8 * 1), rslt_m256i);

		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c0_srlv_msk_m256i[2]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c0_and_msk_m256i);
		    _mm256_storeu_si256((__m256i *)(out + 8 * 2), rslt_m256i);

		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c0_srlv_msk_m256i[3]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c0_and_msk_m256i);
		    _mm256_storeu_si256((__m256i *)(out + 8 * 3), rslt_m256i);

			out += 28;

			break;
		case 1: // 7 * 2-bit + 14 * 1-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c1_srlv_msk_m256i[0]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c1_and_msk_m256i[0]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c1_srlv_msk_m256i[1]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c1_and_msk_m256i[1]);
		    _mm256_storeu_si256((__m256i *)(out + 8 * 1), rslt_m256i);

		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c1_srlv_msk_m256i[2]);
			rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c1_and_msk_m256i[2]);
			_mm256_storeu_si256((__m256i *)(out + 8 * 2), rslt_m256i);

			out += 21;

			break;
		case 2: // 7 * 1-bit + 7 * 2-bit + 7 * 1-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c2_srlv_msk_m256i[0]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c2_and_msk_m256i[0]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c2_srlv_msk_m256i[1]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c2_and_msk_m256i[1]);
		    _mm256_storeu_si256((__m256i *)(out + 8 * 1), rslt_m256i);

		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c2_srlv_msk_m256i[2]);
			rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c2_and_msk_m256i[2]);
			_mm256_storeu_si256((__m256i *)(out + 8 * 2), rslt_m256i);

			out += 21;

			break;
		case 3: // 14 * 1-bit + 7 * 2-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c3_srlv_msk_m256i[0]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c3_and_msk_m256i[0]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c3_srlv_msk_m256i[1]);
			rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c3_and_msk_m256i[1]);
			_mm256_storeu_si256((__m256i *)(out + 8 * 1), rslt_m256i);

			srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c3_srlv_msk_m256i[2]);
			rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c3_and_msk_m256i[2]);
			_mm256_storeu_si256((__m256i *)(out + 8 * 2), rslt_m256i);

			out += 21;

			break;
		case 4: // 14 * 2-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c4_srlv_msk_m256i[0]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c4_and_msk_m256i);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c4_srlv_msk_m256i[1]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c4_and_msk_m256i);
		    _mm256_storeu_si256((__m256i *)(out + 8 * 1), rslt_m256i);

			out += 14;

			break;
		case 5: // 1 * 4-bit + 8 * 3-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c5_srlv_msk_m256i);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c5_and_msk_m256i[0]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

		    rslt_m256i = _mm256_and_si256(set1_rslt_m256i, Simple16_AVX_c5_and_msk_m256i[1]);
		    _mm256_storeu_si256((__m256i *)(out + 8), rslt_m256i);

		    out += 9;

			break;
		case 6: // 1 * 3-bit + 4 * 4-bit + 3 * 3-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c6_srlv_msk_m256i);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c6_and_msk_m256i);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 8;

			break;
		case 7: // 7 * 4-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c7_srlv_msk_m256i);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c7_and_msk_m256i);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 7;

			break;
		case 8: // 4 * 5-bit + 2 * 4-bit
			srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c8_srlv_msk_m256i);
			rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c8_and_msk_m256i);
			_mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 6;

			break;
		case 9: // 2 * 4-bit + 4 * 5-bit
			srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c9_srlv_msk_m256i);
			rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c9_and_msk_m256i);
			_mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 6;

			break;
		case 10: // 3 * 6-bit + 2 * 5-bit
			srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c10_srlv_msk_m256i);
			rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c10_and_msk_m256i);
			_mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 5;

			break;
		case 11: // 2 * 5-bit + 3 * 6-bit
			srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c11_srlv_msk_m256i);
			rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c11_and_msk_m256i);
			_mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 5;

			break;
		case 12: // 4 * 7-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c12_srlv_msk_m256i);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c12_and_msk_m256i);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 4;

			break;
		case 13: // 1 * 10-bit + 2 * 9-bit
			srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c13_srlv_msk_m256i);
			rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c13_and_msk_m256i);
			_mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 3;

			break;
		case 14: // 2 * 14-bit
			srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple16_AVX_c14_srlv_msk_m256i);
			rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple16_AVX_c14_and_msk_m256i);
			_mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 2;

			break;
		case 15: // 1 * 28-bit
			rslt_m256i = _mm256_and_si256(set1_rslt_m256i, Simple16_AVX_c15_and_msk_m256i);
			_mm256_storeu_si256((__m256i *)out, rslt_m256i);

			++out;

			break;
		default: // impossible to get here
			break;
		}
	}

	return in;
}
