/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#include "Simple9_AVX.h"


/************************* shift mask ***************************/

// case 0: 28 * 1-bit
const __m256i Simple9_AVX::Simple9_AVX_c0_shift_msk_m256i[4] = {
		_mm256_set_epi32(20, 21, 22, 23, 24, 25, 26, 27),
		_mm256_set_epi32(12, 13, 14, 15, 16, 17, 18, 19),
		_mm256_set_epi32(4, 5, 6, 7, 8, 9, 10, 11),
		_mm256_set_epi32(0, 0, 0, 0, 0, 1, 2, 3)
};

// case 1: 14 * 2-bit
const __m256i Simple9_AVX::Simple9_AVX_c1_shift_msk_m256i[2] = {
		_mm256_set_epi32(12, 14, 16, 18, 20, 22, 24, 26),
		_mm256_set_epi32(0, 0, 0, 2, 4, 6, 8, 10)
};

// case 2: 9 * 3-bit
const __m256i Simple9_AVX::Simple9_AVX_c2_shift_msk_m256i[2] = {
		_mm256_set_epi32(4, 7, 10, 13, 16, 19, 22, 25),
		_mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 1)
};

// case 3: 7 * 4-bit
const __m256i Simple9_AVX::Simple9_AVX_c3_shift_msk_m256i =
		_mm256_set_epi32(0, 0, 4, 8, 12, 16, 20, 24);

// case 4: 5 * 5-bit
const __m256i Simple9_AVX::Simple9_AVX_c4_shift_msk_m256i =
		_mm256_set_epi32(0, 0, 0, 3, 8, 13, 18, 23);

// case 5: 4 * 7-bit
const __m256i Simple9_AVX::Simple9_AVX_c5_shift_msk_m256i =
		_mm256_set_epi32(0, 0, 0, 0, 0, 7, 14, 21);

// case 6: 3 * 9-bit
const __m256i Simple9_AVX::Simple9_AVX_c6_shift_msk_m256i =
		_mm256_set_epi32(0, 0, 0, 0, 0, 1, 10, 19);

// case 7: 2 * 14-bit
const __m256i Simple9_AVX::Simple9_AVX_c7_shift_msk_m256i =
		_mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 14);

/************************** shift mask ***************************/


/*************************** and mask ****************************/

const __m256i Simple9_AVX::Simple9_AVX_and_msk_m256i[9] = {
		_mm256_set1_epi32(0x01),      // 1-bit
		_mm256_set1_epi32(0x03),      // 2-bit
		_mm256_set1_epi32(0x07),      // 3-bit
		_mm256_set1_epi32(0x0F),      // 4-bit
		_mm256_set1_epi32(0x1F),      // 5-bit
		_mm256_set1_epi32(0x7F),      // 7-bit
		_mm256_set1_epi32(0x01FF),    // 9-bit
		_mm256_set1_epi32(0x3FFF),    // 14-bit
		_mm256_set1_epi32(0xFFFFFFF)  // 28-bit
};

/************************** and mask ***************************/


const uint32_t * Simple9_AVX::decodeArray(const uint32_t *in, size_t csize,
		uint32_t *out, size_t nvalue) {
	__m256i set1_rslt_m256i;
	__m256i srlv_rslt_m256i;
	__m256i rslt_m256i;

	const uint32_t *const endout(out + nvalue);
	while (endout > out) {
		const uint32_t codeword = in[0];
		++in;
		set1_rslt_m256i = _mm256_set1_epi32(codeword);
		const uint32_t descriptor = codeword >> (32 - SIMPLE9_LOGDESC);
		switch (descriptor) {
		case 0: // 28 * 1-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple9_AVX_c0_shift_msk_m256i[0]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple9_AVX_and_msk_m256i[0]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple9_AVX_c0_shift_msk_m256i[1]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple9_AVX_and_msk_m256i[0]);
		    _mm256_storeu_si256((__m256i *)(out + 8 * 1), rslt_m256i);

		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple9_AVX_c0_shift_msk_m256i[2]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple9_AVX_and_msk_m256i[0]);
		    _mm256_storeu_si256((__m256i *)(out + 8 * 2), rslt_m256i);

		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple9_AVX_c0_shift_msk_m256i[3]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple9_AVX_and_msk_m256i[0]);
		    _mm256_storeu_si256((__m256i *)(out + 8 * 3), rslt_m256i);

			out += 28;

			break;
		case 1: // 14 * 2-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple9_AVX_c1_shift_msk_m256i[0]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple9_AVX_and_msk_m256i[1]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple9_AVX_c1_shift_msk_m256i[1]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple9_AVX_and_msk_m256i[1]);
		    _mm256_storeu_si256((__m256i *)(out + 8 * 1), rslt_m256i);

			out += 14;

			break;
		case 2: // 9 * 3-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple9_AVX_c2_shift_msk_m256i[0]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple9_AVX_and_msk_m256i[2]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple9_AVX_c2_shift_msk_m256i[1]);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple9_AVX_and_msk_m256i[2]);
		    _mm256_storeu_si256((__m256i *)(out + 8 * 1), rslt_m256i);

			out += 9;

			break;
		case 3: // 7 * 4-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple9_AVX_c3_shift_msk_m256i);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple9_AVX_and_msk_m256i[3]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 7;

			break;
		case 4: // 5 * 5-bit
			srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple9_AVX_c4_shift_msk_m256i);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple9_AVX_and_msk_m256i[4]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 5;

			break;
		case 5: // 4 * 7-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple9_AVX_c5_shift_msk_m256i);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple9_AVX_and_msk_m256i[5]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 4;

			break;
		case 6: // 3 * 9-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple9_AVX_c6_shift_msk_m256i);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple9_AVX_and_msk_m256i[6]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 3;

			break;
		case 7: // 2 * 14-bit
		    srlv_rslt_m256i = _mm256_srlv_epi32(set1_rslt_m256i, Simple9_AVX_c7_shift_msk_m256i);
		    rslt_m256i = _mm256_and_si256(srlv_rslt_m256i, Simple9_AVX_and_msk_m256i[7]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

			out += 2;

			break;
		case 8: // 1 * 28-bit
		    rslt_m256i = _mm256_and_si256(set1_rslt_m256i, Simple9_AVX_and_msk_m256i[8]);
		    _mm256_storeu_si256((__m256i *)out, rslt_m256i);

			++out;

			break;
		default: // invalid descriptor
			std::cerr << "Invalid descriptor: " << descriptor << std::endl;
			throw std::runtime_error("Invalid descriptor for " + name() + ".");
		}
	}

	return in;
}
