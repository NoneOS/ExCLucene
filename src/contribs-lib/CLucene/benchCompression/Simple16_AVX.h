/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef SIMPLE16_AVX_H_
#define SIMPLE16_AVX_H_

#include "Simple16_Scalar.h"


class Simple16_AVX : public Simple16_Scalar {
public:
	virtual const uint32_t * decodeArray(const uint32_t *in, size_t csize,
			uint32_t *out, size_t nvalue);

	virtual std::string name() const {
		return "Simple16_AVX";
	}

private:
	// shift mask
	static const __m256i Simple16_AVX_c0_srlv_msk_m256i[4];
	static const __m256i Simple16_AVX_c1_srlv_msk_m256i[3];
	static const __m256i Simple16_AVX_c2_srlv_msk_m256i[3];
	static const __m256i Simple16_AVX_c3_srlv_msk_m256i[3];
	static const __m256i Simple16_AVX_c4_srlv_msk_m256i[2];
	static const __m256i Simple16_AVX_c5_srlv_msk_m256i;
	static const __m256i Simple16_AVX_c6_srlv_msk_m256i;
	static const __m256i Simple16_AVX_c7_srlv_msk_m256i;
	static const __m256i Simple16_AVX_c8_srlv_msk_m256i;
	static const __m256i Simple16_AVX_c9_srlv_msk_m256i;
	static const __m256i Simple16_AVX_c10_srlv_msk_m256i;
	static const __m256i Simple16_AVX_c11_srlv_msk_m256i;
	static const __m256i Simple16_AVX_c12_srlv_msk_m256i;
	static const __m256i Simple16_AVX_c13_srlv_msk_m256i;
	static const __m256i Simple16_AVX_c14_srlv_msk_m256i;

	// and mask
	static const __m256i Simple16_AVX_c0_and_msk_m256i;
	static const __m256i Simple16_AVX_c1_and_msk_m256i[3];
	static const __m256i Simple16_AVX_c2_and_msk_m256i[3];
	static const __m256i Simple16_AVX_c3_and_msk_m256i[3];
	static const __m256i Simple16_AVX_c4_and_msk_m256i;
	static const __m256i Simple16_AVX_c5_and_msk_m256i[2];
	static const __m256i Simple16_AVX_c6_and_msk_m256i;
	static const __m256i Simple16_AVX_c7_and_msk_m256i;
	static const __m256i Simple16_AVX_c8_and_msk_m256i;
	static const __m256i Simple16_AVX_c9_and_msk_m256i;
	static const __m256i Simple16_AVX_c10_and_msk_m256i;
	static const __m256i Simple16_AVX_c11_and_msk_m256i;
	static const __m256i Simple16_AVX_c12_and_msk_m256i;
	static const __m256i Simple16_AVX_c13_and_msk_m256i;
	static const __m256i Simple16_AVX_c14_and_msk_m256i;
	static const __m256i Simple16_AVX_c15_and_msk_m256i;
};


#endif /* SIMPLE16_AVX_H_ */
