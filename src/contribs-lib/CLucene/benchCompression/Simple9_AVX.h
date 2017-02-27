/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef SIMPLE9_AVX_H_
#define SIMPLE9_AVX_H_

#include "Simple9_Scalar.h"

class Simple9_AVX : public Simple9_Scalar {
public:
	virtual const uint32_t * decodeArray(const uint32_t *in, size_t csize,
			uint32_t *out, size_t nvalue);

	virtual std::string name() const {
		return "Simple9_AVX";
	}

private:
	// shift mask
	static const __m256i Simple9_AVX_c0_shift_msk_m256i[4];
	static const __m256i Simple9_AVX_c1_shift_msk_m256i[2];
	static const __m256i Simple9_AVX_c2_shift_msk_m256i[2];
	static const __m256i Simple9_AVX_c3_shift_msk_m256i;
	static const __m256i Simple9_AVX_c4_shift_msk_m256i;
	static const __m256i Simple9_AVX_c5_shift_msk_m256i;
	static const __m256i Simple9_AVX_c6_shift_msk_m256i;
	static const __m256i Simple9_AVX_c7_shift_msk_m256i;

	// and mask
	static const __m256i Simple9_AVX_and_msk_m256i[SIMPLE9_LEN];
};

#endif /* SIMPLE9_AVX_H_ */
