/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef SIMPLE16_SSE_H_
#define SIMPLE16_SSE_H_

#include "Simple16_Scalar.h"

class Simple16_SSE : public Simple16_Scalar {
public:
	virtual const uint32_t * decodeArray(const uint32_t *in, size_t csize,
			uint32_t *out, size_t nvalue);

	virtual std::string name() const {
		return "Simple16_SSE";
	}

private:
	// multiplication mask
	static const __m128i Simple16_SSE_mul_msk_m128i[SIMPLE16_LEN];
	static const __m128i Simple16_SSE_c10_mul_m128i;
	static const __m128i Simple16_SSE_c11_mul_m128i;

	// and mask
	static const __m128i Simple16_SSE_and_msk_m128i[SIMPLE16_LEN];
	static const __m128i Simple16_SSE_c6_and_msk_m128i[2];
	static const __m128i Simple16_SSE_c10_and_msk_m128i[2];
	static const __m128i Simple16_SSE_c11_and_msk_m128i[2];
	static const __m128i Simple16_SSE_c13_and_msk_m128i;
};

#endif /* SIMPLE16_SSE_H_ */
