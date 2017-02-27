/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef SIMPLE9_SSE_H_
#define SIMPLE9_SSE_H_

#include "Simple9_Scalar.h"

class Simple9_SSE : public Simple9_Scalar {
public:
	virtual const uint32_t * decodeArray(const uint32_t *in, size_t csize,
			uint32_t *out, size_t nvalue);

	virtual std::string name() const {
		return "Simple9_SSE";
	}

private:
	// multiplication mask
	static const __m128i Simple9_SSE_mul_msk_m128i[SIMPLE9_LEN];

	// and mask
	static const __m128i Simple9_SSE_and_msk_m128i[SIMPLE9_LEN];
};

#endif /* SIMPLE9_SSE_H_ */
