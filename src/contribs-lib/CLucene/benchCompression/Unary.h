/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef UNARY_H_
#define UNARY_H_

#include "IntegerCodec.h"

// Note: 0 indicates continuation, while 1 indicates termination
// Examples: 0 is represented as 1, 1 as 01, 2 as 001, and so on.
class Unary : public IntegerCodec {
public:
	virtual void encodeArray(const uint32_t *in, size_t nvalue,
			uint32_t *out, size_t &csize);

	// for use with OptRice
	// like encodeArray, but does not actually write out the data
	void fakeencodeArray(const uint32_t *in, size_t nvalue,
			size_t &csize);

	virtual const uint32_t * decodeArray(const uint32_t *in, size_t csize,
			uint32_t *out, size_t nvalue);

	std::string name() const {
		return "Unary";
	}
};

#endif /* UNARY_H_ */
