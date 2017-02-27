/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */
/**  Based on code by
 *      Daniel Lemire, <lemire@gmail.com>
 *   which was available under the Apache License, Version 2.0.
 */

#ifndef INTEGERCODEC_H_
#define INTEGERCODEC_H_

#include "common.h"

class IntegerCodec {
public:
    /**
     * You specify input and input length, as well as output and output length.
     * csize gets modified to reflect how many words were used.
     *
     * You are responsible for allocating the memory (nvalue for *in and csize for *out).
     */
	virtual void encodeArray(const uint32_t *in, size_t nvalue,
			uint32_t *out, size_t &csize) = 0;

    /**
     * Usage is similar to decodeArray except that it returns a pointer incremented from in.
     * In theory it should be in + csize. If the returned pointer is less than in + csize,
     * then this generally means that the decompression is not finished (some scheme compress
     * the bulk of the data one way, and then they compress remaining integers using another scheme).
     *
     * As with encodeArray, you need to have csize element allocated for *in and at least nvalue
     * elements allocated for out.
     */
	virtual const uint32_t * decodeArray(const uint32_t *in, size_t csize,
			uint32_t *out, size_t nvalue) = 0;

	virtual std::string name() const = 0;

	virtual ~IntegerCodec() = default;
};

/******************
 * This just copies the data, no compression.
 */
class JustCopy: public IntegerCodec {
public:
    virtual void encodeArray(const uint32_t * in, const size_t nvalue,
    		uint32_t * out, size_t &csize) {
        //if (length > nvalue)
        //    cerr << "It is possible we have a buffer overrun. " << endl;
        memcpy(out, in, sizeof(uint32_t) * nvalue);
        csize = nvalue;
    }

    virtual const uint32_t * decodeArray(const uint32_t *in, const size_t csize,
            uint32_t *out, size_t nvalue) {
        memcpy(out, in, sizeof(uint32_t) * nvalue);
        return in + csize;
    }

    virtual std::string name() const {
        return "JustCopy";
    }
};

#endif /* INTEGERCODEC_H_ */
