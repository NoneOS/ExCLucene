/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */
/*  Based on code by
 *      Daniel Lemire, <lemire@gmail.com>
 *   which was available under the Apache License, Version 2.0.
 */

#ifndef SIMPLE9_SCALAR_H_
#define SIMPLE9_SCALAR_H_

#include "IntegerCodec.h"

class Simple9_Scalar : public IntegerCodec {
public:
	virtual void encodeArray(const uint32_t *in, size_t nvalue,
			uint32_t *out, size_t &csize);

	virtual const uint32_t * decodeArray(const uint32_t *in, size_t csize,
			uint32_t *out, size_t nvalue);

	virtual std::string name() const {
		return "Simple9_Scalar";
	}

protected:
    enum {
        SIMPLE9_LOGDESC = 4, SIMPLE9_LEN = 9
    };

private:
    template <uint32_t num1, uint32_t log1>
    __attribute__ ((pure))
    static bool trymefull(const uint32_t *in) {
        for (uint32_t i = 0; i < num1; ++i) {
            if ((in[i]) >= (1U << log1))
                return false;
        }
        return true;
    }

    template <uint32_t num1, uint32_t log1>
    __attribute__ ((pure))
    static bool tryme(const uint32_t *in, uint32_t nvalue) {
        const uint32_t min = (nvalue < num1) ? nvalue : num1;
        for (uint32_t i = 0; i < min; ++i) {
            if ((in[i]) >= (1U << log1))
                return false;
        }
        return true;
    }


    static void descriptor_writer(uint32_t descriptor, uint32_t &codeword) {
    	codeword = descriptor << (32 - SIMPLE9_LOGDESC);
    }

    template <uint32_t num1, uint32_t log1>
    static void data_writer(const uint32_t *in, uint32_t nvalue, uint32_t &codeword) {
    	uint32_t shift = 32 - SIMPLE9_LOGDESC;
    	for (uint32_t i = 0; i < nvalue; ++i) {
    		shift -= log1;
    		codeword |= (in[i] << shift);
    	}
    }
};

#endif /* SIMPLE9_SCALAR_H_ */
