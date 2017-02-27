/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */
/*   Based on code by
 *      Daniel Lemire, <lemire@gmail.com>
 *   which was available under the Apache License, Version 2.0.
 */

#ifndef SIMPLE16_SCALAR_H_
#define SIMPLE16_SCALAR_H_

#include "IntegerCodec.h"

class Simple16_Scalar : public IntegerCodec {
public:
	virtual void encodeArray(const uint32_t *in, size_t nvalue,
			uint32_t *out, size_t &csize);

	// for use with OptPFor
	// like encodeArray, but does not actually write out the data
	void fakeencodeArray(const uint32_t *in, size_t nvalue,
			size_t &csize);

	virtual const uint32_t * decodeArray(const uint32_t *in, size_t csize,
			uint32_t *out, size_t nvalue);

	virtual std::string name() const {
		return "Simple16_Scalar";
	}

protected:
    enum {
        SIMPLE16_LOGDESC = 4, SIMPLE16_LEN = (1U << SIMPLE16_LOGDESC)
    };

private:
    template <uint32_t num1, uint32_t log1>
    __attribute__ ((pure))
    static bool trymefull(const uint32_t *in) {
        for (uint32_t i = 0; i < num1; ++i) {
            if (in[i] >= (1U << log1))
                return false;
        }
        return true;
    }

    template <uint32_t num1, uint32_t log1, uint32_t num2, uint32_t log2>
    __attribute__ ((pure))
    static bool trymefull(const uint32_t *in) {
        for (uint32_t i = 0; i < num1; ++i) {
            if (in[i] >= (1U << log1))
                return false;
        }

        for (uint32_t i = num1; i < num1 + num2; ++i) {
            if (in[i] >= (1U << log2))
                return false;
        }

        return true;
    }

    template <uint32_t num1, uint32_t log1, uint32_t num2, uint32_t log2,
              uint32_t num3, uint32_t log3>
    __attribute__ ((pure))
    static bool trymefull(const uint32_t *in) {
        for (uint32_t i = 0; i < num1; ++i) {
            if (in[i] >= (1U << log1))
                return false;
        }

        for (uint32_t i = num1; i < num1 + num2; ++i) {
            if (in[i] >= (1U << log2))
                return false;
        }

        for (uint32_t i = num1 + num2; i < num1 + num2 + num3; ++i) {
            if (in[i] >= (1U << log3))
                return false;
        }

        return true;
    }

    template <uint32_t num1, uint32_t log1>
    __attribute__ ((pure))
    static bool tryme(const uint32_t *in, uint32_t nvalue) {
        const uint32_t min1 = (nvalue < num1) ? nvalue : num1;
        for (uint32_t i = 0; i < min1; ++i) {
            if ((in[i]) >= (1U << log1))
                return false;
        }

        return true;
    }

    template <uint32_t num1, uint32_t log1, uint32_t num2, uint32_t log2>
    __attribute__ ((pure))
    static bool tryme(const uint32_t *in, uint32_t nvalue) {
        const uint32_t min1 = (nvalue < num1) ? nvalue : num1;
        for (uint32_t i = 0; i < min1; ++i) {
            if ((in[i]) >= (1U << log1))
                return false;
        }

        nvalue -= min1;
        const uint32_t min2 = (nvalue < num2) ? nvalue : num2;
        for (uint32_t i = min1; i < min1 + min2; ++i) {
            if ((in[i]) >= (1U << log2))
                return false;
        }

        return true;
    }

    template <uint32_t num1, uint32_t log1, uint32_t num2, uint32_t log2,
              uint32_t num3, uint32_t log3>
    __attribute__ ((pure))
    static bool tryme(const uint32_t *in, uint32_t nvalue) {
        const uint32_t min1 = (nvalue < num1) ? nvalue : num1;
        for (uint32_t i = 0; i < min1; ++i) {
            if ((in[i]) >= (1U << log1))
                return false;
        }

        nvalue -= min1;
        const uint32_t min2 = (nvalue < num2) ? nvalue : num2;
        for (uint32_t i = min1; i < min1 + min2; ++i) {
            if ((in[i]) >= (1U << log2))
                return false;
        }

        nvalue -= min2;
        const uint32_t min3 = (nvalue < num3) ? nvalue : num3;
        for (uint32_t i = min1 + min2; i < min1 + min2 + min3; ++i) {
            if ((in[i]) >= (1U << log3))
                return false;
        }

        return true;
    }


    static void descriptor_writer(uint32_t descriptor, uint32_t &codeword) {
    	codeword = descriptor << (32 - SIMPLE16_LOGDESC);
    }

    template <uint32_t num1, uint32_t log1>
    static void data_writer(const uint32_t *in, uint32_t nvalue, uint32_t &codeword) {
    	uint32_t shift = 32 - SIMPLE16_LOGDESC;
    	for (uint32_t i = 0; i < nvalue; ++i) {
    		shift -= log1;
    		codeword |= (in[i] << shift);
    	}
    }

    template <uint32_t num1, uint32_t log1, uint32_t num2, uint32_t log2>
    static void data_writer(const uint32_t *in, uint32_t nvalue, uint32_t &codeword) {
    	uint32_t shift = 32 - SIMPLE16_LOGDESC;
        const uint32_t min1 = (nvalue < num1) ? nvalue : num1;
		for (uint32_t i = 0; i < min1; ++i) {
			shift -= log1;
			codeword |= (in[i] << shift);
		}

    	for (uint32_t i = min1; i < nvalue; ++i) {
    		shift -= log2;
    		codeword |= (in[i] << shift);
    	}
    }

    template <uint32_t num1, uint32_t log1, uint32_t num2, uint32_t log2,
	          uint32_t num3, uint32_t log3>
    static void data_writer(const uint32_t *in, uint32_t nvalue, uint32_t &codeword) {
    	uint32_t shift = 32 - SIMPLE16_LOGDESC;
        const uint32_t min1 = (nvalue < num1) ? nvalue : num1;
		for (uint32_t i = 0; i < min1; ++i) {
			shift -= log1;
			codeword |= (in[i] << shift);
		}

		const uint32_t min2 = (nvalue - min1 < num2) ? nvalue - min1 : num2;
    	for (uint32_t i = min1; i < min1 + min2; ++i) {
    		shift -= log2;
    		codeword |= (in[i] << shift);
    	}

    	for (uint32_t i = min1 + min2; i < nvalue; ++i) {
			shift -= log3;
			codeword |= (in[i] << shift);
		}
    }
};

#endif /* SIMPLE16_SCALAR_H_ */
