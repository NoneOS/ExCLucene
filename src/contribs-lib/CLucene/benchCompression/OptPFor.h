/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */
/**  Based on code by
 *      Daniel Lemire, <lemire@gmail.com>
 *   which was available under the Apache License, Version 2.0.
 */

#ifndef OPTPFOR_H_
#define OPTPFOR_H_

#include "NewPFor.h"

template <uint32_t BlockSize = 256, typename Unpacker = HorizontalScalarUnpacker<false>,
typename TailBlockUnpacker = HorizontalScalarUnpacker<false>, typename ExceptionCoder = Simple16_Scalar> class OptPFor;

template <uint32_t BlockSize>
using OptPFor_Horizontal_Scalar = OptPFor<BlockSize, HorizontalScalarUnpacker<false>, HorizontalScalarUnpacker<false>, Simple16_Scalar>;

template <uint32_t BlockSize>
using OptPFor_Horizontal_SSE = OptPFor<BlockSize, HorizontalSSEUnpacker<false>, HorizontalSSEUnpacker<false>, Simple16_SSE>;

template <uint32_t BlockSize>
using OptPFor_Horizontal_AVX = OptPFor<BlockSize, HorizontalAVXUnpacker<false>, HorizontalAVXUnpacker<false>, Simple16_AVX>;


template <uint32_t BlockSize>
using OptPFor_Vertical_Scalar = OptPFor<BlockSize, VerticalScalarUnpacker<false>, HorizontalScalarUnpacker<false>, Simple16_Scalar>;

template <uint32_t BlockSize>
using OptPFor_Vertical_SSE = OptPFor<BlockSize, VerticalSSEUnpacker<false>, HorizontalSSEUnpacker<false>, Simple16_SSE>;

template <uint32_t BlockSize>
using OptPFor_Vertical_AVX = OptPFor<BlockSize, VerticalAVXUnpacker<false>, HorizontalAVXUnpacker<false>, Simple16_AVX>;

/**
 * OptPFD
 *
 * In a multithreaded context, you may need one OPTPFor per thread.
 *
 * Follows:
 *
 * H. Yan, S. Ding, T. Suel, Inverted index compression and query processing with
 * optimized document ordering, in: WWW '09, 2009, pp. 401-410.
 */
template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
class OptPFor : public NewPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder> {
public:
    virtual std::string name() const {
        std::ostringstream codecname;
        codecname << "OptPFor<"
        		<< BlockSize << ", " << this->unpacker.name() << ", "
				 << this->tbunpacker.name() << ", " << this->ecoder.name()
				<< ">";
        return codecname.str();
    }

    virtual ~OptPFor() = default;

private:
    virtual uint32_t tryB(uint32_t b, const uint32_t *in, uint32_t nvalue);
    virtual uint32_t findBestB(const uint32_t *in, uint32_t nvalue);
};


template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
__attribute__ ((pure))
uint32_t OptPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::tryB(uint32_t b, const uint32_t *in, uint32_t nvalue) {
    assert(b <= 32);

    if (b == 32) {
    	return nvalue;
    }

    uint32_t size = div_roundup(nvalue * b, 32);
    uint32_t nExceptions = 0;
    for (uint32_t i = 0; i < nvalue; ++i) {
        if (in[i] >= (1U << b)) {
            this->exceptionsPositions[nExceptions] = i;
            this->exceptionsValues[nExceptions] = (in[i] >> b);
            ++nExceptions;
        }
    }

    if (nExceptions > 0) {
        for (uint32_t i = nExceptions - 1; i > 0; --i) {
            const uint32_t cur = this->exceptionsPositions[i];
            const uint32_t prev = this->exceptionsPositions[i - 1];
            this->exceptionsPositions[i] = cur - prev - 1;
        }

        for (uint32_t i = 0; i < nExceptions; i++) {
            this->exceptions[i] = this->exceptionsPositions[i];
            this->exceptions[i + nExceptions] = this->exceptionsValues[i] - 1;
        }

        size_t encodedExceptionsSize = 0;
        this->ecoder.fakeencodeArray(&this->exceptions[0], 2 * nExceptions, encodedExceptionsSize);

        size += static_cast<uint32_t>(encodedExceptionsSize);
    }

    return size;
}

template <uint32_t BlockSize, typename HorizontalUnpacker, typename TailBlockUnpacker, typename ExceptionCoder>
__attribute__ ((pure))
uint32_t OptPFor<BlockSize, HorizontalUnpacker, TailBlockUnpacker, ExceptionCoder>::findBestB(const uint32_t *in, uint32_t nvalue) {
    uint32_t b = this->possLogs.back();
    assert(b == 32);

    uint32_t bsize = tryB(b, in, nvalue);
    const uint32_t mb = maxbits(in, in+nvalue);
    uint32_t i = 0;
    while(mb > 28 + this->possLogs[i]) ++i; // some schemes such as Simple16 don't code numbers greater than ((1 << 28) - 1)

    for (; i < this->possLogs.size() - 1; ++i) {
        const uint32_t csize = tryB(this->possLogs[i], in, nvalue);

        if (csize <= bsize) {
            b = this->possLogs[i];
            bsize = csize;
        }
    }
    return b;
}

#endif /* OPTPFOR_H_ */
