/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */
/**  Based on code by
 *      Daniel Lemire, <lemire@gmail.com>
 *   which was available under the Apache License, Version 2.0.
 */

#ifndef NEWPFOR_H_
#define NEWPFOR_H_

#include "common.h"
#include "util.h"

#include "IntegerCodec.h"
#include "Simple16_Scalar.h"
#include "Simple16_SSE.h"
#include "Simple16_AVX.h"

#include "HorizontalScalarUnpacker.h"
#include "HorizontalSSEUnpacker.h"
#include "HorizontalAVXUnpacker.h"

#include "VerticalScalarUnpacker.h"
#include "VerticalSSEUnpacker.h"
#include "VerticalAVXUnpacker.h"


template <uint32_t BlockSize = 256, typename Unpacker = HorizontalScalarUnpacker<false>,
typename TailBlockUnpacker = HorizontalScalarUnpacker<false>, typename ExceptionCoder = Simple16_Scalar> class NewPFor;

template <uint32_t BlockSize>
using NewPFor_Horizontal_Scalar = NewPFor<BlockSize, HorizontalScalarUnpacker<false>, HorizontalScalarUnpacker<false>, Simple16_Scalar>;

template <uint32_t BlockSize>
using NewPFor_Horizontal_SSE = NewPFor<BlockSize, HorizontalSSEUnpacker<false>, HorizontalSSEUnpacker<false>, Simple16_SSE>;

template <uint32_t BlockSize>
using NewPFor_Horizontal_AVX = NewPFor<BlockSize, HorizontalAVXUnpacker<false>, HorizontalAVXUnpacker<false>, Simple16_AVX>;


template <uint32_t BlockSize>
using NewPFor_Vertical_Scalar = NewPFor<BlockSize, VerticalScalarUnpacker<false>, HorizontalScalarUnpacker<false>, Simple16_Scalar>;

template <uint32_t BlockSize>
using NewPFor_Vertical_SSE = NewPFor<BlockSize, VerticalSSEUnpacker<false>, HorizontalSSEUnpacker<false>, Simple16_SSE>;

template <uint32_t BlockSize>
using NewPFor_Vertical_AVX = NewPFor<BlockSize, VerticalAVXUnpacker<false>, HorizontalAVXUnpacker<false>, Simple16_AVX>;

/**
 * NewPFD also known as NewPFOR.
 *
 * In a multithreaded context, you may need one NewPFor per thread.
 *
 * Follows
 *
 * H. Yan, S. Ding, T. Suel, Inverted index compression and query processing with
 * optimized document ordering, in: WWW '09, 2009, pp. 401-410.
 */
template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
class NewPFor : public IntegerCodec {
public:
    virtual void encodeArray(const uint32_t *in, size_t nvalue,
            uint32_t *out, size_t &csize);

    template <bool IsTailBlock = false>
    void encodeBlock(const uint32_t *in, uint32_t nvalue,
    		uint32_t *out, uint32_t &csize);

    virtual const uint32_t * decodeArray(const uint32_t *in, size_t csize,
            uint32_t *out, size_t nvalue);

    template <bool IsTailBlock = false>
    const uint32_t * decodeBlock(const uint32_t* in, uint32_t csize,
    		uint32_t* out, uint32_t nvalue);

    virtual std::string name() const {
        std::ostringstream codecname;
        codecname << "NewPFor<"
        		<< BlockSize << ", " << unpacker.name() << ", "
				<< tbunpacker.name() << ", " << ecoder.name()
				<< ">";
        return codecname.str();
    }

    NewPFor() : exceptionsPositions(BlockSize), exceptionsValues(BlockSize),
	            exceptions(4 * BlockSize + TAIL_MERGIN + 1), tobecoded(BlockSize),
				unpacker(), tbunpacker(), ecoder() {
    	checkifdivisibleby(BlockSize, unpacker.PACKSIZE);
    }

protected:
    enum {
        PFORDELTA_B = 6,
        PFORDELTA_NEXCEPT = 10,
        PFORDELTA_EXCEPTSZ = 16,
        TAIL_MERGIN = 1024,
        PFORDELTA_RATIO = 10,     // exception ratio (expressed as a percent)
    };


    std::vector<uint32_t> exceptionsPositions;
    std::vector<uint32_t> exceptionsValues;
    std::vector<uint32_t> exceptions;
    std::vector<uint32_t> tobecoded;
    static const std::vector<uint32_t> possLogs;

    Unpacker unpacker;            // unpacker for all blocks except the last block
    TailBlockUnpacker tbunpacker; // unpacker for the last block
    ExceptionCoder ecoder;        // coder for exceptions' positions & values

private:
    virtual uint32_t tryB(uint32_t b, const uint32_t *in, uint32_t nvalue);
    virtual uint32_t findBestB(const uint32_t *in, uint32_t nvalue);
};

// nice compilers support this
template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
const std::vector<uint32_t> NewPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::possLogs =
        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
          18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32 };

///// this is for brain dead compilers:
//static inline std::vector<uint32_t> __ihatestupidcompilers() {
//	std::vector<uint32_t> ans;
//	ans.push_back(0); // I
//	ans.push_back(1); // hate
//	ans.push_back(2); // stupid
//	ans.push_back(3); // compilers
//	ans.push_back(4);
//	ans.push_back(5);
//	ans.push_back(6);
//	ans.push_back(7);
//	ans.push_back(8);
//	ans.push_back(9);
//	ans.push_back(10);
//	ans.push_back(11);
//	ans.push_back(12);
//	ans.push_back(13);
//	ans.push_back(16);
//	ans.push_back(20);
//	ans.push_back(32);
//	return ans;
//}
//
//template <uint32_t BlockSize, typename Unpacker, typename ExceptionCoder>
//std::vector<uint32_t> NewPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::possLogs = __ihatestupidcompilers();


template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
__attribute__ ((pure))
uint32_t NewPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::tryB(uint32_t b, const uint32_t *in, uint32_t nvalue) {
    assert(b <= 32);

    if (b == 32)
        return 0;

    uint32_t nExceptions = 0;
    for (uint32_t i = 0; i < nvalue; ++i) {
        if (in[i] >= (1U << b))
        	++nExceptions;
    }

    return nExceptions;
}

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
__attribute__ ((pure))
uint32_t NewPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::findBestB(const uint32_t *in, uint32_t nvalue) {
    const uint32_t mb = maxbits(in, in + nvalue);
    uint32_t i = 0;
    while(mb > 28 + possLogs[i]) ++i; // some schemes such as Simple16 don't code numbers greater than (1 << 28) - 1

    for ( ; i < possLogs.size() - 1; ++i) {
        const uint32_t nExceptions = tryB(possLogs[i], in, nvalue);
        if (nExceptions * 100 <= nvalue * PFORDELTA_RATIO)
            return possLogs[i];
    }
    return possLogs.back();
}

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
template <bool IsTailBlock>
void NewPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::encodeBlock(const uint32_t *in, uint32_t nvalue,
		uint32_t *out, uint32_t &csize) {
    uint32_t b = findBestB(in, nvalue);
    if (b < 32) {
    	uint32_t *const initout(out); // we use this later
    	++out;  // reserve space for the descriptor

    	uint32_t nExceptions = 0;
        for (uint32_t i = 0; i < nvalue; ++i) {
            if (in[i] >= (1U << b)) {
                tobecoded[i] = in[i] & ((1U << b) - 1);
                exceptionsPositions[nExceptions] = i;
                exceptionsValues[nExceptions] = (in[i] >> b);
                ++nExceptions;
            }
            else {
                tobecoded[i] = in[i];
            }
        }

        // pack non-exceptions
        if (!IsTailBlock) { // all blocks except the last block
        	for (uint32_t numberOfValuesPacked = 0; numberOfValuesPacked < BlockSize; numberOfValuesPacked += unpacker.PACKSIZE) {
        		unpacker.packwithoutmask(&tobecoded[numberOfValuesPacked], out, b);
        		out += (unpacker.PACKSIZE * b) / 32;
        	}
        }
        else {  // the last block
        	uint32_t numberOfValuesPacked = 0;
        	for ( ; numberOfValuesPacked + tbunpacker.PACKSIZE < nvalue; numberOfValuesPacked += tbunpacker.PACKSIZE) {
        		tbunpacker.packwithoutmask(&tobecoded[numberOfValuesPacked], out, b);
				out += (tbunpacker.PACKSIZE * b) / 32;
        	}

        	tbunpacker.packwithoutmask_generic(&tobecoded[numberOfValuesPacked], out, b, nvalue - numberOfValuesPacked);
        	out += div_roundup(b * (nvalue - numberOfValuesPacked), 32);
        }

        if (nExceptions > 0) {
            for (uint32_t i = nExceptions - 1; i > 0; --i) {
                const uint32_t cur = exceptionsPositions[i];
                const uint32_t prev = exceptionsPositions[i - 1];
                exceptionsPositions[i] = cur - prev - 1;
            }

            for (uint32_t i = 0; i < nExceptions; ++i) {
                exceptions[i] = exceptionsPositions[i];
                exceptions[i + nExceptions] = exceptionsValues[i] - 1;
            }
        }

        // pack exceptions' positions and values
        size_t encodedExceptionsSize = 0;
        if (nExceptions > 0)
            ecoder.encodeArray(&exceptions[0], 2 * nExceptions, out, encodedExceptionsSize);
        out += static_cast<uint32_t>(encodedExceptionsSize);

        // write descriptor
        *initout = (b << (32 - PFORDELTA_B)) |
        	   (nExceptions << PFORDELTA_EXCEPTSZ) |
				static_cast<uint32_t>(encodedExceptionsSize);

        csize = out - initout;
    }
    else { // b == 32
        *out = (b << (32 - PFORDELTA_B));
        ++out;
        for (uint32_t i = 0; i < nvalue; ++i)
            out[i] = in[i];

        csize =  1 + nvalue;
    }
}

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
void NewPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::encodeArray(const uint32_t *in, size_t nvalue,
		uint32_t *out, size_t &csize) {
	const uint32_t *const initout(out);

	uint32_t blockcsize = 0;
    size_t numBlocks = div_roundup(nvalue, BlockSize); // number of blocks in total; unnecessary to output it

    // for all blocks except the last block
    for (size_t i = 0; i < numBlocks - 1; ++i) {
        encodeBlock(in, BlockSize, out, blockcsize);
        in += BlockSize;
        out += blockcsize;
    }

    // for the last block
    uint32_t tailBlockSize = static_cast<uint32_t>( nvalue - (numBlocks - 1) * BlockSize );
    encodeBlock<true>(in, tailBlockSize, out, blockcsize);
    out += blockcsize;

    csize = out - initout;
}


template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
template <bool IsTailBlock>
const uint32_t * NewPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::decodeBlock(const uint32_t *in, uint32_t csize,
		uint32_t *out, uint32_t nvalue) {
    const uint32_t b = *in >> (32 - PFORDELTA_B);
    const uint32_t nExceptions = (*in >> PFORDELTA_EXCEPTSZ) & ((1U << PFORDELTA_NEXCEPT) - 1);
    const uint32_t encodedExceptionsSize = *in & ((1U << PFORDELTA_EXCEPTSZ) - 1);
    ++in;

    uint32_t *beginout(out); // we use this later

    if (!IsTailBlock) { // all blocks except the last block
    	for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < BlockSize; numberOfValuesUnpacked += unpacker.PACKSIZE) {
    		unpacker.unpack(in, out, b);
    		in += (unpacker.PACKSIZE * b) / 32;
    		out += unpacker.PACKSIZE;
    	}
    }
    else { // the last block
    	uint32_t numberOfValuesUnpacked = 0;
    	for ( ; numberOfValuesUnpacked + tbunpacker.PACKSIZE < nvalue; numberOfValuesUnpacked += tbunpacker.PACKSIZE) {
    		tbunpacker.unpack(in, out, b);
    		in += (tbunpacker.PACKSIZE * b) / 32;
    		out += tbunpacker.PACKSIZE;
    	}

    	tbunpacker.unpack(in, out, b);
    	in += div_roundup((nvalue - numberOfValuesUnpacked) * b, 32);
    }

    if (nExceptions > 0) {
        ecoder.decodeArray(in, encodedExceptionsSize, &exceptions[0], 2 * nExceptions);
		in += encodedExceptionsSize;
	}

    // concatenating
    for (uint32_t e = 0, lpos = -1; e < nExceptions; ++e) {
        lpos += exceptions[e] + 1;
        beginout[lpos] |= (exceptions[e + nExceptions] + 1) << b;
    }

    return in;
}

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
const uint32_t * NewPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::decodeArray(const uint32_t *in, size_t csize,
		uint32_t *out, size_t nvalue) {
    const size_t numBlocks = div_roundup(nvalue, BlockSize); // number of blocks in total

    // for all blocks except the last block
    for (size_t i = 0; i < numBlocks - 1; ++i) {
        in = decodeBlock(in, 0, out, BlockSize);
        out += BlockSize;
    }

    // for the last block
    uint32_t tailBlockSize = nvalue - (numBlocks - 1) * BlockSize;
    in = decodeBlock<true>(in, 0, out, tailBlockSize);

    return in;
}

#endif /* NEWPFOR_H_ */
