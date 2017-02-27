/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef RICE_H_
#define RICE_H_

#include "common.h"
#include "util.h"

#include "IntegerCodec.h"
#include "Unary.h"

#include "HorizontalScalarUnpacker.h"
#include "HorizontalSSEUnpacker.h"
#include "HorizontalAVXUnpacker.h"

#include "VerticalScalarUnpacker.h"
#include "VerticalSSEUnpacker.h"
#include "VerticalAVXUnpacker.h"


template <uint32_t BlockSize = 256, typename Unpacker = HorizontalScalarUnpacker<true>,
typename TailBlockUnpacker = HorizontalScalarUnpacker<true>, typename UnaryCoder = Unary> class Rice;

template <uint32_t BlockSize>
using Rice_Horizontal_Scalar = Rice<BlockSize, HorizontalScalarUnpacker<true>, HorizontalScalarUnpacker<true>, Unary>;

template <uint32_t BlockSize>
using Rice_Horizontal_SSE = Rice<BlockSize, HorizontalSSEUnpacker<true>, HorizontalSSEUnpacker<true>, Unary>;

template <uint32_t BlockSize>
using Rice_Horizontal_AVX = Rice<BlockSize, HorizontalAVXUnpacker<true>, HorizontalAVXUnpacker<true>, Unary>;


template <uint32_t BlockSize>
using Rice_Vertical_Scalar = Rice<BlockSize, VerticalScalarUnpacker<true>, HorizontalScalarUnpacker<true>, Unary>;

template <uint32_t BlockSize>
using Rice_Vertical_SSE = Rice<BlockSize, VerticalSSEUnpacker<true>, HorizontalSSEUnpacker<true>, Unary>;

template <uint32_t BlockSize>
using Rice_Vertical_AVX = Rice<BlockSize, VerticalAVXUnpacker<true>, HorizontalAVXUnpacker<true>, Unary>;


template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
class Rice : public IntegerCodec {
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
		codecname << "Rice<"
				<< BlockSize << ", " << unpacker.name() << ", "
				<< tbunpacker.name() << ", " << ucoder.name()
				<< ">";
		return codecname.str();
	}

	Rice() : quotient(4 * BlockSize + TAIL_MERGIN), remainder(BlockSize),
			 unpacker(&quotient[0]), tbunpacker(&quotient[0]), ucoder() {
		checkifdivisibleby(BlockSize, unpacker.PACKSIZE);
	}

protected:
	enum {
		RICE_B = 6,
		RICE_UNARYSZ = 26,
        TAIL_MERGIN = 1024,
		RICE_RATIO = 69     // expressed as a percent
	};

	std::vector<uint32_t> quotient;
	std::vector<uint32_t> remainder;

	Unpacker unpacker;
	TailBlockUnpacker tbunpacker;
	UnaryCoder ucoder;

private:
	virtual uint32_t findBestB(const uint32_t *in, uint32_t nvalue);
};


template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
uint32_t Rice<BlockSize, Unpacker, TailBlockUnpacker, UnaryCoder>::findBestB(const uint32_t *in, uint32_t nvalue) {
	double avg = std::accumulate(in, in + nvalue, 0.0) / nvalue;
	uint32_t b = gccbits(static_cast<uint32_t>(RICE_RATIO / 100.0 * avg));
	return b;
}

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
template <bool IsTailBlock>
void Rice<BlockSize, Unpacker, TailBlockUnpacker, UnaryCoder>::encodeBlock(const uint32_t *in, uint32_t nvalue,
		uint32_t *out, uint32_t &csize) {
	const uint32_t b = findBestB(in, nvalue);
	if (b < 32) {
		uint32_t *const initout(out); // we use this later
		++out;

		for (uint32_t i = 0; i < nvalue; ++i) {
			quotient[i] = in[i] >> b;
			remainder[i] = in[i] & ((1U << b) - 1);
		}

		size_t encodedQuotientSize = 0;
		ucoder.encodeArray(&quotient[0], nvalue, out, encodedQuotientSize);
		out += encodedQuotientSize;

		if (!IsTailBlock) { // for all blocks except the last block
			for (uint32_t numberOfValuesPacked = 0; numberOfValuesPacked < BlockSize; numberOfValuesPacked += unpacker.PACKSIZE) {
				unpacker.packwithoutmask(&remainder[numberOfValuesPacked], out, b);
				out += (unpacker.PACKSIZE * b) / 32;
			}
		}
		else { // the last block
			uint32_t numberOfValuesPacked = 0;
			for ( ; numberOfValuesPacked + tbunpacker.PACKSIZE < nvalue; numberOfValuesPacked += tbunpacker.PACKSIZE) {
				tbunpacker.packwithoutmask(&remainder[numberOfValuesPacked], out, b);
				out += (tbunpacker.PACKSIZE * b) / 32;
			}

			tbunpacker.packwithoutmask_generic(&remainder[numberOfValuesPacked], out, b, nvalue - numberOfValuesPacked);
			out += div_roundup((nvalue - numberOfValuesPacked) * b, 32);
		}

		// write descriptor
		*initout = (b << (32 - RICE_B)) | static_cast<uint32_t>(encodedQuotientSize);

		csize = out - initout;
	}
	else { // b == 32 (quotient part will be all 0s)
        *out = (b << (32 - RICE_B));
        ++out;
        for (uint32_t i = 0; i < nvalue; ++i) {
            out[i] = in[i];
        }

        csize =  1 + nvalue;
	}
}


template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
void Rice<BlockSize, Unpacker, TailBlockUnpacker, UnaryCoder>::encodeArray(const uint32_t *in, size_t nvalue,
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


template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
template <bool IsTailBlock>
const uint32_t * Rice<BlockSize, Unpacker, TailBlockUnpacker, UnaryCoder>::decodeBlock(const uint32_t *in, uint32_t cszie,
		uint32_t *out, uint32_t nvalue) {
	const uint32_t b = *in >> (32 - RICE_B);
	const size_t encodedQuotientSize = *in & ((1U << RICE_UNARYSZ) - 1);
	++in;

	if (b < 32) {
		ucoder.decodeArray(in, encodedQuotientSize, &quotient[0], nvalue);
		in += encodedQuotientSize;
	}

	if (!IsTailBlock) { // all blocks except the last block
		unpacker.beginquotient = &quotient[0];
		for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < BlockSize; numberOfValuesUnpacked += unpacker.PACKSIZE) {
			unpacker.unpack(in, out, b);
			in += (unpacker.PACKSIZE * b) / 32;
			out += unpacker.PACKSIZE;
			unpacker.beginquotient += unpacker.PACKSIZE;
		}
	}
	else { // the last block
		tbunpacker.beginquotient = &quotient[0];
		uint32_t numberOfValuesUnpacked = 0;
		for ( ; numberOfValuesUnpacked + tbunpacker.PACKSIZE < nvalue; numberOfValuesUnpacked += tbunpacker.PACKSIZE) {
			tbunpacker.unpack(in, out, b);
			in += (tbunpacker.PACKSIZE * b) / 32;
			out += tbunpacker.PACKSIZE;
			tbunpacker.beginquotient += tbunpacker.PACKSIZE;
		}

		tbunpacker.unpack(in, out, b);
		in += div_roundup((nvalue - numberOfValuesUnpacked) * b, 32);
	}

	return in;
}

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
const uint32_t * Rice<BlockSize, Unpacker, TailBlockUnpacker, UnaryCoder>::decodeArray(const uint32_t *in, size_t csize,
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

#endif /* RICE_H_ */
