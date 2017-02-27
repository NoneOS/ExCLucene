/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef OPTRICE_H_
#define OPTRICE_H_

#include "Rice.h"

template <uint32_t BlockSize = 256, typename Unpacker = HorizontalScalarUnpacker<true>,
typename TailBlockUnpacker = HorizontalScalarUnpacker<true>, typename UnaryCoder = Unary> class OptRice;

template <uint32_t BlockSize>
using OptRice_Horizontal_Scalar = OptRice<BlockSize, HorizontalScalarUnpacker<true>, HorizontalScalarUnpacker<true>, Unary>;

template <uint32_t BlockSize>
using OptRice_Horizontal_SSE = OptRice<BlockSize, HorizontalSSEUnpacker<true>, HorizontalSSEUnpacker<true>, Unary>;

template <uint32_t BlockSize>
using OptRice_Horizontal_AVX = OptRice<BlockSize, HorizontalAVXUnpacker<true>, HorizontalAVXUnpacker<true>, Unary>;


template <uint32_t BlockSize>
using OptRice_Vertical_Scalar = OptRice<BlockSize, VerticalScalarUnpacker<true>, HorizontalScalarUnpacker<true>, Unary>;

template <uint32_t BlockSize>
using OptRice_Vertical_SSE = OptRice<BlockSize, VerticalSSEUnpacker<true>, HorizontalSSEUnpacker<true>, Unary>;

template <uint32_t BlockSize>
using OptRice_Vertical_AVX = OptRice<BlockSize, VerticalAVXUnpacker<true>, HorizontalAVXUnpacker<true>, Unary>;


template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
class OptRice : public Rice<BlockSize, Unpacker, TailBlockUnpacker, UnaryCoder> {
public:
	virtual std::string name() const {
		std::ostringstream codecname;
		codecname << "OptRice<"
				<< BlockSize << ", " << this->unpacker.name() << ", "
				<< this->tbunpacker.name() << ", " << this->ucoder.name()
				<< ">";
		return codecname.str();
	}

	virtual ~OptRice() = default;

private:
	virtual uint32_t findBestB(const uint32_t *in, uint32_t nvalue);
};


template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
uint32_t OptRice<BlockSize, Unpacker, TailBlockUnpacker, UnaryCoder>::findBestB(const uint32_t *in, uint32_t nvalue) {
	uint32_t b = 32;
	size_t bsize = nvalue;

	for (uint32_t c = 0; c < 32; ++c) {
		for (uint32_t i = 0; i < nvalue; ++i) {
			this->quotient[i] = in[i] >> c;
			this->remainder[i] = in[i] & ((1U << c) - 1);
		}

		size_t csize = 0;
		this->ucoder.fakeencodeArray(&this->quotient[0], nvalue, csize);
		csize += div_roundup(nvalue * c, 32);
		if (csize < bsize) {
			b = c;
			bsize = csize;
		}
	}
	return b;
}

#endif /* OPTRICE_H_ */
