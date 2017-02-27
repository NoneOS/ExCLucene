/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#include "CodecFactory.h"

// C++11 allows better than this, but neither Microsoft nor Intel support C++11 fully.
static inline CodecMap initializefactory() {
    CodecMap cmap;

    // byte-aligned codecs
    auto p = std::shared_ptr<IntegerCodec>(new VarintG4B);
    cmap[p->name()] = p;
    p = std::shared_ptr<IntegerCodec>(new VarintG8B);
    cmap[p->name()] = p;
    p = std::shared_ptr<IntegerCodec>(new VarintG8IU);
	cmap[p->name()] = p;
	p = std::shared_ptr<IntegerCodec>(new VarintG8CU);
	cmap[p->name()] = p;

	// word-aligned codecs
	cmap["Simple9_Scalar"] = std::shared_ptr<IntegerCodec>(new Simple9_Scalar);
	cmap["Simple9_SSE"] =  std::shared_ptr<IntegerCodec>(new Simple9_SSE);
	cmap["Simple9_AVX"] = std::shared_ptr<IntegerCodec>(new Simple9_AVX);
	cmap["Simple16_Scalar"] = std::shared_ptr<IntegerCodec>(new Simple16_Scalar);
	cmap["Simple16_SSE"] =  std::shared_ptr<IntegerCodec>(new Simple16_SSE);
	cmap["Simple16_AVX"] = std::shared_ptr<IntegerCodec>(new Simple16_AVX);

	// fixed bit-width codecs
	cmap["NewPFor_Horizontal_Scalar"] = std::shared_ptr<IntegerCodec>(new NewPFor_Horizontal_Scalar<256>);
	cmap["NewPFor_Horizontal_SSE"] = std::shared_ptr<IntegerCodec>(new NewPFor_Horizontal_SSE<256>);
	cmap["NewPFor_Horizontal_AVX"] = std::shared_ptr<IntegerCodec>(new NewPFor_Horizontal_AVX<256>);
	cmap["NewPFor_Vertical_Scalar"] = std::shared_ptr<IntegerCodec>(new NewPFor_Vertical_Scalar<256>);
	cmap["NewPFor_Vertical_SSE"] = std::shared_ptr<IntegerCodec>(new NewPFor_Vertical_SSE<256>);
	cmap["NewPFor_Vertical_AVX"] = std::shared_ptr<IntegerCodec>(new NewPFor_Vertical_AVX<256>);

	cmap["OptPFor_Horizontal_Scalar"] = std::shared_ptr<IntegerCodec>(new OptPFor_Horizontal_Scalar<256>);
	cmap["OptPFor_Horizontal_SSE"] = std::shared_ptr<IntegerCodec>(new OptPFor_Horizontal_SSE<256>);
	cmap["OptPFor_Horizontal_AVX"] = std::shared_ptr<IntegerCodec>(new OptPFor_Horizontal_AVX<256>);
	cmap["OptPFor_Vertical_Scalar"] = std::shared_ptr<IntegerCodec>(new OptPFor_Vertical_Scalar<256>);
	cmap["OptPFor_Vertical_SSE"] = std::shared_ptr<IntegerCodec>(new OptPFor_Vertical_SSE<256>);
	cmap["OptPFor_Vertical_AVX"] = std::shared_ptr<IntegerCodec>(new OptPFor_Vertical_AVX<256>);

	// bit-oriented codecs
	cmap["Rice_Horizontal_Scalar"] = std::shared_ptr<IntegerCodec>(new Rice_Horizontal_Scalar<256>);
	cmap["Rice_Horizontal_SSE"] = std::shared_ptr<IntegerCodec>(new Rice_Horizontal_SSE<256>);
	cmap["Rice_Horizontal_AVX"] = std::shared_ptr<IntegerCodec>(new Rice_Horizontal_AVX<256>);
	cmap["Rice_Vertical_Scalar"] = std::shared_ptr<IntegerCodec>(new Rice_Vertical_Scalar<256>);
	cmap["Rice_Vertical_SSE"] = std::shared_ptr<IntegerCodec>(new Rice_Vertical_SSE<256>);
	cmap["Rice_Vertical_AVX"] = std::shared_ptr<IntegerCodec>(new Rice_Vertical_AVX<256>);

	cmap["OptRice_Horizontal_Scalar"] = std::shared_ptr<IntegerCodec>(new OptRice_Horizontal_Scalar<256>);
	cmap["OptRice_Horizontal_SSE"] = std::shared_ptr<IntegerCodec>(new OptRice_Horizontal_SSE<256>);
	cmap["OptRice_Horizontal_AVX"] = std::shared_ptr<IntegerCodec>(new OptRice_Horizontal_AVX<256>);
	cmap["OptRice_Vertical_Scalar"] = std::shared_ptr<IntegerCodec>(new OptRice_Vertical_Scalar<256>);
	cmap["OptRice_Vertical_SSE"] = std::shared_ptr<IntegerCodec>(new OptRice_Vertical_SSE<256>);
	cmap["OptRice_Vertical_AVX"] = std::shared_ptr<IntegerCodec>(new OptRice_Vertical_AVX<256>);

    return cmap;
}

CodecMap CodecFactory::scodecmap = initializefactory();


