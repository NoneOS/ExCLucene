#include "BlockCodecs.hpp"

namespace FastPForLib {
	optpfor_block::codec_type optpfor_block::optpfor_codec;
	Simple16<false> optpfor_block::simple_codec;
	TightVariableByte optpfor_block::vbyte_codec;
}
