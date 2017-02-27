/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef VARINTG8CU_H_
#define VARINTG8CU_H_

#include "common.h"
#include "IntegerCodec.h"
#include "util.h"

#ifdef __SSSE3__
#include <x86intrin.h>
namespace VarintTables {
extern const __m128i VarintG8CUSSEMasks[4][256][2];
} // namespace VarintTables
#endif

namespace VarintTables {
extern const uint8_t VarintG8CUOutputOffset[4][256];
extern const uint8_t VarintG8CUState[256];
extern const uint8_t VarintG8CULengths[256][8];
extern const uint32_t kMask[4];
} // namespace VarintTables

class VarintG8CU : public IntegerCodec {

//#undef __AVX2__
//#undef __SSSE3__

public:
	/**
	 * group details:
	 * kGroupSize - number of bytes for data parts,
	 *              which is also the maximum number of integers that a group can encode.
	 * kHeaderSize - number of bytes for the descriptor.
	 * KFullGroupSize - group size in units of bytes.
	 */
    enum {
    	kHeaderSize = 1,
    	kGroupSize = 8,
		kFullGroupSize = kHeaderSize + kGroupSize
    };

    virtual void encodeArray(const uint32_t *in, size_t nvalue,
                uint32_t *out, size_t &csize) {
        uint8_t *dst = reinterpret_cast<uint8_t *>(out);
        const uint8_t *const initdst = dst;
        uint8_t state = 0;
        while (nvalue > 0) {
            encode(in, nvalue, dst, state);
            dst += kFullGroupSize;
        }

        csize = ((dst - initdst) + 3) / 4; // number of words consumed
    }

    virtual const uint32_t * decodeArray(const uint32_t *in, size_t csize,
                uint32_t *out, size_t nvalue) {
        const uint8_t *src = reinterpret_cast<const uint8_t *>(in);
        const uint8_t *const initsrc = src;
        uint8_t *dst = reinterpret_cast<uint8_t *>(out);
        const uint8_t* const enddst = reinterpret_cast<uint8_t *>(out + nvalue);
        uint8_t state = 0;
        while (enddst > dst) {
            decode(src, dst, state);
    		src += kFullGroupSize;
        }

        csize = ((src - initsrc) + 3) / 4;
        return in + csize;
    }

    virtual std::string name() const {
        std::ostringstream codecname;
        std::string platform = "";
#ifdef __AVX2__
        platform = "AVX";
#else /* !__AVX2__ */
#ifdef __SSSE3__
        platform = "SSE";
#else /* !__AVX2__ && !__SSSE3__ */
        platform = "Scalar";
#endif /* __SSSE3__ */
#endif /* __AVX2__ */
        codecname << "VarintG8CU_" << platform;
        return codecname.str();
    }

    static void encode(const uint32_t * &src, size_t &nvalue, uint8_t *dst, uint8_t &state) {
    	uint8_t *const initdst = dst;
    	++dst; // reserve space for the descriptor byte

    	uint8_t desc = 0;
    	uint32_t totalBytes = 0;

    	while (nvalue > 0) {
			uint8_t bytes = getNumberOfBytes(src[0]) - state;
			totalBytes += bytes;
			if (totalBytes > kGroupSize) {
				break;
			}

			// flip the correct bit in descriptor
			desc |= static_cast<uint8_t>(1 << (totalBytes - 1));

			// write data
			*reinterpret_cast<uint32_t *>(dst) = src[0] >> (state * 8);
			dst += bytes;

			--nvalue;
			++src;

			state = 0;
    	}

    	state = VarintTables::VarintG8CUState[desc];
    	if (state > 0) {
    		*reinterpret_cast<uint32_t *>(dst) = src[0];
    		dst += state;
    	}

    	initdst[0] = desc;
    }

    static void decode(const uint8_t *src, uint8_t * &dst, uint8_t &state) {
    	uint8_t desc = src[0];
    	src += kHeaderSize;
    	const uint8_t outputOffset = VarintTables::VarintG8CUOutputOffset[state][desc];
#ifdef __AVX2__
		const __m128i data = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src));
		const __m256i val = _mm256_inserti128_si256(_mm256_castsi128_si256(data), data, 1);

		const __m128i mask0 = VarintTables::VarintG8CUSSEMasks[state][desc][0];
		const __m128i mask1 = VarintTables::VarintG8CUSSEMasks[state][desc][1];
		const __m256i mask = _mm256_inserti128_si256(_mm256_castsi128_si256(mask0), mask1, 1);
		__m256i result = _mm256_shuffle_epi8(val, mask);
		_mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), result);
#else /* !__AVX2__ */
#ifdef __SSSE3__
		const __m128i val = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src));
		__m128i result = _mm_shuffle_epi8(val, VarintTables::VarintG8CUSSEMasks[state][desc][0]);
		_mm_storeu_si128(reinterpret_cast<__m128i *>(dst), result);

//		if (outputOffset + state > 16) // branch misprediction is costly
		{
			result = _mm_shuffle_epi8(val, VarintTables::VarintG8CUSSEMasks[state][desc][1]);
			_mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 16), result);
		}
#else /* !__AVX2__ && !__SSSE3__ */
		uint8_t length = VarintTables::VarintG8CULengths[desc][0];
		*reinterpret_cast<uint32_t *>(dst) = *reinterpret_cast<const uint32_t *>(src) & VarintTables::kMask[length - 1];
		src += length;

		uint8_t i = 1;
		for (uint8_t offset = 4 - state; offset < outputOffset; offset += 4) {
			length = VarintTables::VarintG8CULengths[desc][i++];
			*reinterpret_cast<uint32_t *>(dst + offset) = *reinterpret_cast<const uint32_t *>(src) & VarintTables::kMask[length - 1];
			src += length;
		}
#endif /* __SSSE3__ */
#endif /* __AVX2__ */
		dst += outputOffset;
		state = VarintTables::VarintG8CUState[desc];
	}
};

#endif /* VARINTG8CU_H_ */
