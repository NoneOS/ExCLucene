/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef VARINTG8IU_H_
#define VARINTG8IU_H_

#include "common.h"
#include "IntegerCodec.h"
#include "util.h"

#ifdef __SSSE3__
#include <x86intrin.h>
namespace VarintTables {
extern const __m128i VarintG8IUSSEMasks[256][2];
}  // namespace VarintTables
#endif

namespace VarintTables {
extern const uint8_t VarintG8IUOutputOffset[256];
extern const uint8_t VarintG8IULengths[256][8];
extern const uint32_t kMask[4];
}  // namespace VarintTables

class VarintG8IU : public IntegerCodec {

//#undef __AVX2__
//#undef __SSSE3__

public:
	/**
	 * group details:
	 * kGroupSize - size of data bytes,
	 *              which is also the maximum number of integers that a group can encode.
	 * kHeaderSize - size of descriptor byte.
	 * KFullGroupSize - group size.
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
        while (nvalue > 0) {
            encode(in, nvalue, dst);
            dst += kFullGroupSize;
        }

        csize = ((dst - initdst) + 3) / 4;
    }

    virtual const uint32_t * decodeArray(const uint32_t *in, size_t csize,
                uint32_t *out, size_t nvalue) {
        const uint32_t *const endout = out + nvalue;
        const uint8_t *src = reinterpret_cast<const uint8_t *>(in);
        const uint8_t *const initsrc = src;
        while (endout > out) {
            decode(src, out);
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
        codecname << "VarintG8IU_" << platform;
        return codecname.str();
    }

    static void encode(const uint32_t * &src, size_t &nvalue, uint8_t *dst) {
    	uint8_t *const initdst = dst;
    	++dst; // reserve space for the descriptor byte

    	uint8_t desc = 0;
    	uint32_t totalLength = 0;
		while (nvalue > 0) {
			uint8_t length = getNumberOfBytes(src[0]);
			totalLength += length;
			if (totalLength > kGroupSize) {
				break;
			}

			// flip the correct bit in descriptor
			desc |= static_cast<uint8_t>(1 << (totalLength - 1));

			// write data
			*reinterpret_cast<uint32_t *>(dst) = src[0];
			dst += length;

			--nvalue;
			++src;
		}

		initdst[0] = desc;
    }

    static void decode(const uint8_t *src, uint32_t * &dst) {
    	uint8_t desc = src[0];
    	src += kHeaderSize;
    	const uint8_t num = VarintTables::VarintG8IUOutputOffset[desc]; // table lookup is faster than __builtin_popcount
//    	const uint8_t num = __builtin_popcount(desc);
#ifdef __AVX2__
    	const __m128i data = _mm_lddqu_si128 (reinterpret_cast<const __m128i *>(src));
    	const __m256i val = _mm256_inserti128_si256(_mm256_castsi128_si256(data), data, 1);

    	const __m128i mask0 = VarintTables::VarintG8IUSSEMasks[desc][0];
    	const __m128i mask1 = VarintTables::VarintG8IUSSEMasks[desc][1];
    	const __m256i mask = _mm256_inserti128_si256(_mm256_castsi128_si256(mask0), mask1, 1);
    	__m256i result = _mm256_shuffle_epi8(val, mask);
    	_mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), result);
#else /* !__AVX2__ */
#ifdef __SSSE3__
    	const __m128i val = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src));
        __m128i result = _mm_shuffle_epi8(val, VarintTables::VarintG8IUSSEMasks[desc][0]);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), result);

//        if (num > 4) // branch misprediction is costly
        {
        	result = _mm_shuffle_epi8(val, VarintTables::VarintG8IUSSEMasks[desc][1]);
        	_mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 4), result);
        }
#else /* !__AVX2__ && !__SSSE3__ */
        for (uint8_t i = 0; i < num; ++i) {
        	uint8_t length = VarintTables::VarintG8IULengths[desc][i];
        	dst[i] = *reinterpret_cast<const uint32_t *>(src) & VarintTables::kMask[length - 1];
        	src += length;
        }

//        // slower alternative
//        uint64_t codeword = *reinterpret_cast<const uint64_t *>(src);
//        for (uint8_t i = 0; i < num; ++i) {
//        	uint8_t length = VarintTables::VarintG8IULengths[desc][i];
//        	dst[i] = codeword & VarintTables::kMask[length - 1];
//        	codeword >>= length * 8;
//        }
#endif /* __SSSE3__ */
#endif /* __AVX2__ */
        dst += num;
    }
};

#endif /* VARINTG8IU_H_ */
