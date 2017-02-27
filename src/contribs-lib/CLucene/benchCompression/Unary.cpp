/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#include "Unary.h"
#include "util.h"

void Unary::encodeArray(const uint32_t *in, size_t nvalue,
		uint32_t *out, size_t &csize) {
	// compute sum[i] = in[0] + ... + in[i] + i, for i = 0, ..., nvalue - 1
	std::vector<uint64_t> sum(in, in + nvalue);
	std::partial_sum(sum.begin(), sum.end(), sum.begin(),
			[](uint64_t x, uint64_t y) { return x + y + 1; } );

	// compute the total number of words needed and set those words to 0
	csize = div_roundup(sum[nvalue - 1] + 1, 32);
	memset(out, 0, csize * sizeof(uint32_t)); // FIXME: memset is not fast enough

	// flip termination bits
	size_t idx(0);
	uint32_t shift(0);
	for (size_t i = 0; i < nvalue; ++i) {
		idx = sum[i] >> 5;
		shift = static_cast<uint32_t>(sum[i] & 0x1f);
		out[idx] |= 1 << shift;
	}
	
	// flip all unused bits
	out[idx] |= ~((1 << shift) - 1);
}

void Unary::fakeencodeArray(const uint32_t *in, size_t nvalue,
		size_t &csize) {
	// compute sum[i] = in[0] + ... + in[i] + i, for i = 0, ..., nvalue - 1
	std::vector<uint64_t> sum(in, in + nvalue);
	std::partial_sum(sum.begin(), sum.end(), sum.begin(),
			[](uint64_t x, uint64_t y) { return x + y + 1; } );

	// compute the total number of words needed and set those words to 0
	csize = div_roundup(sum[nvalue - 1] + 1, 32);
}


const uint32_t * Unary::decodeArray(const uint32_t *in, size_t csize,
		uint32_t *out, size_t nvalue) {
    uint32_t carry(0);
    const uint32_t *const endout(out + nvalue);

    while (endout > out) {
    	uint32_t codeword = in[0], tmpcodeword = codeword;
		++in;
		const int ones = __builtin_popcount(codeword);
		switch (ones) {
		case 0:
			carry += 32;

			break;
		case 32:
			memset32(out);
			out[0] += carry;
			carry = 0;

			break;
		default:
			for (int i = 0; i < ones; ++i) {
				out[i] = __builtin_ctz(codeword);
				codeword >>= out[i] + 1;
			}
			out[0] += carry;
			carry = __builtin_clz(tmpcodeword);

			break;
		}

		out += ones;
	}

    return in;
}

////alternatives
//const uint32_t * Unary::decodeArray(const uint32_t *in, size_t csize,
//		uint32_t *out, size_t nvalue) {
//	const uint64_t *in64 = reinterpret_cast<const uint64_t *>(in);
//	uint32_t carry = 0;
//	const uint32_t *const endout(out + nvalue);
//
//	while (endout > out) {
//		uint64_t codeword = in64[0], tmpcodeword = codeword;
//		++in64;
//		const int ones = __builtin_popcountll(codeword);
//		switch (ones) {
//		case 0:
//			carry += 64;
//
//			break;
//		case 64:
//			memset64(out);
//			out[0] += carry;
//			carry = 0;
//
//			break;
//		default: // FIXME: eliminate for loop
//			for (int i = 0; i < ones; ++i) {
//				out[i] = __builtin_ctzll(codeword);
//				codeword >>= out[i] + 1;
//			}
//			out[0] += carry;
//			carry = __builtin_clzll(tmpcodeword);
//
//			break;
//		}
//
//		out += ones;
//	}
//
//	return reinterpret_cast<const uint32_t *>(in64);
//}

//const uint32_t * Unary::decodeArray(const uint32_t *in, size_t csize,
//		uint32_t *out, size_t nvalue) {
//	const uint16_t *in16 = reinterpret_cast<const uint16_t *>(in);
//	uint32_t carry = 0;
//	const uint32_t *const endout(out + nvalue);
//
//	while (endout > out) {
//		uint16_t codeword = in16[0], tmpcodeword = codeword;
//		++in16;
//		const int ones = __builtin_popcount(codeword);
//		switch (ones) {
//		case 0:
//			carry += 16;
//
//			break;
//		case 16:
//			memset16(out);
//			out[0] += carry;
//			carry = 0;
//
//			break;
//		default: // FIXME: eliminate for loop
//			for (int i = 0; i < ones; ++i) {
//				out[i] = __builtin_ctz(codeword);
//				codeword >>= out[i] + 1;
//			}
//			out[0] += carry;
//			carry = __builtin_clz(tmpcodeword) - 16;
//
//			break;
//		}
//
//		out += ones;
//	}
//
//	return reinterpret_cast<const uint32_t *>(in16);
//}

//const uint32_t * Unary::decodeArray(const uint32_t *in, size_t csize,
//		uint32_t *out, size_t nvalue) {
//    uint32_t carry(0);
//    const uint32_t *const endout(out + nvalue);
//
//    while (endout > out) {
//    	uint32_t *const beginout(out);
//    	uint32_t codeword = in[0], tmpcodeword = codeword;
//		++in;
//		const int ones = __builtin_popcount(codeword);
//		switch (ones) {
//		case 0:
//			carry += 32;
//
//			break;
//		case 32:
//			memset32(out);
//			out[0] += carry;
//			carry = 0;
//
//			out += 32;
//
//			break;
//		case 31:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 30:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 29:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 28:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 27:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 26:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 25:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 24:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 23:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 22:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 21:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 20:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 19:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 18:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 17:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 16:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 15:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 14:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 13:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 12:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 11:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 10:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 9:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 8:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 7:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 6:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 5:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 4:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 3:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 2:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 1:
//			out[0] = __builtin_ctz(codeword);
//			codeword >>= out[0] + 1;
//
//			++out;
//
//			beginout[0] += carry;
//			carry = __builtin_clz(tmpcodeword);
//
//			break;
//		default:
//			break;
//		}
//	}
//
//    return in;
//}
