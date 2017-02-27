/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef VERTICALAVXUNPACKER_H_
#define VERTICALAVXUNPACKER_H_

#include "IntegerUnpacker.h"
#include "util.h"

template <bool IsRiceCoding>
class VerticalAVXUnpacker : public IntegerUnpacker {
	template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
	friend class Rice;
public:
	enum {
		PACKSIZE = 256
	};

	VerticalAVXUnpacker(const uint32_t *q = nullptr) : beginquotient(q), quotient(reinterpret_cast<const __m256i *>(q)) {
		checkifdivisibleby(PACKSIZE, UNITPACKSIZE);
	}

	virtual void unpack(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit);

	/* assumes that integers fit in the prescribed number of bits */
	virtual void packwithoutmask(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit);

	virtual void pack(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit);

	virtual std::string name() const {
		std::ostringstream unpackername;
		unpackername << "VerticalAVXUnpacker<" << PACKSIZE << ">";
		return unpackername.str();
	}

private:
	// for Rice Coding
	const uint32_t *beginquotient;
	const __m256i *quotient;


	enum {
		UNITPACKSIZE = 256
	};


	void __vertical_avx_unpack256(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit);

	void __vertical_avx_unpack256_c0(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c1(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c2(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c3(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c4(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c5(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c6(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c7(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c8(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c9(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c10(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c11(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c12(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c13(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c14(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c15(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c16(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c17(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c18(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c19(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c20(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c21(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c22(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c23(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c24(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c25(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c26(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c27(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c28(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c29(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c30(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c31(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_unpack256_c32(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);


	/* assumes that integers fit in the prescribed number of bits */
	void __vertical_avx_packwithoutmask256(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit);

	void __vertical_avx_packwithoutmask256_c1(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c2(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c3(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c4(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c5(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c6(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c7(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c8(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c9(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c10(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c11(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c12(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c13(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c14(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c15(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c16(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c17(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c18(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c19(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c20(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c21(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c22(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c23(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c24(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c25(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c26(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c27(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c28(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c29(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c30(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c31(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_packwithoutmask256_c32(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);


	void __vertical_avx_pack256(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit);

	void __vertical_avx_pack256_c1(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c2(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c3(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c4(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c5(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c6(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c7(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c8(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c9(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c10(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c11(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c12(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c13(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c14(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c15(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c16(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c17(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c18(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c19(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c20(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c21(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c22(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c23(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c24(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c25(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c26(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c27(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c28(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c29(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c30(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c31(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
	void __vertical_avx_pack256_c32(const __m256i *  __restrict__  in, __m256i *  __restrict__  out);
};


template <bool IsRiceCoding>
inline void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_unpack256(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit) {
    // Could have used function pointers instead of switch.
    // Switch calls do offer the compiler more opportunities for optimization in
    // theory. In this case, it makes no difference with a good compiler.
    const __m256i *in_m256i = reinterpret_cast<const __m256i *>(in);
    __m256i *out_m256i = reinterpret_cast<__m256i *>(out);

    switch(bit) {
    case 0: __vertical_avx_unpack256_c0(in_m256i, out_m256i); return;

	case 1: __vertical_avx_unpack256_c1(in_m256i, out_m256i); return;

	case 2: __vertical_avx_unpack256_c2(in_m256i, out_m256i); return;

	case 3: __vertical_avx_unpack256_c3(in_m256i, out_m256i); return;

	case 4: __vertical_avx_unpack256_c4(in_m256i, out_m256i); return;

	case 5: __vertical_avx_unpack256_c5(in_m256i, out_m256i); return;

	case 6: __vertical_avx_unpack256_c6(in_m256i, out_m256i); return;

	case 7: __vertical_avx_unpack256_c7(in_m256i, out_m256i); return;

	case 8: __vertical_avx_unpack256_c8(in_m256i, out_m256i); return;

	case 9: __vertical_avx_unpack256_c9(in_m256i, out_m256i); return;

	case 10: __vertical_avx_unpack256_c10(in_m256i, out_m256i); return;

	case 11: __vertical_avx_unpack256_c11(in_m256i, out_m256i); return;

	case 12: __vertical_avx_unpack256_c12(in_m256i, out_m256i); return;

	case 13: __vertical_avx_unpack256_c13(in_m256i, out_m256i); return;

	case 14: __vertical_avx_unpack256_c14(in_m256i, out_m256i); return;

	case 15: __vertical_avx_unpack256_c15(in_m256i, out_m256i); return;

	case 16: __vertical_avx_unpack256_c16(in_m256i, out_m256i); return;

	case 17: __vertical_avx_unpack256_c17(in_m256i, out_m256i); return;

	case 18: __vertical_avx_unpack256_c18(in_m256i, out_m256i); return;

	case 19: __vertical_avx_unpack256_c19(in_m256i, out_m256i); return;

	case 20: __vertical_avx_unpack256_c20(in_m256i, out_m256i); return;

	case 21: __vertical_avx_unpack256_c21(in_m256i, out_m256i); return;

	case 22: __vertical_avx_unpack256_c22(in_m256i, out_m256i); return;

	case 23: __vertical_avx_unpack256_c23(in_m256i, out_m256i); return;

	case 24: __vertical_avx_unpack256_c24(in_m256i, out_m256i); return;

	case 25: __vertical_avx_unpack256_c25(in_m256i, out_m256i); return;

	case 26: __vertical_avx_unpack256_c26(in_m256i, out_m256i); return;

	case 27: __vertical_avx_unpack256_c27(in_m256i, out_m256i); return;

	case 28: __vertical_avx_unpack256_c28(in_m256i, out_m256i); return;

	case 29: __vertical_avx_unpack256_c29(in_m256i, out_m256i); return;

	case 30: __vertical_avx_unpack256_c30(in_m256i, out_m256i); return;

	case 31: __vertical_avx_unpack256_c31(in_m256i, out_m256i); return;

	case 32: __vertical_avx_unpack256_c32(in_m256i, out_m256i); return;

	default: return;
    }
}

template <bool IsRiceCoding>
inline void VerticalAVXUnpacker<IsRiceCoding>::unpack(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit) {
	if (!IsRiceCoding) { // NewPFor etc.
		for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < PACKSIZE; numberOfValuesUnpacked += UNITPACKSIZE) {
			__vertical_avx_unpack256(in, out, bit);
			in += (UNITPACKSIZE * bit) / 32;
			out += UNITPACKSIZE;
		}
	}
	else { // Rice Coding
		quotient = reinterpret_cast<const __m256i *>(beginquotient);
		for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < PACKSIZE; numberOfValuesUnpacked += UNITPACKSIZE) {
			__vertical_avx_unpack256(in, out, bit);
			in += (UNITPACKSIZE * bit) / 32;
			out += UNITPACKSIZE;
			quotient += UNITPACKSIZE / 8;
		}
	}
}


template <bool IsRiceCoding>
inline void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_packwithoutmask256(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit) {
    // Could have used function pointers instead of switch.
    // Switch calls do offer the compiler more opportunities for optimization in
    // theory. In this case, it makes no difference with a good compiler.
    const __m256i *in_m256i = reinterpret_cast<const __m256i *>(in);
    __m256i *out_m256i = reinterpret_cast<__m256i *>(out);

    switch(bit) {
	case 0: return;

	case 1: __vertical_avx_packwithoutmask256_c1(in_m256i, out_m256i); return;

	case 2: __vertical_avx_packwithoutmask256_c2(in_m256i, out_m256i); return;

	case 3: __vertical_avx_packwithoutmask256_c3(in_m256i, out_m256i); return;

	case 4: __vertical_avx_packwithoutmask256_c4(in_m256i, out_m256i); return;

	case 5: __vertical_avx_packwithoutmask256_c5(in_m256i, out_m256i); return;

	case 6: __vertical_avx_packwithoutmask256_c6(in_m256i, out_m256i); return;

	case 7: __vertical_avx_packwithoutmask256_c7(in_m256i, out_m256i); return;

	case 8: __vertical_avx_packwithoutmask256_c8(in_m256i, out_m256i); return;

	case 9: __vertical_avx_packwithoutmask256_c9(in_m256i, out_m256i); return;

	case 10: __vertical_avx_packwithoutmask256_c10(in_m256i, out_m256i); return;

	case 11: __vertical_avx_packwithoutmask256_c11(in_m256i, out_m256i); return;

	case 12: __vertical_avx_packwithoutmask256_c12(in_m256i, out_m256i); return;

	case 13: __vertical_avx_packwithoutmask256_c13(in_m256i, out_m256i); return;

	case 14: __vertical_avx_packwithoutmask256_c14(in_m256i, out_m256i); return;

	case 15: __vertical_avx_packwithoutmask256_c15(in_m256i, out_m256i); return;

	case 16: __vertical_avx_packwithoutmask256_c16(in_m256i, out_m256i); return;

	case 17: __vertical_avx_packwithoutmask256_c17(in_m256i, out_m256i); return;

	case 18: __vertical_avx_packwithoutmask256_c18(in_m256i, out_m256i); return;

	case 19: __vertical_avx_packwithoutmask256_c19(in_m256i, out_m256i); return;

	case 20: __vertical_avx_packwithoutmask256_c20(in_m256i, out_m256i); return;

	case 21: __vertical_avx_packwithoutmask256_c21(in_m256i, out_m256i); return;

	case 22: __vertical_avx_packwithoutmask256_c22(in_m256i, out_m256i); return;

	case 23: __vertical_avx_packwithoutmask256_c23(in_m256i, out_m256i); return;

	case 24: __vertical_avx_packwithoutmask256_c24(in_m256i, out_m256i); return;

	case 25: __vertical_avx_packwithoutmask256_c25(in_m256i, out_m256i); return;

	case 26: __vertical_avx_packwithoutmask256_c26(in_m256i, out_m256i); return;

	case 27: __vertical_avx_packwithoutmask256_c27(in_m256i, out_m256i); return;

	case 28: __vertical_avx_packwithoutmask256_c28(in_m256i, out_m256i); return;

	case 29: __vertical_avx_packwithoutmask256_c29(in_m256i, out_m256i); return;

	case 30: __vertical_avx_packwithoutmask256_c30(in_m256i, out_m256i); return;

	case 31: __vertical_avx_packwithoutmask256_c31(in_m256i, out_m256i); return;

	case 32: __vertical_avx_packwithoutmask256_c32(in_m256i, out_m256i); return;

	default: return;
    }
}

template <bool IsRiceCoding>
inline void VerticalAVXUnpacker<IsRiceCoding>::packwithoutmask(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit) {
	for (uint32_t numberOfValuesPacked = 0; numberOfValuesPacked < PACKSIZE; numberOfValuesPacked += UNITPACKSIZE) {
		__vertical_avx_packwithoutmask256(in, out, bit);
		in += UNITPACKSIZE;
		out += (UNITPACKSIZE * bit) / 32;
	}
}


template <bool IsRiceCoding>
inline void VerticalAVXUnpacker<IsRiceCoding>::__vertical_avx_pack256(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit) {
    // Could have used function pointers instead of switch.
    // Switch calls do offer the compiler more opportunities for optimization in
    // theory. In this case, it makes no difference with a good compiler.
    const __m256i *in_m256i = reinterpret_cast<const __m256i *>(in);
    __m256i *out_m256i = reinterpret_cast<__m256i *>(out);

    switch(bit) {
	case 0: return;

	case 1: __vertical_avx_pack256_c1(in_m256i, out_m256i); return;

	case 2: __vertical_avx_pack256_c2(in_m256i, out_m256i); return;

	case 3: __vertical_avx_pack256_c3(in_m256i, out_m256i); return;

	case 4: __vertical_avx_pack256_c4(in_m256i, out_m256i); return;

	case 5: __vertical_avx_pack256_c5(in_m256i, out_m256i); return;

	case 6: __vertical_avx_pack256_c6(in_m256i, out_m256i); return;

	case 7: __vertical_avx_pack256_c7(in_m256i, out_m256i); return;

	case 8: __vertical_avx_pack256_c8(in_m256i, out_m256i); return;

	case 9: __vertical_avx_pack256_c9(in_m256i, out_m256i); return;

	case 10: __vertical_avx_pack256_c10(in_m256i, out_m256i); return;

	case 11: __vertical_avx_pack256_c11(in_m256i, out_m256i); return;

	case 12: __vertical_avx_pack256_c12(in_m256i, out_m256i); return;

	case 13: __vertical_avx_pack256_c13(in_m256i, out_m256i); return;

	case 14: __vertical_avx_pack256_c14(in_m256i, out_m256i); return;

	case 15: __vertical_avx_pack256_c15(in_m256i, out_m256i); return;

	case 16: __vertical_avx_pack256_c16(in_m256i, out_m256i); return;

	case 17: __vertical_avx_pack256_c17(in_m256i, out_m256i); return;

	case 18: __vertical_avx_pack256_c18(in_m256i, out_m256i); return;

	case 19: __vertical_avx_pack256_c19(in_m256i, out_m256i); return;

	case 20: __vertical_avx_pack256_c20(in_m256i, out_m256i); return;

	case 21: __vertical_avx_pack256_c21(in_m256i, out_m256i); return;

	case 22: __vertical_avx_pack256_c22(in_m256i, out_m256i); return;

	case 23: __vertical_avx_pack256_c23(in_m256i, out_m256i); return;

	case 24: __vertical_avx_pack256_c24(in_m256i, out_m256i); return;

	case 25: __vertical_avx_pack256_c25(in_m256i, out_m256i); return;

	case 26: __vertical_avx_pack256_c26(in_m256i, out_m256i); return;

	case 27: __vertical_avx_pack256_c27(in_m256i, out_m256i); return;

	case 28: __vertical_avx_pack256_c28(in_m256i, out_m256i); return;

	case 29: __vertical_avx_pack256_c29(in_m256i, out_m256i); return;

	case 30: __vertical_avx_pack256_c30(in_m256i, out_m256i); return;

	case 31: __vertical_avx_pack256_c31(in_m256i, out_m256i); return;

	case 32: __vertical_avx_pack256_c32(in_m256i, out_m256i); return;

	default: return;
    }
}

template <bool IsRiceCoding>
inline void VerticalAVXUnpacker<IsRiceCoding>::pack(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit) {
	for (uint32_t numberOfValuesPacked = 0; numberOfValuesPacked < PACKSIZE; numberOfValuesPacked += UNITPACKSIZE) {
		__vertical_avx_pack256(in, out, bit);
		in += UNITPACKSIZE;
		out += (UNITPACKSIZE * bit) / 32;
	}
}


#include "VerticalAVXUnpackerIMP.h"


#endif /* VERTICALAVXUNPACKER_H_ */
