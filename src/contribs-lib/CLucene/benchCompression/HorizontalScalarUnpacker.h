/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef HORIZONTALSCALARUNPACKER_H_
#define HORIZONTALSCALARUNPACKER_H_

#include "IntegerUnpacker.h"
#include "util.h"

template <bool IsRiceCoding>
class HorizontalScalarUnpacker : public IntegerUnpacker {
	template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
	friend class Rice;
public:
	enum {
		PACKSIZE = 64  // number of elements to be unpacked for each invocation of unpack
	};

	HorizontalScalarUnpacker(const uint32_t *q = nullptr) : beginquotient(q) {
		checkifdivisibleby(PACKSIZE, UNITPACKSIZE);
	}

	virtual void unpack(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit);

	/* assumes that integers fit in the prescribed number of bits */
	virtual void packwithoutmask(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit);

	virtual void pack(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit);


	virtual void unpack_generic(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit, uint32_t nvalue);

	/* assumes that integers fit in the prescribed number of bits */
	void packwithoutmask_generic(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit, uint32_t nvalue);

	void pack_generic(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit, uint32_t nvalue);


	virtual std::string name() const {
		std::ostringstream unpackername;
		unpackername << "HorizontalScalarUnpacker<" << PACKSIZE << ">";
		return unpackername.str();
	}

	virtual ~HorizontalScalarUnpacker() = default;

private:
	const uint32_t *beginquotient; // for Rice Coding


	enum {
		UNITPACKSIZE = 64
	};


	void __horizontal_scalar_unpack64(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit);

	void __horizontal_scalar_unpack64_c0(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c1(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c2(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c3(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c4(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c5(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c6(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c7(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c8(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c9(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c10(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c11(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c12(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c13(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c14(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c15(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c16(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c17(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c18(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c19(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c20(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c21(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c22(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c23(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c24(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c25(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c26(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c27(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c28(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c29(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c30(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c31(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_unpack64_c32(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);


	/* assumes that integers fit in the prescribed number of bits */
	void __horizontal_scalar_packwithoutmask64(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit);

	void __horizontal_scalar_packwithoutmask64_c1(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c2(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c3(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c4(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c5(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c6(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c7(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c8(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c9(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c10(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c11(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c12(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c13(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c14(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c15(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c16(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c17(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c18(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c19(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c20(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c21(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c22(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c23(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c24(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c25(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c26(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c27(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c28(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c29(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c30(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c31(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_packwithoutmask64_c32(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);


	void __horizontal_scalar_pack64(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit);

	void __horizontal_scalar_pack64_c1(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c2(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c3(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c4(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c5(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c6(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c7(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c8(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c9(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c10(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c11(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c12(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c13(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c14(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c15(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c16(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c17(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c18(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c19(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c20(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c21(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c22(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c23(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c24(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c25(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c26(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c27(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c28(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c29(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c30(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c31(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __horizontal_scalar_pack64_c32(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
};


template <bool IsRiceCoding>
inline void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit) {
    // Could have used function pointers instead of switch.
    // Switch calls do offer the compiler more opportunities for optimization in
    // theory. In this case, it makes no difference with a good compiler.
    switch(bit) {
    case 0: __horizontal_scalar_unpack64_c0(in, out); return;

	case 1: __horizontal_scalar_unpack64_c1(in, out); return;

	case 2: __horizontal_scalar_unpack64_c2(in, out); return;

	case 3: __horizontal_scalar_unpack64_c3(in, out); return;

	case 4: __horizontal_scalar_unpack64_c4(in, out); return;

	case 5: __horizontal_scalar_unpack64_c5(in, out); return;

	case 6: __horizontal_scalar_unpack64_c6(in, out); return;

	case 7: __horizontal_scalar_unpack64_c7(in, out); return;

	case 8: __horizontal_scalar_unpack64_c8(in, out); return;

	case 9: __horizontal_scalar_unpack64_c9(in, out); return;

	case 10: __horizontal_scalar_unpack64_c10(in, out); return;

	case 11: __horizontal_scalar_unpack64_c11(in, out); return;

	case 12: __horizontal_scalar_unpack64_c12(in, out); return;

	case 13: __horizontal_scalar_unpack64_c13(in, out); return;

	case 14: __horizontal_scalar_unpack64_c14(in, out); return;

	case 15: __horizontal_scalar_unpack64_c15(in, out); return;

	case 16: __horizontal_scalar_unpack64_c16(in, out); return;

	case 17: __horizontal_scalar_unpack64_c17(in, out); return;

	case 18: __horizontal_scalar_unpack64_c18(in, out); return;

	case 19: __horizontal_scalar_unpack64_c19(in, out); return;

	case 20: __horizontal_scalar_unpack64_c20(in, out); return;

	case 21: __horizontal_scalar_unpack64_c21(in, out); return;

	case 22: __horizontal_scalar_unpack64_c22(in, out); return;

	case 23: __horizontal_scalar_unpack64_c23(in, out); return;

	case 24: __horizontal_scalar_unpack64_c24(in, out); return;

	case 25: __horizontal_scalar_unpack64_c25(in, out); return;

	case 26: __horizontal_scalar_unpack64_c26(in, out); return;

	case 27: __horizontal_scalar_unpack64_c27(in, out); return;

	case 28: __horizontal_scalar_unpack64_c28(in, out); return;

	case 29: __horizontal_scalar_unpack64_c29(in, out); return;

	case 30: __horizontal_scalar_unpack64_c30(in, out); return;

	case 31: __horizontal_scalar_unpack64_c31(in, out); return;

	case 32: __horizontal_scalar_unpack64_c32(in, out); return;

	default: return;
    }
}

template <bool IsRiceCoding>
inline void HorizontalScalarUnpacker<IsRiceCoding>::unpack(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit) {
	if (!IsRiceCoding) { // NewPFor etc.
		for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < PACKSIZE; numberOfValuesUnpacked += UNITPACKSIZE) {
			__horizontal_scalar_unpack64(in, out, bit);
			in += (UNITPACKSIZE * bit) / 32;
			out += UNITPACKSIZE;
		}
	}
	else { // Rice Coding
		uint32_t *beginout = out;
		for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < PACKSIZE; numberOfValuesUnpacked += UNITPACKSIZE) {
			__horizontal_scalar_unpack64(in, out, bit);
			in += (UNITPACKSIZE * bit) / 32;
			out += UNITPACKSIZE;
		}

		if (bit < 32) { // This is important!
			for (uint32_t i = 0; i < PACKSIZE; ++i)
				beginout[i] |= (beginquotient[i] << bit);
		}
	}
}


template <bool IsRiceCoding>
inline void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit) {
    // Could have used function pointers instead of switch.
    // Switch calls do offer the compiler more opportunities for optimization in
    // theory. In this case, it makes no difference with a good compiler.
    switch(bit) {
	case 0: return;

	case 1: __horizontal_scalar_packwithoutmask64_c1(in, out); return;

	case 2: __horizontal_scalar_packwithoutmask64_c2(in, out); return;

	case 3: __horizontal_scalar_packwithoutmask64_c3(in, out); return;

	case 4: __horizontal_scalar_packwithoutmask64_c4(in, out); return;

	case 5: __horizontal_scalar_packwithoutmask64_c5(in, out); return;

	case 6: __horizontal_scalar_packwithoutmask64_c6(in, out); return;

	case 7: __horizontal_scalar_packwithoutmask64_c7(in, out); return;

	case 8: __horizontal_scalar_packwithoutmask64_c8(in, out); return;

	case 9: __horizontal_scalar_packwithoutmask64_c9(in, out); return;

	case 10: __horizontal_scalar_packwithoutmask64_c10(in, out); return;

	case 11: __horizontal_scalar_packwithoutmask64_c11(in, out); return;

	case 12: __horizontal_scalar_packwithoutmask64_c12(in, out); return;

	case 13: __horizontal_scalar_packwithoutmask64_c13(in, out); return;

	case 14: __horizontal_scalar_packwithoutmask64_c14(in, out); return;

	case 15: __horizontal_scalar_packwithoutmask64_c15(in, out); return;

	case 16: __horizontal_scalar_packwithoutmask64_c16(in, out); return;

	case 17: __horizontal_scalar_packwithoutmask64_c17(in, out); return;

	case 18: __horizontal_scalar_packwithoutmask64_c18(in, out); return;

	case 19: __horizontal_scalar_packwithoutmask64_c19(in, out); return;

	case 20: __horizontal_scalar_packwithoutmask64_c20(in, out); return;

	case 21: __horizontal_scalar_packwithoutmask64_c21(in, out); return;

	case 22: __horizontal_scalar_packwithoutmask64_c22(in, out); return;

	case 23: __horizontal_scalar_packwithoutmask64_c23(in, out); return;

	case 24: __horizontal_scalar_packwithoutmask64_c24(in, out); return;

	case 25: __horizontal_scalar_packwithoutmask64_c25(in, out); return;

	case 26: __horizontal_scalar_packwithoutmask64_c26(in, out); return;

	case 27: __horizontal_scalar_packwithoutmask64_c27(in, out); return;

	case 28: __horizontal_scalar_packwithoutmask64_c28(in, out); return;

	case 29: __horizontal_scalar_packwithoutmask64_c29(in, out); return;

	case 30: __horizontal_scalar_packwithoutmask64_c30(in, out); return;

	case 31: __horizontal_scalar_packwithoutmask64_c31(in, out); return;

	case 32: __horizontal_scalar_packwithoutmask64_c32(in, out); return;

	default: return;
    }
}

template <bool IsRiceCoding>
inline void HorizontalScalarUnpacker<IsRiceCoding>::packwithoutmask(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit) {
	for (uint32_t numberOfValuesPacked = 0; numberOfValuesPacked < PACKSIZE; numberOfValuesPacked += UNITPACKSIZE) {
		__horizontal_scalar_packwithoutmask64(in, out, bit);
		in += UNITPACKSIZE;
		out += (UNITPACKSIZE * bit) / 32;
	}
}


template <bool IsRiceCoding>
inline void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit) {
    // Could have used function pointers instead of switch.
    // Switch calls do offer the compiler more opportunities for optimization in
    // theory. In this case, it makes no difference with a good compiler.
    switch(bit) {
	case 0: return;

	case 1: __horizontal_scalar_pack64_c1(in, out); return;

	case 2: __horizontal_scalar_pack64_c2(in, out); return;

	case 3: __horizontal_scalar_pack64_c3(in, out); return;

	case 4: __horizontal_scalar_pack64_c4(in, out); return;

	case 5: __horizontal_scalar_pack64_c5(in, out); return;

	case 6: __horizontal_scalar_pack64_c6(in, out); return;

	case 7: __horizontal_scalar_pack64_c7(in, out); return;

	case 8: __horizontal_scalar_pack64_c8(in, out); return;

	case 9: __horizontal_scalar_pack64_c9(in, out); return;

	case 10: __horizontal_scalar_pack64_c10(in, out); return;

	case 11: __horizontal_scalar_pack64_c11(in, out); return;

	case 12: __horizontal_scalar_pack64_c12(in, out); return;

	case 13: __horizontal_scalar_pack64_c13(in, out); return;

	case 14: __horizontal_scalar_pack64_c14(in, out); return;

	case 15: __horizontal_scalar_pack64_c15(in, out); return;

	case 16: __horizontal_scalar_pack64_c16(in, out); return;

	case 17: __horizontal_scalar_pack64_c17(in, out); return;

	case 18: __horizontal_scalar_pack64_c18(in, out); return;

	case 19: __horizontal_scalar_pack64_c19(in, out); return;

	case 20: __horizontal_scalar_pack64_c20(in, out); return;

	case 21: __horizontal_scalar_pack64_c21(in, out); return;

	case 22: __horizontal_scalar_pack64_c22(in, out); return;

	case 23: __horizontal_scalar_pack64_c23(in, out); return;

	case 24: __horizontal_scalar_pack64_c24(in, out); return;

	case 25: __horizontal_scalar_pack64_c25(in, out); return;

	case 26: __horizontal_scalar_pack64_c26(in, out); return;

	case 27: __horizontal_scalar_pack64_c27(in, out); return;

	case 28: __horizontal_scalar_pack64_c28(in, out); return;

	case 29: __horizontal_scalar_pack64_c29(in, out); return;

	case 30: __horizontal_scalar_pack64_c30(in, out); return;

	case 31: __horizontal_scalar_pack64_c31(in, out); return;

	case 32: __horizontal_scalar_pack64_c32(in, out); return;

	default: return;
    }
}

template <bool IsRiceCoding>
inline void HorizontalScalarUnpacker<IsRiceCoding>::pack(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit) {
	for (uint32_t numberOfValuesPacked = 0; numberOfValuesPacked < PACKSIZE; numberOfValuesPacked += UNITPACKSIZE) {
		__horizontal_scalar_pack64(in, out, bit);
		in += UNITPACKSIZE;
		out += (UNITPACKSIZE * bit) / 32;
	}
}


template <bool IsRiceCoding>
inline void HorizontalScalarUnpacker<IsRiceCoding>::unpack_generic(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
        const uint32_t bit, uint32_t nvalue) {
    uint64_t mask = (1ULL << bit) - 1;
    for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < nvalue; ++numberOfValuesUnpacked) {
        uint32_t idx = (numberOfValuesUnpacked * bit) >> 5;
        uint32_t shift = (numberOfValuesUnpacked * bit) & 0x1f;
        const uint64_t codeword = (reinterpret_cast<const uint64_t *>(in + idx))[0];
        out[numberOfValuesUnpacked] = static_cast<uint32_t>((codeword >> shift) & mask);
    }
}

template <bool IsRiceCoding>
inline void HorizontalScalarUnpacker<IsRiceCoding>::packwithoutmask_generic(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit, uint32_t nvalue) {
	uint32_t nwords = div_roundup(nvalue * bit, 32);
	memset(out, 0, nwords * sizeof(uint32_t));
	for (uint32_t numberOfValuesPacked = 0; numberOfValuesPacked < nvalue; ++numberOfValuesPacked) {
		uint32_t idx = (numberOfValuesPacked * bit) >> 5;
		uint32_t shift = (numberOfValuesPacked * bit) & 0x1f;
		uint64_t &codeword = (reinterpret_cast<uint64_t *>(out + idx))[0];
		codeword |= static_cast<uint64_t>(in[numberOfValuesPacked]) << shift;
	}
}

template <bool IsRiceCoding>
inline void HorizontalScalarUnpacker<IsRiceCoding>::pack_generic(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit, uint32_t nvalue) {
	uint64_t mask = (1ULL << bit) - 1;
	uint32_t nwords = div_roundup(nvalue * bit, 32);
	memset(out, 0, nwords * sizeof(uint32_t));
	for (uint32_t numberOfValuesPacked = 0; numberOfValuesPacked < nvalue; ++numberOfValuesPacked) {
		uint32_t idx = (numberOfValuesPacked * bit) >> 5;
		uint32_t shift = (numberOfValuesPacked * bit) & 0x1f;
		uint64_t &codeword = (reinterpret_cast<uint64_t *>(out + idx))[0];
		codeword |= static_cast<uint64_t>(in[numberOfValuesPacked] & mask) << shift;
	}
}


#include "HorizontalScalarUnpackerIMP.h"


#endif /* HORIZONTALSCALARUNPACKER_H_ */
