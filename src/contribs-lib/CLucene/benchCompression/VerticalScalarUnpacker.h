/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef VERTICALSCALARUNPACKER_H_
#define VERTICALSCALARUNPACKER_H_

#include "IntegerUnpacker.h"
#include "util.h"

template <bool IsRiceCoding>
class VerticalScalarUnpacker : public IntegerUnpacker {
	template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
	friend class Rice;
public:
	enum {
		PACKSIZE = 128
	};

	VerticalScalarUnpacker(const uint32_t *q = nullptr) : beginquotient(q) {
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
		unpackername << "VerticalScalarUnpacker<" << PACKSIZE << ">";
		return unpackername.str();
	}

private:
	const uint32_t *beginquotient; 	// for Rice Coding


	enum {
		UNITPACKSIZE = 128
	};


	void __vertical_scalar_unpack128(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit);

	void __vertical_scalar_unpack128_c0(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c1(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c2(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c3(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c4(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c5(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c6(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c7(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c8(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c9(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c10(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c11(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c12(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c13(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c14(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c15(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c16(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c17(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c18(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c19(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c20(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c21(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c22(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c23(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c24(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c25(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c26(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c27(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c28(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c29(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c30(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c31(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_unpack128_c32(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);


	/* assumes that integers fit in the prescribed number of bits */
	void __vertical_scalar_packwithoutmask128(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit);

	void __vertical_scalar_packwithoutmask128_c1(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c2(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c3(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c4(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c5(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c6(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c7(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c8(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c9(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c10(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c11(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c12(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c13(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c14(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c15(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c16(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c17(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c18(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c19(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c20(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c21(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c22(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c23(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c24(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c25(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c26(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c27(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c28(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c29(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c30(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c31(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_packwithoutmask128_c32(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);


	void __vertical_scalar_pack128(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit);

	void __vertical_scalar_pack128_c1(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c2(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c3(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c4(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c5(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c6(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c7(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c8(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c9(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c10(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c11(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c12(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c13(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c14(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c15(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c16(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c17(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c18(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c19(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c20(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c21(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c22(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c23(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c24(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c25(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c26(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c27(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c28(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c29(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c30(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c31(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
	void __vertical_scalar_pack128_c32(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out);
};


template <bool IsRiceCoding>
inline void VerticalScalarUnpacker<IsRiceCoding>::__vertical_scalar_unpack128(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit) {
    // Could have used function pointers instead of switch.
    // Switch calls do offer the compiler more opportunities for optimization in
    // theory. In this case, it makes no difference with a good compiler.
    switch(bit) {
    case 0: __vertical_scalar_unpack128_c0(in, out); return;

	case 1: __vertical_scalar_unpack128_c1(in, out); return;

	case 2: __vertical_scalar_unpack128_c2(in, out); return;

	case 3: __vertical_scalar_unpack128_c3(in, out); return;

	case 4: __vertical_scalar_unpack128_c4(in, out); return;

	case 5: __vertical_scalar_unpack128_c5(in, out); return;

	case 6: __vertical_scalar_unpack128_c6(in, out); return;

	case 7: __vertical_scalar_unpack128_c7(in, out); return;

	case 8: __vertical_scalar_unpack128_c8(in, out); return;

	case 9: __vertical_scalar_unpack128_c9(in, out); return;

	case 10: __vertical_scalar_unpack128_c10(in, out); return;

	case 11: __vertical_scalar_unpack128_c11(in, out); return;

	case 12: __vertical_scalar_unpack128_c12(in, out); return;

	case 13: __vertical_scalar_unpack128_c13(in, out); return;

	case 14: __vertical_scalar_unpack128_c14(in, out); return;

	case 15: __vertical_scalar_unpack128_c15(in, out); return;

	case 16: __vertical_scalar_unpack128_c16(in, out); return;

	case 17: __vertical_scalar_unpack128_c17(in, out); return;

	case 18: __vertical_scalar_unpack128_c18(in, out); return;

	case 19: __vertical_scalar_unpack128_c19(in, out); return;

	case 20: __vertical_scalar_unpack128_c20(in, out); return;

	case 21: __vertical_scalar_unpack128_c21(in, out); return;

	case 22: __vertical_scalar_unpack128_c22(in, out); return;

	case 23: __vertical_scalar_unpack128_c23(in, out); return;

	case 24: __vertical_scalar_unpack128_c24(in, out); return;

	case 25: __vertical_scalar_unpack128_c25(in, out); return;

	case 26: __vertical_scalar_unpack128_c26(in, out); return;

	case 27: __vertical_scalar_unpack128_c27(in, out); return;

	case 28: __vertical_scalar_unpack128_c28(in, out); return;

	case 29: __vertical_scalar_unpack128_c29(in, out); return;

	case 30: __vertical_scalar_unpack128_c30(in, out); return;

	case 31: __vertical_scalar_unpack128_c31(in, out); return;

	case 32: __vertical_scalar_unpack128_c32(in, out); return;

	default: return;
    }
}

template <bool IsRiceCoding>
inline void VerticalScalarUnpacker<IsRiceCoding>::unpack(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit) {

	if (!IsRiceCoding) { // NewPFor etc.
		for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < PACKSIZE; numberOfValuesUnpacked += UNITPACKSIZE) {
			__vertical_scalar_unpack128(in, out, bit);
			in += (UNITPACKSIZE * bit) / 32;
			out += UNITPACKSIZE;
		}
	}
	else { // Rice Coding
		uint32_t *beginout = out;
		for (uint32_t numberOfValuesUnpacked = 0; numberOfValuesUnpacked < PACKSIZE; numberOfValuesUnpacked += UNITPACKSIZE) {
			__vertical_scalar_unpack128(in, out, bit);
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
inline void VerticalScalarUnpacker<IsRiceCoding>::__vertical_scalar_packwithoutmask128(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit) {
    // Could have used function pointers instead of switch.
    // Switch calls do offer the compiler more opportunities for optimization in
    // theory. In this case, it makes no difference with a good compiler.
    switch(bit) {
	case 0: return;

	case 1: __vertical_scalar_packwithoutmask128_c1(in, out); return;

	case 2: __vertical_scalar_packwithoutmask128_c2(in, out); return;

	case 3: __vertical_scalar_packwithoutmask128_c3(in, out); return;

	case 4: __vertical_scalar_packwithoutmask128_c4(in, out); return;

	case 5: __vertical_scalar_packwithoutmask128_c5(in, out); return;

	case 6: __vertical_scalar_packwithoutmask128_c6(in, out); return;

	case 7: __vertical_scalar_packwithoutmask128_c7(in, out); return;

	case 8: __vertical_scalar_packwithoutmask128_c8(in, out); return;

	case 9: __vertical_scalar_packwithoutmask128_c9(in, out); return;

	case 10: __vertical_scalar_packwithoutmask128_c10(in, out); return;

	case 11: __vertical_scalar_packwithoutmask128_c11(in, out); return;

	case 12: __vertical_scalar_packwithoutmask128_c12(in, out); return;

	case 13: __vertical_scalar_packwithoutmask128_c13(in, out); return;

	case 14: __vertical_scalar_packwithoutmask128_c14(in, out); return;

	case 15: __vertical_scalar_packwithoutmask128_c15(in, out); return;

	case 16: __vertical_scalar_packwithoutmask128_c16(in, out); return;

	case 17: __vertical_scalar_packwithoutmask128_c17(in, out); return;

	case 18: __vertical_scalar_packwithoutmask128_c18(in, out); return;

	case 19: __vertical_scalar_packwithoutmask128_c19(in, out); return;

	case 20: __vertical_scalar_packwithoutmask128_c20(in, out); return;

	case 21: __vertical_scalar_packwithoutmask128_c21(in, out); return;

	case 22: __vertical_scalar_packwithoutmask128_c22(in, out); return;

	case 23: __vertical_scalar_packwithoutmask128_c23(in, out); return;

	case 24: __vertical_scalar_packwithoutmask128_c24(in, out); return;

	case 25: __vertical_scalar_packwithoutmask128_c25(in, out); return;

	case 26: __vertical_scalar_packwithoutmask128_c26(in, out); return;

	case 27: __vertical_scalar_packwithoutmask128_c27(in, out); return;

	case 28: __vertical_scalar_packwithoutmask128_c28(in, out); return;

	case 29: __vertical_scalar_packwithoutmask128_c29(in, out); return;

	case 30: __vertical_scalar_packwithoutmask128_c30(in, out); return;

	case 31: __vertical_scalar_packwithoutmask128_c31(in, out); return;

	case 32: __vertical_scalar_packwithoutmask128_c32(in, out); return;

	default: return;
    }
}

template <bool IsRiceCoding>
inline void VerticalScalarUnpacker<IsRiceCoding>::packwithoutmask(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit) {
	for (uint32_t numberOfValuesPacked = 0; numberOfValuesPacked < PACKSIZE; numberOfValuesPacked += UNITPACKSIZE) {
		__vertical_scalar_packwithoutmask128(in, out, bit);
		in += UNITPACKSIZE;
		out += (UNITPACKSIZE * bit) / 32;
	}
}


template <bool IsRiceCoding>
inline void VerticalScalarUnpacker<IsRiceCoding>::__vertical_scalar_pack128(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit) {
    // Could have used function pointers instead of switch.
    // Switch calls do offer the compiler more opportunities for optimization in
    // theory. In this case, it makes no difference with a good compiler.
    switch(bit) {
	case 0: return;

	case 1: __vertical_scalar_pack128_c1(in, out); return;

	case 2: __vertical_scalar_pack128_c2(in, out); return;

	case 3: __vertical_scalar_pack128_c3(in, out); return;

	case 4: __vertical_scalar_pack128_c4(in, out); return;

	case 5: __vertical_scalar_pack128_c5(in, out); return;

	case 6: __vertical_scalar_pack128_c6(in, out); return;

	case 7: __vertical_scalar_pack128_c7(in, out); return;

	case 8: __vertical_scalar_pack128_c8(in, out); return;

	case 9: __vertical_scalar_pack128_c9(in, out); return;

	case 10: __vertical_scalar_pack128_c10(in, out); return;

	case 11: __vertical_scalar_pack128_c11(in, out); return;

	case 12: __vertical_scalar_pack128_c12(in, out); return;

	case 13: __vertical_scalar_pack128_c13(in, out); return;

	case 14: __vertical_scalar_pack128_c14(in, out); return;

	case 15: __vertical_scalar_pack128_c15(in, out); return;

	case 16: __vertical_scalar_pack128_c16(in, out); return;

	case 17: __vertical_scalar_pack128_c17(in, out); return;

	case 18: __vertical_scalar_pack128_c18(in, out); return;

	case 19: __vertical_scalar_pack128_c19(in, out); return;

	case 20: __vertical_scalar_pack128_c20(in, out); return;

	case 21: __vertical_scalar_pack128_c21(in, out); return;

	case 22: __vertical_scalar_pack128_c22(in, out); return;

	case 23: __vertical_scalar_pack128_c23(in, out); return;

	case 24: __vertical_scalar_pack128_c24(in, out); return;

	case 25: __vertical_scalar_pack128_c25(in, out); return;

	case 26: __vertical_scalar_pack128_c26(in, out); return;

	case 27: __vertical_scalar_pack128_c27(in, out); return;

	case 28: __vertical_scalar_pack128_c28(in, out); return;

	case 29: __vertical_scalar_pack128_c29(in, out); return;

	case 30: __vertical_scalar_pack128_c30(in, out); return;

	case 31: __vertical_scalar_pack128_c31(in, out); return;

	case 32: __vertical_scalar_pack128_c32(in, out); return;

	default: return;
    }
}

template <bool IsRiceCoding>
inline void VerticalScalarUnpacker<IsRiceCoding>::pack(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out, const uint32_t bit) {
	for (uint32_t numberOfValuesPacked = 0; numberOfValuesPacked < PACKSIZE; numberOfValuesPacked += UNITPACKSIZE) {
		__vertical_scalar_pack128(in, out, bit);
		in += UNITPACKSIZE;
		out += (UNITPACKSIZE * bit) / 32;
	}
}


#include "VerticalScalarUnpackerIMP.h"


#endif /* VERTICALSCALARUNPACKER_H_ */
