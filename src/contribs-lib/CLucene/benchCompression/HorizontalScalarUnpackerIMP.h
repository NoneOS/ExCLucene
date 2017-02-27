/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef HORIZONTALSCALARUNPACKERIMP_H_
#define HORIZONTALSCALARUNPACKERIMP_H_


// 0-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c0(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	memset(out, 0, 64 * sizeof(uint32_t));
}


// 1-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c1(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x01 ;
	out[1] = ( in[0] >> 1 ) & 0x01 ;
	out[2] = ( in[0] >> 2 ) & 0x01 ;
	out[3] = ( in[0] >> 3 ) & 0x01 ;
	out[4] = ( in[0] >> 4 ) & 0x01 ;
	out[5] = ( in[0] >> 5 ) & 0x01 ;
	out[6] = ( in[0] >> 6 ) & 0x01 ;
	out[7] = ( in[0] >> 7 ) & 0x01 ;
	out[8] = ( in[0] >> 8 ) & 0x01 ;
	out[9] = ( in[0] >> 9 ) & 0x01 ;
	out[10] = ( in[0] >> 10 ) & 0x01 ;
	out[11] = ( in[0] >> 11 ) & 0x01 ;
	out[12] = ( in[0] >> 12 ) & 0x01 ;
	out[13] = ( in[0] >> 13 ) & 0x01 ;
	out[14] = ( in[0] >> 14 ) & 0x01 ;
	out[15] = ( in[0] >> 15 ) & 0x01 ;
	out[16] = ( in[0] >> 16 ) & 0x01 ;
	out[17] = ( in[0] >> 17 ) & 0x01 ;
	out[18] = ( in[0] >> 18 ) & 0x01 ;
	out[19] = ( in[0] >> 19 ) & 0x01 ;
	out[20] = ( in[0] >> 20 ) & 0x01 ;
	out[21] = ( in[0] >> 21 ) & 0x01 ;
	out[22] = ( in[0] >> 22 ) & 0x01 ;
	out[23] = ( in[0] >> 23 ) & 0x01 ;
	out[24] = ( in[0] >> 24 ) & 0x01 ;
	out[25] = ( in[0] >> 25 ) & 0x01 ;
	out[26] = ( in[0] >> 26 ) & 0x01 ;
	out[27] = ( in[0] >> 27 ) & 0x01 ;
	out[28] = ( in[0] >> 28 ) & 0x01 ;
	out[29] = ( in[0] >> 29 ) & 0x01 ;
	out[30] = ( in[0] >> 30 ) & 0x01 ;
	out[31] = ( in[0] >> 31 ) ;

	out[32] = ( in[1] >> 0 ) & 0x01 ;
	out[33] = ( in[1] >> 1 ) & 0x01 ;
	out[34] = ( in[1] >> 2 ) & 0x01 ;
	out[35] = ( in[1] >> 3 ) & 0x01 ;
	out[36] = ( in[1] >> 4 ) & 0x01 ;
	out[37] = ( in[1] >> 5 ) & 0x01 ;
	out[38] = ( in[1] >> 6 ) & 0x01 ;
	out[39] = ( in[1] >> 7 ) & 0x01 ;
	out[40] = ( in[1] >> 8 ) & 0x01 ;
	out[41] = ( in[1] >> 9 ) & 0x01 ;
	out[42] = ( in[1] >> 10 ) & 0x01 ;
	out[43] = ( in[1] >> 11 ) & 0x01 ;
	out[44] = ( in[1] >> 12 ) & 0x01 ;
	out[45] = ( in[1] >> 13 ) & 0x01 ;
	out[46] = ( in[1] >> 14 ) & 0x01 ;
	out[47] = ( in[1] >> 15 ) & 0x01 ;
	out[48] = ( in[1] >> 16 ) & 0x01 ;
	out[49] = ( in[1] >> 17 ) & 0x01 ;
	out[50] = ( in[1] >> 18 ) & 0x01 ;
	out[51] = ( in[1] >> 19 ) & 0x01 ;
	out[52] = ( in[1] >> 20 ) & 0x01 ;
	out[53] = ( in[1] >> 21 ) & 0x01 ;
	out[54] = ( in[1] >> 22 ) & 0x01 ;
	out[55] = ( in[1] >> 23 ) & 0x01 ;
	out[56] = ( in[1] >> 24 ) & 0x01 ;
	out[57] = ( in[1] >> 25 ) & 0x01 ;
	out[58] = ( in[1] >> 26 ) & 0x01 ;
	out[59] = ( in[1] >> 27 ) & 0x01 ;
	out[60] = ( in[1] >> 28 ) & 0x01 ;
	out[61] = ( in[1] >> 29 ) & 0x01 ;
	out[62] = ( in[1] >> 30 ) & 0x01 ;
	out[63] = ( in[1] >> 31 ) ;
}


// 2-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c2(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x03 ;
	out[1] = ( in[0] >> 2 ) & 0x03 ;
	out[2] = ( in[0] >> 4 ) & 0x03 ;
	out[3] = ( in[0] >> 6 ) & 0x03 ;
	out[4] = ( in[0] >> 8 ) & 0x03 ;
	out[5] = ( in[0] >> 10 ) & 0x03 ;
	out[6] = ( in[0] >> 12 ) & 0x03 ;
	out[7] = ( in[0] >> 14 ) & 0x03 ;
	out[8] = ( in[0] >> 16 ) & 0x03 ;
	out[9] = ( in[0] >> 18 ) & 0x03 ;
	out[10] = ( in[0] >> 20 ) & 0x03 ;
	out[11] = ( in[0] >> 22 ) & 0x03 ;
	out[12] = ( in[0] >> 24 ) & 0x03 ;
	out[13] = ( in[0] >> 26 ) & 0x03 ;
	out[14] = ( in[0] >> 28 ) & 0x03 ;
	out[15] = ( in[0] >> 30 ) ;

	out[16] = ( in[1] >> 0 ) & 0x03 ;
	out[17] = ( in[1] >> 2 ) & 0x03 ;
	out[18] = ( in[1] >> 4 ) & 0x03 ;
	out[19] = ( in[1] >> 6 ) & 0x03 ;
	out[20] = ( in[1] >> 8 ) & 0x03 ;
	out[21] = ( in[1] >> 10 ) & 0x03 ;
	out[22] = ( in[1] >> 12 ) & 0x03 ;
	out[23] = ( in[1] >> 14 ) & 0x03 ;
	out[24] = ( in[1] >> 16 ) & 0x03 ;
	out[25] = ( in[1] >> 18 ) & 0x03 ;
	out[26] = ( in[1] >> 20 ) & 0x03 ;
	out[27] = ( in[1] >> 22 ) & 0x03 ;
	out[28] = ( in[1] >> 24 ) & 0x03 ;
	out[29] = ( in[1] >> 26 ) & 0x03 ;
	out[30] = ( in[1] >> 28 ) & 0x03 ;
	out[31] = ( in[1] >> 30 ) ;

	out[32] = ( in[2] >> 0 ) & 0x03 ;
	out[33] = ( in[2] >> 2 ) & 0x03 ;
	out[34] = ( in[2] >> 4 ) & 0x03 ;
	out[35] = ( in[2] >> 6 ) & 0x03 ;
	out[36] = ( in[2] >> 8 ) & 0x03 ;
	out[37] = ( in[2] >> 10 ) & 0x03 ;
	out[38] = ( in[2] >> 12 ) & 0x03 ;
	out[39] = ( in[2] >> 14 ) & 0x03 ;
	out[40] = ( in[2] >> 16 ) & 0x03 ;
	out[41] = ( in[2] >> 18 ) & 0x03 ;
	out[42] = ( in[2] >> 20 ) & 0x03 ;
	out[43] = ( in[2] >> 22 ) & 0x03 ;
	out[44] = ( in[2] >> 24 ) & 0x03 ;
	out[45] = ( in[2] >> 26 ) & 0x03 ;
	out[46] = ( in[2] >> 28 ) & 0x03 ;
	out[47] = ( in[2] >> 30 ) ;

	out[48] = ( in[3] >> 0 ) & 0x03 ;
	out[49] = ( in[3] >> 2 ) & 0x03 ;
	out[50] = ( in[3] >> 4 ) & 0x03 ;
	out[51] = ( in[3] >> 6 ) & 0x03 ;
	out[52] = ( in[3] >> 8 ) & 0x03 ;
	out[53] = ( in[3] >> 10 ) & 0x03 ;
	out[54] = ( in[3] >> 12 ) & 0x03 ;
	out[55] = ( in[3] >> 14 ) & 0x03 ;
	out[56] = ( in[3] >> 16 ) & 0x03 ;
	out[57] = ( in[3] >> 18 ) & 0x03 ;
	out[58] = ( in[3] >> 20 ) & 0x03 ;
	out[59] = ( in[3] >> 22 ) & 0x03 ;
	out[60] = ( in[3] >> 24 ) & 0x03 ;
	out[61] = ( in[3] >> 26 ) & 0x03 ;
	out[62] = ( in[3] >> 28 ) & 0x03 ;
	out[63] = ( in[3] >> 30 ) ;
}


// 3-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c3(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x07 ;
	out[1] = ( in[0] >> 3 ) & 0x07 ;
	out[2] = ( in[0] >> 6 ) & 0x07 ;
	out[3] = ( in[0] >> 9 ) & 0x07 ;
	out[4] = ( in[0] >> 12 ) & 0x07 ;
	out[5] = ( in[0] >> 15 ) & 0x07 ;
	out[6] = ( in[0] >> 18 ) & 0x07 ;
	out[7] = ( in[0] >> 21 ) & 0x07 ;
	out[8] = ( in[0] >> 24 ) & 0x07 ;
	out[9] = ( in[0] >> 27 ) & 0x07 ;
	out[10] = ( in[0] >> 30 ) ;

	out[10] |= ( in[1] << ( 32 - 30 ) ) & 0x07 ;
	out[11] = ( in[1] >> 1 ) & 0x07 ;
	out[12] = ( in[1] >> 4 ) & 0x07 ;
	out[13] = ( in[1] >> 7 ) & 0x07 ;
	out[14] = ( in[1] >> 10 ) & 0x07 ;
	out[15] = ( in[1] >> 13 ) & 0x07 ;
	out[16] = ( in[1] >> 16 ) & 0x07 ;
	out[17] = ( in[1] >> 19 ) & 0x07 ;
	out[18] = ( in[1] >> 22 ) & 0x07 ;
	out[19] = ( in[1] >> 25 ) & 0x07 ;
	out[20] = ( in[1] >> 28 ) & 0x07 ;
	out[21] = ( in[1] >> 31 ) ;

	out[21] |= ( in[2] << ( 32 - 31 ) ) & 0x07 ;
	out[22] = ( in[2] >> 2 ) & 0x07 ;
	out[23] = ( in[2] >> 5 ) & 0x07 ;
	out[24] = ( in[2] >> 8 ) & 0x07 ;
	out[25] = ( in[2] >> 11 ) & 0x07 ;
	out[26] = ( in[2] >> 14 ) & 0x07 ;
	out[27] = ( in[2] >> 17 ) & 0x07 ;
	out[28] = ( in[2] >> 20 ) & 0x07 ;
	out[29] = ( in[2] >> 23 ) & 0x07 ;
	out[30] = ( in[2] >> 26 ) & 0x07 ;
	out[31] = ( in[2] >> 29 ) ;

	out[32] = ( in[3] >> 0 ) & 0x07 ;
	out[33] = ( in[3] >> 3 ) & 0x07 ;
	out[34] = ( in[3] >> 6 ) & 0x07 ;
	out[35] = ( in[3] >> 9 ) & 0x07 ;
	out[36] = ( in[3] >> 12 ) & 0x07 ;
	out[37] = ( in[3] >> 15 ) & 0x07 ;
	out[38] = ( in[3] >> 18 ) & 0x07 ;
	out[39] = ( in[3] >> 21 ) & 0x07 ;
	out[40] = ( in[3] >> 24 ) & 0x07 ;
	out[41] = ( in[3] >> 27 ) & 0x07 ;
	out[42] = ( in[3] >> 30 ) ;

	out[42] |= ( in[4] << ( 32 - 30 ) ) & 0x07 ;
	out[43] = ( in[4] >> 1 ) & 0x07 ;
	out[44] = ( in[4] >> 4 ) & 0x07 ;
	out[45] = ( in[4] >> 7 ) & 0x07 ;
	out[46] = ( in[4] >> 10 ) & 0x07 ;
	out[47] = ( in[4] >> 13 ) & 0x07 ;
	out[48] = ( in[4] >> 16 ) & 0x07 ;
	out[49] = ( in[4] >> 19 ) & 0x07 ;
	out[50] = ( in[4] >> 22 ) & 0x07 ;
	out[51] = ( in[4] >> 25 ) & 0x07 ;
	out[52] = ( in[4] >> 28 ) & 0x07 ;
	out[53] = ( in[4] >> 31 ) ;

	out[53] |= ( in[5] << ( 32 - 31 ) ) & 0x07 ;
	out[54] = ( in[5] >> 2 ) & 0x07 ;
	out[55] = ( in[5] >> 5 ) & 0x07 ;
	out[56] = ( in[5] >> 8 ) & 0x07 ;
	out[57] = ( in[5] >> 11 ) & 0x07 ;
	out[58] = ( in[5] >> 14 ) & 0x07 ;
	out[59] = ( in[5] >> 17 ) & 0x07 ;
	out[60] = ( in[5] >> 20 ) & 0x07 ;
	out[61] = ( in[5] >> 23 ) & 0x07 ;
	out[62] = ( in[5] >> 26 ) & 0x07 ;
	out[63] = ( in[5] >> 29 ) ;
}


// 4-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c4(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x0f ;
	out[1] = ( in[0] >> 4 ) & 0x0f ;
	out[2] = ( in[0] >> 8 ) & 0x0f ;
	out[3] = ( in[0] >> 12 ) & 0x0f ;
	out[4] = ( in[0] >> 16 ) & 0x0f ;
	out[5] = ( in[0] >> 20 ) & 0x0f ;
	out[6] = ( in[0] >> 24 ) & 0x0f ;
	out[7] = ( in[0] >> 28 ) ;

	out[8] = ( in[1] >> 0 ) & 0x0f ;
	out[9] = ( in[1] >> 4 ) & 0x0f ;
	out[10] = ( in[1] >> 8 ) & 0x0f ;
	out[11] = ( in[1] >> 12 ) & 0x0f ;
	out[12] = ( in[1] >> 16 ) & 0x0f ;
	out[13] = ( in[1] >> 20 ) & 0x0f ;
	out[14] = ( in[1] >> 24 ) & 0x0f ;
	out[15] = ( in[1] >> 28 ) ;

	out[16] = ( in[2] >> 0 ) & 0x0f ;
	out[17] = ( in[2] >> 4 ) & 0x0f ;
	out[18] = ( in[2] >> 8 ) & 0x0f ;
	out[19] = ( in[2] >> 12 ) & 0x0f ;
	out[20] = ( in[2] >> 16 ) & 0x0f ;
	out[21] = ( in[2] >> 20 ) & 0x0f ;
	out[22] = ( in[2] >> 24 ) & 0x0f ;
	out[23] = ( in[2] >> 28 ) ;

	out[24] = ( in[3] >> 0 ) & 0x0f ;
	out[25] = ( in[3] >> 4 ) & 0x0f ;
	out[26] = ( in[3] >> 8 ) & 0x0f ;
	out[27] = ( in[3] >> 12 ) & 0x0f ;
	out[28] = ( in[3] >> 16 ) & 0x0f ;
	out[29] = ( in[3] >> 20 ) & 0x0f ;
	out[30] = ( in[3] >> 24 ) & 0x0f ;
	out[31] = ( in[3] >> 28 ) ;

	out[32] = ( in[4] >> 0 ) & 0x0f ;
	out[33] = ( in[4] >> 4 ) & 0x0f ;
	out[34] = ( in[4] >> 8 ) & 0x0f ;
	out[35] = ( in[4] >> 12 ) & 0x0f ;
	out[36] = ( in[4] >> 16 ) & 0x0f ;
	out[37] = ( in[4] >> 20 ) & 0x0f ;
	out[38] = ( in[4] >> 24 ) & 0x0f ;
	out[39] = ( in[4] >> 28 ) ;

	out[40] = ( in[5] >> 0 ) & 0x0f ;
	out[41] = ( in[5] >> 4 ) & 0x0f ;
	out[42] = ( in[5] >> 8 ) & 0x0f ;
	out[43] = ( in[5] >> 12 ) & 0x0f ;
	out[44] = ( in[5] >> 16 ) & 0x0f ;
	out[45] = ( in[5] >> 20 ) & 0x0f ;
	out[46] = ( in[5] >> 24 ) & 0x0f ;
	out[47] = ( in[5] >> 28 ) ;

	out[48] = ( in[6] >> 0 ) & 0x0f ;
	out[49] = ( in[6] >> 4 ) & 0x0f ;
	out[50] = ( in[6] >> 8 ) & 0x0f ;
	out[51] = ( in[6] >> 12 ) & 0x0f ;
	out[52] = ( in[6] >> 16 ) & 0x0f ;
	out[53] = ( in[6] >> 20 ) & 0x0f ;
	out[54] = ( in[6] >> 24 ) & 0x0f ;
	out[55] = ( in[6] >> 28 ) ;

	out[56] = ( in[7] >> 0 ) & 0x0f ;
	out[57] = ( in[7] >> 4 ) & 0x0f ;
	out[58] = ( in[7] >> 8 ) & 0x0f ;
	out[59] = ( in[7] >> 12 ) & 0x0f ;
	out[60] = ( in[7] >> 16 ) & 0x0f ;
	out[61] = ( in[7] >> 20 ) & 0x0f ;
	out[62] = ( in[7] >> 24 ) & 0x0f ;
	out[63] = ( in[7] >> 28 ) ;
}


// 5-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c5(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x1f ;
	out[1] = ( in[0] >> 5 ) & 0x1f ;
	out[2] = ( in[0] >> 10 ) & 0x1f ;
	out[3] = ( in[0] >> 15 ) & 0x1f ;
	out[4] = ( in[0] >> 20 ) & 0x1f ;
	out[5] = ( in[0] >> 25 ) & 0x1f ;
	out[6] = ( in[0] >> 30 ) ;

	out[6] |= ( in[1] << ( 32 - 30 ) ) & 0x1f ;
	out[7] = ( in[1] >> 3 ) & 0x1f ;
	out[8] = ( in[1] >> 8 ) & 0x1f ;
	out[9] = ( in[1] >> 13 ) & 0x1f ;
	out[10] = ( in[1] >> 18 ) & 0x1f ;
	out[11] = ( in[1] >> 23 ) & 0x1f ;
	out[12] = ( in[1] >> 28 ) ;

	out[12] |= ( in[2] << ( 32 - 28 ) ) & 0x1f ;
	out[13] = ( in[2] >> 1 ) & 0x1f ;
	out[14] = ( in[2] >> 6 ) & 0x1f ;
	out[15] = ( in[2] >> 11 ) & 0x1f ;
	out[16] = ( in[2] >> 16 ) & 0x1f ;
	out[17] = ( in[2] >> 21 ) & 0x1f ;
	out[18] = ( in[2] >> 26 ) & 0x1f ;
	out[19] = ( in[2] >> 31 ) ;

	out[19] |= ( in[3] << ( 32 - 31 ) ) & 0x1f ;
	out[20] = ( in[3] >> 4 ) & 0x1f ;
	out[21] = ( in[3] >> 9 ) & 0x1f ;
	out[22] = ( in[3] >> 14 ) & 0x1f ;
	out[23] = ( in[3] >> 19 ) & 0x1f ;
	out[24] = ( in[3] >> 24 ) & 0x1f ;
	out[25] = ( in[3] >> 29 ) ;

	out[25] |= ( in[4] << ( 32 - 29 ) ) & 0x1f ;
	out[26] = ( in[4] >> 2 ) & 0x1f ;
	out[27] = ( in[4] >> 7 ) & 0x1f ;
	out[28] = ( in[4] >> 12 ) & 0x1f ;
	out[29] = ( in[4] >> 17 ) & 0x1f ;
	out[30] = ( in[4] >> 22 ) & 0x1f ;
	out[31] = ( in[4] >> 27 ) ;

	out[32] = ( in[5] >> 0 ) & 0x1f ;
	out[33] = ( in[5] >> 5 ) & 0x1f ;
	out[34] = ( in[5] >> 10 ) & 0x1f ;
	out[35] = ( in[5] >> 15 ) & 0x1f ;
	out[36] = ( in[5] >> 20 ) & 0x1f ;
	out[37] = ( in[5] >> 25 ) & 0x1f ;
	out[38] = ( in[5] >> 30 ) ;

	out[38] |= ( in[6] << ( 32 - 30 ) ) & 0x1f ;
	out[39] = ( in[6] >> 3 ) & 0x1f ;
	out[40] = ( in[6] >> 8 ) & 0x1f ;
	out[41] = ( in[6] >> 13 ) & 0x1f ;
	out[42] = ( in[6] >> 18 ) & 0x1f ;
	out[43] = ( in[6] >> 23 ) & 0x1f ;
	out[44] = ( in[6] >> 28 ) ;

	out[44] |= ( in[7] << ( 32 - 28 ) ) & 0x1f ;
	out[45] = ( in[7] >> 1 ) & 0x1f ;
	out[46] = ( in[7] >> 6 ) & 0x1f ;
	out[47] = ( in[7] >> 11 ) & 0x1f ;
	out[48] = ( in[7] >> 16 ) & 0x1f ;
	out[49] = ( in[7] >> 21 ) & 0x1f ;
	out[50] = ( in[7] >> 26 ) & 0x1f ;
	out[51] = ( in[7] >> 31 ) ;

	out[51] |= ( in[8] << ( 32 - 31 ) ) & 0x1f ;
	out[52] = ( in[8] >> 4 ) & 0x1f ;
	out[53] = ( in[8] >> 9 ) & 0x1f ;
	out[54] = ( in[8] >> 14 ) & 0x1f ;
	out[55] = ( in[8] >> 19 ) & 0x1f ;
	out[56] = ( in[8] >> 24 ) & 0x1f ;
	out[57] = ( in[8] >> 29 ) ;

	out[57] |= ( in[9] << ( 32 - 29 ) ) & 0x1f ;
	out[58] = ( in[9] >> 2 ) & 0x1f ;
	out[59] = ( in[9] >> 7 ) & 0x1f ;
	out[60] = ( in[9] >> 12 ) & 0x1f ;
	out[61] = ( in[9] >> 17 ) & 0x1f ;
	out[62] = ( in[9] >> 22 ) & 0x1f ;
	out[63] = ( in[9] >> 27 ) ;
}


// 6-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c6(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x3f ;
	out[1] = ( in[0] >> 6 ) & 0x3f ;
	out[2] = ( in[0] >> 12 ) & 0x3f ;
	out[3] = ( in[0] >> 18 ) & 0x3f ;
	out[4] = ( in[0] >> 24 ) & 0x3f ;
	out[5] = ( in[0] >> 30 ) ;

	out[5] |= ( in[1] << ( 32 - 30 ) ) & 0x3f ;
	out[6] = ( in[1] >> 4 ) & 0x3f ;
	out[7] = ( in[1] >> 10 ) & 0x3f ;
	out[8] = ( in[1] >> 16 ) & 0x3f ;
	out[9] = ( in[1] >> 22 ) & 0x3f ;
	out[10] = ( in[1] >> 28 ) ;

	out[10] |= ( in[2] << ( 32 - 28 ) ) & 0x3f ;
	out[11] = ( in[2] >> 2 ) & 0x3f ;
	out[12] = ( in[2] >> 8 ) & 0x3f ;
	out[13] = ( in[2] >> 14 ) & 0x3f ;
	out[14] = ( in[2] >> 20 ) & 0x3f ;
	out[15] = ( in[2] >> 26 ) ;

	out[16] = ( in[3] >> 0 ) & 0x3f ;
	out[17] = ( in[3] >> 6 ) & 0x3f ;
	out[18] = ( in[3] >> 12 ) & 0x3f ;
	out[19] = ( in[3] >> 18 ) & 0x3f ;
	out[20] = ( in[3] >> 24 ) & 0x3f ;
	out[21] = ( in[3] >> 30 ) ;

	out[21] |= ( in[4] << ( 32 - 30 ) ) & 0x3f ;
	out[22] = ( in[4] >> 4 ) & 0x3f ;
	out[23] = ( in[4] >> 10 ) & 0x3f ;
	out[24] = ( in[4] >> 16 ) & 0x3f ;
	out[25] = ( in[4] >> 22 ) & 0x3f ;
	out[26] = ( in[4] >> 28 ) ;

	out[26] |= ( in[5] << ( 32 - 28 ) ) & 0x3f ;
	out[27] = ( in[5] >> 2 ) & 0x3f ;
	out[28] = ( in[5] >> 8 ) & 0x3f ;
	out[29] = ( in[5] >> 14 ) & 0x3f ;
	out[30] = ( in[5] >> 20 ) & 0x3f ;
	out[31] = ( in[5] >> 26 ) ;

	out[32] = ( in[6] >> 0 ) & 0x3f ;
	out[33] = ( in[6] >> 6 ) & 0x3f ;
	out[34] = ( in[6] >> 12 ) & 0x3f ;
	out[35] = ( in[6] >> 18 ) & 0x3f ;
	out[36] = ( in[6] >> 24 ) & 0x3f ;
	out[37] = ( in[6] >> 30 ) ;

	out[37] |= ( in[7] << ( 32 - 30 ) ) & 0x3f ;
	out[38] = ( in[7] >> 4 ) & 0x3f ;
	out[39] = ( in[7] >> 10 ) & 0x3f ;
	out[40] = ( in[7] >> 16 ) & 0x3f ;
	out[41] = ( in[7] >> 22 ) & 0x3f ;
	out[42] = ( in[7] >> 28 ) ;

	out[42] |= ( in[8] << ( 32 - 28 ) ) & 0x3f ;
	out[43] = ( in[8] >> 2 ) & 0x3f ;
	out[44] = ( in[8] >> 8 ) & 0x3f ;
	out[45] = ( in[8] >> 14 ) & 0x3f ;
	out[46] = ( in[8] >> 20 ) & 0x3f ;
	out[47] = ( in[8] >> 26 ) ;

	out[48] = ( in[9] >> 0 ) & 0x3f ;
	out[49] = ( in[9] >> 6 ) & 0x3f ;
	out[50] = ( in[9] >> 12 ) & 0x3f ;
	out[51] = ( in[9] >> 18 ) & 0x3f ;
	out[52] = ( in[9] >> 24 ) & 0x3f ;
	out[53] = ( in[9] >> 30 ) ;

	out[53] |= ( in[10] << ( 32 - 30 ) ) & 0x3f ;
	out[54] = ( in[10] >> 4 ) & 0x3f ;
	out[55] = ( in[10] >> 10 ) & 0x3f ;
	out[56] = ( in[10] >> 16 ) & 0x3f ;
	out[57] = ( in[10] >> 22 ) & 0x3f ;
	out[58] = ( in[10] >> 28 ) ;

	out[58] |= ( in[11] << ( 32 - 28 ) ) & 0x3f ;
	out[59] = ( in[11] >> 2 ) & 0x3f ;
	out[60] = ( in[11] >> 8 ) & 0x3f ;
	out[61] = ( in[11] >> 14 ) & 0x3f ;
	out[62] = ( in[11] >> 20 ) & 0x3f ;
	out[63] = ( in[11] >> 26 ) ;
}


// 7-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c7(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x7f ;
	out[1] = ( in[0] >> 7 ) & 0x7f ;
	out[2] = ( in[0] >> 14 ) & 0x7f ;
	out[3] = ( in[0] >> 21 ) & 0x7f ;
	out[4] = ( in[0] >> 28 ) ;

	out[4] |= ( in[1] << ( 32 - 28 ) ) & 0x7f ;
	out[5] = ( in[1] >> 3 ) & 0x7f ;
	out[6] = ( in[1] >> 10 ) & 0x7f ;
	out[7] = ( in[1] >> 17 ) & 0x7f ;
	out[8] = ( in[1] >> 24 ) & 0x7f ;
	out[9] = ( in[1] >> 31 ) ;

	out[9] |= ( in[2] << ( 32 - 31 ) ) & 0x7f ;
	out[10] = ( in[2] >> 6 ) & 0x7f ;
	out[11] = ( in[2] >> 13 ) & 0x7f ;
	out[12] = ( in[2] >> 20 ) & 0x7f ;
	out[13] = ( in[2] >> 27 ) ;

	out[13] |= ( in[3] << ( 32 - 27 ) ) & 0x7f ;
	out[14] = ( in[3] >> 2 ) & 0x7f ;
	out[15] = ( in[3] >> 9 ) & 0x7f ;
	out[16] = ( in[3] >> 16 ) & 0x7f ;
	out[17] = ( in[3] >> 23 ) & 0x7f ;
	out[18] = ( in[3] >> 30 ) ;

	out[18] |= ( in[4] << ( 32 - 30 ) ) & 0x7f ;
	out[19] = ( in[4] >> 5 ) & 0x7f ;
	out[20] = ( in[4] >> 12 ) & 0x7f ;
	out[21] = ( in[4] >> 19 ) & 0x7f ;
	out[22] = ( in[4] >> 26 ) ;

	out[22] |= ( in[5] << ( 32 - 26 ) ) & 0x7f ;
	out[23] = ( in[5] >> 1 ) & 0x7f ;
	out[24] = ( in[5] >> 8 ) & 0x7f ;
	out[25] = ( in[5] >> 15 ) & 0x7f ;
	out[26] = ( in[5] >> 22 ) & 0x7f ;
	out[27] = ( in[5] >> 29 ) ;

	out[27] |= ( in[6] << ( 32 - 29 ) ) & 0x7f ;
	out[28] = ( in[6] >> 4 ) & 0x7f ;
	out[29] = ( in[6] >> 11 ) & 0x7f ;
	out[30] = ( in[6] >> 18 ) & 0x7f ;
	out[31] = ( in[6] >> 25 ) ;

	out[32] = ( in[7] >> 0 ) & 0x7f ;
	out[33] = ( in[7] >> 7 ) & 0x7f ;
	out[34] = ( in[7] >> 14 ) & 0x7f ;
	out[35] = ( in[7] >> 21 ) & 0x7f ;
	out[36] = ( in[7] >> 28 ) ;

	out[36] |= ( in[8] << ( 32 - 28 ) ) & 0x7f ;
	out[37] = ( in[8] >> 3 ) & 0x7f ;
	out[38] = ( in[8] >> 10 ) & 0x7f ;
	out[39] = ( in[8] >> 17 ) & 0x7f ;
	out[40] = ( in[8] >> 24 ) & 0x7f ;
	out[41] = ( in[8] >> 31 ) ;

	out[41] |= ( in[9] << ( 32 - 31 ) ) & 0x7f ;
	out[42] = ( in[9] >> 6 ) & 0x7f ;
	out[43] = ( in[9] >> 13 ) & 0x7f ;
	out[44] = ( in[9] >> 20 ) & 0x7f ;
	out[45] = ( in[9] >> 27 ) ;

	out[45] |= ( in[10] << ( 32 - 27 ) ) & 0x7f ;
	out[46] = ( in[10] >> 2 ) & 0x7f ;
	out[47] = ( in[10] >> 9 ) & 0x7f ;
	out[48] = ( in[10] >> 16 ) & 0x7f ;
	out[49] = ( in[10] >> 23 ) & 0x7f ;
	out[50] = ( in[10] >> 30 ) ;

	out[50] |= ( in[11] << ( 32 - 30 ) ) & 0x7f ;
	out[51] = ( in[11] >> 5 ) & 0x7f ;
	out[52] = ( in[11] >> 12 ) & 0x7f ;
	out[53] = ( in[11] >> 19 ) & 0x7f ;
	out[54] = ( in[11] >> 26 ) ;

	out[54] |= ( in[12] << ( 32 - 26 ) ) & 0x7f ;
	out[55] = ( in[12] >> 1 ) & 0x7f ;
	out[56] = ( in[12] >> 8 ) & 0x7f ;
	out[57] = ( in[12] >> 15 ) & 0x7f ;
	out[58] = ( in[12] >> 22 ) & 0x7f ;
	out[59] = ( in[12] >> 29 ) ;

	out[59] |= ( in[13] << ( 32 - 29 ) ) & 0x7f ;
	out[60] = ( in[13] >> 4 ) & 0x7f ;
	out[61] = ( in[13] >> 11 ) & 0x7f ;
	out[62] = ( in[13] >> 18 ) & 0x7f ;
	out[63] = ( in[13] >> 25 ) ;
}


// 8-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c8(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0xff ;
	out[1] = ( in[0] >> 8 ) & 0xff ;
	out[2] = ( in[0] >> 16 ) & 0xff ;
	out[3] = ( in[0] >> 24 ) ;

	out[4] = ( in[1] >> 0 ) & 0xff ;
	out[5] = ( in[1] >> 8 ) & 0xff ;
	out[6] = ( in[1] >> 16 ) & 0xff ;
	out[7] = ( in[1] >> 24 ) ;

	out[8] = ( in[2] >> 0 ) & 0xff ;
	out[9] = ( in[2] >> 8 ) & 0xff ;
	out[10] = ( in[2] >> 16 ) & 0xff ;
	out[11] = ( in[2] >> 24 ) ;

	out[12] = ( in[3] >> 0 ) & 0xff ;
	out[13] = ( in[3] >> 8 ) & 0xff ;
	out[14] = ( in[3] >> 16 ) & 0xff ;
	out[15] = ( in[3] >> 24 ) ;

	out[16] = ( in[4] >> 0 ) & 0xff ;
	out[17] = ( in[4] >> 8 ) & 0xff ;
	out[18] = ( in[4] >> 16 ) & 0xff ;
	out[19] = ( in[4] >> 24 ) ;

	out[20] = ( in[5] >> 0 ) & 0xff ;
	out[21] = ( in[5] >> 8 ) & 0xff ;
	out[22] = ( in[5] >> 16 ) & 0xff ;
	out[23] = ( in[5] >> 24 ) ;

	out[24] = ( in[6] >> 0 ) & 0xff ;
	out[25] = ( in[6] >> 8 ) & 0xff ;
	out[26] = ( in[6] >> 16 ) & 0xff ;
	out[27] = ( in[6] >> 24 ) ;

	out[28] = ( in[7] >> 0 ) & 0xff ;
	out[29] = ( in[7] >> 8 ) & 0xff ;
	out[30] = ( in[7] >> 16 ) & 0xff ;
	out[31] = ( in[7] >> 24 ) ;

	out[32] = ( in[8] >> 0 ) & 0xff ;
	out[33] = ( in[8] >> 8 ) & 0xff ;
	out[34] = ( in[8] >> 16 ) & 0xff ;
	out[35] = ( in[8] >> 24 ) ;

	out[36] = ( in[9] >> 0 ) & 0xff ;
	out[37] = ( in[9] >> 8 ) & 0xff ;
	out[38] = ( in[9] >> 16 ) & 0xff ;
	out[39] = ( in[9] >> 24 ) ;

	out[40] = ( in[10] >> 0 ) & 0xff ;
	out[41] = ( in[10] >> 8 ) & 0xff ;
	out[42] = ( in[10] >> 16 ) & 0xff ;
	out[43] = ( in[10] >> 24 ) ;

	out[44] = ( in[11] >> 0 ) & 0xff ;
	out[45] = ( in[11] >> 8 ) & 0xff ;
	out[46] = ( in[11] >> 16 ) & 0xff ;
	out[47] = ( in[11] >> 24 ) ;

	out[48] = ( in[12] >> 0 ) & 0xff ;
	out[49] = ( in[12] >> 8 ) & 0xff ;
	out[50] = ( in[12] >> 16 ) & 0xff ;
	out[51] = ( in[12] >> 24 ) ;

	out[52] = ( in[13] >> 0 ) & 0xff ;
	out[53] = ( in[13] >> 8 ) & 0xff ;
	out[54] = ( in[13] >> 16 ) & 0xff ;
	out[55] = ( in[13] >> 24 ) ;

	out[56] = ( in[14] >> 0 ) & 0xff ;
	out[57] = ( in[14] >> 8 ) & 0xff ;
	out[58] = ( in[14] >> 16 ) & 0xff ;
	out[59] = ( in[14] >> 24 ) ;

	out[60] = ( in[15] >> 0 ) & 0xff ;
	out[61] = ( in[15] >> 8 ) & 0xff ;
	out[62] = ( in[15] >> 16 ) & 0xff ;
	out[63] = ( in[15] >> 24 ) ;
}


// 9-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c9(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x01ff ;
	out[1] = ( in[0] >> 9 ) & 0x01ff ;
	out[2] = ( in[0] >> 18 ) & 0x01ff ;
	out[3] = ( in[0] >> 27 ) ;

	out[3] |= ( in[1] << ( 32 - 27 ) ) & 0x01ff ;
	out[4] = ( in[1] >> 4 ) & 0x01ff ;
	out[5] = ( in[1] >> 13 ) & 0x01ff ;
	out[6] = ( in[1] >> 22 ) & 0x01ff ;
	out[7] = ( in[1] >> 31 ) ;

	out[7] |= ( in[2] << ( 32 - 31 ) ) & 0x01ff ;
	out[8] = ( in[2] >> 8 ) & 0x01ff ;
	out[9] = ( in[2] >> 17 ) & 0x01ff ;
	out[10] = ( in[2] >> 26 ) ;

	out[10] |= ( in[3] << ( 32 - 26 ) ) & 0x01ff ;
	out[11] = ( in[3] >> 3 ) & 0x01ff ;
	out[12] = ( in[3] >> 12 ) & 0x01ff ;
	out[13] = ( in[3] >> 21 ) & 0x01ff ;
	out[14] = ( in[3] >> 30 ) ;

	out[14] |= ( in[4] << ( 32 - 30 ) ) & 0x01ff ;
	out[15] = ( in[4] >> 7 ) & 0x01ff ;
	out[16] = ( in[4] >> 16 ) & 0x01ff ;
	out[17] = ( in[4] >> 25 ) ;

	out[17] |= ( in[5] << ( 32 - 25 ) ) & 0x01ff ;
	out[18] = ( in[5] >> 2 ) & 0x01ff ;
	out[19] = ( in[5] >> 11 ) & 0x01ff ;
	out[20] = ( in[5] >> 20 ) & 0x01ff ;
	out[21] = ( in[5] >> 29 ) ;

	out[21] |= ( in[6] << ( 32 - 29 ) ) & 0x01ff ;
	out[22] = ( in[6] >> 6 ) & 0x01ff ;
	out[23] = ( in[6] >> 15 ) & 0x01ff ;
	out[24] = ( in[6] >> 24 ) ;

	out[24] |= ( in[7] << ( 32 - 24 ) ) & 0x01ff ;
	out[25] = ( in[7] >> 1 ) & 0x01ff ;
	out[26] = ( in[7] >> 10 ) & 0x01ff ;
	out[27] = ( in[7] >> 19 ) & 0x01ff ;
	out[28] = ( in[7] >> 28 ) ;

	out[28] |= ( in[8] << ( 32 - 28 ) ) & 0x01ff ;
	out[29] = ( in[8] >> 5 ) & 0x01ff ;
	out[30] = ( in[8] >> 14 ) & 0x01ff ;
	out[31] = ( in[8] >> 23 ) ;

	out[32] = ( in[9] >> 0 ) & 0x01ff ;
	out[33] = ( in[9] >> 9 ) & 0x01ff ;
	out[34] = ( in[9] >> 18 ) & 0x01ff ;
	out[35] = ( in[9] >> 27 ) ;

	out[35] |= ( in[10] << ( 32 - 27 ) ) & 0x01ff ;
	out[36] = ( in[10] >> 4 ) & 0x01ff ;
	out[37] = ( in[10] >> 13 ) & 0x01ff ;
	out[38] = ( in[10] >> 22 ) & 0x01ff ;
	out[39] = ( in[10] >> 31 ) ;

	out[39] |= ( in[11] << ( 32 - 31 ) ) & 0x01ff ;
	out[40] = ( in[11] >> 8 ) & 0x01ff ;
	out[41] = ( in[11] >> 17 ) & 0x01ff ;
	out[42] = ( in[11] >> 26 ) ;

	out[42] |= ( in[12] << ( 32 - 26 ) ) & 0x01ff ;
	out[43] = ( in[12] >> 3 ) & 0x01ff ;
	out[44] = ( in[12] >> 12 ) & 0x01ff ;
	out[45] = ( in[12] >> 21 ) & 0x01ff ;
	out[46] = ( in[12] >> 30 ) ;

	out[46] |= ( in[13] << ( 32 - 30 ) ) & 0x01ff ;
	out[47] = ( in[13] >> 7 ) & 0x01ff ;
	out[48] = ( in[13] >> 16 ) & 0x01ff ;
	out[49] = ( in[13] >> 25 ) ;

	out[49] |= ( in[14] << ( 32 - 25 ) ) & 0x01ff ;
	out[50] = ( in[14] >> 2 ) & 0x01ff ;
	out[51] = ( in[14] >> 11 ) & 0x01ff ;
	out[52] = ( in[14] >> 20 ) & 0x01ff ;
	out[53] = ( in[14] >> 29 ) ;

	out[53] |= ( in[15] << ( 32 - 29 ) ) & 0x01ff ;
	out[54] = ( in[15] >> 6 ) & 0x01ff ;
	out[55] = ( in[15] >> 15 ) & 0x01ff ;
	out[56] = ( in[15] >> 24 ) ;

	out[56] |= ( in[16] << ( 32 - 24 ) ) & 0x01ff ;
	out[57] = ( in[16] >> 1 ) & 0x01ff ;
	out[58] = ( in[16] >> 10 ) & 0x01ff ;
	out[59] = ( in[16] >> 19 ) & 0x01ff ;
	out[60] = ( in[16] >> 28 ) ;

	out[60] |= ( in[17] << ( 32 - 28 ) ) & 0x01ff ;
	out[61] = ( in[17] >> 5 ) & 0x01ff ;
	out[62] = ( in[17] >> 14 ) & 0x01ff ;
	out[63] = ( in[17] >> 23 ) ;
}


// 10-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c10(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x03ff ;
	out[1] = ( in[0] >> 10 ) & 0x03ff ;
	out[2] = ( in[0] >> 20 ) & 0x03ff ;
	out[3] = ( in[0] >> 30 ) ;

	out[3] |= ( in[1] << ( 32 - 30 ) ) & 0x03ff ;
	out[4] = ( in[1] >> 8 ) & 0x03ff ;
	out[5] = ( in[1] >> 18 ) & 0x03ff ;
	out[6] = ( in[1] >> 28 ) ;

	out[6] |= ( in[2] << ( 32 - 28 ) ) & 0x03ff ;
	out[7] = ( in[2] >> 6 ) & 0x03ff ;
	out[8] = ( in[2] >> 16 ) & 0x03ff ;
	out[9] = ( in[2] >> 26 ) ;

	out[9] |= ( in[3] << ( 32 - 26 ) ) & 0x03ff ;
	out[10] = ( in[3] >> 4 ) & 0x03ff ;
	out[11] = ( in[3] >> 14 ) & 0x03ff ;
	out[12] = ( in[3] >> 24 ) ;

	out[12] |= ( in[4] << ( 32 - 24 ) ) & 0x03ff ;
	out[13] = ( in[4] >> 2 ) & 0x03ff ;
	out[14] = ( in[4] >> 12 ) & 0x03ff ;
	out[15] = ( in[4] >> 22 ) ;

	out[16] = ( in[5] >> 0 ) & 0x03ff ;
	out[17] = ( in[5] >> 10 ) & 0x03ff ;
	out[18] = ( in[5] >> 20 ) & 0x03ff ;
	out[19] = ( in[5] >> 30 ) ;

	out[19] |= ( in[6] << ( 32 - 30 ) ) & 0x03ff ;
	out[20] = ( in[6] >> 8 ) & 0x03ff ;
	out[21] = ( in[6] >> 18 ) & 0x03ff ;
	out[22] = ( in[6] >> 28 ) ;

	out[22] |= ( in[7] << ( 32 - 28 ) ) & 0x03ff ;
	out[23] = ( in[7] >> 6 ) & 0x03ff ;
	out[24] = ( in[7] >> 16 ) & 0x03ff ;
	out[25] = ( in[7] >> 26 ) ;

	out[25] |= ( in[8] << ( 32 - 26 ) ) & 0x03ff ;
	out[26] = ( in[8] >> 4 ) & 0x03ff ;
	out[27] = ( in[8] >> 14 ) & 0x03ff ;
	out[28] = ( in[8] >> 24 ) ;

	out[28] |= ( in[9] << ( 32 - 24 ) ) & 0x03ff ;
	out[29] = ( in[9] >> 2 ) & 0x03ff ;
	out[30] = ( in[9] >> 12 ) & 0x03ff ;
	out[31] = ( in[9] >> 22 ) ;

	out[32] = ( in[10] >> 0 ) & 0x03ff ;
	out[33] = ( in[10] >> 10 ) & 0x03ff ;
	out[34] = ( in[10] >> 20 ) & 0x03ff ;
	out[35] = ( in[10] >> 30 ) ;

	out[35] |= ( in[11] << ( 32 - 30 ) ) & 0x03ff ;
	out[36] = ( in[11] >> 8 ) & 0x03ff ;
	out[37] = ( in[11] >> 18 ) & 0x03ff ;
	out[38] = ( in[11] >> 28 ) ;

	out[38] |= ( in[12] << ( 32 - 28 ) ) & 0x03ff ;
	out[39] = ( in[12] >> 6 ) & 0x03ff ;
	out[40] = ( in[12] >> 16 ) & 0x03ff ;
	out[41] = ( in[12] >> 26 ) ;

	out[41] |= ( in[13] << ( 32 - 26 ) ) & 0x03ff ;
	out[42] = ( in[13] >> 4 ) & 0x03ff ;
	out[43] = ( in[13] >> 14 ) & 0x03ff ;
	out[44] = ( in[13] >> 24 ) ;

	out[44] |= ( in[14] << ( 32 - 24 ) ) & 0x03ff ;
	out[45] = ( in[14] >> 2 ) & 0x03ff ;
	out[46] = ( in[14] >> 12 ) & 0x03ff ;
	out[47] = ( in[14] >> 22 ) ;

	out[48] = ( in[15] >> 0 ) & 0x03ff ;
	out[49] = ( in[15] >> 10 ) & 0x03ff ;
	out[50] = ( in[15] >> 20 ) & 0x03ff ;
	out[51] = ( in[15] >> 30 ) ;

	out[51] |= ( in[16] << ( 32 - 30 ) ) & 0x03ff ;
	out[52] = ( in[16] >> 8 ) & 0x03ff ;
	out[53] = ( in[16] >> 18 ) & 0x03ff ;
	out[54] = ( in[16] >> 28 ) ;

	out[54] |= ( in[17] << ( 32 - 28 ) ) & 0x03ff ;
	out[55] = ( in[17] >> 6 ) & 0x03ff ;
	out[56] = ( in[17] >> 16 ) & 0x03ff ;
	out[57] = ( in[17] >> 26 ) ;

	out[57] |= ( in[18] << ( 32 - 26 ) ) & 0x03ff ;
	out[58] = ( in[18] >> 4 ) & 0x03ff ;
	out[59] = ( in[18] >> 14 ) & 0x03ff ;
	out[60] = ( in[18] >> 24 ) ;

	out[60] |= ( in[19] << ( 32 - 24 ) ) & 0x03ff ;
	out[61] = ( in[19] >> 2 ) & 0x03ff ;
	out[62] = ( in[19] >> 12 ) & 0x03ff ;
	out[63] = ( in[19] >> 22 ) ;
}


// 11-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c11(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x07ff ;
	out[1] = ( in[0] >> 11 ) & 0x07ff ;
	out[2] = ( in[0] >> 22 ) ;

	out[2] |= ( in[1] << ( 32 - 22 ) ) & 0x07ff ;
	out[3] = ( in[1] >> 1 ) & 0x07ff ;
	out[4] = ( in[1] >> 12 ) & 0x07ff ;
	out[5] = ( in[1] >> 23 ) ;

	out[5] |= ( in[2] << ( 32 - 23 ) ) & 0x07ff ;
	out[6] = ( in[2] >> 2 ) & 0x07ff ;
	out[7] = ( in[2] >> 13 ) & 0x07ff ;
	out[8] = ( in[2] >> 24 ) ;

	out[8] |= ( in[3] << ( 32 - 24 ) ) & 0x07ff ;
	out[9] = ( in[3] >> 3 ) & 0x07ff ;
	out[10] = ( in[3] >> 14 ) & 0x07ff ;
	out[11] = ( in[3] >> 25 ) ;

	out[11] |= ( in[4] << ( 32 - 25 ) ) & 0x07ff ;
	out[12] = ( in[4] >> 4 ) & 0x07ff ;
	out[13] = ( in[4] >> 15 ) & 0x07ff ;
	out[14] = ( in[4] >> 26 ) ;

	out[14] |= ( in[5] << ( 32 - 26 ) ) & 0x07ff ;
	out[15] = ( in[5] >> 5 ) & 0x07ff ;
	out[16] = ( in[5] >> 16 ) & 0x07ff ;
	out[17] = ( in[5] >> 27 ) ;

	out[17] |= ( in[6] << ( 32 - 27 ) ) & 0x07ff ;
	out[18] = ( in[6] >> 6 ) & 0x07ff ;
	out[19] = ( in[6] >> 17 ) & 0x07ff ;
	out[20] = ( in[6] >> 28 ) ;

	out[20] |= ( in[7] << ( 32 - 28 ) ) & 0x07ff ;
	out[21] = ( in[7] >> 7 ) & 0x07ff ;
	out[22] = ( in[7] >> 18 ) & 0x07ff ;
	out[23] = ( in[7] >> 29 ) ;

	out[23] |= ( in[8] << ( 32 - 29 ) ) & 0x07ff ;
	out[24] = ( in[8] >> 8 ) & 0x07ff ;
	out[25] = ( in[8] >> 19 ) & 0x07ff ;
	out[26] = ( in[8] >> 30 ) ;

	out[26] |= ( in[9] << ( 32 - 30 ) ) & 0x07ff ;
	out[27] = ( in[9] >> 9 ) & 0x07ff ;
	out[28] = ( in[9] >> 20 ) & 0x07ff ;
	out[29] = ( in[9] >> 31 ) ;

	out[29] |= ( in[10] << ( 32 - 31 ) ) & 0x07ff ;
	out[30] = ( in[10] >> 10 ) & 0x07ff ;
	out[31] = ( in[10] >> 21 ) ;

	out[32] = ( in[11] >> 0 ) & 0x07ff ;
	out[33] = ( in[11] >> 11 ) & 0x07ff ;
	out[34] = ( in[11] >> 22 ) ;

	out[34] |= ( in[12] << ( 32 - 22 ) ) & 0x07ff ;
	out[35] = ( in[12] >> 1 ) & 0x07ff ;
	out[36] = ( in[12] >> 12 ) & 0x07ff ;
	out[37] = ( in[12] >> 23 ) ;

	out[37] |= ( in[13] << ( 32 - 23 ) ) & 0x07ff ;
	out[38] = ( in[13] >> 2 ) & 0x07ff ;
	out[39] = ( in[13] >> 13 ) & 0x07ff ;
	out[40] = ( in[13] >> 24 ) ;

	out[40] |= ( in[14] << ( 32 - 24 ) ) & 0x07ff ;
	out[41] = ( in[14] >> 3 ) & 0x07ff ;
	out[42] = ( in[14] >> 14 ) & 0x07ff ;
	out[43] = ( in[14] >> 25 ) ;

	out[43] |= ( in[15] << ( 32 - 25 ) ) & 0x07ff ;
	out[44] = ( in[15] >> 4 ) & 0x07ff ;
	out[45] = ( in[15] >> 15 ) & 0x07ff ;
	out[46] = ( in[15] >> 26 ) ;

	out[46] |= ( in[16] << ( 32 - 26 ) ) & 0x07ff ;
	out[47] = ( in[16] >> 5 ) & 0x07ff ;
	out[48] = ( in[16] >> 16 ) & 0x07ff ;
	out[49] = ( in[16] >> 27 ) ;

	out[49] |= ( in[17] << ( 32 - 27 ) ) & 0x07ff ;
	out[50] = ( in[17] >> 6 ) & 0x07ff ;
	out[51] = ( in[17] >> 17 ) & 0x07ff ;
	out[52] = ( in[17] >> 28 ) ;

	out[52] |= ( in[18] << ( 32 - 28 ) ) & 0x07ff ;
	out[53] = ( in[18] >> 7 ) & 0x07ff ;
	out[54] = ( in[18] >> 18 ) & 0x07ff ;
	out[55] = ( in[18] >> 29 ) ;

	out[55] |= ( in[19] << ( 32 - 29 ) ) & 0x07ff ;
	out[56] = ( in[19] >> 8 ) & 0x07ff ;
	out[57] = ( in[19] >> 19 ) & 0x07ff ;
	out[58] = ( in[19] >> 30 ) ;

	out[58] |= ( in[20] << ( 32 - 30 ) ) & 0x07ff ;
	out[59] = ( in[20] >> 9 ) & 0x07ff ;
	out[60] = ( in[20] >> 20 ) & 0x07ff ;
	out[61] = ( in[20] >> 31 ) ;

	out[61] |= ( in[21] << ( 32 - 31 ) ) & 0x07ff ;
	out[62] = ( in[21] >> 10 ) & 0x07ff ;
	out[63] = ( in[21] >> 21 ) ;
}


// 12-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c12(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x0fff ;
	out[1] = ( in[0] >> 12 ) & 0x0fff ;
	out[2] = ( in[0] >> 24 ) ;

	out[2] |= ( in[1] << ( 32 - 24 ) ) & 0x0fff ;
	out[3] = ( in[1] >> 4 ) & 0x0fff ;
	out[4] = ( in[1] >> 16 ) & 0x0fff ;
	out[5] = ( in[1] >> 28 ) ;

	out[5] |= ( in[2] << ( 32 - 28 ) ) & 0x0fff ;
	out[6] = ( in[2] >> 8 ) & 0x0fff ;
	out[7] = ( in[2] >> 20 ) ;

	out[8] = ( in[3] >> 0 ) & 0x0fff ;
	out[9] = ( in[3] >> 12 ) & 0x0fff ;
	out[10] = ( in[3] >> 24 ) ;

	out[10] |= ( in[4] << ( 32 - 24 ) ) & 0x0fff ;
	out[11] = ( in[4] >> 4 ) & 0x0fff ;
	out[12] = ( in[4] >> 16 ) & 0x0fff ;
	out[13] = ( in[4] >> 28 ) ;

	out[13] |= ( in[5] << ( 32 - 28 ) ) & 0x0fff ;
	out[14] = ( in[5] >> 8 ) & 0x0fff ;
	out[15] = ( in[5] >> 20 ) ;

	out[16] = ( in[6] >> 0 ) & 0x0fff ;
	out[17] = ( in[6] >> 12 ) & 0x0fff ;
	out[18] = ( in[6] >> 24 ) ;

	out[18] |= ( in[7] << ( 32 - 24 ) ) & 0x0fff ;
	out[19] = ( in[7] >> 4 ) & 0x0fff ;
	out[20] = ( in[7] >> 16 ) & 0x0fff ;
	out[21] = ( in[7] >> 28 ) ;

	out[21] |= ( in[8] << ( 32 - 28 ) ) & 0x0fff ;
	out[22] = ( in[8] >> 8 ) & 0x0fff ;
	out[23] = ( in[8] >> 20 ) ;

	out[24] = ( in[9] >> 0 ) & 0x0fff ;
	out[25] = ( in[9] >> 12 ) & 0x0fff ;
	out[26] = ( in[9] >> 24 ) ;

	out[26] |= ( in[10] << ( 32 - 24 ) ) & 0x0fff ;
	out[27] = ( in[10] >> 4 ) & 0x0fff ;
	out[28] = ( in[10] >> 16 ) & 0x0fff ;
	out[29] = ( in[10] >> 28 ) ;

	out[29] |= ( in[11] << ( 32 - 28 ) ) & 0x0fff ;
	out[30] = ( in[11] >> 8 ) & 0x0fff ;
	out[31] = ( in[11] >> 20 ) ;

	out[32] = ( in[12] >> 0 ) & 0x0fff ;
	out[33] = ( in[12] >> 12 ) & 0x0fff ;
	out[34] = ( in[12] >> 24 ) ;

	out[34] |= ( in[13] << ( 32 - 24 ) ) & 0x0fff ;
	out[35] = ( in[13] >> 4 ) & 0x0fff ;
	out[36] = ( in[13] >> 16 ) & 0x0fff ;
	out[37] = ( in[13] >> 28 ) ;

	out[37] |= ( in[14] << ( 32 - 28 ) ) & 0x0fff ;
	out[38] = ( in[14] >> 8 ) & 0x0fff ;
	out[39] = ( in[14] >> 20 ) ;

	out[40] = ( in[15] >> 0 ) & 0x0fff ;
	out[41] = ( in[15] >> 12 ) & 0x0fff ;
	out[42] = ( in[15] >> 24 ) ;

	out[42] |= ( in[16] << ( 32 - 24 ) ) & 0x0fff ;
	out[43] = ( in[16] >> 4 ) & 0x0fff ;
	out[44] = ( in[16] >> 16 ) & 0x0fff ;
	out[45] = ( in[16] >> 28 ) ;

	out[45] |= ( in[17] << ( 32 - 28 ) ) & 0x0fff ;
	out[46] = ( in[17] >> 8 ) & 0x0fff ;
	out[47] = ( in[17] >> 20 ) ;

	out[48] = ( in[18] >> 0 ) & 0x0fff ;
	out[49] = ( in[18] >> 12 ) & 0x0fff ;
	out[50] = ( in[18] >> 24 ) ;

	out[50] |= ( in[19] << ( 32 - 24 ) ) & 0x0fff ;
	out[51] = ( in[19] >> 4 ) & 0x0fff ;
	out[52] = ( in[19] >> 16 ) & 0x0fff ;
	out[53] = ( in[19] >> 28 ) ;

	out[53] |= ( in[20] << ( 32 - 28 ) ) & 0x0fff ;
	out[54] = ( in[20] >> 8 ) & 0x0fff ;
	out[55] = ( in[20] >> 20 ) ;

	out[56] = ( in[21] >> 0 ) & 0x0fff ;
	out[57] = ( in[21] >> 12 ) & 0x0fff ;
	out[58] = ( in[21] >> 24 ) ;

	out[58] |= ( in[22] << ( 32 - 24 ) ) & 0x0fff ;
	out[59] = ( in[22] >> 4 ) & 0x0fff ;
	out[60] = ( in[22] >> 16 ) & 0x0fff ;
	out[61] = ( in[22] >> 28 ) ;

	out[61] |= ( in[23] << ( 32 - 28 ) ) & 0x0fff ;
	out[62] = ( in[23] >> 8 ) & 0x0fff ;
	out[63] = ( in[23] >> 20 ) ;
}


// 13-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c13(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x1fff ;
	out[1] = ( in[0] >> 13 ) & 0x1fff ;
	out[2] = ( in[0] >> 26 ) ;

	out[2] |= ( in[1] << ( 32 - 26 ) ) & 0x1fff ;
	out[3] = ( in[1] >> 7 ) & 0x1fff ;
	out[4] = ( in[1] >> 20 ) ;

	out[4] |= ( in[2] << ( 32 - 20 ) ) & 0x1fff ;
	out[5] = ( in[2] >> 1 ) & 0x1fff ;
	out[6] = ( in[2] >> 14 ) & 0x1fff ;
	out[7] = ( in[2] >> 27 ) ;

	out[7] |= ( in[3] << ( 32 - 27 ) ) & 0x1fff ;
	out[8] = ( in[3] >> 8 ) & 0x1fff ;
	out[9] = ( in[3] >> 21 ) ;

	out[9] |= ( in[4] << ( 32 - 21 ) ) & 0x1fff ;
	out[10] = ( in[4] >> 2 ) & 0x1fff ;
	out[11] = ( in[4] >> 15 ) & 0x1fff ;
	out[12] = ( in[4] >> 28 ) ;

	out[12] |= ( in[5] << ( 32 - 28 ) ) & 0x1fff ;
	out[13] = ( in[5] >> 9 ) & 0x1fff ;
	out[14] = ( in[5] >> 22 ) ;

	out[14] |= ( in[6] << ( 32 - 22 ) ) & 0x1fff ;
	out[15] = ( in[6] >> 3 ) & 0x1fff ;
	out[16] = ( in[6] >> 16 ) & 0x1fff ;
	out[17] = ( in[6] >> 29 ) ;

	out[17] |= ( in[7] << ( 32 - 29 ) ) & 0x1fff ;
	out[18] = ( in[7] >> 10 ) & 0x1fff ;
	out[19] = ( in[7] >> 23 ) ;

	out[19] |= ( in[8] << ( 32 - 23 ) ) & 0x1fff ;
	out[20] = ( in[8] >> 4 ) & 0x1fff ;
	out[21] = ( in[8] >> 17 ) & 0x1fff ;
	out[22] = ( in[8] >> 30 ) ;

	out[22] |= ( in[9] << ( 32 - 30 ) ) & 0x1fff ;
	out[23] = ( in[9] >> 11 ) & 0x1fff ;
	out[24] = ( in[9] >> 24 ) ;

	out[24] |= ( in[10] << ( 32 - 24 ) ) & 0x1fff ;
	out[25] = ( in[10] >> 5 ) & 0x1fff ;
	out[26] = ( in[10] >> 18 ) & 0x1fff ;
	out[27] = ( in[10] >> 31 ) ;

	out[27] |= ( in[11] << ( 32 - 31 ) ) & 0x1fff ;
	out[28] = ( in[11] >> 12 ) & 0x1fff ;
	out[29] = ( in[11] >> 25 ) ;

	out[29] |= ( in[12] << ( 32 - 25 ) ) & 0x1fff ;
	out[30] = ( in[12] >> 6 ) & 0x1fff ;
	out[31] = ( in[12] >> 19 ) ;

	out[32] = ( in[13] >> 0 ) & 0x1fff ;
	out[33] = ( in[13] >> 13 ) & 0x1fff ;
	out[34] = ( in[13] >> 26 ) ;

	out[34] |= ( in[14] << ( 32 - 26 ) ) & 0x1fff ;
	out[35] = ( in[14] >> 7 ) & 0x1fff ;
	out[36] = ( in[14] >> 20 ) ;

	out[36] |= ( in[15] << ( 32 - 20 ) ) & 0x1fff ;
	out[37] = ( in[15] >> 1 ) & 0x1fff ;
	out[38] = ( in[15] >> 14 ) & 0x1fff ;
	out[39] = ( in[15] >> 27 ) ;

	out[39] |= ( in[16] << ( 32 - 27 ) ) & 0x1fff ;
	out[40] = ( in[16] >> 8 ) & 0x1fff ;
	out[41] = ( in[16] >> 21 ) ;

	out[41] |= ( in[17] << ( 32 - 21 ) ) & 0x1fff ;
	out[42] = ( in[17] >> 2 ) & 0x1fff ;
	out[43] = ( in[17] >> 15 ) & 0x1fff ;
	out[44] = ( in[17] >> 28 ) ;

	out[44] |= ( in[18] << ( 32 - 28 ) ) & 0x1fff ;
	out[45] = ( in[18] >> 9 ) & 0x1fff ;
	out[46] = ( in[18] >> 22 ) ;

	out[46] |= ( in[19] << ( 32 - 22 ) ) & 0x1fff ;
	out[47] = ( in[19] >> 3 ) & 0x1fff ;
	out[48] = ( in[19] >> 16 ) & 0x1fff ;
	out[49] = ( in[19] >> 29 ) ;

	out[49] |= ( in[20] << ( 32 - 29 ) ) & 0x1fff ;
	out[50] = ( in[20] >> 10 ) & 0x1fff ;
	out[51] = ( in[20] >> 23 ) ;

	out[51] |= ( in[21] << ( 32 - 23 ) ) & 0x1fff ;
	out[52] = ( in[21] >> 4 ) & 0x1fff ;
	out[53] = ( in[21] >> 17 ) & 0x1fff ;
	out[54] = ( in[21] >> 30 ) ;

	out[54] |= ( in[22] << ( 32 - 30 ) ) & 0x1fff ;
	out[55] = ( in[22] >> 11 ) & 0x1fff ;
	out[56] = ( in[22] >> 24 ) ;

	out[56] |= ( in[23] << ( 32 - 24 ) ) & 0x1fff ;
	out[57] = ( in[23] >> 5 ) & 0x1fff ;
	out[58] = ( in[23] >> 18 ) & 0x1fff ;
	out[59] = ( in[23] >> 31 ) ;

	out[59] |= ( in[24] << ( 32 - 31 ) ) & 0x1fff ;
	out[60] = ( in[24] >> 12 ) & 0x1fff ;
	out[61] = ( in[24] >> 25 ) ;

	out[61] |= ( in[25] << ( 32 - 25 ) ) & 0x1fff ;
	out[62] = ( in[25] >> 6 ) & 0x1fff ;
	out[63] = ( in[25] >> 19 ) ;
}


// 14-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c14(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x3fff ;
	out[1] = ( in[0] >> 14 ) & 0x3fff ;
	out[2] = ( in[0] >> 28 ) ;

	out[2] |= ( in[1] << ( 32 - 28 ) ) & 0x3fff ;
	out[3] = ( in[1] >> 10 ) & 0x3fff ;
	out[4] = ( in[1] >> 24 ) ;

	out[4] |= ( in[2] << ( 32 - 24 ) ) & 0x3fff ;
	out[5] = ( in[2] >> 6 ) & 0x3fff ;
	out[6] = ( in[2] >> 20 ) ;

	out[6] |= ( in[3] << ( 32 - 20 ) ) & 0x3fff ;
	out[7] = ( in[3] >> 2 ) & 0x3fff ;
	out[8] = ( in[3] >> 16 ) & 0x3fff ;
	out[9] = ( in[3] >> 30 ) ;

	out[9] |= ( in[4] << ( 32 - 30 ) ) & 0x3fff ;
	out[10] = ( in[4] >> 12 ) & 0x3fff ;
	out[11] = ( in[4] >> 26 ) ;

	out[11] |= ( in[5] << ( 32 - 26 ) ) & 0x3fff ;
	out[12] = ( in[5] >> 8 ) & 0x3fff ;
	out[13] = ( in[5] >> 22 ) ;

	out[13] |= ( in[6] << ( 32 - 22 ) ) & 0x3fff ;
	out[14] = ( in[6] >> 4 ) & 0x3fff ;
	out[15] = ( in[6] >> 18 ) ;

	out[16] = ( in[7] >> 0 ) & 0x3fff ;
	out[17] = ( in[7] >> 14 ) & 0x3fff ;
	out[18] = ( in[7] >> 28 ) ;

	out[18] |= ( in[8] << ( 32 - 28 ) ) & 0x3fff ;
	out[19] = ( in[8] >> 10 ) & 0x3fff ;
	out[20] = ( in[8] >> 24 ) ;

	out[20] |= ( in[9] << ( 32 - 24 ) ) & 0x3fff ;
	out[21] = ( in[9] >> 6 ) & 0x3fff ;
	out[22] = ( in[9] >> 20 ) ;

	out[22] |= ( in[10] << ( 32 - 20 ) ) & 0x3fff ;
	out[23] = ( in[10] >> 2 ) & 0x3fff ;
	out[24] = ( in[10] >> 16 ) & 0x3fff ;
	out[25] = ( in[10] >> 30 ) ;

	out[25] |= ( in[11] << ( 32 - 30 ) ) & 0x3fff ;
	out[26] = ( in[11] >> 12 ) & 0x3fff ;
	out[27] = ( in[11] >> 26 ) ;

	out[27] |= ( in[12] << ( 32 - 26 ) ) & 0x3fff ;
	out[28] = ( in[12] >> 8 ) & 0x3fff ;
	out[29] = ( in[12] >> 22 ) ;

	out[29] |= ( in[13] << ( 32 - 22 ) ) & 0x3fff ;
	out[30] = ( in[13] >> 4 ) & 0x3fff ;
	out[31] = ( in[13] >> 18 ) ;

	out[32] = ( in[14] >> 0 ) & 0x3fff ;
	out[33] = ( in[14] >> 14 ) & 0x3fff ;
	out[34] = ( in[14] >> 28 ) ;

	out[34] |= ( in[15] << ( 32 - 28 ) ) & 0x3fff ;
	out[35] = ( in[15] >> 10 ) & 0x3fff ;
	out[36] = ( in[15] >> 24 ) ;

	out[36] |= ( in[16] << ( 32 - 24 ) ) & 0x3fff ;
	out[37] = ( in[16] >> 6 ) & 0x3fff ;
	out[38] = ( in[16] >> 20 ) ;

	out[38] |= ( in[17] << ( 32 - 20 ) ) & 0x3fff ;
	out[39] = ( in[17] >> 2 ) & 0x3fff ;
	out[40] = ( in[17] >> 16 ) & 0x3fff ;
	out[41] = ( in[17] >> 30 ) ;

	out[41] |= ( in[18] << ( 32 - 30 ) ) & 0x3fff ;
	out[42] = ( in[18] >> 12 ) & 0x3fff ;
	out[43] = ( in[18] >> 26 ) ;

	out[43] |= ( in[19] << ( 32 - 26 ) ) & 0x3fff ;
	out[44] = ( in[19] >> 8 ) & 0x3fff ;
	out[45] = ( in[19] >> 22 ) ;

	out[45] |= ( in[20] << ( 32 - 22 ) ) & 0x3fff ;
	out[46] = ( in[20] >> 4 ) & 0x3fff ;
	out[47] = ( in[20] >> 18 ) ;

	out[48] = ( in[21] >> 0 ) & 0x3fff ;
	out[49] = ( in[21] >> 14 ) & 0x3fff ;
	out[50] = ( in[21] >> 28 ) ;

	out[50] |= ( in[22] << ( 32 - 28 ) ) & 0x3fff ;
	out[51] = ( in[22] >> 10 ) & 0x3fff ;
	out[52] = ( in[22] >> 24 ) ;

	out[52] |= ( in[23] << ( 32 - 24 ) ) & 0x3fff ;
	out[53] = ( in[23] >> 6 ) & 0x3fff ;
	out[54] = ( in[23] >> 20 ) ;

	out[54] |= ( in[24] << ( 32 - 20 ) ) & 0x3fff ;
	out[55] = ( in[24] >> 2 ) & 0x3fff ;
	out[56] = ( in[24] >> 16 ) & 0x3fff ;
	out[57] = ( in[24] >> 30 ) ;

	out[57] |= ( in[25] << ( 32 - 30 ) ) & 0x3fff ;
	out[58] = ( in[25] >> 12 ) & 0x3fff ;
	out[59] = ( in[25] >> 26 ) ;

	out[59] |= ( in[26] << ( 32 - 26 ) ) & 0x3fff ;
	out[60] = ( in[26] >> 8 ) & 0x3fff ;
	out[61] = ( in[26] >> 22 ) ;

	out[61] |= ( in[27] << ( 32 - 22 ) ) & 0x3fff ;
	out[62] = ( in[27] >> 4 ) & 0x3fff ;
	out[63] = ( in[27] >> 18 ) ;
}


// 15-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c15(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x7fff ;
	out[1] = ( in[0] >> 15 ) & 0x7fff ;
	out[2] = ( in[0] >> 30 ) ;

	out[2] |= ( in[1] << ( 32 - 30 ) ) & 0x7fff ;
	out[3] = ( in[1] >> 13 ) & 0x7fff ;
	out[4] = ( in[1] >> 28 ) ;

	out[4] |= ( in[2] << ( 32 - 28 ) ) & 0x7fff ;
	out[5] = ( in[2] >> 11 ) & 0x7fff ;
	out[6] = ( in[2] >> 26 ) ;

	out[6] |= ( in[3] << ( 32 - 26 ) ) & 0x7fff ;
	out[7] = ( in[3] >> 9 ) & 0x7fff ;
	out[8] = ( in[3] >> 24 ) ;

	out[8] |= ( in[4] << ( 32 - 24 ) ) & 0x7fff ;
	out[9] = ( in[4] >> 7 ) & 0x7fff ;
	out[10] = ( in[4] >> 22 ) ;

	out[10] |= ( in[5] << ( 32 - 22 ) ) & 0x7fff ;
	out[11] = ( in[5] >> 5 ) & 0x7fff ;
	out[12] = ( in[5] >> 20 ) ;

	out[12] |= ( in[6] << ( 32 - 20 ) ) & 0x7fff ;
	out[13] = ( in[6] >> 3 ) & 0x7fff ;
	out[14] = ( in[6] >> 18 ) ;

	out[14] |= ( in[7] << ( 32 - 18 ) ) & 0x7fff ;
	out[15] = ( in[7] >> 1 ) & 0x7fff ;
	out[16] = ( in[7] >> 16 ) & 0x7fff ;
	out[17] = ( in[7] >> 31 ) ;

	out[17] |= ( in[8] << ( 32 - 31 ) ) & 0x7fff ;
	out[18] = ( in[8] >> 14 ) & 0x7fff ;
	out[19] = ( in[8] >> 29 ) ;

	out[19] |= ( in[9] << ( 32 - 29 ) ) & 0x7fff ;
	out[20] = ( in[9] >> 12 ) & 0x7fff ;
	out[21] = ( in[9] >> 27 ) ;

	out[21] |= ( in[10] << ( 32 - 27 ) ) & 0x7fff ;
	out[22] = ( in[10] >> 10 ) & 0x7fff ;
	out[23] = ( in[10] >> 25 ) ;

	out[23] |= ( in[11] << ( 32 - 25 ) ) & 0x7fff ;
	out[24] = ( in[11] >> 8 ) & 0x7fff ;
	out[25] = ( in[11] >> 23 ) ;

	out[25] |= ( in[12] << ( 32 - 23 ) ) & 0x7fff ;
	out[26] = ( in[12] >> 6 ) & 0x7fff ;
	out[27] = ( in[12] >> 21 ) ;

	out[27] |= ( in[13] << ( 32 - 21 ) ) & 0x7fff ;
	out[28] = ( in[13] >> 4 ) & 0x7fff ;
	out[29] = ( in[13] >> 19 ) ;

	out[29] |= ( in[14] << ( 32 - 19 ) ) & 0x7fff ;
	out[30] = ( in[14] >> 2 ) & 0x7fff ;
	out[31] = ( in[14] >> 17 ) ;

	out[32] = ( in[15] >> 0 ) & 0x7fff ;
	out[33] = ( in[15] >> 15 ) & 0x7fff ;
	out[34] = ( in[15] >> 30 ) ;

	out[34] |= ( in[16] << ( 32 - 30 ) ) & 0x7fff ;
	out[35] = ( in[16] >> 13 ) & 0x7fff ;
	out[36] = ( in[16] >> 28 ) ;

	out[36] |= ( in[17] << ( 32 - 28 ) ) & 0x7fff ;
	out[37] = ( in[17] >> 11 ) & 0x7fff ;
	out[38] = ( in[17] >> 26 ) ;

	out[38] |= ( in[18] << ( 32 - 26 ) ) & 0x7fff ;
	out[39] = ( in[18] >> 9 ) & 0x7fff ;
	out[40] = ( in[18] >> 24 ) ;

	out[40] |= ( in[19] << ( 32 - 24 ) ) & 0x7fff ;
	out[41] = ( in[19] >> 7 ) & 0x7fff ;
	out[42] = ( in[19] >> 22 ) ;

	out[42] |= ( in[20] << ( 32 - 22 ) ) & 0x7fff ;
	out[43] = ( in[20] >> 5 ) & 0x7fff ;
	out[44] = ( in[20] >> 20 ) ;

	out[44] |= ( in[21] << ( 32 - 20 ) ) & 0x7fff ;
	out[45] = ( in[21] >> 3 ) & 0x7fff ;
	out[46] = ( in[21] >> 18 ) ;

	out[46] |= ( in[22] << ( 32 - 18 ) ) & 0x7fff ;
	out[47] = ( in[22] >> 1 ) & 0x7fff ;
	out[48] = ( in[22] >> 16 ) & 0x7fff ;
	out[49] = ( in[22] >> 31 ) ;

	out[49] |= ( in[23] << ( 32 - 31 ) ) & 0x7fff ;
	out[50] = ( in[23] >> 14 ) & 0x7fff ;
	out[51] = ( in[23] >> 29 ) ;

	out[51] |= ( in[24] << ( 32 - 29 ) ) & 0x7fff ;
	out[52] = ( in[24] >> 12 ) & 0x7fff ;
	out[53] = ( in[24] >> 27 ) ;

	out[53] |= ( in[25] << ( 32 - 27 ) ) & 0x7fff ;
	out[54] = ( in[25] >> 10 ) & 0x7fff ;
	out[55] = ( in[25] >> 25 ) ;

	out[55] |= ( in[26] << ( 32 - 25 ) ) & 0x7fff ;
	out[56] = ( in[26] >> 8 ) & 0x7fff ;
	out[57] = ( in[26] >> 23 ) ;

	out[57] |= ( in[27] << ( 32 - 23 ) ) & 0x7fff ;
	out[58] = ( in[27] >> 6 ) & 0x7fff ;
	out[59] = ( in[27] >> 21 ) ;

	out[59] |= ( in[28] << ( 32 - 21 ) ) & 0x7fff ;
	out[60] = ( in[28] >> 4 ) & 0x7fff ;
	out[61] = ( in[28] >> 19 ) ;

	out[61] |= ( in[29] << ( 32 - 19 ) ) & 0x7fff ;
	out[62] = ( in[29] >> 2 ) & 0x7fff ;
	out[63] = ( in[29] >> 17 ) ;
}


// 16-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c16(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0xffff ;
	out[1] = ( in[0] >> 16 ) ;

	out[2] = ( in[1] >> 0 ) & 0xffff ;
	out[3] = ( in[1] >> 16 ) ;

	out[4] = ( in[2] >> 0 ) & 0xffff ;
	out[5] = ( in[2] >> 16 ) ;

	out[6] = ( in[3] >> 0 ) & 0xffff ;
	out[7] = ( in[3] >> 16 ) ;

	out[8] = ( in[4] >> 0 ) & 0xffff ;
	out[9] = ( in[4] >> 16 ) ;

	out[10] = ( in[5] >> 0 ) & 0xffff ;
	out[11] = ( in[5] >> 16 ) ;

	out[12] = ( in[6] >> 0 ) & 0xffff ;
	out[13] = ( in[6] >> 16 ) ;

	out[14] = ( in[7] >> 0 ) & 0xffff ;
	out[15] = ( in[7] >> 16 ) ;

	out[16] = ( in[8] >> 0 ) & 0xffff ;
	out[17] = ( in[8] >> 16 ) ;

	out[18] = ( in[9] >> 0 ) & 0xffff ;
	out[19] = ( in[9] >> 16 ) ;

	out[20] = ( in[10] >> 0 ) & 0xffff ;
	out[21] = ( in[10] >> 16 ) ;

	out[22] = ( in[11] >> 0 ) & 0xffff ;
	out[23] = ( in[11] >> 16 ) ;

	out[24] = ( in[12] >> 0 ) & 0xffff ;
	out[25] = ( in[12] >> 16 ) ;

	out[26] = ( in[13] >> 0 ) & 0xffff ;
	out[27] = ( in[13] >> 16 ) ;

	out[28] = ( in[14] >> 0 ) & 0xffff ;
	out[29] = ( in[14] >> 16 ) ;

	out[30] = ( in[15] >> 0 ) & 0xffff ;
	out[31] = ( in[15] >> 16 ) ;

	out[32] = ( in[16] >> 0 ) & 0xffff ;
	out[33] = ( in[16] >> 16 ) ;

	out[34] = ( in[17] >> 0 ) & 0xffff ;
	out[35] = ( in[17] >> 16 ) ;

	out[36] = ( in[18] >> 0 ) & 0xffff ;
	out[37] = ( in[18] >> 16 ) ;

	out[38] = ( in[19] >> 0 ) & 0xffff ;
	out[39] = ( in[19] >> 16 ) ;

	out[40] = ( in[20] >> 0 ) & 0xffff ;
	out[41] = ( in[20] >> 16 ) ;

	out[42] = ( in[21] >> 0 ) & 0xffff ;
	out[43] = ( in[21] >> 16 ) ;

	out[44] = ( in[22] >> 0 ) & 0xffff ;
	out[45] = ( in[22] >> 16 ) ;

	out[46] = ( in[23] >> 0 ) & 0xffff ;
	out[47] = ( in[23] >> 16 ) ;

	out[48] = ( in[24] >> 0 ) & 0xffff ;
	out[49] = ( in[24] >> 16 ) ;

	out[50] = ( in[25] >> 0 ) & 0xffff ;
	out[51] = ( in[25] >> 16 ) ;

	out[52] = ( in[26] >> 0 ) & 0xffff ;
	out[53] = ( in[26] >> 16 ) ;

	out[54] = ( in[27] >> 0 ) & 0xffff ;
	out[55] = ( in[27] >> 16 ) ;

	out[56] = ( in[28] >> 0 ) & 0xffff ;
	out[57] = ( in[28] >> 16 ) ;

	out[58] = ( in[29] >> 0 ) & 0xffff ;
	out[59] = ( in[29] >> 16 ) ;

	out[60] = ( in[30] >> 0 ) & 0xffff ;
	out[61] = ( in[30] >> 16 ) ;

	out[62] = ( in[31] >> 0 ) & 0xffff ;
	out[63] = ( in[31] >> 16 ) ;
}


// 17-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c17(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x01ffff ;
	out[1] = ( in[0] >> 17 ) ;

	out[1] |= ( in[1] << ( 32 - 17 ) ) & 0x01ffff ;
	out[2] = ( in[1] >> 2 ) & 0x01ffff ;
	out[3] = ( in[1] >> 19 ) ;

	out[3] |= ( in[2] << ( 32 - 19 ) ) & 0x01ffff ;
	out[4] = ( in[2] >> 4 ) & 0x01ffff ;
	out[5] = ( in[2] >> 21 ) ;

	out[5] |= ( in[3] << ( 32 - 21 ) ) & 0x01ffff ;
	out[6] = ( in[3] >> 6 ) & 0x01ffff ;
	out[7] = ( in[3] >> 23 ) ;

	out[7] |= ( in[4] << ( 32 - 23 ) ) & 0x01ffff ;
	out[8] = ( in[4] >> 8 ) & 0x01ffff ;
	out[9] = ( in[4] >> 25 ) ;

	out[9] |= ( in[5] << ( 32 - 25 ) ) & 0x01ffff ;
	out[10] = ( in[5] >> 10 ) & 0x01ffff ;
	out[11] = ( in[5] >> 27 ) ;

	out[11] |= ( in[6] << ( 32 - 27 ) ) & 0x01ffff ;
	out[12] = ( in[6] >> 12 ) & 0x01ffff ;
	out[13] = ( in[6] >> 29 ) ;

	out[13] |= ( in[7] << ( 32 - 29 ) ) & 0x01ffff ;
	out[14] = ( in[7] >> 14 ) & 0x01ffff ;
	out[15] = ( in[7] >> 31 ) ;

	out[15] |= ( in[8] << ( 32 - 31 ) ) & 0x01ffff ;
	out[16] = ( in[8] >> 16 ) ;

	out[16] |= ( in[9] << ( 32 - 16 ) ) & 0x01ffff ;
	out[17] = ( in[9] >> 1 ) & 0x01ffff ;
	out[18] = ( in[9] >> 18 ) ;

	out[18] |= ( in[10] << ( 32 - 18 ) ) & 0x01ffff ;
	out[19] = ( in[10] >> 3 ) & 0x01ffff ;
	out[20] = ( in[10] >> 20 ) ;

	out[20] |= ( in[11] << ( 32 - 20 ) ) & 0x01ffff ;
	out[21] = ( in[11] >> 5 ) & 0x01ffff ;
	out[22] = ( in[11] >> 22 ) ;

	out[22] |= ( in[12] << ( 32 - 22 ) ) & 0x01ffff ;
	out[23] = ( in[12] >> 7 ) & 0x01ffff ;
	out[24] = ( in[12] >> 24 ) ;

	out[24] |= ( in[13] << ( 32 - 24 ) ) & 0x01ffff ;
	out[25] = ( in[13] >> 9 ) & 0x01ffff ;
	out[26] = ( in[13] >> 26 ) ;

	out[26] |= ( in[14] << ( 32 - 26 ) ) & 0x01ffff ;
	out[27] = ( in[14] >> 11 ) & 0x01ffff ;
	out[28] = ( in[14] >> 28 ) ;

	out[28] |= ( in[15] << ( 32 - 28 ) ) & 0x01ffff ;
	out[29] = ( in[15] >> 13 ) & 0x01ffff ;
	out[30] = ( in[15] >> 30 ) ;

	out[30] |= ( in[16] << ( 32 - 30 ) ) & 0x01ffff ;
	out[31] = ( in[16] >> 15 ) ;

	out[32] = ( in[17] >> 0 ) & 0x01ffff ;
	out[33] = ( in[17] >> 17 ) ;

	out[33] |= ( in[18] << ( 32 - 17 ) ) & 0x01ffff ;
	out[34] = ( in[18] >> 2 ) & 0x01ffff ;
	out[35] = ( in[18] >> 19 ) ;

	out[35] |= ( in[19] << ( 32 - 19 ) ) & 0x01ffff ;
	out[36] = ( in[19] >> 4 ) & 0x01ffff ;
	out[37] = ( in[19] >> 21 ) ;

	out[37] |= ( in[20] << ( 32 - 21 ) ) & 0x01ffff ;
	out[38] = ( in[20] >> 6 ) & 0x01ffff ;
	out[39] = ( in[20] >> 23 ) ;

	out[39] |= ( in[21] << ( 32 - 23 ) ) & 0x01ffff ;
	out[40] = ( in[21] >> 8 ) & 0x01ffff ;
	out[41] = ( in[21] >> 25 ) ;

	out[41] |= ( in[22] << ( 32 - 25 ) ) & 0x01ffff ;
	out[42] = ( in[22] >> 10 ) & 0x01ffff ;
	out[43] = ( in[22] >> 27 ) ;

	out[43] |= ( in[23] << ( 32 - 27 ) ) & 0x01ffff ;
	out[44] = ( in[23] >> 12 ) & 0x01ffff ;
	out[45] = ( in[23] >> 29 ) ;

	out[45] |= ( in[24] << ( 32 - 29 ) ) & 0x01ffff ;
	out[46] = ( in[24] >> 14 ) & 0x01ffff ;
	out[47] = ( in[24] >> 31 ) ;

	out[47] |= ( in[25] << ( 32 - 31 ) ) & 0x01ffff ;
	out[48] = ( in[25] >> 16 ) ;

	out[48] |= ( in[26] << ( 32 - 16 ) ) & 0x01ffff ;
	out[49] = ( in[26] >> 1 ) & 0x01ffff ;
	out[50] = ( in[26] >> 18 ) ;

	out[50] |= ( in[27] << ( 32 - 18 ) ) & 0x01ffff ;
	out[51] = ( in[27] >> 3 ) & 0x01ffff ;
	out[52] = ( in[27] >> 20 ) ;

	out[52] |= ( in[28] << ( 32 - 20 ) ) & 0x01ffff ;
	out[53] = ( in[28] >> 5 ) & 0x01ffff ;
	out[54] = ( in[28] >> 22 ) ;

	out[54] |= ( in[29] << ( 32 - 22 ) ) & 0x01ffff ;
	out[55] = ( in[29] >> 7 ) & 0x01ffff ;
	out[56] = ( in[29] >> 24 ) ;

	out[56] |= ( in[30] << ( 32 - 24 ) ) & 0x01ffff ;
	out[57] = ( in[30] >> 9 ) & 0x01ffff ;
	out[58] = ( in[30] >> 26 ) ;

	out[58] |= ( in[31] << ( 32 - 26 ) ) & 0x01ffff ;
	out[59] = ( in[31] >> 11 ) & 0x01ffff ;
	out[60] = ( in[31] >> 28 ) ;

	out[60] |= ( in[32] << ( 32 - 28 ) ) & 0x01ffff ;
	out[61] = ( in[32] >> 13 ) & 0x01ffff ;
	out[62] = ( in[32] >> 30 ) ;

	out[62] |= ( in[33] << ( 32 - 30 ) ) & 0x01ffff ;
	out[63] = ( in[33] >> 15 ) ;
}


// 18-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c18(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x03ffff ;
	out[1] = ( in[0] >> 18 ) ;

	out[1] |= ( in[1] << ( 32 - 18 ) ) & 0x03ffff ;
	out[2] = ( in[1] >> 4 ) & 0x03ffff ;
	out[3] = ( in[1] >> 22 ) ;

	out[3] |= ( in[2] << ( 32 - 22 ) ) & 0x03ffff ;
	out[4] = ( in[2] >> 8 ) & 0x03ffff ;
	out[5] = ( in[2] >> 26 ) ;

	out[5] |= ( in[3] << ( 32 - 26 ) ) & 0x03ffff ;
	out[6] = ( in[3] >> 12 ) & 0x03ffff ;
	out[7] = ( in[3] >> 30 ) ;

	out[7] |= ( in[4] << ( 32 - 30 ) ) & 0x03ffff ;
	out[8] = ( in[4] >> 16 ) ;

	out[8] |= ( in[5] << ( 32 - 16 ) ) & 0x03ffff ;
	out[9] = ( in[5] >> 2 ) & 0x03ffff ;
	out[10] = ( in[5] >> 20 ) ;

	out[10] |= ( in[6] << ( 32 - 20 ) ) & 0x03ffff ;
	out[11] = ( in[6] >> 6 ) & 0x03ffff ;
	out[12] = ( in[6] >> 24 ) ;

	out[12] |= ( in[7] << ( 32 - 24 ) ) & 0x03ffff ;
	out[13] = ( in[7] >> 10 ) & 0x03ffff ;
	out[14] = ( in[7] >> 28 ) ;

	out[14] |= ( in[8] << ( 32 - 28 ) ) & 0x03ffff ;
	out[15] = ( in[8] >> 14 ) ;

	out[16] = ( in[9] >> 0 ) & 0x03ffff ;
	out[17] = ( in[9] >> 18 ) ;

	out[17] |= ( in[10] << ( 32 - 18 ) ) & 0x03ffff ;
	out[18] = ( in[10] >> 4 ) & 0x03ffff ;
	out[19] = ( in[10] >> 22 ) ;

	out[19] |= ( in[11] << ( 32 - 22 ) ) & 0x03ffff ;
	out[20] = ( in[11] >> 8 ) & 0x03ffff ;
	out[21] = ( in[11] >> 26 ) ;

	out[21] |= ( in[12] << ( 32 - 26 ) ) & 0x03ffff ;
	out[22] = ( in[12] >> 12 ) & 0x03ffff ;
	out[23] = ( in[12] >> 30 ) ;

	out[23] |= ( in[13] << ( 32 - 30 ) ) & 0x03ffff ;
	out[24] = ( in[13] >> 16 ) ;

	out[24] |= ( in[14] << ( 32 - 16 ) ) & 0x03ffff ;
	out[25] = ( in[14] >> 2 ) & 0x03ffff ;
	out[26] = ( in[14] >> 20 ) ;

	out[26] |= ( in[15] << ( 32 - 20 ) ) & 0x03ffff ;
	out[27] = ( in[15] >> 6 ) & 0x03ffff ;
	out[28] = ( in[15] >> 24 ) ;

	out[28] |= ( in[16] << ( 32 - 24 ) ) & 0x03ffff ;
	out[29] = ( in[16] >> 10 ) & 0x03ffff ;
	out[30] = ( in[16] >> 28 ) ;

	out[30] |= ( in[17] << ( 32 - 28 ) ) & 0x03ffff ;
	out[31] = ( in[17] >> 14 ) ;

	out[32] = ( in[18] >> 0 ) & 0x03ffff ;
	out[33] = ( in[18] >> 18 ) ;

	out[33] |= ( in[19] << ( 32 - 18 ) ) & 0x03ffff ;
	out[34] = ( in[19] >> 4 ) & 0x03ffff ;
	out[35] = ( in[19] >> 22 ) ;

	out[35] |= ( in[20] << ( 32 - 22 ) ) & 0x03ffff ;
	out[36] = ( in[20] >> 8 ) & 0x03ffff ;
	out[37] = ( in[20] >> 26 ) ;

	out[37] |= ( in[21] << ( 32 - 26 ) ) & 0x03ffff ;
	out[38] = ( in[21] >> 12 ) & 0x03ffff ;
	out[39] = ( in[21] >> 30 ) ;

	out[39] |= ( in[22] << ( 32 - 30 ) ) & 0x03ffff ;
	out[40] = ( in[22] >> 16 ) ;

	out[40] |= ( in[23] << ( 32 - 16 ) ) & 0x03ffff ;
	out[41] = ( in[23] >> 2 ) & 0x03ffff ;
	out[42] = ( in[23] >> 20 ) ;

	out[42] |= ( in[24] << ( 32 - 20 ) ) & 0x03ffff ;
	out[43] = ( in[24] >> 6 ) & 0x03ffff ;
	out[44] = ( in[24] >> 24 ) ;

	out[44] |= ( in[25] << ( 32 - 24 ) ) & 0x03ffff ;
	out[45] = ( in[25] >> 10 ) & 0x03ffff ;
	out[46] = ( in[25] >> 28 ) ;

	out[46] |= ( in[26] << ( 32 - 28 ) ) & 0x03ffff ;
	out[47] = ( in[26] >> 14 ) ;

	out[48] = ( in[27] >> 0 ) & 0x03ffff ;
	out[49] = ( in[27] >> 18 ) ;

	out[49] |= ( in[28] << ( 32 - 18 ) ) & 0x03ffff ;
	out[50] = ( in[28] >> 4 ) & 0x03ffff ;
	out[51] = ( in[28] >> 22 ) ;

	out[51] |= ( in[29] << ( 32 - 22 ) ) & 0x03ffff ;
	out[52] = ( in[29] >> 8 ) & 0x03ffff ;
	out[53] = ( in[29] >> 26 ) ;

	out[53] |= ( in[30] << ( 32 - 26 ) ) & 0x03ffff ;
	out[54] = ( in[30] >> 12 ) & 0x03ffff ;
	out[55] = ( in[30] >> 30 ) ;

	out[55] |= ( in[31] << ( 32 - 30 ) ) & 0x03ffff ;
	out[56] = ( in[31] >> 16 ) ;

	out[56] |= ( in[32] << ( 32 - 16 ) ) & 0x03ffff ;
	out[57] = ( in[32] >> 2 ) & 0x03ffff ;
	out[58] = ( in[32] >> 20 ) ;

	out[58] |= ( in[33] << ( 32 - 20 ) ) & 0x03ffff ;
	out[59] = ( in[33] >> 6 ) & 0x03ffff ;
	out[60] = ( in[33] >> 24 ) ;

	out[60] |= ( in[34] << ( 32 - 24 ) ) & 0x03ffff ;
	out[61] = ( in[34] >> 10 ) & 0x03ffff ;
	out[62] = ( in[34] >> 28 ) ;

	out[62] |= ( in[35] << ( 32 - 28 ) ) & 0x03ffff ;
	out[63] = ( in[35] >> 14 ) ;
}


// 19-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c19(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x07ffff ;
	out[1] = ( in[0] >> 19 ) ;

	out[1] |= ( in[1] << ( 32 - 19 ) ) & 0x07ffff ;
	out[2] = ( in[1] >> 6 ) & 0x07ffff ;
	out[3] = ( in[1] >> 25 ) ;

	out[3] |= ( in[2] << ( 32 - 25 ) ) & 0x07ffff ;
	out[4] = ( in[2] >> 12 ) & 0x07ffff ;
	out[5] = ( in[2] >> 31 ) ;

	out[5] |= ( in[3] << ( 32 - 31 ) ) & 0x07ffff ;
	out[6] = ( in[3] >> 18 ) ;

	out[6] |= ( in[4] << ( 32 - 18 ) ) & 0x07ffff ;
	out[7] = ( in[4] >> 5 ) & 0x07ffff ;
	out[8] = ( in[4] >> 24 ) ;

	out[8] |= ( in[5] << ( 32 - 24 ) ) & 0x07ffff ;
	out[9] = ( in[5] >> 11 ) & 0x07ffff ;
	out[10] = ( in[5] >> 30 ) ;

	out[10] |= ( in[6] << ( 32 - 30 ) ) & 0x07ffff ;
	out[11] = ( in[6] >> 17 ) ;

	out[11] |= ( in[7] << ( 32 - 17 ) ) & 0x07ffff ;
	out[12] = ( in[7] >> 4 ) & 0x07ffff ;
	out[13] = ( in[7] >> 23 ) ;

	out[13] |= ( in[8] << ( 32 - 23 ) ) & 0x07ffff ;
	out[14] = ( in[8] >> 10 ) & 0x07ffff ;
	out[15] = ( in[8] >> 29 ) ;

	out[15] |= ( in[9] << ( 32 - 29 ) ) & 0x07ffff ;
	out[16] = ( in[9] >> 16 ) ;

	out[16] |= ( in[10] << ( 32 - 16 ) ) & 0x07ffff ;
	out[17] = ( in[10] >> 3 ) & 0x07ffff ;
	out[18] = ( in[10] >> 22 ) ;

	out[18] |= ( in[11] << ( 32 - 22 ) ) & 0x07ffff ;
	out[19] = ( in[11] >> 9 ) & 0x07ffff ;
	out[20] = ( in[11] >> 28 ) ;

	out[20] |= ( in[12] << ( 32 - 28 ) ) & 0x07ffff ;
	out[21] = ( in[12] >> 15 ) ;

	out[21] |= ( in[13] << ( 32 - 15 ) ) & 0x07ffff ;
	out[22] = ( in[13] >> 2 ) & 0x07ffff ;
	out[23] = ( in[13] >> 21 ) ;

	out[23] |= ( in[14] << ( 32 - 21 ) ) & 0x07ffff ;
	out[24] = ( in[14] >> 8 ) & 0x07ffff ;
	out[25] = ( in[14] >> 27 ) ;

	out[25] |= ( in[15] << ( 32 - 27 ) ) & 0x07ffff ;
	out[26] = ( in[15] >> 14 ) ;

	out[26] |= ( in[16] << ( 32 - 14 ) ) & 0x07ffff ;
	out[27] = ( in[16] >> 1 ) & 0x07ffff ;
	out[28] = ( in[16] >> 20 ) ;

	out[28] |= ( in[17] << ( 32 - 20 ) ) & 0x07ffff ;
	out[29] = ( in[17] >> 7 ) & 0x07ffff ;
	out[30] = ( in[17] >> 26 ) ;

	out[30] |= ( in[18] << ( 32 - 26 ) ) & 0x07ffff ;
	out[31] = ( in[18] >> 13 ) ;

	out[32] = ( in[19] >> 0 ) & 0x07ffff ;
	out[33] = ( in[19] >> 19 ) ;

	out[33] |= ( in[20] << ( 32 - 19 ) ) & 0x07ffff ;
	out[34] = ( in[20] >> 6 ) & 0x07ffff ;
	out[35] = ( in[20] >> 25 ) ;

	out[35] |= ( in[21] << ( 32 - 25 ) ) & 0x07ffff ;
	out[36] = ( in[21] >> 12 ) & 0x07ffff ;
	out[37] = ( in[21] >> 31 ) ;

	out[37] |= ( in[22] << ( 32 - 31 ) ) & 0x07ffff ;
	out[38] = ( in[22] >> 18 ) ;

	out[38] |= ( in[23] << ( 32 - 18 ) ) & 0x07ffff ;
	out[39] = ( in[23] >> 5 ) & 0x07ffff ;
	out[40] = ( in[23] >> 24 ) ;

	out[40] |= ( in[24] << ( 32 - 24 ) ) & 0x07ffff ;
	out[41] = ( in[24] >> 11 ) & 0x07ffff ;
	out[42] = ( in[24] >> 30 ) ;

	out[42] |= ( in[25] << ( 32 - 30 ) ) & 0x07ffff ;
	out[43] = ( in[25] >> 17 ) ;

	out[43] |= ( in[26] << ( 32 - 17 ) ) & 0x07ffff ;
	out[44] = ( in[26] >> 4 ) & 0x07ffff ;
	out[45] = ( in[26] >> 23 ) ;

	out[45] |= ( in[27] << ( 32 - 23 ) ) & 0x07ffff ;
	out[46] = ( in[27] >> 10 ) & 0x07ffff ;
	out[47] = ( in[27] >> 29 ) ;

	out[47] |= ( in[28] << ( 32 - 29 ) ) & 0x07ffff ;
	out[48] = ( in[28] >> 16 ) ;

	out[48] |= ( in[29] << ( 32 - 16 ) ) & 0x07ffff ;
	out[49] = ( in[29] >> 3 ) & 0x07ffff ;
	out[50] = ( in[29] >> 22 ) ;

	out[50] |= ( in[30] << ( 32 - 22 ) ) & 0x07ffff ;
	out[51] = ( in[30] >> 9 ) & 0x07ffff ;
	out[52] = ( in[30] >> 28 ) ;

	out[52] |= ( in[31] << ( 32 - 28 ) ) & 0x07ffff ;
	out[53] = ( in[31] >> 15 ) ;

	out[53] |= ( in[32] << ( 32 - 15 ) ) & 0x07ffff ;
	out[54] = ( in[32] >> 2 ) & 0x07ffff ;
	out[55] = ( in[32] >> 21 ) ;

	out[55] |= ( in[33] << ( 32 - 21 ) ) & 0x07ffff ;
	out[56] = ( in[33] >> 8 ) & 0x07ffff ;
	out[57] = ( in[33] >> 27 ) ;

	out[57] |= ( in[34] << ( 32 - 27 ) ) & 0x07ffff ;
	out[58] = ( in[34] >> 14 ) ;

	out[58] |= ( in[35] << ( 32 - 14 ) ) & 0x07ffff ;
	out[59] = ( in[35] >> 1 ) & 0x07ffff ;
	out[60] = ( in[35] >> 20 ) ;

	out[60] |= ( in[36] << ( 32 - 20 ) ) & 0x07ffff ;
	out[61] = ( in[36] >> 7 ) & 0x07ffff ;
	out[62] = ( in[36] >> 26 ) ;

	out[62] |= ( in[37] << ( 32 - 26 ) ) & 0x07ffff ;
	out[63] = ( in[37] >> 13 ) ;
}


// 20-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c20(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x0fffff ;
	out[1] = ( in[0] >> 20 ) ;

	out[1] |= ( in[1] << ( 32 - 20 ) ) & 0x0fffff ;
	out[2] = ( in[1] >> 8 ) & 0x0fffff ;
	out[3] = ( in[1] >> 28 ) ;

	out[3] |= ( in[2] << ( 32 - 28 ) ) & 0x0fffff ;
	out[4] = ( in[2] >> 16 ) ;

	out[4] |= ( in[3] << ( 32 - 16 ) ) & 0x0fffff ;
	out[5] = ( in[3] >> 4 ) & 0x0fffff ;
	out[6] = ( in[3] >> 24 ) ;

	out[6] |= ( in[4] << ( 32 - 24 ) ) & 0x0fffff ;
	out[7] = ( in[4] >> 12 ) ;

	out[8] = ( in[5] >> 0 ) & 0x0fffff ;
	out[9] = ( in[5] >> 20 ) ;

	out[9] |= ( in[6] << ( 32 - 20 ) ) & 0x0fffff ;
	out[10] = ( in[6] >> 8 ) & 0x0fffff ;
	out[11] = ( in[6] >> 28 ) ;

	out[11] |= ( in[7] << ( 32 - 28 ) ) & 0x0fffff ;
	out[12] = ( in[7] >> 16 ) ;

	out[12] |= ( in[8] << ( 32 - 16 ) ) & 0x0fffff ;
	out[13] = ( in[8] >> 4 ) & 0x0fffff ;
	out[14] = ( in[8] >> 24 ) ;

	out[14] |= ( in[9] << ( 32 - 24 ) ) & 0x0fffff ;
	out[15] = ( in[9] >> 12 ) ;

	out[16] = ( in[10] >> 0 ) & 0x0fffff ;
	out[17] = ( in[10] >> 20 ) ;

	out[17] |= ( in[11] << ( 32 - 20 ) ) & 0x0fffff ;
	out[18] = ( in[11] >> 8 ) & 0x0fffff ;
	out[19] = ( in[11] >> 28 ) ;

	out[19] |= ( in[12] << ( 32 - 28 ) ) & 0x0fffff ;
	out[20] = ( in[12] >> 16 ) ;

	out[20] |= ( in[13] << ( 32 - 16 ) ) & 0x0fffff ;
	out[21] = ( in[13] >> 4 ) & 0x0fffff ;
	out[22] = ( in[13] >> 24 ) ;

	out[22] |= ( in[14] << ( 32 - 24 ) ) & 0x0fffff ;
	out[23] = ( in[14] >> 12 ) ;

	out[24] = ( in[15] >> 0 ) & 0x0fffff ;
	out[25] = ( in[15] >> 20 ) ;

	out[25] |= ( in[16] << ( 32 - 20 ) ) & 0x0fffff ;
	out[26] = ( in[16] >> 8 ) & 0x0fffff ;
	out[27] = ( in[16] >> 28 ) ;

	out[27] |= ( in[17] << ( 32 - 28 ) ) & 0x0fffff ;
	out[28] = ( in[17] >> 16 ) ;

	out[28] |= ( in[18] << ( 32 - 16 ) ) & 0x0fffff ;
	out[29] = ( in[18] >> 4 ) & 0x0fffff ;
	out[30] = ( in[18] >> 24 ) ;

	out[30] |= ( in[19] << ( 32 - 24 ) ) & 0x0fffff ;
	out[31] = ( in[19] >> 12 ) ;

	out[32] = ( in[20] >> 0 ) & 0x0fffff ;
	out[33] = ( in[20] >> 20 ) ;

	out[33] |= ( in[21] << ( 32 - 20 ) ) & 0x0fffff ;
	out[34] = ( in[21] >> 8 ) & 0x0fffff ;
	out[35] = ( in[21] >> 28 ) ;

	out[35] |= ( in[22] << ( 32 - 28 ) ) & 0x0fffff ;
	out[36] = ( in[22] >> 16 ) ;

	out[36] |= ( in[23] << ( 32 - 16 ) ) & 0x0fffff ;
	out[37] = ( in[23] >> 4 ) & 0x0fffff ;
	out[38] = ( in[23] >> 24 ) ;

	out[38] |= ( in[24] << ( 32 - 24 ) ) & 0x0fffff ;
	out[39] = ( in[24] >> 12 ) ;

	out[40] = ( in[25] >> 0 ) & 0x0fffff ;
	out[41] = ( in[25] >> 20 ) ;

	out[41] |= ( in[26] << ( 32 - 20 ) ) & 0x0fffff ;
	out[42] = ( in[26] >> 8 ) & 0x0fffff ;
	out[43] = ( in[26] >> 28 ) ;

	out[43] |= ( in[27] << ( 32 - 28 ) ) & 0x0fffff ;
	out[44] = ( in[27] >> 16 ) ;

	out[44] |= ( in[28] << ( 32 - 16 ) ) & 0x0fffff ;
	out[45] = ( in[28] >> 4 ) & 0x0fffff ;
	out[46] = ( in[28] >> 24 ) ;

	out[46] |= ( in[29] << ( 32 - 24 ) ) & 0x0fffff ;
	out[47] = ( in[29] >> 12 ) ;

	out[48] = ( in[30] >> 0 ) & 0x0fffff ;
	out[49] = ( in[30] >> 20 ) ;

	out[49] |= ( in[31] << ( 32 - 20 ) ) & 0x0fffff ;
	out[50] = ( in[31] >> 8 ) & 0x0fffff ;
	out[51] = ( in[31] >> 28 ) ;

	out[51] |= ( in[32] << ( 32 - 28 ) ) & 0x0fffff ;
	out[52] = ( in[32] >> 16 ) ;

	out[52] |= ( in[33] << ( 32 - 16 ) ) & 0x0fffff ;
	out[53] = ( in[33] >> 4 ) & 0x0fffff ;
	out[54] = ( in[33] >> 24 ) ;

	out[54] |= ( in[34] << ( 32 - 24 ) ) & 0x0fffff ;
	out[55] = ( in[34] >> 12 ) ;

	out[56] = ( in[35] >> 0 ) & 0x0fffff ;
	out[57] = ( in[35] >> 20 ) ;

	out[57] |= ( in[36] << ( 32 - 20 ) ) & 0x0fffff ;
	out[58] = ( in[36] >> 8 ) & 0x0fffff ;
	out[59] = ( in[36] >> 28 ) ;

	out[59] |= ( in[37] << ( 32 - 28 ) ) & 0x0fffff ;
	out[60] = ( in[37] >> 16 ) ;

	out[60] |= ( in[38] << ( 32 - 16 ) ) & 0x0fffff ;
	out[61] = ( in[38] >> 4 ) & 0x0fffff ;
	out[62] = ( in[38] >> 24 ) ;

	out[62] |= ( in[39] << ( 32 - 24 ) ) & 0x0fffff ;
	out[63] = ( in[39] >> 12 ) ;
}


// 21-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c21(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x1fffff ;
	out[1] = ( in[0] >> 21 ) ;

	out[1] |= ( in[1] << ( 32 - 21 ) ) & 0x1fffff ;
	out[2] = ( in[1] >> 10 ) & 0x1fffff ;
	out[3] = ( in[1] >> 31 ) ;

	out[3] |= ( in[2] << ( 32 - 31 ) ) & 0x1fffff ;
	out[4] = ( in[2] >> 20 ) ;

	out[4] |= ( in[3] << ( 32 - 20 ) ) & 0x1fffff ;
	out[5] = ( in[3] >> 9 ) & 0x1fffff ;
	out[6] = ( in[3] >> 30 ) ;

	out[6] |= ( in[4] << ( 32 - 30 ) ) & 0x1fffff ;
	out[7] = ( in[4] >> 19 ) ;

	out[7] |= ( in[5] << ( 32 - 19 ) ) & 0x1fffff ;
	out[8] = ( in[5] >> 8 ) & 0x1fffff ;
	out[9] = ( in[5] >> 29 ) ;

	out[9] |= ( in[6] << ( 32 - 29 ) ) & 0x1fffff ;
	out[10] = ( in[6] >> 18 ) ;

	out[10] |= ( in[7] << ( 32 - 18 ) ) & 0x1fffff ;
	out[11] = ( in[7] >> 7 ) & 0x1fffff ;
	out[12] = ( in[7] >> 28 ) ;

	out[12] |= ( in[8] << ( 32 - 28 ) ) & 0x1fffff ;
	out[13] = ( in[8] >> 17 ) ;

	out[13] |= ( in[9] << ( 32 - 17 ) ) & 0x1fffff ;
	out[14] = ( in[9] >> 6 ) & 0x1fffff ;
	out[15] = ( in[9] >> 27 ) ;

	out[15] |= ( in[10] << ( 32 - 27 ) ) & 0x1fffff ;
	out[16] = ( in[10] >> 16 ) ;

	out[16] |= ( in[11] << ( 32 - 16 ) ) & 0x1fffff ;
	out[17] = ( in[11] >> 5 ) & 0x1fffff ;
	out[18] = ( in[11] >> 26 ) ;

	out[18] |= ( in[12] << ( 32 - 26 ) ) & 0x1fffff ;
	out[19] = ( in[12] >> 15 ) ;

	out[19] |= ( in[13] << ( 32 - 15 ) ) & 0x1fffff ;
	out[20] = ( in[13] >> 4 ) & 0x1fffff ;
	out[21] = ( in[13] >> 25 ) ;

	out[21] |= ( in[14] << ( 32 - 25 ) ) & 0x1fffff ;
	out[22] = ( in[14] >> 14 ) ;

	out[22] |= ( in[15] << ( 32 - 14 ) ) & 0x1fffff ;
	out[23] = ( in[15] >> 3 ) & 0x1fffff ;
	out[24] = ( in[15] >> 24 ) ;

	out[24] |= ( in[16] << ( 32 - 24 ) ) & 0x1fffff ;
	out[25] = ( in[16] >> 13 ) ;

	out[25] |= ( in[17] << ( 32 - 13 ) ) & 0x1fffff ;
	out[26] = ( in[17] >> 2 ) & 0x1fffff ;
	out[27] = ( in[17] >> 23 ) ;

	out[27] |= ( in[18] << ( 32 - 23 ) ) & 0x1fffff ;
	out[28] = ( in[18] >> 12 ) ;

	out[28] |= ( in[19] << ( 32 - 12 ) ) & 0x1fffff ;
	out[29] = ( in[19] >> 1 ) & 0x1fffff ;
	out[30] = ( in[19] >> 22 ) ;

	out[30] |= ( in[20] << ( 32 - 22 ) ) & 0x1fffff ;
	out[31] = ( in[20] >> 11 ) ;

	out[32] = ( in[21] >> 0 ) & 0x1fffff ;
	out[33] = ( in[21] >> 21 ) ;

	out[33] |= ( in[22] << ( 32 - 21 ) ) & 0x1fffff ;
	out[34] = ( in[22] >> 10 ) & 0x1fffff ;
	out[35] = ( in[22] >> 31 ) ;

	out[35] |= ( in[23] << ( 32 - 31 ) ) & 0x1fffff ;
	out[36] = ( in[23] >> 20 ) ;

	out[36] |= ( in[24] << ( 32 - 20 ) ) & 0x1fffff ;
	out[37] = ( in[24] >> 9 ) & 0x1fffff ;
	out[38] = ( in[24] >> 30 ) ;

	out[38] |= ( in[25] << ( 32 - 30 ) ) & 0x1fffff ;
	out[39] = ( in[25] >> 19 ) ;

	out[39] |= ( in[26] << ( 32 - 19 ) ) & 0x1fffff ;
	out[40] = ( in[26] >> 8 ) & 0x1fffff ;
	out[41] = ( in[26] >> 29 ) ;

	out[41] |= ( in[27] << ( 32 - 29 ) ) & 0x1fffff ;
	out[42] = ( in[27] >> 18 ) ;

	out[42] |= ( in[28] << ( 32 - 18 ) ) & 0x1fffff ;
	out[43] = ( in[28] >> 7 ) & 0x1fffff ;
	out[44] = ( in[28] >> 28 ) ;

	out[44] |= ( in[29] << ( 32 - 28 ) ) & 0x1fffff ;
	out[45] = ( in[29] >> 17 ) ;

	out[45] |= ( in[30] << ( 32 - 17 ) ) & 0x1fffff ;
	out[46] = ( in[30] >> 6 ) & 0x1fffff ;
	out[47] = ( in[30] >> 27 ) ;

	out[47] |= ( in[31] << ( 32 - 27 ) ) & 0x1fffff ;
	out[48] = ( in[31] >> 16 ) ;

	out[48] |= ( in[32] << ( 32 - 16 ) ) & 0x1fffff ;
	out[49] = ( in[32] >> 5 ) & 0x1fffff ;
	out[50] = ( in[32] >> 26 ) ;

	out[50] |= ( in[33] << ( 32 - 26 ) ) & 0x1fffff ;
	out[51] = ( in[33] >> 15 ) ;

	out[51] |= ( in[34] << ( 32 - 15 ) ) & 0x1fffff ;
	out[52] = ( in[34] >> 4 ) & 0x1fffff ;
	out[53] = ( in[34] >> 25 ) ;

	out[53] |= ( in[35] << ( 32 - 25 ) ) & 0x1fffff ;
	out[54] = ( in[35] >> 14 ) ;

	out[54] |= ( in[36] << ( 32 - 14 ) ) & 0x1fffff ;
	out[55] = ( in[36] >> 3 ) & 0x1fffff ;
	out[56] = ( in[36] >> 24 ) ;

	out[56] |= ( in[37] << ( 32 - 24 ) ) & 0x1fffff ;
	out[57] = ( in[37] >> 13 ) ;

	out[57] |= ( in[38] << ( 32 - 13 ) ) & 0x1fffff ;
	out[58] = ( in[38] >> 2 ) & 0x1fffff ;
	out[59] = ( in[38] >> 23 ) ;

	out[59] |= ( in[39] << ( 32 - 23 ) ) & 0x1fffff ;
	out[60] = ( in[39] >> 12 ) ;

	out[60] |= ( in[40] << ( 32 - 12 ) ) & 0x1fffff ;
	out[61] = ( in[40] >> 1 ) & 0x1fffff ;
	out[62] = ( in[40] >> 22 ) ;

	out[62] |= ( in[41] << ( 32 - 22 ) ) & 0x1fffff ;
	out[63] = ( in[41] >> 11 ) ;
}


// 22-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c22(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x3fffff ;
	out[1] = ( in[0] >> 22 ) ;

	out[1] |= ( in[1] << ( 32 - 22 ) ) & 0x3fffff ;
	out[2] = ( in[1] >> 12 ) ;

	out[2] |= ( in[2] << ( 32 - 12 ) ) & 0x3fffff ;
	out[3] = ( in[2] >> 2 ) & 0x3fffff ;
	out[4] = ( in[2] >> 24 ) ;

	out[4] |= ( in[3] << ( 32 - 24 ) ) & 0x3fffff ;
	out[5] = ( in[3] >> 14 ) ;

	out[5] |= ( in[4] << ( 32 - 14 ) ) & 0x3fffff ;
	out[6] = ( in[4] >> 4 ) & 0x3fffff ;
	out[7] = ( in[4] >> 26 ) ;

	out[7] |= ( in[5] << ( 32 - 26 ) ) & 0x3fffff ;
	out[8] = ( in[5] >> 16 ) ;

	out[8] |= ( in[6] << ( 32 - 16 ) ) & 0x3fffff ;
	out[9] = ( in[6] >> 6 ) & 0x3fffff ;
	out[10] = ( in[6] >> 28 ) ;

	out[10] |= ( in[7] << ( 32 - 28 ) ) & 0x3fffff ;
	out[11] = ( in[7] >> 18 ) ;

	out[11] |= ( in[8] << ( 32 - 18 ) ) & 0x3fffff ;
	out[12] = ( in[8] >> 8 ) & 0x3fffff ;
	out[13] = ( in[8] >> 30 ) ;

	out[13] |= ( in[9] << ( 32 - 30 ) ) & 0x3fffff ;
	out[14] = ( in[9] >> 20 ) ;

	out[14] |= ( in[10] << ( 32 - 20 ) ) & 0x3fffff ;
	out[15] = ( in[10] >> 10 ) ;

	out[16] = ( in[11] >> 0 ) & 0x3fffff ;
	out[17] = ( in[11] >> 22 ) ;

	out[17] |= ( in[12] << ( 32 - 22 ) ) & 0x3fffff ;
	out[18] = ( in[12] >> 12 ) ;

	out[18] |= ( in[13] << ( 32 - 12 ) ) & 0x3fffff ;
	out[19] = ( in[13] >> 2 ) & 0x3fffff ;
	out[20] = ( in[13] >> 24 ) ;

	out[20] |= ( in[14] << ( 32 - 24 ) ) & 0x3fffff ;
	out[21] = ( in[14] >> 14 ) ;

	out[21] |= ( in[15] << ( 32 - 14 ) ) & 0x3fffff ;
	out[22] = ( in[15] >> 4 ) & 0x3fffff ;
	out[23] = ( in[15] >> 26 ) ;

	out[23] |= ( in[16] << ( 32 - 26 ) ) & 0x3fffff ;
	out[24] = ( in[16] >> 16 ) ;

	out[24] |= ( in[17] << ( 32 - 16 ) ) & 0x3fffff ;
	out[25] = ( in[17] >> 6 ) & 0x3fffff ;
	out[26] = ( in[17] >> 28 ) ;

	out[26] |= ( in[18] << ( 32 - 28 ) ) & 0x3fffff ;
	out[27] = ( in[18] >> 18 ) ;

	out[27] |= ( in[19] << ( 32 - 18 ) ) & 0x3fffff ;
	out[28] = ( in[19] >> 8 ) & 0x3fffff ;
	out[29] = ( in[19] >> 30 ) ;

	out[29] |= ( in[20] << ( 32 - 30 ) ) & 0x3fffff ;
	out[30] = ( in[20] >> 20 ) ;

	out[30] |= ( in[21] << ( 32 - 20 ) ) & 0x3fffff ;
	out[31] = ( in[21] >> 10 ) ;

	out[32] = ( in[22] >> 0 ) & 0x3fffff ;
	out[33] = ( in[22] >> 22 ) ;

	out[33] |= ( in[23] << ( 32 - 22 ) ) & 0x3fffff ;
	out[34] = ( in[23] >> 12 ) ;

	out[34] |= ( in[24] << ( 32 - 12 ) ) & 0x3fffff ;
	out[35] = ( in[24] >> 2 ) & 0x3fffff ;
	out[36] = ( in[24] >> 24 ) ;

	out[36] |= ( in[25] << ( 32 - 24 ) ) & 0x3fffff ;
	out[37] = ( in[25] >> 14 ) ;

	out[37] |= ( in[26] << ( 32 - 14 ) ) & 0x3fffff ;
	out[38] = ( in[26] >> 4 ) & 0x3fffff ;
	out[39] = ( in[26] >> 26 ) ;

	out[39] |= ( in[27] << ( 32 - 26 ) ) & 0x3fffff ;
	out[40] = ( in[27] >> 16 ) ;

	out[40] |= ( in[28] << ( 32 - 16 ) ) & 0x3fffff ;
	out[41] = ( in[28] >> 6 ) & 0x3fffff ;
	out[42] = ( in[28] >> 28 ) ;

	out[42] |= ( in[29] << ( 32 - 28 ) ) & 0x3fffff ;
	out[43] = ( in[29] >> 18 ) ;

	out[43] |= ( in[30] << ( 32 - 18 ) ) & 0x3fffff ;
	out[44] = ( in[30] >> 8 ) & 0x3fffff ;
	out[45] = ( in[30] >> 30 ) ;

	out[45] |= ( in[31] << ( 32 - 30 ) ) & 0x3fffff ;
	out[46] = ( in[31] >> 20 ) ;

	out[46] |= ( in[32] << ( 32 - 20 ) ) & 0x3fffff ;
	out[47] = ( in[32] >> 10 ) ;

	out[48] = ( in[33] >> 0 ) & 0x3fffff ;
	out[49] = ( in[33] >> 22 ) ;

	out[49] |= ( in[34] << ( 32 - 22 ) ) & 0x3fffff ;
	out[50] = ( in[34] >> 12 ) ;

	out[50] |= ( in[35] << ( 32 - 12 ) ) & 0x3fffff ;
	out[51] = ( in[35] >> 2 ) & 0x3fffff ;
	out[52] = ( in[35] >> 24 ) ;

	out[52] |= ( in[36] << ( 32 - 24 ) ) & 0x3fffff ;
	out[53] = ( in[36] >> 14 ) ;

	out[53] |= ( in[37] << ( 32 - 14 ) ) & 0x3fffff ;
	out[54] = ( in[37] >> 4 ) & 0x3fffff ;
	out[55] = ( in[37] >> 26 ) ;

	out[55] |= ( in[38] << ( 32 - 26 ) ) & 0x3fffff ;
	out[56] = ( in[38] >> 16 ) ;

	out[56] |= ( in[39] << ( 32 - 16 ) ) & 0x3fffff ;
	out[57] = ( in[39] >> 6 ) & 0x3fffff ;
	out[58] = ( in[39] >> 28 ) ;

	out[58] |= ( in[40] << ( 32 - 28 ) ) & 0x3fffff ;
	out[59] = ( in[40] >> 18 ) ;

	out[59] |= ( in[41] << ( 32 - 18 ) ) & 0x3fffff ;
	out[60] = ( in[41] >> 8 ) & 0x3fffff ;
	out[61] = ( in[41] >> 30 ) ;

	out[61] |= ( in[42] << ( 32 - 30 ) ) & 0x3fffff ;
	out[62] = ( in[42] >> 20 ) ;

	out[62] |= ( in[43] << ( 32 - 20 ) ) & 0x3fffff ;
	out[63] = ( in[43] >> 10 ) ;
}


// 23-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c23(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x7fffff ;
	out[1] = ( in[0] >> 23 ) ;

	out[1] |= ( in[1] << ( 32 - 23 ) ) & 0x7fffff ;
	out[2] = ( in[1] >> 14 ) ;

	out[2] |= ( in[2] << ( 32 - 14 ) ) & 0x7fffff ;
	out[3] = ( in[2] >> 5 ) & 0x7fffff ;
	out[4] = ( in[2] >> 28 ) ;

	out[4] |= ( in[3] << ( 32 - 28 ) ) & 0x7fffff ;
	out[5] = ( in[3] >> 19 ) ;

	out[5] |= ( in[4] << ( 32 - 19 ) ) & 0x7fffff ;
	out[6] = ( in[4] >> 10 ) ;

	out[6] |= ( in[5] << ( 32 - 10 ) ) & 0x7fffff ;
	out[7] = ( in[5] >> 1 ) & 0x7fffff ;
	out[8] = ( in[5] >> 24 ) ;

	out[8] |= ( in[6] << ( 32 - 24 ) ) & 0x7fffff ;
	out[9] = ( in[6] >> 15 ) ;

	out[9] |= ( in[7] << ( 32 - 15 ) ) & 0x7fffff ;
	out[10] = ( in[7] >> 6 ) & 0x7fffff ;
	out[11] = ( in[7] >> 29 ) ;

	out[11] |= ( in[8] << ( 32 - 29 ) ) & 0x7fffff ;
	out[12] = ( in[8] >> 20 ) ;

	out[12] |= ( in[9] << ( 32 - 20 ) ) & 0x7fffff ;
	out[13] = ( in[9] >> 11 ) ;

	out[13] |= ( in[10] << ( 32 - 11 ) ) & 0x7fffff ;
	out[14] = ( in[10] >> 2 ) & 0x7fffff ;
	out[15] = ( in[10] >> 25 ) ;

	out[15] |= ( in[11] << ( 32 - 25 ) ) & 0x7fffff ;
	out[16] = ( in[11] >> 16 ) ;

	out[16] |= ( in[12] << ( 32 - 16 ) ) & 0x7fffff ;
	out[17] = ( in[12] >> 7 ) & 0x7fffff ;
	out[18] = ( in[12] >> 30 ) ;

	out[18] |= ( in[13] << ( 32 - 30 ) ) & 0x7fffff ;
	out[19] = ( in[13] >> 21 ) ;

	out[19] |= ( in[14] << ( 32 - 21 ) ) & 0x7fffff ;
	out[20] = ( in[14] >> 12 ) ;

	out[20] |= ( in[15] << ( 32 - 12 ) ) & 0x7fffff ;
	out[21] = ( in[15] >> 3 ) & 0x7fffff ;
	out[22] = ( in[15] >> 26 ) ;

	out[22] |= ( in[16] << ( 32 - 26 ) ) & 0x7fffff ;
	out[23] = ( in[16] >> 17 ) ;

	out[23] |= ( in[17] << ( 32 - 17 ) ) & 0x7fffff ;
	out[24] = ( in[17] >> 8 ) & 0x7fffff ;
	out[25] = ( in[17] >> 31 ) ;

	out[25] |= ( in[18] << ( 32 - 31 ) ) & 0x7fffff ;
	out[26] = ( in[18] >> 22 ) ;

	out[26] |= ( in[19] << ( 32 - 22 ) ) & 0x7fffff ;
	out[27] = ( in[19] >> 13 ) ;

	out[27] |= ( in[20] << ( 32 - 13 ) ) & 0x7fffff ;
	out[28] = ( in[20] >> 4 ) & 0x7fffff ;
	out[29] = ( in[20] >> 27 ) ;

	out[29] |= ( in[21] << ( 32 - 27 ) ) & 0x7fffff ;
	out[30] = ( in[21] >> 18 ) ;

	out[30] |= ( in[22] << ( 32 - 18 ) ) & 0x7fffff ;
	out[31] = ( in[22] >> 9 ) ;

	out[32] = ( in[23] >> 0 ) & 0x7fffff ;
	out[33] = ( in[23] >> 23 ) ;

	out[33] |= ( in[24] << ( 32 - 23 ) ) & 0x7fffff ;
	out[34] = ( in[24] >> 14 ) ;

	out[34] |= ( in[25] << ( 32 - 14 ) ) & 0x7fffff ;
	out[35] = ( in[25] >> 5 ) & 0x7fffff ;
	out[36] = ( in[25] >> 28 ) ;

	out[36] |= ( in[26] << ( 32 - 28 ) ) & 0x7fffff ;
	out[37] = ( in[26] >> 19 ) ;

	out[37] |= ( in[27] << ( 32 - 19 ) ) & 0x7fffff ;
	out[38] = ( in[27] >> 10 ) ;

	out[38] |= ( in[28] << ( 32 - 10 ) ) & 0x7fffff ;
	out[39] = ( in[28] >> 1 ) & 0x7fffff ;
	out[40] = ( in[28] >> 24 ) ;

	out[40] |= ( in[29] << ( 32 - 24 ) ) & 0x7fffff ;
	out[41] = ( in[29] >> 15 ) ;

	out[41] |= ( in[30] << ( 32 - 15 ) ) & 0x7fffff ;
	out[42] = ( in[30] >> 6 ) & 0x7fffff ;
	out[43] = ( in[30] >> 29 ) ;

	out[43] |= ( in[31] << ( 32 - 29 ) ) & 0x7fffff ;
	out[44] = ( in[31] >> 20 ) ;

	out[44] |= ( in[32] << ( 32 - 20 ) ) & 0x7fffff ;
	out[45] = ( in[32] >> 11 ) ;

	out[45] |= ( in[33] << ( 32 - 11 ) ) & 0x7fffff ;
	out[46] = ( in[33] >> 2 ) & 0x7fffff ;
	out[47] = ( in[33] >> 25 ) ;

	out[47] |= ( in[34] << ( 32 - 25 ) ) & 0x7fffff ;
	out[48] = ( in[34] >> 16 ) ;

	out[48] |= ( in[35] << ( 32 - 16 ) ) & 0x7fffff ;
	out[49] = ( in[35] >> 7 ) & 0x7fffff ;
	out[50] = ( in[35] >> 30 ) ;

	out[50] |= ( in[36] << ( 32 - 30 ) ) & 0x7fffff ;
	out[51] = ( in[36] >> 21 ) ;

	out[51] |= ( in[37] << ( 32 - 21 ) ) & 0x7fffff ;
	out[52] = ( in[37] >> 12 ) ;

	out[52] |= ( in[38] << ( 32 - 12 ) ) & 0x7fffff ;
	out[53] = ( in[38] >> 3 ) & 0x7fffff ;
	out[54] = ( in[38] >> 26 ) ;

	out[54] |= ( in[39] << ( 32 - 26 ) ) & 0x7fffff ;
	out[55] = ( in[39] >> 17 ) ;

	out[55] |= ( in[40] << ( 32 - 17 ) ) & 0x7fffff ;
	out[56] = ( in[40] >> 8 ) & 0x7fffff ;
	out[57] = ( in[40] >> 31 ) ;

	out[57] |= ( in[41] << ( 32 - 31 ) ) & 0x7fffff ;
	out[58] = ( in[41] >> 22 ) ;

	out[58] |= ( in[42] << ( 32 - 22 ) ) & 0x7fffff ;
	out[59] = ( in[42] >> 13 ) ;

	out[59] |= ( in[43] << ( 32 - 13 ) ) & 0x7fffff ;
	out[60] = ( in[43] >> 4 ) & 0x7fffff ;
	out[61] = ( in[43] >> 27 ) ;

	out[61] |= ( in[44] << ( 32 - 27 ) ) & 0x7fffff ;
	out[62] = ( in[44] >> 18 ) ;

	out[62] |= ( in[45] << ( 32 - 18 ) ) & 0x7fffff ;
	out[63] = ( in[45] >> 9 ) ;
}


// 24-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c24(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0xffffff ;
	out[1] = ( in[0] >> 24 ) ;

	out[1] |= ( in[1] << ( 32 - 24 ) ) & 0xffffff ;
	out[2] = ( in[1] >> 16 ) ;

	out[2] |= ( in[2] << ( 32 - 16 ) ) & 0xffffff ;
	out[3] = ( in[2] >> 8 ) ;

	out[4] = ( in[3] >> 0 ) & 0xffffff ;
	out[5] = ( in[3] >> 24 ) ;

	out[5] |= ( in[4] << ( 32 - 24 ) ) & 0xffffff ;
	out[6] = ( in[4] >> 16 ) ;

	out[6] |= ( in[5] << ( 32 - 16 ) ) & 0xffffff ;
	out[7] = ( in[5] >> 8 ) ;

	out[8] = ( in[6] >> 0 ) & 0xffffff ;
	out[9] = ( in[6] >> 24 ) ;

	out[9] |= ( in[7] << ( 32 - 24 ) ) & 0xffffff ;
	out[10] = ( in[7] >> 16 ) ;

	out[10] |= ( in[8] << ( 32 - 16 ) ) & 0xffffff ;
	out[11] = ( in[8] >> 8 ) ;

	out[12] = ( in[9] >> 0 ) & 0xffffff ;
	out[13] = ( in[9] >> 24 ) ;

	out[13] |= ( in[10] << ( 32 - 24 ) ) & 0xffffff ;
	out[14] = ( in[10] >> 16 ) ;

	out[14] |= ( in[11] << ( 32 - 16 ) ) & 0xffffff ;
	out[15] = ( in[11] >> 8 ) ;

	out[16] = ( in[12] >> 0 ) & 0xffffff ;
	out[17] = ( in[12] >> 24 ) ;

	out[17] |= ( in[13] << ( 32 - 24 ) ) & 0xffffff ;
	out[18] = ( in[13] >> 16 ) ;

	out[18] |= ( in[14] << ( 32 - 16 ) ) & 0xffffff ;
	out[19] = ( in[14] >> 8 ) ;

	out[20] = ( in[15] >> 0 ) & 0xffffff ;
	out[21] = ( in[15] >> 24 ) ;

	out[21] |= ( in[16] << ( 32 - 24 ) ) & 0xffffff ;
	out[22] = ( in[16] >> 16 ) ;

	out[22] |= ( in[17] << ( 32 - 16 ) ) & 0xffffff ;
	out[23] = ( in[17] >> 8 ) ;

	out[24] = ( in[18] >> 0 ) & 0xffffff ;
	out[25] = ( in[18] >> 24 ) ;

	out[25] |= ( in[19] << ( 32 - 24 ) ) & 0xffffff ;
	out[26] = ( in[19] >> 16 ) ;

	out[26] |= ( in[20] << ( 32 - 16 ) ) & 0xffffff ;
	out[27] = ( in[20] >> 8 ) ;

	out[28] = ( in[21] >> 0 ) & 0xffffff ;
	out[29] = ( in[21] >> 24 ) ;

	out[29] |= ( in[22] << ( 32 - 24 ) ) & 0xffffff ;
	out[30] = ( in[22] >> 16 ) ;

	out[30] |= ( in[23] << ( 32 - 16 ) ) & 0xffffff ;
	out[31] = ( in[23] >> 8 ) ;

	out[32] = ( in[24] >> 0 ) & 0xffffff ;
	out[33] = ( in[24] >> 24 ) ;

	out[33] |= ( in[25] << ( 32 - 24 ) ) & 0xffffff ;
	out[34] = ( in[25] >> 16 ) ;

	out[34] |= ( in[26] << ( 32 - 16 ) ) & 0xffffff ;
	out[35] = ( in[26] >> 8 ) ;

	out[36] = ( in[27] >> 0 ) & 0xffffff ;
	out[37] = ( in[27] >> 24 ) ;

	out[37] |= ( in[28] << ( 32 - 24 ) ) & 0xffffff ;
	out[38] = ( in[28] >> 16 ) ;

	out[38] |= ( in[29] << ( 32 - 16 ) ) & 0xffffff ;
	out[39] = ( in[29] >> 8 ) ;

	out[40] = ( in[30] >> 0 ) & 0xffffff ;
	out[41] = ( in[30] >> 24 ) ;

	out[41] |= ( in[31] << ( 32 - 24 ) ) & 0xffffff ;
	out[42] = ( in[31] >> 16 ) ;

	out[42] |= ( in[32] << ( 32 - 16 ) ) & 0xffffff ;
	out[43] = ( in[32] >> 8 ) ;

	out[44] = ( in[33] >> 0 ) & 0xffffff ;
	out[45] = ( in[33] >> 24 ) ;

	out[45] |= ( in[34] << ( 32 - 24 ) ) & 0xffffff ;
	out[46] = ( in[34] >> 16 ) ;

	out[46] |= ( in[35] << ( 32 - 16 ) ) & 0xffffff ;
	out[47] = ( in[35] >> 8 ) ;

	out[48] = ( in[36] >> 0 ) & 0xffffff ;
	out[49] = ( in[36] >> 24 ) ;

	out[49] |= ( in[37] << ( 32 - 24 ) ) & 0xffffff ;
	out[50] = ( in[37] >> 16 ) ;

	out[50] |= ( in[38] << ( 32 - 16 ) ) & 0xffffff ;
	out[51] = ( in[38] >> 8 ) ;

	out[52] = ( in[39] >> 0 ) & 0xffffff ;
	out[53] = ( in[39] >> 24 ) ;

	out[53] |= ( in[40] << ( 32 - 24 ) ) & 0xffffff ;
	out[54] = ( in[40] >> 16 ) ;

	out[54] |= ( in[41] << ( 32 - 16 ) ) & 0xffffff ;
	out[55] = ( in[41] >> 8 ) ;

	out[56] = ( in[42] >> 0 ) & 0xffffff ;
	out[57] = ( in[42] >> 24 ) ;

	out[57] |= ( in[43] << ( 32 - 24 ) ) & 0xffffff ;
	out[58] = ( in[43] >> 16 ) ;

	out[58] |= ( in[44] << ( 32 - 16 ) ) & 0xffffff ;
	out[59] = ( in[44] >> 8 ) ;

	out[60] = ( in[45] >> 0 ) & 0xffffff ;
	out[61] = ( in[45] >> 24 ) ;

	out[61] |= ( in[46] << ( 32 - 24 ) ) & 0xffffff ;
	out[62] = ( in[46] >> 16 ) ;

	out[62] |= ( in[47] << ( 32 - 16 ) ) & 0xffffff ;
	out[63] = ( in[47] >> 8 ) ;
}


// 25-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c25(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x01ffffff ;
	out[1] = ( in[0] >> 25 ) ;

	out[1] |= ( in[1] << ( 32 - 25 ) ) & 0x01ffffff ;
	out[2] = ( in[1] >> 18 ) ;

	out[2] |= ( in[2] << ( 32 - 18 ) ) & 0x01ffffff ;
	out[3] = ( in[2] >> 11 ) ;

	out[3] |= ( in[3] << ( 32 - 11 ) ) & 0x01ffffff ;
	out[4] = ( in[3] >> 4 ) & 0x01ffffff ;
	out[5] = ( in[3] >> 29 ) ;

	out[5] |= ( in[4] << ( 32 - 29 ) ) & 0x01ffffff ;
	out[6] = ( in[4] >> 22 ) ;

	out[6] |= ( in[5] << ( 32 - 22 ) ) & 0x01ffffff ;
	out[7] = ( in[5] >> 15 ) ;

	out[7] |= ( in[6] << ( 32 - 15 ) ) & 0x01ffffff ;
	out[8] = ( in[6] >> 8 ) ;

	out[8] |= ( in[7] << ( 32 - 8 ) ) & 0x01ffffff ;
	out[9] = ( in[7] >> 1 ) & 0x01ffffff ;
	out[10] = ( in[7] >> 26 ) ;

	out[10] |= ( in[8] << ( 32 - 26 ) ) & 0x01ffffff ;
	out[11] = ( in[8] >> 19 ) ;

	out[11] |= ( in[9] << ( 32 - 19 ) ) & 0x01ffffff ;
	out[12] = ( in[9] >> 12 ) ;

	out[12] |= ( in[10] << ( 32 - 12 ) ) & 0x01ffffff ;
	out[13] = ( in[10] >> 5 ) & 0x01ffffff ;
	out[14] = ( in[10] >> 30 ) ;

	out[14] |= ( in[11] << ( 32 - 30 ) ) & 0x01ffffff ;
	out[15] = ( in[11] >> 23 ) ;

	out[15] |= ( in[12] << ( 32 - 23 ) ) & 0x01ffffff ;
	out[16] = ( in[12] >> 16 ) ;

	out[16] |= ( in[13] << ( 32 - 16 ) ) & 0x01ffffff ;
	out[17] = ( in[13] >> 9 ) ;

	out[17] |= ( in[14] << ( 32 - 9 ) ) & 0x01ffffff ;
	out[18] = ( in[14] >> 2 ) & 0x01ffffff ;
	out[19] = ( in[14] >> 27 ) ;

	out[19] |= ( in[15] << ( 32 - 27 ) ) & 0x01ffffff ;
	out[20] = ( in[15] >> 20 ) ;

	out[20] |= ( in[16] << ( 32 - 20 ) ) & 0x01ffffff ;
	out[21] = ( in[16] >> 13 ) ;

	out[21] |= ( in[17] << ( 32 - 13 ) ) & 0x01ffffff ;
	out[22] = ( in[17] >> 6 ) & 0x01ffffff ;
	out[23] = ( in[17] >> 31 ) ;

	out[23] |= ( in[18] << ( 32 - 31 ) ) & 0x01ffffff ;
	out[24] = ( in[18] >> 24 ) ;

	out[24] |= ( in[19] << ( 32 - 24 ) ) & 0x01ffffff ;
	out[25] = ( in[19] >> 17 ) ;

	out[25] |= ( in[20] << ( 32 - 17 ) ) & 0x01ffffff ;
	out[26] = ( in[20] >> 10 ) ;

	out[26] |= ( in[21] << ( 32 - 10 ) ) & 0x01ffffff ;
	out[27] = ( in[21] >> 3 ) & 0x01ffffff ;
	out[28] = ( in[21] >> 28 ) ;

	out[28] |= ( in[22] << ( 32 - 28 ) ) & 0x01ffffff ;
	out[29] = ( in[22] >> 21 ) ;

	out[29] |= ( in[23] << ( 32 - 21 ) ) & 0x01ffffff ;
	out[30] = ( in[23] >> 14 ) ;

	out[30] |= ( in[24] << ( 32 - 14 ) ) & 0x01ffffff ;
	out[31] = ( in[24] >> 7 ) ;

	out[32] = ( in[25] >> 0 ) & 0x01ffffff ;
	out[33] = ( in[25] >> 25 ) ;

	out[33] |= ( in[26] << ( 32 - 25 ) ) & 0x01ffffff ;
	out[34] = ( in[26] >> 18 ) ;

	out[34] |= ( in[27] << ( 32 - 18 ) ) & 0x01ffffff ;
	out[35] = ( in[27] >> 11 ) ;

	out[35] |= ( in[28] << ( 32 - 11 ) ) & 0x01ffffff ;
	out[36] = ( in[28] >> 4 ) & 0x01ffffff ;
	out[37] = ( in[28] >> 29 ) ;

	out[37] |= ( in[29] << ( 32 - 29 ) ) & 0x01ffffff ;
	out[38] = ( in[29] >> 22 ) ;

	out[38] |= ( in[30] << ( 32 - 22 ) ) & 0x01ffffff ;
	out[39] = ( in[30] >> 15 ) ;

	out[39] |= ( in[31] << ( 32 - 15 ) ) & 0x01ffffff ;
	out[40] = ( in[31] >> 8 ) ;

	out[40] |= ( in[32] << ( 32 - 8 ) ) & 0x01ffffff ;
	out[41] = ( in[32] >> 1 ) & 0x01ffffff ;
	out[42] = ( in[32] >> 26 ) ;

	out[42] |= ( in[33] << ( 32 - 26 ) ) & 0x01ffffff ;
	out[43] = ( in[33] >> 19 ) ;

	out[43] |= ( in[34] << ( 32 - 19 ) ) & 0x01ffffff ;
	out[44] = ( in[34] >> 12 ) ;

	out[44] |= ( in[35] << ( 32 - 12 ) ) & 0x01ffffff ;
	out[45] = ( in[35] >> 5 ) & 0x01ffffff ;
	out[46] = ( in[35] >> 30 ) ;

	out[46] |= ( in[36] << ( 32 - 30 ) ) & 0x01ffffff ;
	out[47] = ( in[36] >> 23 ) ;

	out[47] |= ( in[37] << ( 32 - 23 ) ) & 0x01ffffff ;
	out[48] = ( in[37] >> 16 ) ;

	out[48] |= ( in[38] << ( 32 - 16 ) ) & 0x01ffffff ;
	out[49] = ( in[38] >> 9 ) ;

	out[49] |= ( in[39] << ( 32 - 9 ) ) & 0x01ffffff ;
	out[50] = ( in[39] >> 2 ) & 0x01ffffff ;
	out[51] = ( in[39] >> 27 ) ;

	out[51] |= ( in[40] << ( 32 - 27 ) ) & 0x01ffffff ;
	out[52] = ( in[40] >> 20 ) ;

	out[52] |= ( in[41] << ( 32 - 20 ) ) & 0x01ffffff ;
	out[53] = ( in[41] >> 13 ) ;

	out[53] |= ( in[42] << ( 32 - 13 ) ) & 0x01ffffff ;
	out[54] = ( in[42] >> 6 ) & 0x01ffffff ;
	out[55] = ( in[42] >> 31 ) ;

	out[55] |= ( in[43] << ( 32 - 31 ) ) & 0x01ffffff ;
	out[56] = ( in[43] >> 24 ) ;

	out[56] |= ( in[44] << ( 32 - 24 ) ) & 0x01ffffff ;
	out[57] = ( in[44] >> 17 ) ;

	out[57] |= ( in[45] << ( 32 - 17 ) ) & 0x01ffffff ;
	out[58] = ( in[45] >> 10 ) ;

	out[58] |= ( in[46] << ( 32 - 10 ) ) & 0x01ffffff ;
	out[59] = ( in[46] >> 3 ) & 0x01ffffff ;
	out[60] = ( in[46] >> 28 ) ;

	out[60] |= ( in[47] << ( 32 - 28 ) ) & 0x01ffffff ;
	out[61] = ( in[47] >> 21 ) ;

	out[61] |= ( in[48] << ( 32 - 21 ) ) & 0x01ffffff ;
	out[62] = ( in[48] >> 14 ) ;

	out[62] |= ( in[49] << ( 32 - 14 ) ) & 0x01ffffff ;
	out[63] = ( in[49] >> 7 ) ;
}


// 26-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c26(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x03ffffff ;
	out[1] = ( in[0] >> 26 ) ;

	out[1] |= ( in[1] << ( 32 - 26 ) ) & 0x03ffffff ;
	out[2] = ( in[1] >> 20 ) ;

	out[2] |= ( in[2] << ( 32 - 20 ) ) & 0x03ffffff ;
	out[3] = ( in[2] >> 14 ) ;

	out[3] |= ( in[3] << ( 32 - 14 ) ) & 0x03ffffff ;
	out[4] = ( in[3] >> 8 ) ;

	out[4] |= ( in[4] << ( 32 - 8 ) ) & 0x03ffffff ;
	out[5] = ( in[4] >> 2 ) & 0x03ffffff ;
	out[6] = ( in[4] >> 28 ) ;

	out[6] |= ( in[5] << ( 32 - 28 ) ) & 0x03ffffff ;
	out[7] = ( in[5] >> 22 ) ;

	out[7] |= ( in[6] << ( 32 - 22 ) ) & 0x03ffffff ;
	out[8] = ( in[6] >> 16 ) ;

	out[8] |= ( in[7] << ( 32 - 16 ) ) & 0x03ffffff ;
	out[9] = ( in[7] >> 10 ) ;

	out[9] |= ( in[8] << ( 32 - 10 ) ) & 0x03ffffff ;
	out[10] = ( in[8] >> 4 ) & 0x03ffffff ;
	out[11] = ( in[8] >> 30 ) ;

	out[11] |= ( in[9] << ( 32 - 30 ) ) & 0x03ffffff ;
	out[12] = ( in[9] >> 24 ) ;

	out[12] |= ( in[10] << ( 32 - 24 ) ) & 0x03ffffff ;
	out[13] = ( in[10] >> 18 ) ;

	out[13] |= ( in[11] << ( 32 - 18 ) ) & 0x03ffffff ;
	out[14] = ( in[11] >> 12 ) ;

	out[14] |= ( in[12] << ( 32 - 12 ) ) & 0x03ffffff ;
	out[15] = ( in[12] >> 6 ) ;

	out[16] = ( in[13] >> 0 ) & 0x03ffffff ;
	out[17] = ( in[13] >> 26 ) ;

	out[17] |= ( in[14] << ( 32 - 26 ) ) & 0x03ffffff ;
	out[18] = ( in[14] >> 20 ) ;

	out[18] |= ( in[15] << ( 32 - 20 ) ) & 0x03ffffff ;
	out[19] = ( in[15] >> 14 ) ;

	out[19] |= ( in[16] << ( 32 - 14 ) ) & 0x03ffffff ;
	out[20] = ( in[16] >> 8 ) ;

	out[20] |= ( in[17] << ( 32 - 8 ) ) & 0x03ffffff ;
	out[21] = ( in[17] >> 2 ) & 0x03ffffff ;
	out[22] = ( in[17] >> 28 ) ;

	out[22] |= ( in[18] << ( 32 - 28 ) ) & 0x03ffffff ;
	out[23] = ( in[18] >> 22 ) ;

	out[23] |= ( in[19] << ( 32 - 22 ) ) & 0x03ffffff ;
	out[24] = ( in[19] >> 16 ) ;

	out[24] |= ( in[20] << ( 32 - 16 ) ) & 0x03ffffff ;
	out[25] = ( in[20] >> 10 ) ;

	out[25] |= ( in[21] << ( 32 - 10 ) ) & 0x03ffffff ;
	out[26] = ( in[21] >> 4 ) & 0x03ffffff ;
	out[27] = ( in[21] >> 30 ) ;

	out[27] |= ( in[22] << ( 32 - 30 ) ) & 0x03ffffff ;
	out[28] = ( in[22] >> 24 ) ;

	out[28] |= ( in[23] << ( 32 - 24 ) ) & 0x03ffffff ;
	out[29] = ( in[23] >> 18 ) ;

	out[29] |= ( in[24] << ( 32 - 18 ) ) & 0x03ffffff ;
	out[30] = ( in[24] >> 12 ) ;

	out[30] |= ( in[25] << ( 32 - 12 ) ) & 0x03ffffff ;
	out[31] = ( in[25] >> 6 ) ;

	out[32] = ( in[26] >> 0 ) & 0x03ffffff ;
	out[33] = ( in[26] >> 26 ) ;

	out[33] |= ( in[27] << ( 32 - 26 ) ) & 0x03ffffff ;
	out[34] = ( in[27] >> 20 ) ;

	out[34] |= ( in[28] << ( 32 - 20 ) ) & 0x03ffffff ;
	out[35] = ( in[28] >> 14 ) ;

	out[35] |= ( in[29] << ( 32 - 14 ) ) & 0x03ffffff ;
	out[36] = ( in[29] >> 8 ) ;

	out[36] |= ( in[30] << ( 32 - 8 ) ) & 0x03ffffff ;
	out[37] = ( in[30] >> 2 ) & 0x03ffffff ;
	out[38] = ( in[30] >> 28 ) ;

	out[38] |= ( in[31] << ( 32 - 28 ) ) & 0x03ffffff ;
	out[39] = ( in[31] >> 22 ) ;

	out[39] |= ( in[32] << ( 32 - 22 ) ) & 0x03ffffff ;
	out[40] = ( in[32] >> 16 ) ;

	out[40] |= ( in[33] << ( 32 - 16 ) ) & 0x03ffffff ;
	out[41] = ( in[33] >> 10 ) ;

	out[41] |= ( in[34] << ( 32 - 10 ) ) & 0x03ffffff ;
	out[42] = ( in[34] >> 4 ) & 0x03ffffff ;
	out[43] = ( in[34] >> 30 ) ;

	out[43] |= ( in[35] << ( 32 - 30 ) ) & 0x03ffffff ;
	out[44] = ( in[35] >> 24 ) ;

	out[44] |= ( in[36] << ( 32 - 24 ) ) & 0x03ffffff ;
	out[45] = ( in[36] >> 18 ) ;

	out[45] |= ( in[37] << ( 32 - 18 ) ) & 0x03ffffff ;
	out[46] = ( in[37] >> 12 ) ;

	out[46] |= ( in[38] << ( 32 - 12 ) ) & 0x03ffffff ;
	out[47] = ( in[38] >> 6 ) ;

	out[48] = ( in[39] >> 0 ) & 0x03ffffff ;
	out[49] = ( in[39] >> 26 ) ;

	out[49] |= ( in[40] << ( 32 - 26 ) ) & 0x03ffffff ;
	out[50] = ( in[40] >> 20 ) ;

	out[50] |= ( in[41] << ( 32 - 20 ) ) & 0x03ffffff ;
	out[51] = ( in[41] >> 14 ) ;

	out[51] |= ( in[42] << ( 32 - 14 ) ) & 0x03ffffff ;
	out[52] = ( in[42] >> 8 ) ;

	out[52] |= ( in[43] << ( 32 - 8 ) ) & 0x03ffffff ;
	out[53] = ( in[43] >> 2 ) & 0x03ffffff ;
	out[54] = ( in[43] >> 28 ) ;

	out[54] |= ( in[44] << ( 32 - 28 ) ) & 0x03ffffff ;
	out[55] = ( in[44] >> 22 ) ;

	out[55] |= ( in[45] << ( 32 - 22 ) ) & 0x03ffffff ;
	out[56] = ( in[45] >> 16 ) ;

	out[56] |= ( in[46] << ( 32 - 16 ) ) & 0x03ffffff ;
	out[57] = ( in[46] >> 10 ) ;

	out[57] |= ( in[47] << ( 32 - 10 ) ) & 0x03ffffff ;
	out[58] = ( in[47] >> 4 ) & 0x03ffffff ;
	out[59] = ( in[47] >> 30 ) ;

	out[59] |= ( in[48] << ( 32 - 30 ) ) & 0x03ffffff ;
	out[60] = ( in[48] >> 24 ) ;

	out[60] |= ( in[49] << ( 32 - 24 ) ) & 0x03ffffff ;
	out[61] = ( in[49] >> 18 ) ;

	out[61] |= ( in[50] << ( 32 - 18 ) ) & 0x03ffffff ;
	out[62] = ( in[50] >> 12 ) ;

	out[62] |= ( in[51] << ( 32 - 12 ) ) & 0x03ffffff ;
	out[63] = ( in[51] >> 6 ) ;
}


// 27-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c27(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x07ffffff ;
	out[1] = ( in[0] >> 27 ) ;

	out[1] |= ( in[1] << ( 32 - 27 ) ) & 0x07ffffff ;
	out[2] = ( in[1] >> 22 ) ;

	out[2] |= ( in[2] << ( 32 - 22 ) ) & 0x07ffffff ;
	out[3] = ( in[2] >> 17 ) ;

	out[3] |= ( in[3] << ( 32 - 17 ) ) & 0x07ffffff ;
	out[4] = ( in[3] >> 12 ) ;

	out[4] |= ( in[4] << ( 32 - 12 ) ) & 0x07ffffff ;
	out[5] = ( in[4] >> 7 ) ;

	out[5] |= ( in[5] << ( 32 - 7 ) ) & 0x07ffffff ;
	out[6] = ( in[5] >> 2 ) & 0x07ffffff ;
	out[7] = ( in[5] >> 29 ) ;

	out[7] |= ( in[6] << ( 32 - 29 ) ) & 0x07ffffff ;
	out[8] = ( in[6] >> 24 ) ;

	out[8] |= ( in[7] << ( 32 - 24 ) ) & 0x07ffffff ;
	out[9] = ( in[7] >> 19 ) ;

	out[9] |= ( in[8] << ( 32 - 19 ) ) & 0x07ffffff ;
	out[10] = ( in[8] >> 14 ) ;

	out[10] |= ( in[9] << ( 32 - 14 ) ) & 0x07ffffff ;
	out[11] = ( in[9] >> 9 ) ;

	out[11] |= ( in[10] << ( 32 - 9 ) ) & 0x07ffffff ;
	out[12] = ( in[10] >> 4 ) & 0x07ffffff ;
	out[13] = ( in[10] >> 31 ) ;

	out[13] |= ( in[11] << ( 32 - 31 ) ) & 0x07ffffff ;
	out[14] = ( in[11] >> 26 ) ;

	out[14] |= ( in[12] << ( 32 - 26 ) ) & 0x07ffffff ;
	out[15] = ( in[12] >> 21 ) ;

	out[15] |= ( in[13] << ( 32 - 21 ) ) & 0x07ffffff ;
	out[16] = ( in[13] >> 16 ) ;

	out[16] |= ( in[14] << ( 32 - 16 ) ) & 0x07ffffff ;
	out[17] = ( in[14] >> 11 ) ;

	out[17] |= ( in[15] << ( 32 - 11 ) ) & 0x07ffffff ;
	out[18] = ( in[15] >> 6 ) ;

	out[18] |= ( in[16] << ( 32 - 6 ) ) & 0x07ffffff ;
	out[19] = ( in[16] >> 1 ) & 0x07ffffff ;
	out[20] = ( in[16] >> 28 ) ;

	out[20] |= ( in[17] << ( 32 - 28 ) ) & 0x07ffffff ;
	out[21] = ( in[17] >> 23 ) ;

	out[21] |= ( in[18] << ( 32 - 23 ) ) & 0x07ffffff ;
	out[22] = ( in[18] >> 18 ) ;

	out[22] |= ( in[19] << ( 32 - 18 ) ) & 0x07ffffff ;
	out[23] = ( in[19] >> 13 ) ;

	out[23] |= ( in[20] << ( 32 - 13 ) ) & 0x07ffffff ;
	out[24] = ( in[20] >> 8 ) ;

	out[24] |= ( in[21] << ( 32 - 8 ) ) & 0x07ffffff ;
	out[25] = ( in[21] >> 3 ) & 0x07ffffff ;
	out[26] = ( in[21] >> 30 ) ;

	out[26] |= ( in[22] << ( 32 - 30 ) ) & 0x07ffffff ;
	out[27] = ( in[22] >> 25 ) ;

	out[27] |= ( in[23] << ( 32 - 25 ) ) & 0x07ffffff ;
	out[28] = ( in[23] >> 20 ) ;

	out[28] |= ( in[24] << ( 32 - 20 ) ) & 0x07ffffff ;
	out[29] = ( in[24] >> 15 ) ;

	out[29] |= ( in[25] << ( 32 - 15 ) ) & 0x07ffffff ;
	out[30] = ( in[25] >> 10 ) ;

	out[30] |= ( in[26] << ( 32 - 10 ) ) & 0x07ffffff ;
	out[31] = ( in[26] >> 5 ) ;

	out[32] = ( in[27] >> 0 ) & 0x07ffffff ;
	out[33] = ( in[27] >> 27 ) ;

	out[33] |= ( in[28] << ( 32 - 27 ) ) & 0x07ffffff ;
	out[34] = ( in[28] >> 22 ) ;

	out[34] |= ( in[29] << ( 32 - 22 ) ) & 0x07ffffff ;
	out[35] = ( in[29] >> 17 ) ;

	out[35] |= ( in[30] << ( 32 - 17 ) ) & 0x07ffffff ;
	out[36] = ( in[30] >> 12 ) ;

	out[36] |= ( in[31] << ( 32 - 12 ) ) & 0x07ffffff ;
	out[37] = ( in[31] >> 7 ) ;

	out[37] |= ( in[32] << ( 32 - 7 ) ) & 0x07ffffff ;
	out[38] = ( in[32] >> 2 ) & 0x07ffffff ;
	out[39] = ( in[32] >> 29 ) ;

	out[39] |= ( in[33] << ( 32 - 29 ) ) & 0x07ffffff ;
	out[40] = ( in[33] >> 24 ) ;

	out[40] |= ( in[34] << ( 32 - 24 ) ) & 0x07ffffff ;
	out[41] = ( in[34] >> 19 ) ;

	out[41] |= ( in[35] << ( 32 - 19 ) ) & 0x07ffffff ;
	out[42] = ( in[35] >> 14 ) ;

	out[42] |= ( in[36] << ( 32 - 14 ) ) & 0x07ffffff ;
	out[43] = ( in[36] >> 9 ) ;

	out[43] |= ( in[37] << ( 32 - 9 ) ) & 0x07ffffff ;
	out[44] = ( in[37] >> 4 ) & 0x07ffffff ;
	out[45] = ( in[37] >> 31 ) ;

	out[45] |= ( in[38] << ( 32 - 31 ) ) & 0x07ffffff ;
	out[46] = ( in[38] >> 26 ) ;

	out[46] |= ( in[39] << ( 32 - 26 ) ) & 0x07ffffff ;
	out[47] = ( in[39] >> 21 ) ;

	out[47] |= ( in[40] << ( 32 - 21 ) ) & 0x07ffffff ;
	out[48] = ( in[40] >> 16 ) ;

	out[48] |= ( in[41] << ( 32 - 16 ) ) & 0x07ffffff ;
	out[49] = ( in[41] >> 11 ) ;

	out[49] |= ( in[42] << ( 32 - 11 ) ) & 0x07ffffff ;
	out[50] = ( in[42] >> 6 ) ;

	out[50] |= ( in[43] << ( 32 - 6 ) ) & 0x07ffffff ;
	out[51] = ( in[43] >> 1 ) & 0x07ffffff ;
	out[52] = ( in[43] >> 28 ) ;

	out[52] |= ( in[44] << ( 32 - 28 ) ) & 0x07ffffff ;
	out[53] = ( in[44] >> 23 ) ;

	out[53] |= ( in[45] << ( 32 - 23 ) ) & 0x07ffffff ;
	out[54] = ( in[45] >> 18 ) ;

	out[54] |= ( in[46] << ( 32 - 18 ) ) & 0x07ffffff ;
	out[55] = ( in[46] >> 13 ) ;

	out[55] |= ( in[47] << ( 32 - 13 ) ) & 0x07ffffff ;
	out[56] = ( in[47] >> 8 ) ;

	out[56] |= ( in[48] << ( 32 - 8 ) ) & 0x07ffffff ;
	out[57] = ( in[48] >> 3 ) & 0x07ffffff ;
	out[58] = ( in[48] >> 30 ) ;

	out[58] |= ( in[49] << ( 32 - 30 ) ) & 0x07ffffff ;
	out[59] = ( in[49] >> 25 ) ;

	out[59] |= ( in[50] << ( 32 - 25 ) ) & 0x07ffffff ;
	out[60] = ( in[50] >> 20 ) ;

	out[60] |= ( in[51] << ( 32 - 20 ) ) & 0x07ffffff ;
	out[61] = ( in[51] >> 15 ) ;

	out[61] |= ( in[52] << ( 32 - 15 ) ) & 0x07ffffff ;
	out[62] = ( in[52] >> 10 ) ;

	out[62] |= ( in[53] << ( 32 - 10 ) ) & 0x07ffffff ;
	out[63] = ( in[53] >> 5 ) ;
}


// 28-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c28(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x0fffffff ;
	out[1] = ( in[0] >> 28 ) ;

	out[1] |= ( in[1] << ( 32 - 28 ) ) & 0x0fffffff ;
	out[2] = ( in[1] >> 24 ) ;

	out[2] |= ( in[2] << ( 32 - 24 ) ) & 0x0fffffff ;
	out[3] = ( in[2] >> 20 ) ;

	out[3] |= ( in[3] << ( 32 - 20 ) ) & 0x0fffffff ;
	out[4] = ( in[3] >> 16 ) ;

	out[4] |= ( in[4] << ( 32 - 16 ) ) & 0x0fffffff ;
	out[5] = ( in[4] >> 12 ) ;

	out[5] |= ( in[5] << ( 32 - 12 ) ) & 0x0fffffff ;
	out[6] = ( in[5] >> 8 ) ;

	out[6] |= ( in[6] << ( 32 - 8 ) ) & 0x0fffffff ;
	out[7] = ( in[6] >> 4 ) ;

	out[8] = ( in[7] >> 0 ) & 0x0fffffff ;
	out[9] = ( in[7] >> 28 ) ;

	out[9] |= ( in[8] << ( 32 - 28 ) ) & 0x0fffffff ;
	out[10] = ( in[8] >> 24 ) ;

	out[10] |= ( in[9] << ( 32 - 24 ) ) & 0x0fffffff ;
	out[11] = ( in[9] >> 20 ) ;

	out[11] |= ( in[10] << ( 32 - 20 ) ) & 0x0fffffff ;
	out[12] = ( in[10] >> 16 ) ;

	out[12] |= ( in[11] << ( 32 - 16 ) ) & 0x0fffffff ;
	out[13] = ( in[11] >> 12 ) ;

	out[13] |= ( in[12] << ( 32 - 12 ) ) & 0x0fffffff ;
	out[14] = ( in[12] >> 8 ) ;

	out[14] |= ( in[13] << ( 32 - 8 ) ) & 0x0fffffff ;
	out[15] = ( in[13] >> 4 ) ;

	out[16] = ( in[14] >> 0 ) & 0x0fffffff ;
	out[17] = ( in[14] >> 28 ) ;

	out[17] |= ( in[15] << ( 32 - 28 ) ) & 0x0fffffff ;
	out[18] = ( in[15] >> 24 ) ;

	out[18] |= ( in[16] << ( 32 - 24 ) ) & 0x0fffffff ;
	out[19] = ( in[16] >> 20 ) ;

	out[19] |= ( in[17] << ( 32 - 20 ) ) & 0x0fffffff ;
	out[20] = ( in[17] >> 16 ) ;

	out[20] |= ( in[18] << ( 32 - 16 ) ) & 0x0fffffff ;
	out[21] = ( in[18] >> 12 ) ;

	out[21] |= ( in[19] << ( 32 - 12 ) ) & 0x0fffffff ;
	out[22] = ( in[19] >> 8 ) ;

	out[22] |= ( in[20] << ( 32 - 8 ) ) & 0x0fffffff ;
	out[23] = ( in[20] >> 4 ) ;

	out[24] = ( in[21] >> 0 ) & 0x0fffffff ;
	out[25] = ( in[21] >> 28 ) ;

	out[25] |= ( in[22] << ( 32 - 28 ) ) & 0x0fffffff ;
	out[26] = ( in[22] >> 24 ) ;

	out[26] |= ( in[23] << ( 32 - 24 ) ) & 0x0fffffff ;
	out[27] = ( in[23] >> 20 ) ;

	out[27] |= ( in[24] << ( 32 - 20 ) ) & 0x0fffffff ;
	out[28] = ( in[24] >> 16 ) ;

	out[28] |= ( in[25] << ( 32 - 16 ) ) & 0x0fffffff ;
	out[29] = ( in[25] >> 12 ) ;

	out[29] |= ( in[26] << ( 32 - 12 ) ) & 0x0fffffff ;
	out[30] = ( in[26] >> 8 ) ;

	out[30] |= ( in[27] << ( 32 - 8 ) ) & 0x0fffffff ;
	out[31] = ( in[27] >> 4 ) ;

	out[32] = ( in[28] >> 0 ) & 0x0fffffff ;
	out[33] = ( in[28] >> 28 ) ;

	out[33] |= ( in[29] << ( 32 - 28 ) ) & 0x0fffffff ;
	out[34] = ( in[29] >> 24 ) ;

	out[34] |= ( in[30] << ( 32 - 24 ) ) & 0x0fffffff ;
	out[35] = ( in[30] >> 20 ) ;

	out[35] |= ( in[31] << ( 32 - 20 ) ) & 0x0fffffff ;
	out[36] = ( in[31] >> 16 ) ;

	out[36] |= ( in[32] << ( 32 - 16 ) ) & 0x0fffffff ;
	out[37] = ( in[32] >> 12 ) ;

	out[37] |= ( in[33] << ( 32 - 12 ) ) & 0x0fffffff ;
	out[38] = ( in[33] >> 8 ) ;

	out[38] |= ( in[34] << ( 32 - 8 ) ) & 0x0fffffff ;
	out[39] = ( in[34] >> 4 ) ;

	out[40] = ( in[35] >> 0 ) & 0x0fffffff ;
	out[41] = ( in[35] >> 28 ) ;

	out[41] |= ( in[36] << ( 32 - 28 ) ) & 0x0fffffff ;
	out[42] = ( in[36] >> 24 ) ;

	out[42] |= ( in[37] << ( 32 - 24 ) ) & 0x0fffffff ;
	out[43] = ( in[37] >> 20 ) ;

	out[43] |= ( in[38] << ( 32 - 20 ) ) & 0x0fffffff ;
	out[44] = ( in[38] >> 16 ) ;

	out[44] |= ( in[39] << ( 32 - 16 ) ) & 0x0fffffff ;
	out[45] = ( in[39] >> 12 ) ;

	out[45] |= ( in[40] << ( 32 - 12 ) ) & 0x0fffffff ;
	out[46] = ( in[40] >> 8 ) ;

	out[46] |= ( in[41] << ( 32 - 8 ) ) & 0x0fffffff ;
	out[47] = ( in[41] >> 4 ) ;

	out[48] = ( in[42] >> 0 ) & 0x0fffffff ;
	out[49] = ( in[42] >> 28 ) ;

	out[49] |= ( in[43] << ( 32 - 28 ) ) & 0x0fffffff ;
	out[50] = ( in[43] >> 24 ) ;

	out[50] |= ( in[44] << ( 32 - 24 ) ) & 0x0fffffff ;
	out[51] = ( in[44] >> 20 ) ;

	out[51] |= ( in[45] << ( 32 - 20 ) ) & 0x0fffffff ;
	out[52] = ( in[45] >> 16 ) ;

	out[52] |= ( in[46] << ( 32 - 16 ) ) & 0x0fffffff ;
	out[53] = ( in[46] >> 12 ) ;

	out[53] |= ( in[47] << ( 32 - 12 ) ) & 0x0fffffff ;
	out[54] = ( in[47] >> 8 ) ;

	out[54] |= ( in[48] << ( 32 - 8 ) ) & 0x0fffffff ;
	out[55] = ( in[48] >> 4 ) ;

	out[56] = ( in[49] >> 0 ) & 0x0fffffff ;
	out[57] = ( in[49] >> 28 ) ;

	out[57] |= ( in[50] << ( 32 - 28 ) ) & 0x0fffffff ;
	out[58] = ( in[50] >> 24 ) ;

	out[58] |= ( in[51] << ( 32 - 24 ) ) & 0x0fffffff ;
	out[59] = ( in[51] >> 20 ) ;

	out[59] |= ( in[52] << ( 32 - 20 ) ) & 0x0fffffff ;
	out[60] = ( in[52] >> 16 ) ;

	out[60] |= ( in[53] << ( 32 - 16 ) ) & 0x0fffffff ;
	out[61] = ( in[53] >> 12 ) ;

	out[61] |= ( in[54] << ( 32 - 12 ) ) & 0x0fffffff ;
	out[62] = ( in[54] >> 8 ) ;

	out[62] |= ( in[55] << ( 32 - 8 ) ) & 0x0fffffff ;
	out[63] = ( in[55] >> 4 ) ;
}


// 29-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c29(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x1fffffff ;
	out[1] = ( in[0] >> 29 ) ;

	out[1] |= ( in[1] << ( 32 - 29 ) ) & 0x1fffffff ;
	out[2] = ( in[1] >> 26 ) ;

	out[2] |= ( in[2] << ( 32 - 26 ) ) & 0x1fffffff ;
	out[3] = ( in[2] >> 23 ) ;

	out[3] |= ( in[3] << ( 32 - 23 ) ) & 0x1fffffff ;
	out[4] = ( in[3] >> 20 ) ;

	out[4] |= ( in[4] << ( 32 - 20 ) ) & 0x1fffffff ;
	out[5] = ( in[4] >> 17 ) ;

	out[5] |= ( in[5] << ( 32 - 17 ) ) & 0x1fffffff ;
	out[6] = ( in[5] >> 14 ) ;

	out[6] |= ( in[6] << ( 32 - 14 ) ) & 0x1fffffff ;
	out[7] = ( in[6] >> 11 ) ;

	out[7] |= ( in[7] << ( 32 - 11 ) ) & 0x1fffffff ;
	out[8] = ( in[7] >> 8 ) ;

	out[8] |= ( in[8] << ( 32 - 8 ) ) & 0x1fffffff ;
	out[9] = ( in[8] >> 5 ) ;

	out[9] |= ( in[9] << ( 32 - 5 ) ) & 0x1fffffff ;
	out[10] = ( in[9] >> 2 ) & 0x1fffffff ;
	out[11] = ( in[9] >> 31 ) ;

	out[11] |= ( in[10] << ( 32 - 31 ) ) & 0x1fffffff ;
	out[12] = ( in[10] >> 28 ) ;

	out[12] |= ( in[11] << ( 32 - 28 ) ) & 0x1fffffff ;
	out[13] = ( in[11] >> 25 ) ;

	out[13] |= ( in[12] << ( 32 - 25 ) ) & 0x1fffffff ;
	out[14] = ( in[12] >> 22 ) ;

	out[14] |= ( in[13] << ( 32 - 22 ) ) & 0x1fffffff ;
	out[15] = ( in[13] >> 19 ) ;

	out[15] |= ( in[14] << ( 32 - 19 ) ) & 0x1fffffff ;
	out[16] = ( in[14] >> 16 ) ;

	out[16] |= ( in[15] << ( 32 - 16 ) ) & 0x1fffffff ;
	out[17] = ( in[15] >> 13 ) ;

	out[17] |= ( in[16] << ( 32 - 13 ) ) & 0x1fffffff ;
	out[18] = ( in[16] >> 10 ) ;

	out[18] |= ( in[17] << ( 32 - 10 ) ) & 0x1fffffff ;
	out[19] = ( in[17] >> 7 ) ;

	out[19] |= ( in[18] << ( 32 - 7 ) ) & 0x1fffffff ;
	out[20] = ( in[18] >> 4 ) ;

	out[20] |= ( in[19] << ( 32 - 4 ) ) & 0x1fffffff ;
	out[21] = ( in[19] >> 1 ) & 0x1fffffff ;
	out[22] = ( in[19] >> 30 ) ;

	out[22] |= ( in[20] << ( 32 - 30 ) ) & 0x1fffffff ;
	out[23] = ( in[20] >> 27 ) ;

	out[23] |= ( in[21] << ( 32 - 27 ) ) & 0x1fffffff ;
	out[24] = ( in[21] >> 24 ) ;

	out[24] |= ( in[22] << ( 32 - 24 ) ) & 0x1fffffff ;
	out[25] = ( in[22] >> 21 ) ;

	out[25] |= ( in[23] << ( 32 - 21 ) ) & 0x1fffffff ;
	out[26] = ( in[23] >> 18 ) ;

	out[26] |= ( in[24] << ( 32 - 18 ) ) & 0x1fffffff ;
	out[27] = ( in[24] >> 15 ) ;

	out[27] |= ( in[25] << ( 32 - 15 ) ) & 0x1fffffff ;
	out[28] = ( in[25] >> 12 ) ;

	out[28] |= ( in[26] << ( 32 - 12 ) ) & 0x1fffffff ;
	out[29] = ( in[26] >> 9 ) ;

	out[29] |= ( in[27] << ( 32 - 9 ) ) & 0x1fffffff ;
	out[30] = ( in[27] >> 6 ) ;

	out[30] |= ( in[28] << ( 32 - 6 ) ) & 0x1fffffff ;
	out[31] = ( in[28] >> 3 ) ;

	out[32] = ( in[29] >> 0 ) & 0x1fffffff ;
	out[33] = ( in[29] >> 29 ) ;

	out[33] |= ( in[30] << ( 32 - 29 ) ) & 0x1fffffff ;
	out[34] = ( in[30] >> 26 ) ;

	out[34] |= ( in[31] << ( 32 - 26 ) ) & 0x1fffffff ;
	out[35] = ( in[31] >> 23 ) ;

	out[35] |= ( in[32] << ( 32 - 23 ) ) & 0x1fffffff ;
	out[36] = ( in[32] >> 20 ) ;

	out[36] |= ( in[33] << ( 32 - 20 ) ) & 0x1fffffff ;
	out[37] = ( in[33] >> 17 ) ;

	out[37] |= ( in[34] << ( 32 - 17 ) ) & 0x1fffffff ;
	out[38] = ( in[34] >> 14 ) ;

	out[38] |= ( in[35] << ( 32 - 14 ) ) & 0x1fffffff ;
	out[39] = ( in[35] >> 11 ) ;

	out[39] |= ( in[36] << ( 32 - 11 ) ) & 0x1fffffff ;
	out[40] = ( in[36] >> 8 ) ;

	out[40] |= ( in[37] << ( 32 - 8 ) ) & 0x1fffffff ;
	out[41] = ( in[37] >> 5 ) ;

	out[41] |= ( in[38] << ( 32 - 5 ) ) & 0x1fffffff ;
	out[42] = ( in[38] >> 2 ) & 0x1fffffff ;
	out[43] = ( in[38] >> 31 ) ;

	out[43] |= ( in[39] << ( 32 - 31 ) ) & 0x1fffffff ;
	out[44] = ( in[39] >> 28 ) ;

	out[44] |= ( in[40] << ( 32 - 28 ) ) & 0x1fffffff ;
	out[45] = ( in[40] >> 25 ) ;

	out[45] |= ( in[41] << ( 32 - 25 ) ) & 0x1fffffff ;
	out[46] = ( in[41] >> 22 ) ;

	out[46] |= ( in[42] << ( 32 - 22 ) ) & 0x1fffffff ;
	out[47] = ( in[42] >> 19 ) ;

	out[47] |= ( in[43] << ( 32 - 19 ) ) & 0x1fffffff ;
	out[48] = ( in[43] >> 16 ) ;

	out[48] |= ( in[44] << ( 32 - 16 ) ) & 0x1fffffff ;
	out[49] = ( in[44] >> 13 ) ;

	out[49] |= ( in[45] << ( 32 - 13 ) ) & 0x1fffffff ;
	out[50] = ( in[45] >> 10 ) ;

	out[50] |= ( in[46] << ( 32 - 10 ) ) & 0x1fffffff ;
	out[51] = ( in[46] >> 7 ) ;

	out[51] |= ( in[47] << ( 32 - 7 ) ) & 0x1fffffff ;
	out[52] = ( in[47] >> 4 ) ;

	out[52] |= ( in[48] << ( 32 - 4 ) ) & 0x1fffffff ;
	out[53] = ( in[48] >> 1 ) & 0x1fffffff ;
	out[54] = ( in[48] >> 30 ) ;

	out[54] |= ( in[49] << ( 32 - 30 ) ) & 0x1fffffff ;
	out[55] = ( in[49] >> 27 ) ;

	out[55] |= ( in[50] << ( 32 - 27 ) ) & 0x1fffffff ;
	out[56] = ( in[50] >> 24 ) ;

	out[56] |= ( in[51] << ( 32 - 24 ) ) & 0x1fffffff ;
	out[57] = ( in[51] >> 21 ) ;

	out[57] |= ( in[52] << ( 32 - 21 ) ) & 0x1fffffff ;
	out[58] = ( in[52] >> 18 ) ;

	out[58] |= ( in[53] << ( 32 - 18 ) ) & 0x1fffffff ;
	out[59] = ( in[53] >> 15 ) ;

	out[59] |= ( in[54] << ( 32 - 15 ) ) & 0x1fffffff ;
	out[60] = ( in[54] >> 12 ) ;

	out[60] |= ( in[55] << ( 32 - 12 ) ) & 0x1fffffff ;
	out[61] = ( in[55] >> 9 ) ;

	out[61] |= ( in[56] << ( 32 - 9 ) ) & 0x1fffffff ;
	out[62] = ( in[56] >> 6 ) ;

	out[62] |= ( in[57] << ( 32 - 6 ) ) & 0x1fffffff ;
	out[63] = ( in[57] >> 3 ) ;
}


// 30-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c30(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x3fffffff ;
	out[1] = ( in[0] >> 30 ) ;

	out[1] |= ( in[1] << ( 32 - 30 ) ) & 0x3fffffff ;
	out[2] = ( in[1] >> 28 ) ;

	out[2] |= ( in[2] << ( 32 - 28 ) ) & 0x3fffffff ;
	out[3] = ( in[2] >> 26 ) ;

	out[3] |= ( in[3] << ( 32 - 26 ) ) & 0x3fffffff ;
	out[4] = ( in[3] >> 24 ) ;

	out[4] |= ( in[4] << ( 32 - 24 ) ) & 0x3fffffff ;
	out[5] = ( in[4] >> 22 ) ;

	out[5] |= ( in[5] << ( 32 - 22 ) ) & 0x3fffffff ;
	out[6] = ( in[5] >> 20 ) ;

	out[6] |= ( in[6] << ( 32 - 20 ) ) & 0x3fffffff ;
	out[7] = ( in[6] >> 18 ) ;

	out[7] |= ( in[7] << ( 32 - 18 ) ) & 0x3fffffff ;
	out[8] = ( in[7] >> 16 ) ;

	out[8] |= ( in[8] << ( 32 - 16 ) ) & 0x3fffffff ;
	out[9] = ( in[8] >> 14 ) ;

	out[9] |= ( in[9] << ( 32 - 14 ) ) & 0x3fffffff ;
	out[10] = ( in[9] >> 12 ) ;

	out[10] |= ( in[10] << ( 32 - 12 ) ) & 0x3fffffff ;
	out[11] = ( in[10] >> 10 ) ;

	out[11] |= ( in[11] << ( 32 - 10 ) ) & 0x3fffffff ;
	out[12] = ( in[11] >> 8 ) ;

	out[12] |= ( in[12] << ( 32 - 8 ) ) & 0x3fffffff ;
	out[13] = ( in[12] >> 6 ) ;

	out[13] |= ( in[13] << ( 32 - 6 ) ) & 0x3fffffff ;
	out[14] = ( in[13] >> 4 ) ;

	out[14] |= ( in[14] << ( 32 - 4 ) ) & 0x3fffffff ;
	out[15] = ( in[14] >> 2 ) ;

	out[16] = ( in[15] >> 0 ) & 0x3fffffff ;
	out[17] = ( in[15] >> 30 ) ;

	out[17] |= ( in[16] << ( 32 - 30 ) ) & 0x3fffffff ;
	out[18] = ( in[16] >> 28 ) ;

	out[18] |= ( in[17] << ( 32 - 28 ) ) & 0x3fffffff ;
	out[19] = ( in[17] >> 26 ) ;

	out[19] |= ( in[18] << ( 32 - 26 ) ) & 0x3fffffff ;
	out[20] = ( in[18] >> 24 ) ;

	out[20] |= ( in[19] << ( 32 - 24 ) ) & 0x3fffffff ;
	out[21] = ( in[19] >> 22 ) ;

	out[21] |= ( in[20] << ( 32 - 22 ) ) & 0x3fffffff ;
	out[22] = ( in[20] >> 20 ) ;

	out[22] |= ( in[21] << ( 32 - 20 ) ) & 0x3fffffff ;
	out[23] = ( in[21] >> 18 ) ;

	out[23] |= ( in[22] << ( 32 - 18 ) ) & 0x3fffffff ;
	out[24] = ( in[22] >> 16 ) ;

	out[24] |= ( in[23] << ( 32 - 16 ) ) & 0x3fffffff ;
	out[25] = ( in[23] >> 14 ) ;

	out[25] |= ( in[24] << ( 32 - 14 ) ) & 0x3fffffff ;
	out[26] = ( in[24] >> 12 ) ;

	out[26] |= ( in[25] << ( 32 - 12 ) ) & 0x3fffffff ;
	out[27] = ( in[25] >> 10 ) ;

	out[27] |= ( in[26] << ( 32 - 10 ) ) & 0x3fffffff ;
	out[28] = ( in[26] >> 8 ) ;

	out[28] |= ( in[27] << ( 32 - 8 ) ) & 0x3fffffff ;
	out[29] = ( in[27] >> 6 ) ;

	out[29] |= ( in[28] << ( 32 - 6 ) ) & 0x3fffffff ;
	out[30] = ( in[28] >> 4 ) ;

	out[30] |= ( in[29] << ( 32 - 4 ) ) & 0x3fffffff ;
	out[31] = ( in[29] >> 2 ) ;

	out[32] = ( in[30] >> 0 ) & 0x3fffffff ;
	out[33] = ( in[30] >> 30 ) ;

	out[33] |= ( in[31] << ( 32 - 30 ) ) & 0x3fffffff ;
	out[34] = ( in[31] >> 28 ) ;

	out[34] |= ( in[32] << ( 32 - 28 ) ) & 0x3fffffff ;
	out[35] = ( in[32] >> 26 ) ;

	out[35] |= ( in[33] << ( 32 - 26 ) ) & 0x3fffffff ;
	out[36] = ( in[33] >> 24 ) ;

	out[36] |= ( in[34] << ( 32 - 24 ) ) & 0x3fffffff ;
	out[37] = ( in[34] >> 22 ) ;

	out[37] |= ( in[35] << ( 32 - 22 ) ) & 0x3fffffff ;
	out[38] = ( in[35] >> 20 ) ;

	out[38] |= ( in[36] << ( 32 - 20 ) ) & 0x3fffffff ;
	out[39] = ( in[36] >> 18 ) ;

	out[39] |= ( in[37] << ( 32 - 18 ) ) & 0x3fffffff ;
	out[40] = ( in[37] >> 16 ) ;

	out[40] |= ( in[38] << ( 32 - 16 ) ) & 0x3fffffff ;
	out[41] = ( in[38] >> 14 ) ;

	out[41] |= ( in[39] << ( 32 - 14 ) ) & 0x3fffffff ;
	out[42] = ( in[39] >> 12 ) ;

	out[42] |= ( in[40] << ( 32 - 12 ) ) & 0x3fffffff ;
	out[43] = ( in[40] >> 10 ) ;

	out[43] |= ( in[41] << ( 32 - 10 ) ) & 0x3fffffff ;
	out[44] = ( in[41] >> 8 ) ;

	out[44] |= ( in[42] << ( 32 - 8 ) ) & 0x3fffffff ;
	out[45] = ( in[42] >> 6 ) ;

	out[45] |= ( in[43] << ( 32 - 6 ) ) & 0x3fffffff ;
	out[46] = ( in[43] >> 4 ) ;

	out[46] |= ( in[44] << ( 32 - 4 ) ) & 0x3fffffff ;
	out[47] = ( in[44] >> 2 ) ;

	out[48] = ( in[45] >> 0 ) & 0x3fffffff ;
	out[49] = ( in[45] >> 30 ) ;

	out[49] |= ( in[46] << ( 32 - 30 ) ) & 0x3fffffff ;
	out[50] = ( in[46] >> 28 ) ;

	out[50] |= ( in[47] << ( 32 - 28 ) ) & 0x3fffffff ;
	out[51] = ( in[47] >> 26 ) ;

	out[51] |= ( in[48] << ( 32 - 26 ) ) & 0x3fffffff ;
	out[52] = ( in[48] >> 24 ) ;

	out[52] |= ( in[49] << ( 32 - 24 ) ) & 0x3fffffff ;
	out[53] = ( in[49] >> 22 ) ;

	out[53] |= ( in[50] << ( 32 - 22 ) ) & 0x3fffffff ;
	out[54] = ( in[50] >> 20 ) ;

	out[54] |= ( in[51] << ( 32 - 20 ) ) & 0x3fffffff ;
	out[55] = ( in[51] >> 18 ) ;

	out[55] |= ( in[52] << ( 32 - 18 ) ) & 0x3fffffff ;
	out[56] = ( in[52] >> 16 ) ;

	out[56] |= ( in[53] << ( 32 - 16 ) ) & 0x3fffffff ;
	out[57] = ( in[53] >> 14 ) ;

	out[57] |= ( in[54] << ( 32 - 14 ) ) & 0x3fffffff ;
	out[58] = ( in[54] >> 12 ) ;

	out[58] |= ( in[55] << ( 32 - 12 ) ) & 0x3fffffff ;
	out[59] = ( in[55] >> 10 ) ;

	out[59] |= ( in[56] << ( 32 - 10 ) ) & 0x3fffffff ;
	out[60] = ( in[56] >> 8 ) ;

	out[60] |= ( in[57] << ( 32 - 8 ) ) & 0x3fffffff ;
	out[61] = ( in[57] >> 6 ) ;

	out[61] |= ( in[58] << ( 32 - 6 ) ) & 0x3fffffff ;
	out[62] = ( in[58] >> 4 ) ;

	out[62] |= ( in[59] << ( 32 - 4 ) ) & 0x3fffffff ;
	out[63] = ( in[59] >> 2 ) ;
}


// 31-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c31(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] >> 0 ) & 0x7fffffff ;
	out[1] = ( in[0] >> 31 ) ;

	out[1] |= ( in[1] << ( 32 - 31 ) ) & 0x7fffffff ;
	out[2] = ( in[1] >> 30 ) ;

	out[2] |= ( in[2] << ( 32 - 30 ) ) & 0x7fffffff ;
	out[3] = ( in[2] >> 29 ) ;

	out[3] |= ( in[3] << ( 32 - 29 ) ) & 0x7fffffff ;
	out[4] = ( in[3] >> 28 ) ;

	out[4] |= ( in[4] << ( 32 - 28 ) ) & 0x7fffffff ;
	out[5] = ( in[4] >> 27 ) ;

	out[5] |= ( in[5] << ( 32 - 27 ) ) & 0x7fffffff ;
	out[6] = ( in[5] >> 26 ) ;

	out[6] |= ( in[6] << ( 32 - 26 ) ) & 0x7fffffff ;
	out[7] = ( in[6] >> 25 ) ;

	out[7] |= ( in[7] << ( 32 - 25 ) ) & 0x7fffffff ;
	out[8] = ( in[7] >> 24 ) ;

	out[8] |= ( in[8] << ( 32 - 24 ) ) & 0x7fffffff ;
	out[9] = ( in[8] >> 23 ) ;

	out[9] |= ( in[9] << ( 32 - 23 ) ) & 0x7fffffff ;
	out[10] = ( in[9] >> 22 ) ;

	out[10] |= ( in[10] << ( 32 - 22 ) ) & 0x7fffffff ;
	out[11] = ( in[10] >> 21 ) ;

	out[11] |= ( in[11] << ( 32 - 21 ) ) & 0x7fffffff ;
	out[12] = ( in[11] >> 20 ) ;

	out[12] |= ( in[12] << ( 32 - 20 ) ) & 0x7fffffff ;
	out[13] = ( in[12] >> 19 ) ;

	out[13] |= ( in[13] << ( 32 - 19 ) ) & 0x7fffffff ;
	out[14] = ( in[13] >> 18 ) ;

	out[14] |= ( in[14] << ( 32 - 18 ) ) & 0x7fffffff ;
	out[15] = ( in[14] >> 17 ) ;

	out[15] |= ( in[15] << ( 32 - 17 ) ) & 0x7fffffff ;
	out[16] = ( in[15] >> 16 ) ;

	out[16] |= ( in[16] << ( 32 - 16 ) ) & 0x7fffffff ;
	out[17] = ( in[16] >> 15 ) ;

	out[17] |= ( in[17] << ( 32 - 15 ) ) & 0x7fffffff ;
	out[18] = ( in[17] >> 14 ) ;

	out[18] |= ( in[18] << ( 32 - 14 ) ) & 0x7fffffff ;
	out[19] = ( in[18] >> 13 ) ;

	out[19] |= ( in[19] << ( 32 - 13 ) ) & 0x7fffffff ;
	out[20] = ( in[19] >> 12 ) ;

	out[20] |= ( in[20] << ( 32 - 12 ) ) & 0x7fffffff ;
	out[21] = ( in[20] >> 11 ) ;

	out[21] |= ( in[21] << ( 32 - 11 ) ) & 0x7fffffff ;
	out[22] = ( in[21] >> 10 ) ;

	out[22] |= ( in[22] << ( 32 - 10 ) ) & 0x7fffffff ;
	out[23] = ( in[22] >> 9 ) ;

	out[23] |= ( in[23] << ( 32 - 9 ) ) & 0x7fffffff ;
	out[24] = ( in[23] >> 8 ) ;

	out[24] |= ( in[24] << ( 32 - 8 ) ) & 0x7fffffff ;
	out[25] = ( in[24] >> 7 ) ;

	out[25] |= ( in[25] << ( 32 - 7 ) ) & 0x7fffffff ;
	out[26] = ( in[25] >> 6 ) ;

	out[26] |= ( in[26] << ( 32 - 6 ) ) & 0x7fffffff ;
	out[27] = ( in[26] >> 5 ) ;

	out[27] |= ( in[27] << ( 32 - 5 ) ) & 0x7fffffff ;
	out[28] = ( in[27] >> 4 ) ;

	out[28] |= ( in[28] << ( 32 - 4 ) ) & 0x7fffffff ;
	out[29] = ( in[28] >> 3 ) ;

	out[29] |= ( in[29] << ( 32 - 3 ) ) & 0x7fffffff ;
	out[30] = ( in[29] >> 2 ) ;

	out[30] |= ( in[30] << ( 32 - 2 ) ) & 0x7fffffff ;
	out[31] = ( in[30] >> 1 ) ;

	out[32] = ( in[31] >> 0 ) & 0x7fffffff ;
	out[33] = ( in[31] >> 31 ) ;

	out[33] |= ( in[32] << ( 32 - 31 ) ) & 0x7fffffff ;
	out[34] = ( in[32] >> 30 ) ;

	out[34] |= ( in[33] << ( 32 - 30 ) ) & 0x7fffffff ;
	out[35] = ( in[33] >> 29 ) ;

	out[35] |= ( in[34] << ( 32 - 29 ) ) & 0x7fffffff ;
	out[36] = ( in[34] >> 28 ) ;

	out[36] |= ( in[35] << ( 32 - 28 ) ) & 0x7fffffff ;
	out[37] = ( in[35] >> 27 ) ;

	out[37] |= ( in[36] << ( 32 - 27 ) ) & 0x7fffffff ;
	out[38] = ( in[36] >> 26 ) ;

	out[38] |= ( in[37] << ( 32 - 26 ) ) & 0x7fffffff ;
	out[39] = ( in[37] >> 25 ) ;

	out[39] |= ( in[38] << ( 32 - 25 ) ) & 0x7fffffff ;
	out[40] = ( in[38] >> 24 ) ;

	out[40] |= ( in[39] << ( 32 - 24 ) ) & 0x7fffffff ;
	out[41] = ( in[39] >> 23 ) ;

	out[41] |= ( in[40] << ( 32 - 23 ) ) & 0x7fffffff ;
	out[42] = ( in[40] >> 22 ) ;

	out[42] |= ( in[41] << ( 32 - 22 ) ) & 0x7fffffff ;
	out[43] = ( in[41] >> 21 ) ;

	out[43] |= ( in[42] << ( 32 - 21 ) ) & 0x7fffffff ;
	out[44] = ( in[42] >> 20 ) ;

	out[44] |= ( in[43] << ( 32 - 20 ) ) & 0x7fffffff ;
	out[45] = ( in[43] >> 19 ) ;

	out[45] |= ( in[44] << ( 32 - 19 ) ) & 0x7fffffff ;
	out[46] = ( in[44] >> 18 ) ;

	out[46] |= ( in[45] << ( 32 - 18 ) ) & 0x7fffffff ;
	out[47] = ( in[45] >> 17 ) ;

	out[47] |= ( in[46] << ( 32 - 17 ) ) & 0x7fffffff ;
	out[48] = ( in[46] >> 16 ) ;

	out[48] |= ( in[47] << ( 32 - 16 ) ) & 0x7fffffff ;
	out[49] = ( in[47] >> 15 ) ;

	out[49] |= ( in[48] << ( 32 - 15 ) ) & 0x7fffffff ;
	out[50] = ( in[48] >> 14 ) ;

	out[50] |= ( in[49] << ( 32 - 14 ) ) & 0x7fffffff ;
	out[51] = ( in[49] >> 13 ) ;

	out[51] |= ( in[50] << ( 32 - 13 ) ) & 0x7fffffff ;
	out[52] = ( in[50] >> 12 ) ;

	out[52] |= ( in[51] << ( 32 - 12 ) ) & 0x7fffffff ;
	out[53] = ( in[51] >> 11 ) ;

	out[53] |= ( in[52] << ( 32 - 11 ) ) & 0x7fffffff ;
	out[54] = ( in[52] >> 10 ) ;

	out[54] |= ( in[53] << ( 32 - 10 ) ) & 0x7fffffff ;
	out[55] = ( in[53] >> 9 ) ;

	out[55] |= ( in[54] << ( 32 - 9 ) ) & 0x7fffffff ;
	out[56] = ( in[54] >> 8 ) ;

	out[56] |= ( in[55] << ( 32 - 8 ) ) & 0x7fffffff ;
	out[57] = ( in[55] >> 7 ) ;

	out[57] |= ( in[56] << ( 32 - 7 ) ) & 0x7fffffff ;
	out[58] = ( in[56] >> 6 ) ;

	out[58] |= ( in[57] << ( 32 - 6 ) ) & 0x7fffffff ;
	out[59] = ( in[57] >> 5 ) ;

	out[59] |= ( in[58] << ( 32 - 5 ) ) & 0x7fffffff ;
	out[60] = ( in[58] >> 4 ) ;

	out[60] |= ( in[59] << ( 32 - 4 ) ) & 0x7fffffff ;
	out[61] = ( in[59] >> 3 ) ;

	out[61] |= ( in[60] << ( 32 - 3 ) ) & 0x7fffffff ;
	out[62] = ( in[60] >> 2 ) ;

	out[62] |= ( in[61] << ( 32 - 2 ) ) & 0x7fffffff ;
	out[63] = ( in[61] >> 1 ) ;
}


// 32-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_unpack64_c32(const uint32_t *  __restrict__  in,
		uint32_t *  __restrict__  out) {
	memcpy(out, in, 64 * sizeof(uint32_t));
}


// 1-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c1(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 1 ;
	out[0] |= in[2] << 2 ;
	out[0] |= in[3] << 3 ;
	out[0] |= in[4] << 4 ;
	out[0] |= in[5] << 5 ;
	out[0] |= in[6] << 6 ;
	out[0] |= in[7] << 7 ;
	out[0] |= in[8] << 8 ;
	out[0] |= in[9] << 9 ;
	out[0] |= in[10] << 10 ;
	out[0] |= in[11] << 11 ;
	out[0] |= in[12] << 12 ;
	out[0] |= in[13] << 13 ;
	out[0] |= in[14] << 14 ;
	out[0] |= in[15] << 15 ;
	out[0] |= in[16] << 16 ;
	out[0] |= in[17] << 17 ;
	out[0] |= in[18] << 18 ;
	out[0] |= in[19] << 19 ;
	out[0] |= in[20] << 20 ;
	out[0] |= in[21] << 21 ;
	out[0] |= in[22] << 22 ;
	out[0] |= in[23] << 23 ;
	out[0] |= in[24] << 24 ;
	out[0] |= in[25] << 25 ;
	out[0] |= in[26] << 26 ;
	out[0] |= in[27] << 27 ;
	out[0] |= in[28] << 28 ;
	out[0] |= in[29] << 29 ;
	out[0] |= in[30] << 30 ;
	out[0] |= in[31] << 31 ;

	out[1] = in[32] << 0 ;
	out[1] |= in[33] << 1 ;
	out[1] |= in[34] << 2 ;
	out[1] |= in[35] << 3 ;
	out[1] |= in[36] << 4 ;
	out[1] |= in[37] << 5 ;
	out[1] |= in[38] << 6 ;
	out[1] |= in[39] << 7 ;
	out[1] |= in[40] << 8 ;
	out[1] |= in[41] << 9 ;
	out[1] |= in[42] << 10 ;
	out[1] |= in[43] << 11 ;
	out[1] |= in[44] << 12 ;
	out[1] |= in[45] << 13 ;
	out[1] |= in[46] << 14 ;
	out[1] |= in[47] << 15 ;
	out[1] |= in[48] << 16 ;
	out[1] |= in[49] << 17 ;
	out[1] |= in[50] << 18 ;
	out[1] |= in[51] << 19 ;
	out[1] |= in[52] << 20 ;
	out[1] |= in[53] << 21 ;
	out[1] |= in[54] << 22 ;
	out[1] |= in[55] << 23 ;
	out[1] |= in[56] << 24 ;
	out[1] |= in[57] << 25 ;
	out[1] |= in[58] << 26 ;
	out[1] |= in[59] << 27 ;
	out[1] |= in[60] << 28 ;
	out[1] |= in[61] << 29 ;
	out[1] |= in[62] << 30 ;
	out[1] |= in[63] << 31 ;
}


// 2-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c2(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 2 ;
	out[0] |= in[2] << 4 ;
	out[0] |= in[3] << 6 ;
	out[0] |= in[4] << 8 ;
	out[0] |= in[5] << 10 ;
	out[0] |= in[6] << 12 ;
	out[0] |= in[7] << 14 ;
	out[0] |= in[8] << 16 ;
	out[0] |= in[9] << 18 ;
	out[0] |= in[10] << 20 ;
	out[0] |= in[11] << 22 ;
	out[0] |= in[12] << 24 ;
	out[0] |= in[13] << 26 ;
	out[0] |= in[14] << 28 ;
	out[0] |= in[15] << 30 ;

	out[1] = in[16] << 0 ;
	out[1] |= in[17] << 2 ;
	out[1] |= in[18] << 4 ;
	out[1] |= in[19] << 6 ;
	out[1] |= in[20] << 8 ;
	out[1] |= in[21] << 10 ;
	out[1] |= in[22] << 12 ;
	out[1] |= in[23] << 14 ;
	out[1] |= in[24] << 16 ;
	out[1] |= in[25] << 18 ;
	out[1] |= in[26] << 20 ;
	out[1] |= in[27] << 22 ;
	out[1] |= in[28] << 24 ;
	out[1] |= in[29] << 26 ;
	out[1] |= in[30] << 28 ;
	out[1] |= in[31] << 30 ;

	out[2] = in[32] << 0 ;
	out[2] |= in[33] << 2 ;
	out[2] |= in[34] << 4 ;
	out[2] |= in[35] << 6 ;
	out[2] |= in[36] << 8 ;
	out[2] |= in[37] << 10 ;
	out[2] |= in[38] << 12 ;
	out[2] |= in[39] << 14 ;
	out[2] |= in[40] << 16 ;
	out[2] |= in[41] << 18 ;
	out[2] |= in[42] << 20 ;
	out[2] |= in[43] << 22 ;
	out[2] |= in[44] << 24 ;
	out[2] |= in[45] << 26 ;
	out[2] |= in[46] << 28 ;
	out[2] |= in[47] << 30 ;

	out[3] = in[48] << 0 ;
	out[3] |= in[49] << 2 ;
	out[3] |= in[50] << 4 ;
	out[3] |= in[51] << 6 ;
	out[3] |= in[52] << 8 ;
	out[3] |= in[53] << 10 ;
	out[3] |= in[54] << 12 ;
	out[3] |= in[55] << 14 ;
	out[3] |= in[56] << 16 ;
	out[3] |= in[57] << 18 ;
	out[3] |= in[58] << 20 ;
	out[3] |= in[59] << 22 ;
	out[3] |= in[60] << 24 ;
	out[3] |= in[61] << 26 ;
	out[3] |= in[62] << 28 ;
	out[3] |= in[63] << 30 ;
}


// 3-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c3(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 3 ;
	out[0] |= in[2] << 6 ;
	out[0] |= in[3] << 9 ;
	out[0] |= in[4] << 12 ;
	out[0] |= in[5] << 15 ;
	out[0] |= in[6] << 18 ;
	out[0] |= in[7] << 21 ;
	out[0] |= in[8] << 24 ;
	out[0] |= in[9] << 27 ;
	out[0] |= in[10] << 30 ;

	out[1] = in[10] >> ( 32 - 30 ) ;
	out[1] |= in[11] << 1 ;
	out[1] |= in[12] << 4 ;
	out[1] |= in[13] << 7 ;
	out[1] |= in[14] << 10 ;
	out[1] |= in[15] << 13 ;
	out[1] |= in[16] << 16 ;
	out[1] |= in[17] << 19 ;
	out[1] |= in[18] << 22 ;
	out[1] |= in[19] << 25 ;
	out[1] |= in[20] << 28 ;
	out[1] |= in[21] << 31 ;

	out[2] = in[21] >> ( 32 - 31 ) ;
	out[2] |= in[22] << 2 ;
	out[2] |= in[23] << 5 ;
	out[2] |= in[24] << 8 ;
	out[2] |= in[25] << 11 ;
	out[2] |= in[26] << 14 ;
	out[2] |= in[27] << 17 ;
	out[2] |= in[28] << 20 ;
	out[2] |= in[29] << 23 ;
	out[2] |= in[30] << 26 ;
	out[2] |= in[31] << 29 ;

	out[3] = in[32] << 0 ;
	out[3] |= in[33] << 3 ;
	out[3] |= in[34] << 6 ;
	out[3] |= in[35] << 9 ;
	out[3] |= in[36] << 12 ;
	out[3] |= in[37] << 15 ;
	out[3] |= in[38] << 18 ;
	out[3] |= in[39] << 21 ;
	out[3] |= in[40] << 24 ;
	out[3] |= in[41] << 27 ;
	out[3] |= in[42] << 30 ;

	out[4] = in[42] >> ( 32 - 30 ) ;
	out[4] |= in[43] << 1 ;
	out[4] |= in[44] << 4 ;
	out[4] |= in[45] << 7 ;
	out[4] |= in[46] << 10 ;
	out[4] |= in[47] << 13 ;
	out[4] |= in[48] << 16 ;
	out[4] |= in[49] << 19 ;
	out[4] |= in[50] << 22 ;
	out[4] |= in[51] << 25 ;
	out[4] |= in[52] << 28 ;
	out[4] |= in[53] << 31 ;

	out[5] = in[53] >> ( 32 - 31 ) ;
	out[5] |= in[54] << 2 ;
	out[5] |= in[55] << 5 ;
	out[5] |= in[56] << 8 ;
	out[5] |= in[57] << 11 ;
	out[5] |= in[58] << 14 ;
	out[5] |= in[59] << 17 ;
	out[5] |= in[60] << 20 ;
	out[5] |= in[61] << 23 ;
	out[5] |= in[62] << 26 ;
	out[5] |= in[63] << 29 ;
}


// 4-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c4(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 4 ;
	out[0] |= in[2] << 8 ;
	out[0] |= in[3] << 12 ;
	out[0] |= in[4] << 16 ;
	out[0] |= in[5] << 20 ;
	out[0] |= in[6] << 24 ;
	out[0] |= in[7] << 28 ;

	out[1] = in[8] << 0 ;
	out[1] |= in[9] << 4 ;
	out[1] |= in[10] << 8 ;
	out[1] |= in[11] << 12 ;
	out[1] |= in[12] << 16 ;
	out[1] |= in[13] << 20 ;
	out[1] |= in[14] << 24 ;
	out[1] |= in[15] << 28 ;

	out[2] = in[16] << 0 ;
	out[2] |= in[17] << 4 ;
	out[2] |= in[18] << 8 ;
	out[2] |= in[19] << 12 ;
	out[2] |= in[20] << 16 ;
	out[2] |= in[21] << 20 ;
	out[2] |= in[22] << 24 ;
	out[2] |= in[23] << 28 ;

	out[3] = in[24] << 0 ;
	out[3] |= in[25] << 4 ;
	out[3] |= in[26] << 8 ;
	out[3] |= in[27] << 12 ;
	out[3] |= in[28] << 16 ;
	out[3] |= in[29] << 20 ;
	out[3] |= in[30] << 24 ;
	out[3] |= in[31] << 28 ;

	out[4] = in[32] << 0 ;
	out[4] |= in[33] << 4 ;
	out[4] |= in[34] << 8 ;
	out[4] |= in[35] << 12 ;
	out[4] |= in[36] << 16 ;
	out[4] |= in[37] << 20 ;
	out[4] |= in[38] << 24 ;
	out[4] |= in[39] << 28 ;

	out[5] = in[40] << 0 ;
	out[5] |= in[41] << 4 ;
	out[5] |= in[42] << 8 ;
	out[5] |= in[43] << 12 ;
	out[5] |= in[44] << 16 ;
	out[5] |= in[45] << 20 ;
	out[5] |= in[46] << 24 ;
	out[5] |= in[47] << 28 ;

	out[6] = in[48] << 0 ;
	out[6] |= in[49] << 4 ;
	out[6] |= in[50] << 8 ;
	out[6] |= in[51] << 12 ;
	out[6] |= in[52] << 16 ;
	out[6] |= in[53] << 20 ;
	out[6] |= in[54] << 24 ;
	out[6] |= in[55] << 28 ;

	out[7] = in[56] << 0 ;
	out[7] |= in[57] << 4 ;
	out[7] |= in[58] << 8 ;
	out[7] |= in[59] << 12 ;
	out[7] |= in[60] << 16 ;
	out[7] |= in[61] << 20 ;
	out[7] |= in[62] << 24 ;
	out[7] |= in[63] << 28 ;
}


// 5-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c5(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 5 ;
	out[0] |= in[2] << 10 ;
	out[0] |= in[3] << 15 ;
	out[0] |= in[4] << 20 ;
	out[0] |= in[5] << 25 ;
	out[0] |= in[6] << 30 ;

	out[1] = in[6] >> ( 32 - 30 ) ;
	out[1] |= in[7] << 3 ;
	out[1] |= in[8] << 8 ;
	out[1] |= in[9] << 13 ;
	out[1] |= in[10] << 18 ;
	out[1] |= in[11] << 23 ;
	out[1] |= in[12] << 28 ;

	out[2] = in[12] >> ( 32 - 28 ) ;
	out[2] |= in[13] << 1 ;
	out[2] |= in[14] << 6 ;
	out[2] |= in[15] << 11 ;
	out[2] |= in[16] << 16 ;
	out[2] |= in[17] << 21 ;
	out[2] |= in[18] << 26 ;
	out[2] |= in[19] << 31 ;

	out[3] = in[19] >> ( 32 - 31 ) ;
	out[3] |= in[20] << 4 ;
	out[3] |= in[21] << 9 ;
	out[3] |= in[22] << 14 ;
	out[3] |= in[23] << 19 ;
	out[3] |= in[24] << 24 ;
	out[3] |= in[25] << 29 ;

	out[4] = in[25] >> ( 32 - 29 ) ;
	out[4] |= in[26] << 2 ;
	out[4] |= in[27] << 7 ;
	out[4] |= in[28] << 12 ;
	out[4] |= in[29] << 17 ;
	out[4] |= in[30] << 22 ;
	out[4] |= in[31] << 27 ;

	out[5] = in[32] << 0 ;
	out[5] |= in[33] << 5 ;
	out[5] |= in[34] << 10 ;
	out[5] |= in[35] << 15 ;
	out[5] |= in[36] << 20 ;
	out[5] |= in[37] << 25 ;
	out[5] |= in[38] << 30 ;

	out[6] = in[38] >> ( 32 - 30 ) ;
	out[6] |= in[39] << 3 ;
	out[6] |= in[40] << 8 ;
	out[6] |= in[41] << 13 ;
	out[6] |= in[42] << 18 ;
	out[6] |= in[43] << 23 ;
	out[6] |= in[44] << 28 ;

	out[7] = in[44] >> ( 32 - 28 ) ;
	out[7] |= in[45] << 1 ;
	out[7] |= in[46] << 6 ;
	out[7] |= in[47] << 11 ;
	out[7] |= in[48] << 16 ;
	out[7] |= in[49] << 21 ;
	out[7] |= in[50] << 26 ;
	out[7] |= in[51] << 31 ;

	out[8] = in[51] >> ( 32 - 31 ) ;
	out[8] |= in[52] << 4 ;
	out[8] |= in[53] << 9 ;
	out[8] |= in[54] << 14 ;
	out[8] |= in[55] << 19 ;
	out[8] |= in[56] << 24 ;
	out[8] |= in[57] << 29 ;

	out[9] = in[57] >> ( 32 - 29 ) ;
	out[9] |= in[58] << 2 ;
	out[9] |= in[59] << 7 ;
	out[9] |= in[60] << 12 ;
	out[9] |= in[61] << 17 ;
	out[9] |= in[62] << 22 ;
	out[9] |= in[63] << 27 ;
}


// 6-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c6(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 6 ;
	out[0] |= in[2] << 12 ;
	out[0] |= in[3] << 18 ;
	out[0] |= in[4] << 24 ;
	out[0] |= in[5] << 30 ;

	out[1] = in[5] >> ( 32 - 30 ) ;
	out[1] |= in[6] << 4 ;
	out[1] |= in[7] << 10 ;
	out[1] |= in[8] << 16 ;
	out[1] |= in[9] << 22 ;
	out[1] |= in[10] << 28 ;

	out[2] = in[10] >> ( 32 - 28 ) ;
	out[2] |= in[11] << 2 ;
	out[2] |= in[12] << 8 ;
	out[2] |= in[13] << 14 ;
	out[2] |= in[14] << 20 ;
	out[2] |= in[15] << 26 ;

	out[3] = in[16] << 0 ;
	out[3] |= in[17] << 6 ;
	out[3] |= in[18] << 12 ;
	out[3] |= in[19] << 18 ;
	out[3] |= in[20] << 24 ;
	out[3] |= in[21] << 30 ;

	out[4] = in[21] >> ( 32 - 30 ) ;
	out[4] |= in[22] << 4 ;
	out[4] |= in[23] << 10 ;
	out[4] |= in[24] << 16 ;
	out[4] |= in[25] << 22 ;
	out[4] |= in[26] << 28 ;

	out[5] = in[26] >> ( 32 - 28 ) ;
	out[5] |= in[27] << 2 ;
	out[5] |= in[28] << 8 ;
	out[5] |= in[29] << 14 ;
	out[5] |= in[30] << 20 ;
	out[5] |= in[31] << 26 ;

	out[6] = in[32] << 0 ;
	out[6] |= in[33] << 6 ;
	out[6] |= in[34] << 12 ;
	out[6] |= in[35] << 18 ;
	out[6] |= in[36] << 24 ;
	out[6] |= in[37] << 30 ;

	out[7] = in[37] >> ( 32 - 30 ) ;
	out[7] |= in[38] << 4 ;
	out[7] |= in[39] << 10 ;
	out[7] |= in[40] << 16 ;
	out[7] |= in[41] << 22 ;
	out[7] |= in[42] << 28 ;

	out[8] = in[42] >> ( 32 - 28 ) ;
	out[8] |= in[43] << 2 ;
	out[8] |= in[44] << 8 ;
	out[8] |= in[45] << 14 ;
	out[8] |= in[46] << 20 ;
	out[8] |= in[47] << 26 ;

	out[9] = in[48] << 0 ;
	out[9] |= in[49] << 6 ;
	out[9] |= in[50] << 12 ;
	out[9] |= in[51] << 18 ;
	out[9] |= in[52] << 24 ;
	out[9] |= in[53] << 30 ;

	out[10] = in[53] >> ( 32 - 30 ) ;
	out[10] |= in[54] << 4 ;
	out[10] |= in[55] << 10 ;
	out[10] |= in[56] << 16 ;
	out[10] |= in[57] << 22 ;
	out[10] |= in[58] << 28 ;

	out[11] = in[58] >> ( 32 - 28 ) ;
	out[11] |= in[59] << 2 ;
	out[11] |= in[60] << 8 ;
	out[11] |= in[61] << 14 ;
	out[11] |= in[62] << 20 ;
	out[11] |= in[63] << 26 ;
}


// 7-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c7(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 7 ;
	out[0] |= in[2] << 14 ;
	out[0] |= in[3] << 21 ;
	out[0] |= in[4] << 28 ;

	out[1] = in[4] >> ( 32 - 28 ) ;
	out[1] |= in[5] << 3 ;
	out[1] |= in[6] << 10 ;
	out[1] |= in[7] << 17 ;
	out[1] |= in[8] << 24 ;
	out[1] |= in[9] << 31 ;

	out[2] = in[9] >> ( 32 - 31 ) ;
	out[2] |= in[10] << 6 ;
	out[2] |= in[11] << 13 ;
	out[2] |= in[12] << 20 ;
	out[2] |= in[13] << 27 ;

	out[3] = in[13] >> ( 32 - 27 ) ;
	out[3] |= in[14] << 2 ;
	out[3] |= in[15] << 9 ;
	out[3] |= in[16] << 16 ;
	out[3] |= in[17] << 23 ;
	out[3] |= in[18] << 30 ;

	out[4] = in[18] >> ( 32 - 30 ) ;
	out[4] |= in[19] << 5 ;
	out[4] |= in[20] << 12 ;
	out[4] |= in[21] << 19 ;
	out[4] |= in[22] << 26 ;

	out[5] = in[22] >> ( 32 - 26 ) ;
	out[5] |= in[23] << 1 ;
	out[5] |= in[24] << 8 ;
	out[5] |= in[25] << 15 ;
	out[5] |= in[26] << 22 ;
	out[5] |= in[27] << 29 ;

	out[6] = in[27] >> ( 32 - 29 ) ;
	out[6] |= in[28] << 4 ;
	out[6] |= in[29] << 11 ;
	out[6] |= in[30] << 18 ;
	out[6] |= in[31] << 25 ;

	out[7] = in[32] << 0 ;
	out[7] |= in[33] << 7 ;
	out[7] |= in[34] << 14 ;
	out[7] |= in[35] << 21 ;
	out[7] |= in[36] << 28 ;

	out[8] = in[36] >> ( 32 - 28 ) ;
	out[8] |= in[37] << 3 ;
	out[8] |= in[38] << 10 ;
	out[8] |= in[39] << 17 ;
	out[8] |= in[40] << 24 ;
	out[8] |= in[41] << 31 ;

	out[9] = in[41] >> ( 32 - 31 ) ;
	out[9] |= in[42] << 6 ;
	out[9] |= in[43] << 13 ;
	out[9] |= in[44] << 20 ;
	out[9] |= in[45] << 27 ;

	out[10] = in[45] >> ( 32 - 27 ) ;
	out[10] |= in[46] << 2 ;
	out[10] |= in[47] << 9 ;
	out[10] |= in[48] << 16 ;
	out[10] |= in[49] << 23 ;
	out[10] |= in[50] << 30 ;

	out[11] = in[50] >> ( 32 - 30 ) ;
	out[11] |= in[51] << 5 ;
	out[11] |= in[52] << 12 ;
	out[11] |= in[53] << 19 ;
	out[11] |= in[54] << 26 ;

	out[12] = in[54] >> ( 32 - 26 ) ;
	out[12] |= in[55] << 1 ;
	out[12] |= in[56] << 8 ;
	out[12] |= in[57] << 15 ;
	out[12] |= in[58] << 22 ;
	out[12] |= in[59] << 29 ;

	out[13] = in[59] >> ( 32 - 29 ) ;
	out[13] |= in[60] << 4 ;
	out[13] |= in[61] << 11 ;
	out[13] |= in[62] << 18 ;
	out[13] |= in[63] << 25 ;
}


// 8-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c8(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 8 ;
	out[0] |= in[2] << 16 ;
	out[0] |= in[3] << 24 ;

	out[1] = in[4] << 0 ;
	out[1] |= in[5] << 8 ;
	out[1] |= in[6] << 16 ;
	out[1] |= in[7] << 24 ;

	out[2] = in[8] << 0 ;
	out[2] |= in[9] << 8 ;
	out[2] |= in[10] << 16 ;
	out[2] |= in[11] << 24 ;

	out[3] = in[12] << 0 ;
	out[3] |= in[13] << 8 ;
	out[3] |= in[14] << 16 ;
	out[3] |= in[15] << 24 ;

	out[4] = in[16] << 0 ;
	out[4] |= in[17] << 8 ;
	out[4] |= in[18] << 16 ;
	out[4] |= in[19] << 24 ;

	out[5] = in[20] << 0 ;
	out[5] |= in[21] << 8 ;
	out[5] |= in[22] << 16 ;
	out[5] |= in[23] << 24 ;

	out[6] = in[24] << 0 ;
	out[6] |= in[25] << 8 ;
	out[6] |= in[26] << 16 ;
	out[6] |= in[27] << 24 ;

	out[7] = in[28] << 0 ;
	out[7] |= in[29] << 8 ;
	out[7] |= in[30] << 16 ;
	out[7] |= in[31] << 24 ;

	out[8] = in[32] << 0 ;
	out[8] |= in[33] << 8 ;
	out[8] |= in[34] << 16 ;
	out[8] |= in[35] << 24 ;

	out[9] = in[36] << 0 ;
	out[9] |= in[37] << 8 ;
	out[9] |= in[38] << 16 ;
	out[9] |= in[39] << 24 ;

	out[10] = in[40] << 0 ;
	out[10] |= in[41] << 8 ;
	out[10] |= in[42] << 16 ;
	out[10] |= in[43] << 24 ;

	out[11] = in[44] << 0 ;
	out[11] |= in[45] << 8 ;
	out[11] |= in[46] << 16 ;
	out[11] |= in[47] << 24 ;

	out[12] = in[48] << 0 ;
	out[12] |= in[49] << 8 ;
	out[12] |= in[50] << 16 ;
	out[12] |= in[51] << 24 ;

	out[13] = in[52] << 0 ;
	out[13] |= in[53] << 8 ;
	out[13] |= in[54] << 16 ;
	out[13] |= in[55] << 24 ;

	out[14] = in[56] << 0 ;
	out[14] |= in[57] << 8 ;
	out[14] |= in[58] << 16 ;
	out[14] |= in[59] << 24 ;

	out[15] = in[60] << 0 ;
	out[15] |= in[61] << 8 ;
	out[15] |= in[62] << 16 ;
	out[15] |= in[63] << 24 ;
}


// 9-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c9(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 9 ;
	out[0] |= in[2] << 18 ;
	out[0] |= in[3] << 27 ;

	out[1] = in[3] >> ( 32 - 27 ) ;
	out[1] |= in[4] << 4 ;
	out[1] |= in[5] << 13 ;
	out[1] |= in[6] << 22 ;
	out[1] |= in[7] << 31 ;

	out[2] = in[7] >> ( 32 - 31 ) ;
	out[2] |= in[8] << 8 ;
	out[2] |= in[9] << 17 ;
	out[2] |= in[10] << 26 ;

	out[3] = in[10] >> ( 32 - 26 ) ;
	out[3] |= in[11] << 3 ;
	out[3] |= in[12] << 12 ;
	out[3] |= in[13] << 21 ;
	out[3] |= in[14] << 30 ;

	out[4] = in[14] >> ( 32 - 30 ) ;
	out[4] |= in[15] << 7 ;
	out[4] |= in[16] << 16 ;
	out[4] |= in[17] << 25 ;

	out[5] = in[17] >> ( 32 - 25 ) ;
	out[5] |= in[18] << 2 ;
	out[5] |= in[19] << 11 ;
	out[5] |= in[20] << 20 ;
	out[5] |= in[21] << 29 ;

	out[6] = in[21] >> ( 32 - 29 ) ;
	out[6] |= in[22] << 6 ;
	out[6] |= in[23] << 15 ;
	out[6] |= in[24] << 24 ;

	out[7] = in[24] >> ( 32 - 24 ) ;
	out[7] |= in[25] << 1 ;
	out[7] |= in[26] << 10 ;
	out[7] |= in[27] << 19 ;
	out[7] |= in[28] << 28 ;

	out[8] = in[28] >> ( 32 - 28 ) ;
	out[8] |= in[29] << 5 ;
	out[8] |= in[30] << 14 ;
	out[8] |= in[31] << 23 ;

	out[9] = in[32] << 0 ;
	out[9] |= in[33] << 9 ;
	out[9] |= in[34] << 18 ;
	out[9] |= in[35] << 27 ;

	out[10] = in[35] >> ( 32 - 27 ) ;
	out[10] |= in[36] << 4 ;
	out[10] |= in[37] << 13 ;
	out[10] |= in[38] << 22 ;
	out[10] |= in[39] << 31 ;

	out[11] = in[39] >> ( 32 - 31 ) ;
	out[11] |= in[40] << 8 ;
	out[11] |= in[41] << 17 ;
	out[11] |= in[42] << 26 ;

	out[12] = in[42] >> ( 32 - 26 ) ;
	out[12] |= in[43] << 3 ;
	out[12] |= in[44] << 12 ;
	out[12] |= in[45] << 21 ;
	out[12] |= in[46] << 30 ;

	out[13] = in[46] >> ( 32 - 30 ) ;
	out[13] |= in[47] << 7 ;
	out[13] |= in[48] << 16 ;
	out[13] |= in[49] << 25 ;

	out[14] = in[49] >> ( 32 - 25 ) ;
	out[14] |= in[50] << 2 ;
	out[14] |= in[51] << 11 ;
	out[14] |= in[52] << 20 ;
	out[14] |= in[53] << 29 ;

	out[15] = in[53] >> ( 32 - 29 ) ;
	out[15] |= in[54] << 6 ;
	out[15] |= in[55] << 15 ;
	out[15] |= in[56] << 24 ;

	out[16] = in[56] >> ( 32 - 24 ) ;
	out[16] |= in[57] << 1 ;
	out[16] |= in[58] << 10 ;
	out[16] |= in[59] << 19 ;
	out[16] |= in[60] << 28 ;

	out[17] = in[60] >> ( 32 - 28 ) ;
	out[17] |= in[61] << 5 ;
	out[17] |= in[62] << 14 ;
	out[17] |= in[63] << 23 ;
}


// 10-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c10(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 10 ;
	out[0] |= in[2] << 20 ;
	out[0] |= in[3] << 30 ;

	out[1] = in[3] >> ( 32 - 30 ) ;
	out[1] |= in[4] << 8 ;
	out[1] |= in[5] << 18 ;
	out[1] |= in[6] << 28 ;

	out[2] = in[6] >> ( 32 - 28 ) ;
	out[2] |= in[7] << 6 ;
	out[2] |= in[8] << 16 ;
	out[2] |= in[9] << 26 ;

	out[3] = in[9] >> ( 32 - 26 ) ;
	out[3] |= in[10] << 4 ;
	out[3] |= in[11] << 14 ;
	out[3] |= in[12] << 24 ;

	out[4] = in[12] >> ( 32 - 24 ) ;
	out[4] |= in[13] << 2 ;
	out[4] |= in[14] << 12 ;
	out[4] |= in[15] << 22 ;

	out[5] = in[16] << 0 ;
	out[5] |= in[17] << 10 ;
	out[5] |= in[18] << 20 ;
	out[5] |= in[19] << 30 ;

	out[6] = in[19] >> ( 32 - 30 ) ;
	out[6] |= in[20] << 8 ;
	out[6] |= in[21] << 18 ;
	out[6] |= in[22] << 28 ;

	out[7] = in[22] >> ( 32 - 28 ) ;
	out[7] |= in[23] << 6 ;
	out[7] |= in[24] << 16 ;
	out[7] |= in[25] << 26 ;

	out[8] = in[25] >> ( 32 - 26 ) ;
	out[8] |= in[26] << 4 ;
	out[8] |= in[27] << 14 ;
	out[8] |= in[28] << 24 ;

	out[9] = in[28] >> ( 32 - 24 ) ;
	out[9] |= in[29] << 2 ;
	out[9] |= in[30] << 12 ;
	out[9] |= in[31] << 22 ;

	out[10] = in[32] << 0 ;
	out[10] |= in[33] << 10 ;
	out[10] |= in[34] << 20 ;
	out[10] |= in[35] << 30 ;

	out[11] = in[35] >> ( 32 - 30 ) ;
	out[11] |= in[36] << 8 ;
	out[11] |= in[37] << 18 ;
	out[11] |= in[38] << 28 ;

	out[12] = in[38] >> ( 32 - 28 ) ;
	out[12] |= in[39] << 6 ;
	out[12] |= in[40] << 16 ;
	out[12] |= in[41] << 26 ;

	out[13] = in[41] >> ( 32 - 26 ) ;
	out[13] |= in[42] << 4 ;
	out[13] |= in[43] << 14 ;
	out[13] |= in[44] << 24 ;

	out[14] = in[44] >> ( 32 - 24 ) ;
	out[14] |= in[45] << 2 ;
	out[14] |= in[46] << 12 ;
	out[14] |= in[47] << 22 ;

	out[15] = in[48] << 0 ;
	out[15] |= in[49] << 10 ;
	out[15] |= in[50] << 20 ;
	out[15] |= in[51] << 30 ;

	out[16] = in[51] >> ( 32 - 30 ) ;
	out[16] |= in[52] << 8 ;
	out[16] |= in[53] << 18 ;
	out[16] |= in[54] << 28 ;

	out[17] = in[54] >> ( 32 - 28 ) ;
	out[17] |= in[55] << 6 ;
	out[17] |= in[56] << 16 ;
	out[17] |= in[57] << 26 ;

	out[18] = in[57] >> ( 32 - 26 ) ;
	out[18] |= in[58] << 4 ;
	out[18] |= in[59] << 14 ;
	out[18] |= in[60] << 24 ;

	out[19] = in[60] >> ( 32 - 24 ) ;
	out[19] |= in[61] << 2 ;
	out[19] |= in[62] << 12 ;
	out[19] |= in[63] << 22 ;
}


// 11-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c11(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 11 ;
	out[0] |= in[2] << 22 ;

	out[1] = in[2] >> ( 32 - 22 ) ;
	out[1] |= in[3] << 1 ;
	out[1] |= in[4] << 12 ;
	out[1] |= in[5] << 23 ;

	out[2] = in[5] >> ( 32 - 23 ) ;
	out[2] |= in[6] << 2 ;
	out[2] |= in[7] << 13 ;
	out[2] |= in[8] << 24 ;

	out[3] = in[8] >> ( 32 - 24 ) ;
	out[3] |= in[9] << 3 ;
	out[3] |= in[10] << 14 ;
	out[3] |= in[11] << 25 ;

	out[4] = in[11] >> ( 32 - 25 ) ;
	out[4] |= in[12] << 4 ;
	out[4] |= in[13] << 15 ;
	out[4] |= in[14] << 26 ;

	out[5] = in[14] >> ( 32 - 26 ) ;
	out[5] |= in[15] << 5 ;
	out[5] |= in[16] << 16 ;
	out[5] |= in[17] << 27 ;

	out[6] = in[17] >> ( 32 - 27 ) ;
	out[6] |= in[18] << 6 ;
	out[6] |= in[19] << 17 ;
	out[6] |= in[20] << 28 ;

	out[7] = in[20] >> ( 32 - 28 ) ;
	out[7] |= in[21] << 7 ;
	out[7] |= in[22] << 18 ;
	out[7] |= in[23] << 29 ;

	out[8] = in[23] >> ( 32 - 29 ) ;
	out[8] |= in[24] << 8 ;
	out[8] |= in[25] << 19 ;
	out[8] |= in[26] << 30 ;

	out[9] = in[26] >> ( 32 - 30 ) ;
	out[9] |= in[27] << 9 ;
	out[9] |= in[28] << 20 ;
	out[9] |= in[29] << 31 ;

	out[10] = in[29] >> ( 32 - 31 ) ;
	out[10] |= in[30] << 10 ;
	out[10] |= in[31] << 21 ;

	out[11] = in[32] << 0 ;
	out[11] |= in[33] << 11 ;
	out[11] |= in[34] << 22 ;

	out[12] = in[34] >> ( 32 - 22 ) ;
	out[12] |= in[35] << 1 ;
	out[12] |= in[36] << 12 ;
	out[12] |= in[37] << 23 ;

	out[13] = in[37] >> ( 32 - 23 ) ;
	out[13] |= in[38] << 2 ;
	out[13] |= in[39] << 13 ;
	out[13] |= in[40] << 24 ;

	out[14] = in[40] >> ( 32 - 24 ) ;
	out[14] |= in[41] << 3 ;
	out[14] |= in[42] << 14 ;
	out[14] |= in[43] << 25 ;

	out[15] = in[43] >> ( 32 - 25 ) ;
	out[15] |= in[44] << 4 ;
	out[15] |= in[45] << 15 ;
	out[15] |= in[46] << 26 ;

	out[16] = in[46] >> ( 32 - 26 ) ;
	out[16] |= in[47] << 5 ;
	out[16] |= in[48] << 16 ;
	out[16] |= in[49] << 27 ;

	out[17] = in[49] >> ( 32 - 27 ) ;
	out[17] |= in[50] << 6 ;
	out[17] |= in[51] << 17 ;
	out[17] |= in[52] << 28 ;

	out[18] = in[52] >> ( 32 - 28 ) ;
	out[18] |= in[53] << 7 ;
	out[18] |= in[54] << 18 ;
	out[18] |= in[55] << 29 ;

	out[19] = in[55] >> ( 32 - 29 ) ;
	out[19] |= in[56] << 8 ;
	out[19] |= in[57] << 19 ;
	out[19] |= in[58] << 30 ;

	out[20] = in[58] >> ( 32 - 30 ) ;
	out[20] |= in[59] << 9 ;
	out[20] |= in[60] << 20 ;
	out[20] |= in[61] << 31 ;

	out[21] = in[61] >> ( 32 - 31 ) ;
	out[21] |= in[62] << 10 ;
	out[21] |= in[63] << 21 ;
}


// 12-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c12(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 12 ;
	out[0] |= in[2] << 24 ;

	out[1] = in[2] >> ( 32 - 24 ) ;
	out[1] |= in[3] << 4 ;
	out[1] |= in[4] << 16 ;
	out[1] |= in[5] << 28 ;

	out[2] = in[5] >> ( 32 - 28 ) ;
	out[2] |= in[6] << 8 ;
	out[2] |= in[7] << 20 ;

	out[3] = in[8] << 0 ;
	out[3] |= in[9] << 12 ;
	out[3] |= in[10] << 24 ;

	out[4] = in[10] >> ( 32 - 24 ) ;
	out[4] |= in[11] << 4 ;
	out[4] |= in[12] << 16 ;
	out[4] |= in[13] << 28 ;

	out[5] = in[13] >> ( 32 - 28 ) ;
	out[5] |= in[14] << 8 ;
	out[5] |= in[15] << 20 ;

	out[6] = in[16] << 0 ;
	out[6] |= in[17] << 12 ;
	out[6] |= in[18] << 24 ;

	out[7] = in[18] >> ( 32 - 24 ) ;
	out[7] |= in[19] << 4 ;
	out[7] |= in[20] << 16 ;
	out[7] |= in[21] << 28 ;

	out[8] = in[21] >> ( 32 - 28 ) ;
	out[8] |= in[22] << 8 ;
	out[8] |= in[23] << 20 ;

	out[9] = in[24] << 0 ;
	out[9] |= in[25] << 12 ;
	out[9] |= in[26] << 24 ;

	out[10] = in[26] >> ( 32 - 24 ) ;
	out[10] |= in[27] << 4 ;
	out[10] |= in[28] << 16 ;
	out[10] |= in[29] << 28 ;

	out[11] = in[29] >> ( 32 - 28 ) ;
	out[11] |= in[30] << 8 ;
	out[11] |= in[31] << 20 ;

	out[12] = in[32] << 0 ;
	out[12] |= in[33] << 12 ;
	out[12] |= in[34] << 24 ;

	out[13] = in[34] >> ( 32 - 24 ) ;
	out[13] |= in[35] << 4 ;
	out[13] |= in[36] << 16 ;
	out[13] |= in[37] << 28 ;

	out[14] = in[37] >> ( 32 - 28 ) ;
	out[14] |= in[38] << 8 ;
	out[14] |= in[39] << 20 ;

	out[15] = in[40] << 0 ;
	out[15] |= in[41] << 12 ;
	out[15] |= in[42] << 24 ;

	out[16] = in[42] >> ( 32 - 24 ) ;
	out[16] |= in[43] << 4 ;
	out[16] |= in[44] << 16 ;
	out[16] |= in[45] << 28 ;

	out[17] = in[45] >> ( 32 - 28 ) ;
	out[17] |= in[46] << 8 ;
	out[17] |= in[47] << 20 ;

	out[18] = in[48] << 0 ;
	out[18] |= in[49] << 12 ;
	out[18] |= in[50] << 24 ;

	out[19] = in[50] >> ( 32 - 24 ) ;
	out[19] |= in[51] << 4 ;
	out[19] |= in[52] << 16 ;
	out[19] |= in[53] << 28 ;

	out[20] = in[53] >> ( 32 - 28 ) ;
	out[20] |= in[54] << 8 ;
	out[20] |= in[55] << 20 ;

	out[21] = in[56] << 0 ;
	out[21] |= in[57] << 12 ;
	out[21] |= in[58] << 24 ;

	out[22] = in[58] >> ( 32 - 24 ) ;
	out[22] |= in[59] << 4 ;
	out[22] |= in[60] << 16 ;
	out[22] |= in[61] << 28 ;

	out[23] = in[61] >> ( 32 - 28 ) ;
	out[23] |= in[62] << 8 ;
	out[23] |= in[63] << 20 ;
}


// 13-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c13(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 13 ;
	out[0] |= in[2] << 26 ;

	out[1] = in[2] >> ( 32 - 26 ) ;
	out[1] |= in[3] << 7 ;
	out[1] |= in[4] << 20 ;

	out[2] = in[4] >> ( 32 - 20 ) ;
	out[2] |= in[5] << 1 ;
	out[2] |= in[6] << 14 ;
	out[2] |= in[7] << 27 ;

	out[3] = in[7] >> ( 32 - 27 ) ;
	out[3] |= in[8] << 8 ;
	out[3] |= in[9] << 21 ;

	out[4] = in[9] >> ( 32 - 21 ) ;
	out[4] |= in[10] << 2 ;
	out[4] |= in[11] << 15 ;
	out[4] |= in[12] << 28 ;

	out[5] = in[12] >> ( 32 - 28 ) ;
	out[5] |= in[13] << 9 ;
	out[5] |= in[14] << 22 ;

	out[6] = in[14] >> ( 32 - 22 ) ;
	out[6] |= in[15] << 3 ;
	out[6] |= in[16] << 16 ;
	out[6] |= in[17] << 29 ;

	out[7] = in[17] >> ( 32 - 29 ) ;
	out[7] |= in[18] << 10 ;
	out[7] |= in[19] << 23 ;

	out[8] = in[19] >> ( 32 - 23 ) ;
	out[8] |= in[20] << 4 ;
	out[8] |= in[21] << 17 ;
	out[8] |= in[22] << 30 ;

	out[9] = in[22] >> ( 32 - 30 ) ;
	out[9] |= in[23] << 11 ;
	out[9] |= in[24] << 24 ;

	out[10] = in[24] >> ( 32 - 24 ) ;
	out[10] |= in[25] << 5 ;
	out[10] |= in[26] << 18 ;
	out[10] |= in[27] << 31 ;

	out[11] = in[27] >> ( 32 - 31 ) ;
	out[11] |= in[28] << 12 ;
	out[11] |= in[29] << 25 ;

	out[12] = in[29] >> ( 32 - 25 ) ;
	out[12] |= in[30] << 6 ;
	out[12] |= in[31] << 19 ;

	out[13] = in[32] << 0 ;
	out[13] |= in[33] << 13 ;
	out[13] |= in[34] << 26 ;

	out[14] = in[34] >> ( 32 - 26 ) ;
	out[14] |= in[35] << 7 ;
	out[14] |= in[36] << 20 ;

	out[15] = in[36] >> ( 32 - 20 ) ;
	out[15] |= in[37] << 1 ;
	out[15] |= in[38] << 14 ;
	out[15] |= in[39] << 27 ;

	out[16] = in[39] >> ( 32 - 27 ) ;
	out[16] |= in[40] << 8 ;
	out[16] |= in[41] << 21 ;

	out[17] = in[41] >> ( 32 - 21 ) ;
	out[17] |= in[42] << 2 ;
	out[17] |= in[43] << 15 ;
	out[17] |= in[44] << 28 ;

	out[18] = in[44] >> ( 32 - 28 ) ;
	out[18] |= in[45] << 9 ;
	out[18] |= in[46] << 22 ;

	out[19] = in[46] >> ( 32 - 22 ) ;
	out[19] |= in[47] << 3 ;
	out[19] |= in[48] << 16 ;
	out[19] |= in[49] << 29 ;

	out[20] = in[49] >> ( 32 - 29 ) ;
	out[20] |= in[50] << 10 ;
	out[20] |= in[51] << 23 ;

	out[21] = in[51] >> ( 32 - 23 ) ;
	out[21] |= in[52] << 4 ;
	out[21] |= in[53] << 17 ;
	out[21] |= in[54] << 30 ;

	out[22] = in[54] >> ( 32 - 30 ) ;
	out[22] |= in[55] << 11 ;
	out[22] |= in[56] << 24 ;

	out[23] = in[56] >> ( 32 - 24 ) ;
	out[23] |= in[57] << 5 ;
	out[23] |= in[58] << 18 ;
	out[23] |= in[59] << 31 ;

	out[24] = in[59] >> ( 32 - 31 ) ;
	out[24] |= in[60] << 12 ;
	out[24] |= in[61] << 25 ;

	out[25] = in[61] >> ( 32 - 25 ) ;
	out[25] |= in[62] << 6 ;
	out[25] |= in[63] << 19 ;
}


// 14-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c14(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 14 ;
	out[0] |= in[2] << 28 ;

	out[1] = in[2] >> ( 32 - 28 ) ;
	out[1] |= in[3] << 10 ;
	out[1] |= in[4] << 24 ;

	out[2] = in[4] >> ( 32 - 24 ) ;
	out[2] |= in[5] << 6 ;
	out[2] |= in[6] << 20 ;

	out[3] = in[6] >> ( 32 - 20 ) ;
	out[3] |= in[7] << 2 ;
	out[3] |= in[8] << 16 ;
	out[3] |= in[9] << 30 ;

	out[4] = in[9] >> ( 32 - 30 ) ;
	out[4] |= in[10] << 12 ;
	out[4] |= in[11] << 26 ;

	out[5] = in[11] >> ( 32 - 26 ) ;
	out[5] |= in[12] << 8 ;
	out[5] |= in[13] << 22 ;

	out[6] = in[13] >> ( 32 - 22 ) ;
	out[6] |= in[14] << 4 ;
	out[6] |= in[15] << 18 ;

	out[7] = in[16] << 0 ;
	out[7] |= in[17] << 14 ;
	out[7] |= in[18] << 28 ;

	out[8] = in[18] >> ( 32 - 28 ) ;
	out[8] |= in[19] << 10 ;
	out[8] |= in[20] << 24 ;

	out[9] = in[20] >> ( 32 - 24 ) ;
	out[9] |= in[21] << 6 ;
	out[9] |= in[22] << 20 ;

	out[10] = in[22] >> ( 32 - 20 ) ;
	out[10] |= in[23] << 2 ;
	out[10] |= in[24] << 16 ;
	out[10] |= in[25] << 30 ;

	out[11] = in[25] >> ( 32 - 30 ) ;
	out[11] |= in[26] << 12 ;
	out[11] |= in[27] << 26 ;

	out[12] = in[27] >> ( 32 - 26 ) ;
	out[12] |= in[28] << 8 ;
	out[12] |= in[29] << 22 ;

	out[13] = in[29] >> ( 32 - 22 ) ;
	out[13] |= in[30] << 4 ;
	out[13] |= in[31] << 18 ;

	out[14] = in[32] << 0 ;
	out[14] |= in[33] << 14 ;
	out[14] |= in[34] << 28 ;

	out[15] = in[34] >> ( 32 - 28 ) ;
	out[15] |= in[35] << 10 ;
	out[15] |= in[36] << 24 ;

	out[16] = in[36] >> ( 32 - 24 ) ;
	out[16] |= in[37] << 6 ;
	out[16] |= in[38] << 20 ;

	out[17] = in[38] >> ( 32 - 20 ) ;
	out[17] |= in[39] << 2 ;
	out[17] |= in[40] << 16 ;
	out[17] |= in[41] << 30 ;

	out[18] = in[41] >> ( 32 - 30 ) ;
	out[18] |= in[42] << 12 ;
	out[18] |= in[43] << 26 ;

	out[19] = in[43] >> ( 32 - 26 ) ;
	out[19] |= in[44] << 8 ;
	out[19] |= in[45] << 22 ;

	out[20] = in[45] >> ( 32 - 22 ) ;
	out[20] |= in[46] << 4 ;
	out[20] |= in[47] << 18 ;

	out[21] = in[48] << 0 ;
	out[21] |= in[49] << 14 ;
	out[21] |= in[50] << 28 ;

	out[22] = in[50] >> ( 32 - 28 ) ;
	out[22] |= in[51] << 10 ;
	out[22] |= in[52] << 24 ;

	out[23] = in[52] >> ( 32 - 24 ) ;
	out[23] |= in[53] << 6 ;
	out[23] |= in[54] << 20 ;

	out[24] = in[54] >> ( 32 - 20 ) ;
	out[24] |= in[55] << 2 ;
	out[24] |= in[56] << 16 ;
	out[24] |= in[57] << 30 ;

	out[25] = in[57] >> ( 32 - 30 ) ;
	out[25] |= in[58] << 12 ;
	out[25] |= in[59] << 26 ;

	out[26] = in[59] >> ( 32 - 26 ) ;
	out[26] |= in[60] << 8 ;
	out[26] |= in[61] << 22 ;

	out[27] = in[61] >> ( 32 - 22 ) ;
	out[27] |= in[62] << 4 ;
	out[27] |= in[63] << 18 ;
}


// 15-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c15(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 15 ;
	out[0] |= in[2] << 30 ;

	out[1] = in[2] >> ( 32 - 30 ) ;
	out[1] |= in[3] << 13 ;
	out[1] |= in[4] << 28 ;

	out[2] = in[4] >> ( 32 - 28 ) ;
	out[2] |= in[5] << 11 ;
	out[2] |= in[6] << 26 ;

	out[3] = in[6] >> ( 32 - 26 ) ;
	out[3] |= in[7] << 9 ;
	out[3] |= in[8] << 24 ;

	out[4] = in[8] >> ( 32 - 24 ) ;
	out[4] |= in[9] << 7 ;
	out[4] |= in[10] << 22 ;

	out[5] = in[10] >> ( 32 - 22 ) ;
	out[5] |= in[11] << 5 ;
	out[5] |= in[12] << 20 ;

	out[6] = in[12] >> ( 32 - 20 ) ;
	out[6] |= in[13] << 3 ;
	out[6] |= in[14] << 18 ;

	out[7] = in[14] >> ( 32 - 18 ) ;
	out[7] |= in[15] << 1 ;
	out[7] |= in[16] << 16 ;
	out[7] |= in[17] << 31 ;

	out[8] = in[17] >> ( 32 - 31 ) ;
	out[8] |= in[18] << 14 ;
	out[8] |= in[19] << 29 ;

	out[9] = in[19] >> ( 32 - 29 ) ;
	out[9] |= in[20] << 12 ;
	out[9] |= in[21] << 27 ;

	out[10] = in[21] >> ( 32 - 27 ) ;
	out[10] |= in[22] << 10 ;
	out[10] |= in[23] << 25 ;

	out[11] = in[23] >> ( 32 - 25 ) ;
	out[11] |= in[24] << 8 ;
	out[11] |= in[25] << 23 ;

	out[12] = in[25] >> ( 32 - 23 ) ;
	out[12] |= in[26] << 6 ;
	out[12] |= in[27] << 21 ;

	out[13] = in[27] >> ( 32 - 21 ) ;
	out[13] |= in[28] << 4 ;
	out[13] |= in[29] << 19 ;

	out[14] = in[29] >> ( 32 - 19 ) ;
	out[14] |= in[30] << 2 ;
	out[14] |= in[31] << 17 ;

	out[15] = in[32] << 0 ;
	out[15] |= in[33] << 15 ;
	out[15] |= in[34] << 30 ;

	out[16] = in[34] >> ( 32 - 30 ) ;
	out[16] |= in[35] << 13 ;
	out[16] |= in[36] << 28 ;

	out[17] = in[36] >> ( 32 - 28 ) ;
	out[17] |= in[37] << 11 ;
	out[17] |= in[38] << 26 ;

	out[18] = in[38] >> ( 32 - 26 ) ;
	out[18] |= in[39] << 9 ;
	out[18] |= in[40] << 24 ;

	out[19] = in[40] >> ( 32 - 24 ) ;
	out[19] |= in[41] << 7 ;
	out[19] |= in[42] << 22 ;

	out[20] = in[42] >> ( 32 - 22 ) ;
	out[20] |= in[43] << 5 ;
	out[20] |= in[44] << 20 ;

	out[21] = in[44] >> ( 32 - 20 ) ;
	out[21] |= in[45] << 3 ;
	out[21] |= in[46] << 18 ;

	out[22] = in[46] >> ( 32 - 18 ) ;
	out[22] |= in[47] << 1 ;
	out[22] |= in[48] << 16 ;
	out[22] |= in[49] << 31 ;

	out[23] = in[49] >> ( 32 - 31 ) ;
	out[23] |= in[50] << 14 ;
	out[23] |= in[51] << 29 ;

	out[24] = in[51] >> ( 32 - 29 ) ;
	out[24] |= in[52] << 12 ;
	out[24] |= in[53] << 27 ;

	out[25] = in[53] >> ( 32 - 27 ) ;
	out[25] |= in[54] << 10 ;
	out[25] |= in[55] << 25 ;

	out[26] = in[55] >> ( 32 - 25 ) ;
	out[26] |= in[56] << 8 ;
	out[26] |= in[57] << 23 ;

	out[27] = in[57] >> ( 32 - 23 ) ;
	out[27] |= in[58] << 6 ;
	out[27] |= in[59] << 21 ;

	out[28] = in[59] >> ( 32 - 21 ) ;
	out[28] |= in[60] << 4 ;
	out[28] |= in[61] << 19 ;

	out[29] = in[61] >> ( 32 - 19 ) ;
	out[29] |= in[62] << 2 ;
	out[29] |= in[63] << 17 ;
}


// 16-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c16(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 16 ;

	out[1] = in[2] << 0 ;
	out[1] |= in[3] << 16 ;

	out[2] = in[4] << 0 ;
	out[2] |= in[5] << 16 ;

	out[3] = in[6] << 0 ;
	out[3] |= in[7] << 16 ;

	out[4] = in[8] << 0 ;
	out[4] |= in[9] << 16 ;

	out[5] = in[10] << 0 ;
	out[5] |= in[11] << 16 ;

	out[6] = in[12] << 0 ;
	out[6] |= in[13] << 16 ;

	out[7] = in[14] << 0 ;
	out[7] |= in[15] << 16 ;

	out[8] = in[16] << 0 ;
	out[8] |= in[17] << 16 ;

	out[9] = in[18] << 0 ;
	out[9] |= in[19] << 16 ;

	out[10] = in[20] << 0 ;
	out[10] |= in[21] << 16 ;

	out[11] = in[22] << 0 ;
	out[11] |= in[23] << 16 ;

	out[12] = in[24] << 0 ;
	out[12] |= in[25] << 16 ;

	out[13] = in[26] << 0 ;
	out[13] |= in[27] << 16 ;

	out[14] = in[28] << 0 ;
	out[14] |= in[29] << 16 ;

	out[15] = in[30] << 0 ;
	out[15] |= in[31] << 16 ;

	out[16] = in[32] << 0 ;
	out[16] |= in[33] << 16 ;

	out[17] = in[34] << 0 ;
	out[17] |= in[35] << 16 ;

	out[18] = in[36] << 0 ;
	out[18] |= in[37] << 16 ;

	out[19] = in[38] << 0 ;
	out[19] |= in[39] << 16 ;

	out[20] = in[40] << 0 ;
	out[20] |= in[41] << 16 ;

	out[21] = in[42] << 0 ;
	out[21] |= in[43] << 16 ;

	out[22] = in[44] << 0 ;
	out[22] |= in[45] << 16 ;

	out[23] = in[46] << 0 ;
	out[23] |= in[47] << 16 ;

	out[24] = in[48] << 0 ;
	out[24] |= in[49] << 16 ;

	out[25] = in[50] << 0 ;
	out[25] |= in[51] << 16 ;

	out[26] = in[52] << 0 ;
	out[26] |= in[53] << 16 ;

	out[27] = in[54] << 0 ;
	out[27] |= in[55] << 16 ;

	out[28] = in[56] << 0 ;
	out[28] |= in[57] << 16 ;

	out[29] = in[58] << 0 ;
	out[29] |= in[59] << 16 ;

	out[30] = in[60] << 0 ;
	out[30] |= in[61] << 16 ;

	out[31] = in[62] << 0 ;
	out[31] |= in[63] << 16 ;
}


// 17-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c17(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 17 ;

	out[1] = in[1] >> ( 32 - 17 ) ;
	out[1] |= in[2] << 2 ;
	out[1] |= in[3] << 19 ;

	out[2] = in[3] >> ( 32 - 19 ) ;
	out[2] |= in[4] << 4 ;
	out[2] |= in[5] << 21 ;

	out[3] = in[5] >> ( 32 - 21 ) ;
	out[3] |= in[6] << 6 ;
	out[3] |= in[7] << 23 ;

	out[4] = in[7] >> ( 32 - 23 ) ;
	out[4] |= in[8] << 8 ;
	out[4] |= in[9] << 25 ;

	out[5] = in[9] >> ( 32 - 25 ) ;
	out[5] |= in[10] << 10 ;
	out[5] |= in[11] << 27 ;

	out[6] = in[11] >> ( 32 - 27 ) ;
	out[6] |= in[12] << 12 ;
	out[6] |= in[13] << 29 ;

	out[7] = in[13] >> ( 32 - 29 ) ;
	out[7] |= in[14] << 14 ;
	out[7] |= in[15] << 31 ;

	out[8] = in[15] >> ( 32 - 31 ) ;
	out[8] |= in[16] << 16 ;

	out[9] = in[16] >> ( 32 - 16 ) ;
	out[9] |= in[17] << 1 ;
	out[9] |= in[18] << 18 ;

	out[10] = in[18] >> ( 32 - 18 ) ;
	out[10] |= in[19] << 3 ;
	out[10] |= in[20] << 20 ;

	out[11] = in[20] >> ( 32 - 20 ) ;
	out[11] |= in[21] << 5 ;
	out[11] |= in[22] << 22 ;

	out[12] = in[22] >> ( 32 - 22 ) ;
	out[12] |= in[23] << 7 ;
	out[12] |= in[24] << 24 ;

	out[13] = in[24] >> ( 32 - 24 ) ;
	out[13] |= in[25] << 9 ;
	out[13] |= in[26] << 26 ;

	out[14] = in[26] >> ( 32 - 26 ) ;
	out[14] |= in[27] << 11 ;
	out[14] |= in[28] << 28 ;

	out[15] = in[28] >> ( 32 - 28 ) ;
	out[15] |= in[29] << 13 ;
	out[15] |= in[30] << 30 ;

	out[16] = in[30] >> ( 32 - 30 ) ;
	out[16] |= in[31] << 15 ;

	out[17] = in[32] << 0 ;
	out[17] |= in[33] << 17 ;

	out[18] = in[33] >> ( 32 - 17 ) ;
	out[18] |= in[34] << 2 ;
	out[18] |= in[35] << 19 ;

	out[19] = in[35] >> ( 32 - 19 ) ;
	out[19] |= in[36] << 4 ;
	out[19] |= in[37] << 21 ;

	out[20] = in[37] >> ( 32 - 21 ) ;
	out[20] |= in[38] << 6 ;
	out[20] |= in[39] << 23 ;

	out[21] = in[39] >> ( 32 - 23 ) ;
	out[21] |= in[40] << 8 ;
	out[21] |= in[41] << 25 ;

	out[22] = in[41] >> ( 32 - 25 ) ;
	out[22] |= in[42] << 10 ;
	out[22] |= in[43] << 27 ;

	out[23] = in[43] >> ( 32 - 27 ) ;
	out[23] |= in[44] << 12 ;
	out[23] |= in[45] << 29 ;

	out[24] = in[45] >> ( 32 - 29 ) ;
	out[24] |= in[46] << 14 ;
	out[24] |= in[47] << 31 ;

	out[25] = in[47] >> ( 32 - 31 ) ;
	out[25] |= in[48] << 16 ;

	out[26] = in[48] >> ( 32 - 16 ) ;
	out[26] |= in[49] << 1 ;
	out[26] |= in[50] << 18 ;

	out[27] = in[50] >> ( 32 - 18 ) ;
	out[27] |= in[51] << 3 ;
	out[27] |= in[52] << 20 ;

	out[28] = in[52] >> ( 32 - 20 ) ;
	out[28] |= in[53] << 5 ;
	out[28] |= in[54] << 22 ;

	out[29] = in[54] >> ( 32 - 22 ) ;
	out[29] |= in[55] << 7 ;
	out[29] |= in[56] << 24 ;

	out[30] = in[56] >> ( 32 - 24 ) ;
	out[30] |= in[57] << 9 ;
	out[30] |= in[58] << 26 ;

	out[31] = in[58] >> ( 32 - 26 ) ;
	out[31] |= in[59] << 11 ;
	out[31] |= in[60] << 28 ;

	out[32] = in[60] >> ( 32 - 28 ) ;
	out[32] |= in[61] << 13 ;
	out[32] |= in[62] << 30 ;

	out[33] = in[62] >> ( 32 - 30 ) ;
	out[33] |= in[63] << 15 ;
}


// 18-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c18(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 18 ;

	out[1] = in[1] >> ( 32 - 18 ) ;
	out[1] |= in[2] << 4 ;
	out[1] |= in[3] << 22 ;

	out[2] = in[3] >> ( 32 - 22 ) ;
	out[2] |= in[4] << 8 ;
	out[2] |= in[5] << 26 ;

	out[3] = in[5] >> ( 32 - 26 ) ;
	out[3] |= in[6] << 12 ;
	out[3] |= in[7] << 30 ;

	out[4] = in[7] >> ( 32 - 30 ) ;
	out[4] |= in[8] << 16 ;

	out[5] = in[8] >> ( 32 - 16 ) ;
	out[5] |= in[9] << 2 ;
	out[5] |= in[10] << 20 ;

	out[6] = in[10] >> ( 32 - 20 ) ;
	out[6] |= in[11] << 6 ;
	out[6] |= in[12] << 24 ;

	out[7] = in[12] >> ( 32 - 24 ) ;
	out[7] |= in[13] << 10 ;
	out[7] |= in[14] << 28 ;

	out[8] = in[14] >> ( 32 - 28 ) ;
	out[8] |= in[15] << 14 ;

	out[9] = in[16] << 0 ;
	out[9] |= in[17] << 18 ;

	out[10] = in[17] >> ( 32 - 18 ) ;
	out[10] |= in[18] << 4 ;
	out[10] |= in[19] << 22 ;

	out[11] = in[19] >> ( 32 - 22 ) ;
	out[11] |= in[20] << 8 ;
	out[11] |= in[21] << 26 ;

	out[12] = in[21] >> ( 32 - 26 ) ;
	out[12] |= in[22] << 12 ;
	out[12] |= in[23] << 30 ;

	out[13] = in[23] >> ( 32 - 30 ) ;
	out[13] |= in[24] << 16 ;

	out[14] = in[24] >> ( 32 - 16 ) ;
	out[14] |= in[25] << 2 ;
	out[14] |= in[26] << 20 ;

	out[15] = in[26] >> ( 32 - 20 ) ;
	out[15] |= in[27] << 6 ;
	out[15] |= in[28] << 24 ;

	out[16] = in[28] >> ( 32 - 24 ) ;
	out[16] |= in[29] << 10 ;
	out[16] |= in[30] << 28 ;

	out[17] = in[30] >> ( 32 - 28 ) ;
	out[17] |= in[31] << 14 ;

	out[18] = in[32] << 0 ;
	out[18] |= in[33] << 18 ;

	out[19] = in[33] >> ( 32 - 18 ) ;
	out[19] |= in[34] << 4 ;
	out[19] |= in[35] << 22 ;

	out[20] = in[35] >> ( 32 - 22 ) ;
	out[20] |= in[36] << 8 ;
	out[20] |= in[37] << 26 ;

	out[21] = in[37] >> ( 32 - 26 ) ;
	out[21] |= in[38] << 12 ;
	out[21] |= in[39] << 30 ;

	out[22] = in[39] >> ( 32 - 30 ) ;
	out[22] |= in[40] << 16 ;

	out[23] = in[40] >> ( 32 - 16 ) ;
	out[23] |= in[41] << 2 ;
	out[23] |= in[42] << 20 ;

	out[24] = in[42] >> ( 32 - 20 ) ;
	out[24] |= in[43] << 6 ;
	out[24] |= in[44] << 24 ;

	out[25] = in[44] >> ( 32 - 24 ) ;
	out[25] |= in[45] << 10 ;
	out[25] |= in[46] << 28 ;

	out[26] = in[46] >> ( 32 - 28 ) ;
	out[26] |= in[47] << 14 ;

	out[27] = in[48] << 0 ;
	out[27] |= in[49] << 18 ;

	out[28] = in[49] >> ( 32 - 18 ) ;
	out[28] |= in[50] << 4 ;
	out[28] |= in[51] << 22 ;

	out[29] = in[51] >> ( 32 - 22 ) ;
	out[29] |= in[52] << 8 ;
	out[29] |= in[53] << 26 ;

	out[30] = in[53] >> ( 32 - 26 ) ;
	out[30] |= in[54] << 12 ;
	out[30] |= in[55] << 30 ;

	out[31] = in[55] >> ( 32 - 30 ) ;
	out[31] |= in[56] << 16 ;

	out[32] = in[56] >> ( 32 - 16 ) ;
	out[32] |= in[57] << 2 ;
	out[32] |= in[58] << 20 ;

	out[33] = in[58] >> ( 32 - 20 ) ;
	out[33] |= in[59] << 6 ;
	out[33] |= in[60] << 24 ;

	out[34] = in[60] >> ( 32 - 24 ) ;
	out[34] |= in[61] << 10 ;
	out[34] |= in[62] << 28 ;

	out[35] = in[62] >> ( 32 - 28 ) ;
	out[35] |= in[63] << 14 ;
}


// 19-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c19(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 19 ;

	out[1] = in[1] >> ( 32 - 19 ) ;
	out[1] |= in[2] << 6 ;
	out[1] |= in[3] << 25 ;

	out[2] = in[3] >> ( 32 - 25 ) ;
	out[2] |= in[4] << 12 ;
	out[2] |= in[5] << 31 ;

	out[3] = in[5] >> ( 32 - 31 ) ;
	out[3] |= in[6] << 18 ;

	out[4] = in[6] >> ( 32 - 18 ) ;
	out[4] |= in[7] << 5 ;
	out[4] |= in[8] << 24 ;

	out[5] = in[8] >> ( 32 - 24 ) ;
	out[5] |= in[9] << 11 ;
	out[5] |= in[10] << 30 ;

	out[6] = in[10] >> ( 32 - 30 ) ;
	out[6] |= in[11] << 17 ;

	out[7] = in[11] >> ( 32 - 17 ) ;
	out[7] |= in[12] << 4 ;
	out[7] |= in[13] << 23 ;

	out[8] = in[13] >> ( 32 - 23 ) ;
	out[8] |= in[14] << 10 ;
	out[8] |= in[15] << 29 ;

	out[9] = in[15] >> ( 32 - 29 ) ;
	out[9] |= in[16] << 16 ;

	out[10] = in[16] >> ( 32 - 16 ) ;
	out[10] |= in[17] << 3 ;
	out[10] |= in[18] << 22 ;

	out[11] = in[18] >> ( 32 - 22 ) ;
	out[11] |= in[19] << 9 ;
	out[11] |= in[20] << 28 ;

	out[12] = in[20] >> ( 32 - 28 ) ;
	out[12] |= in[21] << 15 ;

	out[13] = in[21] >> ( 32 - 15 ) ;
	out[13] |= in[22] << 2 ;
	out[13] |= in[23] << 21 ;

	out[14] = in[23] >> ( 32 - 21 ) ;
	out[14] |= in[24] << 8 ;
	out[14] |= in[25] << 27 ;

	out[15] = in[25] >> ( 32 - 27 ) ;
	out[15] |= in[26] << 14 ;

	out[16] = in[26] >> ( 32 - 14 ) ;
	out[16] |= in[27] << 1 ;
	out[16] |= in[28] << 20 ;

	out[17] = in[28] >> ( 32 - 20 ) ;
	out[17] |= in[29] << 7 ;
	out[17] |= in[30] << 26 ;

	out[18] = in[30] >> ( 32 - 26 ) ;
	out[18] |= in[31] << 13 ;

	out[19] = in[32] << 0 ;
	out[19] |= in[33] << 19 ;

	out[20] = in[33] >> ( 32 - 19 ) ;
	out[20] |= in[34] << 6 ;
	out[20] |= in[35] << 25 ;

	out[21] = in[35] >> ( 32 - 25 ) ;
	out[21] |= in[36] << 12 ;
	out[21] |= in[37] << 31 ;

	out[22] = in[37] >> ( 32 - 31 ) ;
	out[22] |= in[38] << 18 ;

	out[23] = in[38] >> ( 32 - 18 ) ;
	out[23] |= in[39] << 5 ;
	out[23] |= in[40] << 24 ;

	out[24] = in[40] >> ( 32 - 24 ) ;
	out[24] |= in[41] << 11 ;
	out[24] |= in[42] << 30 ;

	out[25] = in[42] >> ( 32 - 30 ) ;
	out[25] |= in[43] << 17 ;

	out[26] = in[43] >> ( 32 - 17 ) ;
	out[26] |= in[44] << 4 ;
	out[26] |= in[45] << 23 ;

	out[27] = in[45] >> ( 32 - 23 ) ;
	out[27] |= in[46] << 10 ;
	out[27] |= in[47] << 29 ;

	out[28] = in[47] >> ( 32 - 29 ) ;
	out[28] |= in[48] << 16 ;

	out[29] = in[48] >> ( 32 - 16 ) ;
	out[29] |= in[49] << 3 ;
	out[29] |= in[50] << 22 ;

	out[30] = in[50] >> ( 32 - 22 ) ;
	out[30] |= in[51] << 9 ;
	out[30] |= in[52] << 28 ;

	out[31] = in[52] >> ( 32 - 28 ) ;
	out[31] |= in[53] << 15 ;

	out[32] = in[53] >> ( 32 - 15 ) ;
	out[32] |= in[54] << 2 ;
	out[32] |= in[55] << 21 ;

	out[33] = in[55] >> ( 32 - 21 ) ;
	out[33] |= in[56] << 8 ;
	out[33] |= in[57] << 27 ;

	out[34] = in[57] >> ( 32 - 27 ) ;
	out[34] |= in[58] << 14 ;

	out[35] = in[58] >> ( 32 - 14 ) ;
	out[35] |= in[59] << 1 ;
	out[35] |= in[60] << 20 ;

	out[36] = in[60] >> ( 32 - 20 ) ;
	out[36] |= in[61] << 7 ;
	out[36] |= in[62] << 26 ;

	out[37] = in[62] >> ( 32 - 26 ) ;
	out[37] |= in[63] << 13 ;
}


// 20-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c20(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 20 ;

	out[1] = in[1] >> ( 32 - 20 ) ;
	out[1] |= in[2] << 8 ;
	out[1] |= in[3] << 28 ;

	out[2] = in[3] >> ( 32 - 28 ) ;
	out[2] |= in[4] << 16 ;

	out[3] = in[4] >> ( 32 - 16 ) ;
	out[3] |= in[5] << 4 ;
	out[3] |= in[6] << 24 ;

	out[4] = in[6] >> ( 32 - 24 ) ;
	out[4] |= in[7] << 12 ;

	out[5] = in[8] << 0 ;
	out[5] |= in[9] << 20 ;

	out[6] = in[9] >> ( 32 - 20 ) ;
	out[6] |= in[10] << 8 ;
	out[6] |= in[11] << 28 ;

	out[7] = in[11] >> ( 32 - 28 ) ;
	out[7] |= in[12] << 16 ;

	out[8] = in[12] >> ( 32 - 16 ) ;
	out[8] |= in[13] << 4 ;
	out[8] |= in[14] << 24 ;

	out[9] = in[14] >> ( 32 - 24 ) ;
	out[9] |= in[15] << 12 ;

	out[10] = in[16] << 0 ;
	out[10] |= in[17] << 20 ;

	out[11] = in[17] >> ( 32 - 20 ) ;
	out[11] |= in[18] << 8 ;
	out[11] |= in[19] << 28 ;

	out[12] = in[19] >> ( 32 - 28 ) ;
	out[12] |= in[20] << 16 ;

	out[13] = in[20] >> ( 32 - 16 ) ;
	out[13] |= in[21] << 4 ;
	out[13] |= in[22] << 24 ;

	out[14] = in[22] >> ( 32 - 24 ) ;
	out[14] |= in[23] << 12 ;

	out[15] = in[24] << 0 ;
	out[15] |= in[25] << 20 ;

	out[16] = in[25] >> ( 32 - 20 ) ;
	out[16] |= in[26] << 8 ;
	out[16] |= in[27] << 28 ;

	out[17] = in[27] >> ( 32 - 28 ) ;
	out[17] |= in[28] << 16 ;

	out[18] = in[28] >> ( 32 - 16 ) ;
	out[18] |= in[29] << 4 ;
	out[18] |= in[30] << 24 ;

	out[19] = in[30] >> ( 32 - 24 ) ;
	out[19] |= in[31] << 12 ;

	out[20] = in[32] << 0 ;
	out[20] |= in[33] << 20 ;

	out[21] = in[33] >> ( 32 - 20 ) ;
	out[21] |= in[34] << 8 ;
	out[21] |= in[35] << 28 ;

	out[22] = in[35] >> ( 32 - 28 ) ;
	out[22] |= in[36] << 16 ;

	out[23] = in[36] >> ( 32 - 16 ) ;
	out[23] |= in[37] << 4 ;
	out[23] |= in[38] << 24 ;

	out[24] = in[38] >> ( 32 - 24 ) ;
	out[24] |= in[39] << 12 ;

	out[25] = in[40] << 0 ;
	out[25] |= in[41] << 20 ;

	out[26] = in[41] >> ( 32 - 20 ) ;
	out[26] |= in[42] << 8 ;
	out[26] |= in[43] << 28 ;

	out[27] = in[43] >> ( 32 - 28 ) ;
	out[27] |= in[44] << 16 ;

	out[28] = in[44] >> ( 32 - 16 ) ;
	out[28] |= in[45] << 4 ;
	out[28] |= in[46] << 24 ;

	out[29] = in[46] >> ( 32 - 24 ) ;
	out[29] |= in[47] << 12 ;

	out[30] = in[48] << 0 ;
	out[30] |= in[49] << 20 ;

	out[31] = in[49] >> ( 32 - 20 ) ;
	out[31] |= in[50] << 8 ;
	out[31] |= in[51] << 28 ;

	out[32] = in[51] >> ( 32 - 28 ) ;
	out[32] |= in[52] << 16 ;

	out[33] = in[52] >> ( 32 - 16 ) ;
	out[33] |= in[53] << 4 ;
	out[33] |= in[54] << 24 ;

	out[34] = in[54] >> ( 32 - 24 ) ;
	out[34] |= in[55] << 12 ;

	out[35] = in[56] << 0 ;
	out[35] |= in[57] << 20 ;

	out[36] = in[57] >> ( 32 - 20 ) ;
	out[36] |= in[58] << 8 ;
	out[36] |= in[59] << 28 ;

	out[37] = in[59] >> ( 32 - 28 ) ;
	out[37] |= in[60] << 16 ;

	out[38] = in[60] >> ( 32 - 16 ) ;
	out[38] |= in[61] << 4 ;
	out[38] |= in[62] << 24 ;

	out[39] = in[62] >> ( 32 - 24 ) ;
	out[39] |= in[63] << 12 ;
}


// 21-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c21(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 21 ;

	out[1] = in[1] >> ( 32 - 21 ) ;
	out[1] |= in[2] << 10 ;
	out[1] |= in[3] << 31 ;

	out[2] = in[3] >> ( 32 - 31 ) ;
	out[2] |= in[4] << 20 ;

	out[3] = in[4] >> ( 32 - 20 ) ;
	out[3] |= in[5] << 9 ;
	out[3] |= in[6] << 30 ;

	out[4] = in[6] >> ( 32 - 30 ) ;
	out[4] |= in[7] << 19 ;

	out[5] = in[7] >> ( 32 - 19 ) ;
	out[5] |= in[8] << 8 ;
	out[5] |= in[9] << 29 ;

	out[6] = in[9] >> ( 32 - 29 ) ;
	out[6] |= in[10] << 18 ;

	out[7] = in[10] >> ( 32 - 18 ) ;
	out[7] |= in[11] << 7 ;
	out[7] |= in[12] << 28 ;

	out[8] = in[12] >> ( 32 - 28 ) ;
	out[8] |= in[13] << 17 ;

	out[9] = in[13] >> ( 32 - 17 ) ;
	out[9] |= in[14] << 6 ;
	out[9] |= in[15] << 27 ;

	out[10] = in[15] >> ( 32 - 27 ) ;
	out[10] |= in[16] << 16 ;

	out[11] = in[16] >> ( 32 - 16 ) ;
	out[11] |= in[17] << 5 ;
	out[11] |= in[18] << 26 ;

	out[12] = in[18] >> ( 32 - 26 ) ;
	out[12] |= in[19] << 15 ;

	out[13] = in[19] >> ( 32 - 15 ) ;
	out[13] |= in[20] << 4 ;
	out[13] |= in[21] << 25 ;

	out[14] = in[21] >> ( 32 - 25 ) ;
	out[14] |= in[22] << 14 ;

	out[15] = in[22] >> ( 32 - 14 ) ;
	out[15] |= in[23] << 3 ;
	out[15] |= in[24] << 24 ;

	out[16] = in[24] >> ( 32 - 24 ) ;
	out[16] |= in[25] << 13 ;

	out[17] = in[25] >> ( 32 - 13 ) ;
	out[17] |= in[26] << 2 ;
	out[17] |= in[27] << 23 ;

	out[18] = in[27] >> ( 32 - 23 ) ;
	out[18] |= in[28] << 12 ;

	out[19] = in[28] >> ( 32 - 12 ) ;
	out[19] |= in[29] << 1 ;
	out[19] |= in[30] << 22 ;

	out[20] = in[30] >> ( 32 - 22 ) ;
	out[20] |= in[31] << 11 ;

	out[21] = in[32] << 0 ;
	out[21] |= in[33] << 21 ;

	out[22] = in[33] >> ( 32 - 21 ) ;
	out[22] |= in[34] << 10 ;
	out[22] |= in[35] << 31 ;

	out[23] = in[35] >> ( 32 - 31 ) ;
	out[23] |= in[36] << 20 ;

	out[24] = in[36] >> ( 32 - 20 ) ;
	out[24] |= in[37] << 9 ;
	out[24] |= in[38] << 30 ;

	out[25] = in[38] >> ( 32 - 30 ) ;
	out[25] |= in[39] << 19 ;

	out[26] = in[39] >> ( 32 - 19 ) ;
	out[26] |= in[40] << 8 ;
	out[26] |= in[41] << 29 ;

	out[27] = in[41] >> ( 32 - 29 ) ;
	out[27] |= in[42] << 18 ;

	out[28] = in[42] >> ( 32 - 18 ) ;
	out[28] |= in[43] << 7 ;
	out[28] |= in[44] << 28 ;

	out[29] = in[44] >> ( 32 - 28 ) ;
	out[29] |= in[45] << 17 ;

	out[30] = in[45] >> ( 32 - 17 ) ;
	out[30] |= in[46] << 6 ;
	out[30] |= in[47] << 27 ;

	out[31] = in[47] >> ( 32 - 27 ) ;
	out[31] |= in[48] << 16 ;

	out[32] = in[48] >> ( 32 - 16 ) ;
	out[32] |= in[49] << 5 ;
	out[32] |= in[50] << 26 ;

	out[33] = in[50] >> ( 32 - 26 ) ;
	out[33] |= in[51] << 15 ;

	out[34] = in[51] >> ( 32 - 15 ) ;
	out[34] |= in[52] << 4 ;
	out[34] |= in[53] << 25 ;

	out[35] = in[53] >> ( 32 - 25 ) ;
	out[35] |= in[54] << 14 ;

	out[36] = in[54] >> ( 32 - 14 ) ;
	out[36] |= in[55] << 3 ;
	out[36] |= in[56] << 24 ;

	out[37] = in[56] >> ( 32 - 24 ) ;
	out[37] |= in[57] << 13 ;

	out[38] = in[57] >> ( 32 - 13 ) ;
	out[38] |= in[58] << 2 ;
	out[38] |= in[59] << 23 ;

	out[39] = in[59] >> ( 32 - 23 ) ;
	out[39] |= in[60] << 12 ;

	out[40] = in[60] >> ( 32 - 12 ) ;
	out[40] |= in[61] << 1 ;
	out[40] |= in[62] << 22 ;

	out[41] = in[62] >> ( 32 - 22 ) ;
	out[41] |= in[63] << 11 ;
}


// 22-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c22(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 22 ;

	out[1] = in[1] >> ( 32 - 22 ) ;
	out[1] |= in[2] << 12 ;

	out[2] = in[2] >> ( 32 - 12 ) ;
	out[2] |= in[3] << 2 ;
	out[2] |= in[4] << 24 ;

	out[3] = in[4] >> ( 32 - 24 ) ;
	out[3] |= in[5] << 14 ;

	out[4] = in[5] >> ( 32 - 14 ) ;
	out[4] |= in[6] << 4 ;
	out[4] |= in[7] << 26 ;

	out[5] = in[7] >> ( 32 - 26 ) ;
	out[5] |= in[8] << 16 ;

	out[6] = in[8] >> ( 32 - 16 ) ;
	out[6] |= in[9] << 6 ;
	out[6] |= in[10] << 28 ;

	out[7] = in[10] >> ( 32 - 28 ) ;
	out[7] |= in[11] << 18 ;

	out[8] = in[11] >> ( 32 - 18 ) ;
	out[8] |= in[12] << 8 ;
	out[8] |= in[13] << 30 ;

	out[9] = in[13] >> ( 32 - 30 ) ;
	out[9] |= in[14] << 20 ;

	out[10] = in[14] >> ( 32 - 20 ) ;
	out[10] |= in[15] << 10 ;

	out[11] = in[16] << 0 ;
	out[11] |= in[17] << 22 ;

	out[12] = in[17] >> ( 32 - 22 ) ;
	out[12] |= in[18] << 12 ;

	out[13] = in[18] >> ( 32 - 12 ) ;
	out[13] |= in[19] << 2 ;
	out[13] |= in[20] << 24 ;

	out[14] = in[20] >> ( 32 - 24 ) ;
	out[14] |= in[21] << 14 ;

	out[15] = in[21] >> ( 32 - 14 ) ;
	out[15] |= in[22] << 4 ;
	out[15] |= in[23] << 26 ;

	out[16] = in[23] >> ( 32 - 26 ) ;
	out[16] |= in[24] << 16 ;

	out[17] = in[24] >> ( 32 - 16 ) ;
	out[17] |= in[25] << 6 ;
	out[17] |= in[26] << 28 ;

	out[18] = in[26] >> ( 32 - 28 ) ;
	out[18] |= in[27] << 18 ;

	out[19] = in[27] >> ( 32 - 18 ) ;
	out[19] |= in[28] << 8 ;
	out[19] |= in[29] << 30 ;

	out[20] = in[29] >> ( 32 - 30 ) ;
	out[20] |= in[30] << 20 ;

	out[21] = in[30] >> ( 32 - 20 ) ;
	out[21] |= in[31] << 10 ;

	out[22] = in[32] << 0 ;
	out[22] |= in[33] << 22 ;

	out[23] = in[33] >> ( 32 - 22 ) ;
	out[23] |= in[34] << 12 ;

	out[24] = in[34] >> ( 32 - 12 ) ;
	out[24] |= in[35] << 2 ;
	out[24] |= in[36] << 24 ;

	out[25] = in[36] >> ( 32 - 24 ) ;
	out[25] |= in[37] << 14 ;

	out[26] = in[37] >> ( 32 - 14 ) ;
	out[26] |= in[38] << 4 ;
	out[26] |= in[39] << 26 ;

	out[27] = in[39] >> ( 32 - 26 ) ;
	out[27] |= in[40] << 16 ;

	out[28] = in[40] >> ( 32 - 16 ) ;
	out[28] |= in[41] << 6 ;
	out[28] |= in[42] << 28 ;

	out[29] = in[42] >> ( 32 - 28 ) ;
	out[29] |= in[43] << 18 ;

	out[30] = in[43] >> ( 32 - 18 ) ;
	out[30] |= in[44] << 8 ;
	out[30] |= in[45] << 30 ;

	out[31] = in[45] >> ( 32 - 30 ) ;
	out[31] |= in[46] << 20 ;

	out[32] = in[46] >> ( 32 - 20 ) ;
	out[32] |= in[47] << 10 ;

	out[33] = in[48] << 0 ;
	out[33] |= in[49] << 22 ;

	out[34] = in[49] >> ( 32 - 22 ) ;
	out[34] |= in[50] << 12 ;

	out[35] = in[50] >> ( 32 - 12 ) ;
	out[35] |= in[51] << 2 ;
	out[35] |= in[52] << 24 ;

	out[36] = in[52] >> ( 32 - 24 ) ;
	out[36] |= in[53] << 14 ;

	out[37] = in[53] >> ( 32 - 14 ) ;
	out[37] |= in[54] << 4 ;
	out[37] |= in[55] << 26 ;

	out[38] = in[55] >> ( 32 - 26 ) ;
	out[38] |= in[56] << 16 ;

	out[39] = in[56] >> ( 32 - 16 ) ;
	out[39] |= in[57] << 6 ;
	out[39] |= in[58] << 28 ;

	out[40] = in[58] >> ( 32 - 28 ) ;
	out[40] |= in[59] << 18 ;

	out[41] = in[59] >> ( 32 - 18 ) ;
	out[41] |= in[60] << 8 ;
	out[41] |= in[61] << 30 ;

	out[42] = in[61] >> ( 32 - 30 ) ;
	out[42] |= in[62] << 20 ;

	out[43] = in[62] >> ( 32 - 20 ) ;
	out[43] |= in[63] << 10 ;
}


// 23-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c23(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 23 ;

	out[1] = in[1] >> ( 32 - 23 ) ;
	out[1] |= in[2] << 14 ;

	out[2] = in[2] >> ( 32 - 14 ) ;
	out[2] |= in[3] << 5 ;
	out[2] |= in[4] << 28 ;

	out[3] = in[4] >> ( 32 - 28 ) ;
	out[3] |= in[5] << 19 ;

	out[4] = in[5] >> ( 32 - 19 ) ;
	out[4] |= in[6] << 10 ;

	out[5] = in[6] >> ( 32 - 10 ) ;
	out[5] |= in[7] << 1 ;
	out[5] |= in[8] << 24 ;

	out[6] = in[8] >> ( 32 - 24 ) ;
	out[6] |= in[9] << 15 ;

	out[7] = in[9] >> ( 32 - 15 ) ;
	out[7] |= in[10] << 6 ;
	out[7] |= in[11] << 29 ;

	out[8] = in[11] >> ( 32 - 29 ) ;
	out[8] |= in[12] << 20 ;

	out[9] = in[12] >> ( 32 - 20 ) ;
	out[9] |= in[13] << 11 ;

	out[10] = in[13] >> ( 32 - 11 ) ;
	out[10] |= in[14] << 2 ;
	out[10] |= in[15] << 25 ;

	out[11] = in[15] >> ( 32 - 25 ) ;
	out[11] |= in[16] << 16 ;

	out[12] = in[16] >> ( 32 - 16 ) ;
	out[12] |= in[17] << 7 ;
	out[12] |= in[18] << 30 ;

	out[13] = in[18] >> ( 32 - 30 ) ;
	out[13] |= in[19] << 21 ;

	out[14] = in[19] >> ( 32 - 21 ) ;
	out[14] |= in[20] << 12 ;

	out[15] = in[20] >> ( 32 - 12 ) ;
	out[15] |= in[21] << 3 ;
	out[15] |= in[22] << 26 ;

	out[16] = in[22] >> ( 32 - 26 ) ;
	out[16] |= in[23] << 17 ;

	out[17] = in[23] >> ( 32 - 17 ) ;
	out[17] |= in[24] << 8 ;
	out[17] |= in[25] << 31 ;

	out[18] = in[25] >> ( 32 - 31 ) ;
	out[18] |= in[26] << 22 ;

	out[19] = in[26] >> ( 32 - 22 ) ;
	out[19] |= in[27] << 13 ;

	out[20] = in[27] >> ( 32 - 13 ) ;
	out[20] |= in[28] << 4 ;
	out[20] |= in[29] << 27 ;

	out[21] = in[29] >> ( 32 - 27 ) ;
	out[21] |= in[30] << 18 ;

	out[22] = in[30] >> ( 32 - 18 ) ;
	out[22] |= in[31] << 9 ;

	out[23] = in[32] << 0 ;
	out[23] |= in[33] << 23 ;

	out[24] = in[33] >> ( 32 - 23 ) ;
	out[24] |= in[34] << 14 ;

	out[25] = in[34] >> ( 32 - 14 ) ;
	out[25] |= in[35] << 5 ;
	out[25] |= in[36] << 28 ;

	out[26] = in[36] >> ( 32 - 28 ) ;
	out[26] |= in[37] << 19 ;

	out[27] = in[37] >> ( 32 - 19 ) ;
	out[27] |= in[38] << 10 ;

	out[28] = in[38] >> ( 32 - 10 ) ;
	out[28] |= in[39] << 1 ;
	out[28] |= in[40] << 24 ;

	out[29] = in[40] >> ( 32 - 24 ) ;
	out[29] |= in[41] << 15 ;

	out[30] = in[41] >> ( 32 - 15 ) ;
	out[30] |= in[42] << 6 ;
	out[30] |= in[43] << 29 ;

	out[31] = in[43] >> ( 32 - 29 ) ;
	out[31] |= in[44] << 20 ;

	out[32] = in[44] >> ( 32 - 20 ) ;
	out[32] |= in[45] << 11 ;

	out[33] = in[45] >> ( 32 - 11 ) ;
	out[33] |= in[46] << 2 ;
	out[33] |= in[47] << 25 ;

	out[34] = in[47] >> ( 32 - 25 ) ;
	out[34] |= in[48] << 16 ;

	out[35] = in[48] >> ( 32 - 16 ) ;
	out[35] |= in[49] << 7 ;
	out[35] |= in[50] << 30 ;

	out[36] = in[50] >> ( 32 - 30 ) ;
	out[36] |= in[51] << 21 ;

	out[37] = in[51] >> ( 32 - 21 ) ;
	out[37] |= in[52] << 12 ;

	out[38] = in[52] >> ( 32 - 12 ) ;
	out[38] |= in[53] << 3 ;
	out[38] |= in[54] << 26 ;

	out[39] = in[54] >> ( 32 - 26 ) ;
	out[39] |= in[55] << 17 ;

	out[40] = in[55] >> ( 32 - 17 ) ;
	out[40] |= in[56] << 8 ;
	out[40] |= in[57] << 31 ;

	out[41] = in[57] >> ( 32 - 31 ) ;
	out[41] |= in[58] << 22 ;

	out[42] = in[58] >> ( 32 - 22 ) ;
	out[42] |= in[59] << 13 ;

	out[43] = in[59] >> ( 32 - 13 ) ;
	out[43] |= in[60] << 4 ;
	out[43] |= in[61] << 27 ;

	out[44] = in[61] >> ( 32 - 27 ) ;
	out[44] |= in[62] << 18 ;

	out[45] = in[62] >> ( 32 - 18 ) ;
	out[45] |= in[63] << 9 ;
}


// 24-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c24(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 24 ;

	out[1] = in[1] >> ( 32 - 24 ) ;
	out[1] |= in[2] << 16 ;

	out[2] = in[2] >> ( 32 - 16 ) ;
	out[2] |= in[3] << 8 ;

	out[3] = in[4] << 0 ;
	out[3] |= in[5] << 24 ;

	out[4] = in[5] >> ( 32 - 24 ) ;
	out[4] |= in[6] << 16 ;

	out[5] = in[6] >> ( 32 - 16 ) ;
	out[5] |= in[7] << 8 ;

	out[6] = in[8] << 0 ;
	out[6] |= in[9] << 24 ;

	out[7] = in[9] >> ( 32 - 24 ) ;
	out[7] |= in[10] << 16 ;

	out[8] = in[10] >> ( 32 - 16 ) ;
	out[8] |= in[11] << 8 ;

	out[9] = in[12] << 0 ;
	out[9] |= in[13] << 24 ;

	out[10] = in[13] >> ( 32 - 24 ) ;
	out[10] |= in[14] << 16 ;

	out[11] = in[14] >> ( 32 - 16 ) ;
	out[11] |= in[15] << 8 ;

	out[12] = in[16] << 0 ;
	out[12] |= in[17] << 24 ;

	out[13] = in[17] >> ( 32 - 24 ) ;
	out[13] |= in[18] << 16 ;

	out[14] = in[18] >> ( 32 - 16 ) ;
	out[14] |= in[19] << 8 ;

	out[15] = in[20] << 0 ;
	out[15] |= in[21] << 24 ;

	out[16] = in[21] >> ( 32 - 24 ) ;
	out[16] |= in[22] << 16 ;

	out[17] = in[22] >> ( 32 - 16 ) ;
	out[17] |= in[23] << 8 ;

	out[18] = in[24] << 0 ;
	out[18] |= in[25] << 24 ;

	out[19] = in[25] >> ( 32 - 24 ) ;
	out[19] |= in[26] << 16 ;

	out[20] = in[26] >> ( 32 - 16 ) ;
	out[20] |= in[27] << 8 ;

	out[21] = in[28] << 0 ;
	out[21] |= in[29] << 24 ;

	out[22] = in[29] >> ( 32 - 24 ) ;
	out[22] |= in[30] << 16 ;

	out[23] = in[30] >> ( 32 - 16 ) ;
	out[23] |= in[31] << 8 ;

	out[24] = in[32] << 0 ;
	out[24] |= in[33] << 24 ;

	out[25] = in[33] >> ( 32 - 24 ) ;
	out[25] |= in[34] << 16 ;

	out[26] = in[34] >> ( 32 - 16 ) ;
	out[26] |= in[35] << 8 ;

	out[27] = in[36] << 0 ;
	out[27] |= in[37] << 24 ;

	out[28] = in[37] >> ( 32 - 24 ) ;
	out[28] |= in[38] << 16 ;

	out[29] = in[38] >> ( 32 - 16 ) ;
	out[29] |= in[39] << 8 ;

	out[30] = in[40] << 0 ;
	out[30] |= in[41] << 24 ;

	out[31] = in[41] >> ( 32 - 24 ) ;
	out[31] |= in[42] << 16 ;

	out[32] = in[42] >> ( 32 - 16 ) ;
	out[32] |= in[43] << 8 ;

	out[33] = in[44] << 0 ;
	out[33] |= in[45] << 24 ;

	out[34] = in[45] >> ( 32 - 24 ) ;
	out[34] |= in[46] << 16 ;

	out[35] = in[46] >> ( 32 - 16 ) ;
	out[35] |= in[47] << 8 ;

	out[36] = in[48] << 0 ;
	out[36] |= in[49] << 24 ;

	out[37] = in[49] >> ( 32 - 24 ) ;
	out[37] |= in[50] << 16 ;

	out[38] = in[50] >> ( 32 - 16 ) ;
	out[38] |= in[51] << 8 ;

	out[39] = in[52] << 0 ;
	out[39] |= in[53] << 24 ;

	out[40] = in[53] >> ( 32 - 24 ) ;
	out[40] |= in[54] << 16 ;

	out[41] = in[54] >> ( 32 - 16 ) ;
	out[41] |= in[55] << 8 ;

	out[42] = in[56] << 0 ;
	out[42] |= in[57] << 24 ;

	out[43] = in[57] >> ( 32 - 24 ) ;
	out[43] |= in[58] << 16 ;

	out[44] = in[58] >> ( 32 - 16 ) ;
	out[44] |= in[59] << 8 ;

	out[45] = in[60] << 0 ;
	out[45] |= in[61] << 24 ;

	out[46] = in[61] >> ( 32 - 24 ) ;
	out[46] |= in[62] << 16 ;

	out[47] = in[62] >> ( 32 - 16 ) ;
	out[47] |= in[63] << 8 ;
}


// 25-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c25(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 25 ;

	out[1] = in[1] >> ( 32 - 25 ) ;
	out[1] |= in[2] << 18 ;

	out[2] = in[2] >> ( 32 - 18 ) ;
	out[2] |= in[3] << 11 ;

	out[3] = in[3] >> ( 32 - 11 ) ;
	out[3] |= in[4] << 4 ;
	out[3] |= in[5] << 29 ;

	out[4] = in[5] >> ( 32 - 29 ) ;
	out[4] |= in[6] << 22 ;

	out[5] = in[6] >> ( 32 - 22 ) ;
	out[5] |= in[7] << 15 ;

	out[6] = in[7] >> ( 32 - 15 ) ;
	out[6] |= in[8] << 8 ;

	out[7] = in[8] >> ( 32 - 8 ) ;
	out[7] |= in[9] << 1 ;
	out[7] |= in[10] << 26 ;

	out[8] = in[10] >> ( 32 - 26 ) ;
	out[8] |= in[11] << 19 ;

	out[9] = in[11] >> ( 32 - 19 ) ;
	out[9] |= in[12] << 12 ;

	out[10] = in[12] >> ( 32 - 12 ) ;
	out[10] |= in[13] << 5 ;
	out[10] |= in[14] << 30 ;

	out[11] = in[14] >> ( 32 - 30 ) ;
	out[11] |= in[15] << 23 ;

	out[12] = in[15] >> ( 32 - 23 ) ;
	out[12] |= in[16] << 16 ;

	out[13] = in[16] >> ( 32 - 16 ) ;
	out[13] |= in[17] << 9 ;

	out[14] = in[17] >> ( 32 - 9 ) ;
	out[14] |= in[18] << 2 ;
	out[14] |= in[19] << 27 ;

	out[15] = in[19] >> ( 32 - 27 ) ;
	out[15] |= in[20] << 20 ;

	out[16] = in[20] >> ( 32 - 20 ) ;
	out[16] |= in[21] << 13 ;

	out[17] = in[21] >> ( 32 - 13 ) ;
	out[17] |= in[22] << 6 ;
	out[17] |= in[23] << 31 ;

	out[18] = in[23] >> ( 32 - 31 ) ;
	out[18] |= in[24] << 24 ;

	out[19] = in[24] >> ( 32 - 24 ) ;
	out[19] |= in[25] << 17 ;

	out[20] = in[25] >> ( 32 - 17 ) ;
	out[20] |= in[26] << 10 ;

	out[21] = in[26] >> ( 32 - 10 ) ;
	out[21] |= in[27] << 3 ;
	out[21] |= in[28] << 28 ;

	out[22] = in[28] >> ( 32 - 28 ) ;
	out[22] |= in[29] << 21 ;

	out[23] = in[29] >> ( 32 - 21 ) ;
	out[23] |= in[30] << 14 ;

	out[24] = in[30] >> ( 32 - 14 ) ;
	out[24] |= in[31] << 7 ;

	out[25] = in[32] << 0 ;
	out[25] |= in[33] << 25 ;

	out[26] = in[33] >> ( 32 - 25 ) ;
	out[26] |= in[34] << 18 ;

	out[27] = in[34] >> ( 32 - 18 ) ;
	out[27] |= in[35] << 11 ;

	out[28] = in[35] >> ( 32 - 11 ) ;
	out[28] |= in[36] << 4 ;
	out[28] |= in[37] << 29 ;

	out[29] = in[37] >> ( 32 - 29 ) ;
	out[29] |= in[38] << 22 ;

	out[30] = in[38] >> ( 32 - 22 ) ;
	out[30] |= in[39] << 15 ;

	out[31] = in[39] >> ( 32 - 15 ) ;
	out[31] |= in[40] << 8 ;

	out[32] = in[40] >> ( 32 - 8 ) ;
	out[32] |= in[41] << 1 ;
	out[32] |= in[42] << 26 ;

	out[33] = in[42] >> ( 32 - 26 ) ;
	out[33] |= in[43] << 19 ;

	out[34] = in[43] >> ( 32 - 19 ) ;
	out[34] |= in[44] << 12 ;

	out[35] = in[44] >> ( 32 - 12 ) ;
	out[35] |= in[45] << 5 ;
	out[35] |= in[46] << 30 ;

	out[36] = in[46] >> ( 32 - 30 ) ;
	out[36] |= in[47] << 23 ;

	out[37] = in[47] >> ( 32 - 23 ) ;
	out[37] |= in[48] << 16 ;

	out[38] = in[48] >> ( 32 - 16 ) ;
	out[38] |= in[49] << 9 ;

	out[39] = in[49] >> ( 32 - 9 ) ;
	out[39] |= in[50] << 2 ;
	out[39] |= in[51] << 27 ;

	out[40] = in[51] >> ( 32 - 27 ) ;
	out[40] |= in[52] << 20 ;

	out[41] = in[52] >> ( 32 - 20 ) ;
	out[41] |= in[53] << 13 ;

	out[42] = in[53] >> ( 32 - 13 ) ;
	out[42] |= in[54] << 6 ;
	out[42] |= in[55] << 31 ;

	out[43] = in[55] >> ( 32 - 31 ) ;
	out[43] |= in[56] << 24 ;

	out[44] = in[56] >> ( 32 - 24 ) ;
	out[44] |= in[57] << 17 ;

	out[45] = in[57] >> ( 32 - 17 ) ;
	out[45] |= in[58] << 10 ;

	out[46] = in[58] >> ( 32 - 10 ) ;
	out[46] |= in[59] << 3 ;
	out[46] |= in[60] << 28 ;

	out[47] = in[60] >> ( 32 - 28 ) ;
	out[47] |= in[61] << 21 ;

	out[48] = in[61] >> ( 32 - 21 ) ;
	out[48] |= in[62] << 14 ;

	out[49] = in[62] >> ( 32 - 14 ) ;
	out[49] |= in[63] << 7 ;
}


// 26-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c26(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 26 ;

	out[1] = in[1] >> ( 32 - 26 ) ;
	out[1] |= in[2] << 20 ;

	out[2] = in[2] >> ( 32 - 20 ) ;
	out[2] |= in[3] << 14 ;

	out[3] = in[3] >> ( 32 - 14 ) ;
	out[3] |= in[4] << 8 ;

	out[4] = in[4] >> ( 32 - 8 ) ;
	out[4] |= in[5] << 2 ;
	out[4] |= in[6] << 28 ;

	out[5] = in[6] >> ( 32 - 28 ) ;
	out[5] |= in[7] << 22 ;

	out[6] = in[7] >> ( 32 - 22 ) ;
	out[6] |= in[8] << 16 ;

	out[7] = in[8] >> ( 32 - 16 ) ;
	out[7] |= in[9] << 10 ;

	out[8] = in[9] >> ( 32 - 10 ) ;
	out[8] |= in[10] << 4 ;
	out[8] |= in[11] << 30 ;

	out[9] = in[11] >> ( 32 - 30 ) ;
	out[9] |= in[12] << 24 ;

	out[10] = in[12] >> ( 32 - 24 ) ;
	out[10] |= in[13] << 18 ;

	out[11] = in[13] >> ( 32 - 18 ) ;
	out[11] |= in[14] << 12 ;

	out[12] = in[14] >> ( 32 - 12 ) ;
	out[12] |= in[15] << 6 ;

	out[13] = in[16] << 0 ;
	out[13] |= in[17] << 26 ;

	out[14] = in[17] >> ( 32 - 26 ) ;
	out[14] |= in[18] << 20 ;

	out[15] = in[18] >> ( 32 - 20 ) ;
	out[15] |= in[19] << 14 ;

	out[16] = in[19] >> ( 32 - 14 ) ;
	out[16] |= in[20] << 8 ;

	out[17] = in[20] >> ( 32 - 8 ) ;
	out[17] |= in[21] << 2 ;
	out[17] |= in[22] << 28 ;

	out[18] = in[22] >> ( 32 - 28 ) ;
	out[18] |= in[23] << 22 ;

	out[19] = in[23] >> ( 32 - 22 ) ;
	out[19] |= in[24] << 16 ;

	out[20] = in[24] >> ( 32 - 16 ) ;
	out[20] |= in[25] << 10 ;

	out[21] = in[25] >> ( 32 - 10 ) ;
	out[21] |= in[26] << 4 ;
	out[21] |= in[27] << 30 ;

	out[22] = in[27] >> ( 32 - 30 ) ;
	out[22] |= in[28] << 24 ;

	out[23] = in[28] >> ( 32 - 24 ) ;
	out[23] |= in[29] << 18 ;

	out[24] = in[29] >> ( 32 - 18 ) ;
	out[24] |= in[30] << 12 ;

	out[25] = in[30] >> ( 32 - 12 ) ;
	out[25] |= in[31] << 6 ;

	out[26] = in[32] << 0 ;
	out[26] |= in[33] << 26 ;

	out[27] = in[33] >> ( 32 - 26 ) ;
	out[27] |= in[34] << 20 ;

	out[28] = in[34] >> ( 32 - 20 ) ;
	out[28] |= in[35] << 14 ;

	out[29] = in[35] >> ( 32 - 14 ) ;
	out[29] |= in[36] << 8 ;

	out[30] = in[36] >> ( 32 - 8 ) ;
	out[30] |= in[37] << 2 ;
	out[30] |= in[38] << 28 ;

	out[31] = in[38] >> ( 32 - 28 ) ;
	out[31] |= in[39] << 22 ;

	out[32] = in[39] >> ( 32 - 22 ) ;
	out[32] |= in[40] << 16 ;

	out[33] = in[40] >> ( 32 - 16 ) ;
	out[33] |= in[41] << 10 ;

	out[34] = in[41] >> ( 32 - 10 ) ;
	out[34] |= in[42] << 4 ;
	out[34] |= in[43] << 30 ;

	out[35] = in[43] >> ( 32 - 30 ) ;
	out[35] |= in[44] << 24 ;

	out[36] = in[44] >> ( 32 - 24 ) ;
	out[36] |= in[45] << 18 ;

	out[37] = in[45] >> ( 32 - 18 ) ;
	out[37] |= in[46] << 12 ;

	out[38] = in[46] >> ( 32 - 12 ) ;
	out[38] |= in[47] << 6 ;

	out[39] = in[48] << 0 ;
	out[39] |= in[49] << 26 ;

	out[40] = in[49] >> ( 32 - 26 ) ;
	out[40] |= in[50] << 20 ;

	out[41] = in[50] >> ( 32 - 20 ) ;
	out[41] |= in[51] << 14 ;

	out[42] = in[51] >> ( 32 - 14 ) ;
	out[42] |= in[52] << 8 ;

	out[43] = in[52] >> ( 32 - 8 ) ;
	out[43] |= in[53] << 2 ;
	out[43] |= in[54] << 28 ;

	out[44] = in[54] >> ( 32 - 28 ) ;
	out[44] |= in[55] << 22 ;

	out[45] = in[55] >> ( 32 - 22 ) ;
	out[45] |= in[56] << 16 ;

	out[46] = in[56] >> ( 32 - 16 ) ;
	out[46] |= in[57] << 10 ;

	out[47] = in[57] >> ( 32 - 10 ) ;
	out[47] |= in[58] << 4 ;
	out[47] |= in[59] << 30 ;

	out[48] = in[59] >> ( 32 - 30 ) ;
	out[48] |= in[60] << 24 ;

	out[49] = in[60] >> ( 32 - 24 ) ;
	out[49] |= in[61] << 18 ;

	out[50] = in[61] >> ( 32 - 18 ) ;
	out[50] |= in[62] << 12 ;

	out[51] = in[62] >> ( 32 - 12 ) ;
	out[51] |= in[63] << 6 ;
}


// 27-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c27(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 27 ;

	out[1] = in[1] >> ( 32 - 27 ) ;
	out[1] |= in[2] << 22 ;

	out[2] = in[2] >> ( 32 - 22 ) ;
	out[2] |= in[3] << 17 ;

	out[3] = in[3] >> ( 32 - 17 ) ;
	out[3] |= in[4] << 12 ;

	out[4] = in[4] >> ( 32 - 12 ) ;
	out[4] |= in[5] << 7 ;

	out[5] = in[5] >> ( 32 - 7 ) ;
	out[5] |= in[6] << 2 ;
	out[5] |= in[7] << 29 ;

	out[6] = in[7] >> ( 32 - 29 ) ;
	out[6] |= in[8] << 24 ;

	out[7] = in[8] >> ( 32 - 24 ) ;
	out[7] |= in[9] << 19 ;

	out[8] = in[9] >> ( 32 - 19 ) ;
	out[8] |= in[10] << 14 ;

	out[9] = in[10] >> ( 32 - 14 ) ;
	out[9] |= in[11] << 9 ;

	out[10] = in[11] >> ( 32 - 9 ) ;
	out[10] |= in[12] << 4 ;
	out[10] |= in[13] << 31 ;

	out[11] = in[13] >> ( 32 - 31 ) ;
	out[11] |= in[14] << 26 ;

	out[12] = in[14] >> ( 32 - 26 ) ;
	out[12] |= in[15] << 21 ;

	out[13] = in[15] >> ( 32 - 21 ) ;
	out[13] |= in[16] << 16 ;

	out[14] = in[16] >> ( 32 - 16 ) ;
	out[14] |= in[17] << 11 ;

	out[15] = in[17] >> ( 32 - 11 ) ;
	out[15] |= in[18] << 6 ;

	out[16] = in[18] >> ( 32 - 6 ) ;
	out[16] |= in[19] << 1 ;
	out[16] |= in[20] << 28 ;

	out[17] = in[20] >> ( 32 - 28 ) ;
	out[17] |= in[21] << 23 ;

	out[18] = in[21] >> ( 32 - 23 ) ;
	out[18] |= in[22] << 18 ;

	out[19] = in[22] >> ( 32 - 18 ) ;
	out[19] |= in[23] << 13 ;

	out[20] = in[23] >> ( 32 - 13 ) ;
	out[20] |= in[24] << 8 ;

	out[21] = in[24] >> ( 32 - 8 ) ;
	out[21] |= in[25] << 3 ;
	out[21] |= in[26] << 30 ;

	out[22] = in[26] >> ( 32 - 30 ) ;
	out[22] |= in[27] << 25 ;

	out[23] = in[27] >> ( 32 - 25 ) ;
	out[23] |= in[28] << 20 ;

	out[24] = in[28] >> ( 32 - 20 ) ;
	out[24] |= in[29] << 15 ;

	out[25] = in[29] >> ( 32 - 15 ) ;
	out[25] |= in[30] << 10 ;

	out[26] = in[30] >> ( 32 - 10 ) ;
	out[26] |= in[31] << 5 ;

	out[27] = in[32] << 0 ;
	out[27] |= in[33] << 27 ;

	out[28] = in[33] >> ( 32 - 27 ) ;
	out[28] |= in[34] << 22 ;

	out[29] = in[34] >> ( 32 - 22 ) ;
	out[29] |= in[35] << 17 ;

	out[30] = in[35] >> ( 32 - 17 ) ;
	out[30] |= in[36] << 12 ;

	out[31] = in[36] >> ( 32 - 12 ) ;
	out[31] |= in[37] << 7 ;

	out[32] = in[37] >> ( 32 - 7 ) ;
	out[32] |= in[38] << 2 ;
	out[32] |= in[39] << 29 ;

	out[33] = in[39] >> ( 32 - 29 ) ;
	out[33] |= in[40] << 24 ;

	out[34] = in[40] >> ( 32 - 24 ) ;
	out[34] |= in[41] << 19 ;

	out[35] = in[41] >> ( 32 - 19 ) ;
	out[35] |= in[42] << 14 ;

	out[36] = in[42] >> ( 32 - 14 ) ;
	out[36] |= in[43] << 9 ;

	out[37] = in[43] >> ( 32 - 9 ) ;
	out[37] |= in[44] << 4 ;
	out[37] |= in[45] << 31 ;

	out[38] = in[45] >> ( 32 - 31 ) ;
	out[38] |= in[46] << 26 ;

	out[39] = in[46] >> ( 32 - 26 ) ;
	out[39] |= in[47] << 21 ;

	out[40] = in[47] >> ( 32 - 21 ) ;
	out[40] |= in[48] << 16 ;

	out[41] = in[48] >> ( 32 - 16 ) ;
	out[41] |= in[49] << 11 ;

	out[42] = in[49] >> ( 32 - 11 ) ;
	out[42] |= in[50] << 6 ;

	out[43] = in[50] >> ( 32 - 6 ) ;
	out[43] |= in[51] << 1 ;
	out[43] |= in[52] << 28 ;

	out[44] = in[52] >> ( 32 - 28 ) ;
	out[44] |= in[53] << 23 ;

	out[45] = in[53] >> ( 32 - 23 ) ;
	out[45] |= in[54] << 18 ;

	out[46] = in[54] >> ( 32 - 18 ) ;
	out[46] |= in[55] << 13 ;

	out[47] = in[55] >> ( 32 - 13 ) ;
	out[47] |= in[56] << 8 ;

	out[48] = in[56] >> ( 32 - 8 ) ;
	out[48] |= in[57] << 3 ;
	out[48] |= in[58] << 30 ;

	out[49] = in[58] >> ( 32 - 30 ) ;
	out[49] |= in[59] << 25 ;

	out[50] = in[59] >> ( 32 - 25 ) ;
	out[50] |= in[60] << 20 ;

	out[51] = in[60] >> ( 32 - 20 ) ;
	out[51] |= in[61] << 15 ;

	out[52] = in[61] >> ( 32 - 15 ) ;
	out[52] |= in[62] << 10 ;

	out[53] = in[62] >> ( 32 - 10 ) ;
	out[53] |= in[63] << 5 ;
}


// 28-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c28(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 28 ;

	out[1] = in[1] >> ( 32 - 28 ) ;
	out[1] |= in[2] << 24 ;

	out[2] = in[2] >> ( 32 - 24 ) ;
	out[2] |= in[3] << 20 ;

	out[3] = in[3] >> ( 32 - 20 ) ;
	out[3] |= in[4] << 16 ;

	out[4] = in[4] >> ( 32 - 16 ) ;
	out[4] |= in[5] << 12 ;

	out[5] = in[5] >> ( 32 - 12 ) ;
	out[5] |= in[6] << 8 ;

	out[6] = in[6] >> ( 32 - 8 ) ;
	out[6] |= in[7] << 4 ;

	out[7] = in[8] << 0 ;
	out[7] |= in[9] << 28 ;

	out[8] = in[9] >> ( 32 - 28 ) ;
	out[8] |= in[10] << 24 ;

	out[9] = in[10] >> ( 32 - 24 ) ;
	out[9] |= in[11] << 20 ;

	out[10] = in[11] >> ( 32 - 20 ) ;
	out[10] |= in[12] << 16 ;

	out[11] = in[12] >> ( 32 - 16 ) ;
	out[11] |= in[13] << 12 ;

	out[12] = in[13] >> ( 32 - 12 ) ;
	out[12] |= in[14] << 8 ;

	out[13] = in[14] >> ( 32 - 8 ) ;
	out[13] |= in[15] << 4 ;

	out[14] = in[16] << 0 ;
	out[14] |= in[17] << 28 ;

	out[15] = in[17] >> ( 32 - 28 ) ;
	out[15] |= in[18] << 24 ;

	out[16] = in[18] >> ( 32 - 24 ) ;
	out[16] |= in[19] << 20 ;

	out[17] = in[19] >> ( 32 - 20 ) ;
	out[17] |= in[20] << 16 ;

	out[18] = in[20] >> ( 32 - 16 ) ;
	out[18] |= in[21] << 12 ;

	out[19] = in[21] >> ( 32 - 12 ) ;
	out[19] |= in[22] << 8 ;

	out[20] = in[22] >> ( 32 - 8 ) ;
	out[20] |= in[23] << 4 ;

	out[21] = in[24] << 0 ;
	out[21] |= in[25] << 28 ;

	out[22] = in[25] >> ( 32 - 28 ) ;
	out[22] |= in[26] << 24 ;

	out[23] = in[26] >> ( 32 - 24 ) ;
	out[23] |= in[27] << 20 ;

	out[24] = in[27] >> ( 32 - 20 ) ;
	out[24] |= in[28] << 16 ;

	out[25] = in[28] >> ( 32 - 16 ) ;
	out[25] |= in[29] << 12 ;

	out[26] = in[29] >> ( 32 - 12 ) ;
	out[26] |= in[30] << 8 ;

	out[27] = in[30] >> ( 32 - 8 ) ;
	out[27] |= in[31] << 4 ;

	out[28] = in[32] << 0 ;
	out[28] |= in[33] << 28 ;

	out[29] = in[33] >> ( 32 - 28 ) ;
	out[29] |= in[34] << 24 ;

	out[30] = in[34] >> ( 32 - 24 ) ;
	out[30] |= in[35] << 20 ;

	out[31] = in[35] >> ( 32 - 20 ) ;
	out[31] |= in[36] << 16 ;

	out[32] = in[36] >> ( 32 - 16 ) ;
	out[32] |= in[37] << 12 ;

	out[33] = in[37] >> ( 32 - 12 ) ;
	out[33] |= in[38] << 8 ;

	out[34] = in[38] >> ( 32 - 8 ) ;
	out[34] |= in[39] << 4 ;

	out[35] = in[40] << 0 ;
	out[35] |= in[41] << 28 ;

	out[36] = in[41] >> ( 32 - 28 ) ;
	out[36] |= in[42] << 24 ;

	out[37] = in[42] >> ( 32 - 24 ) ;
	out[37] |= in[43] << 20 ;

	out[38] = in[43] >> ( 32 - 20 ) ;
	out[38] |= in[44] << 16 ;

	out[39] = in[44] >> ( 32 - 16 ) ;
	out[39] |= in[45] << 12 ;

	out[40] = in[45] >> ( 32 - 12 ) ;
	out[40] |= in[46] << 8 ;

	out[41] = in[46] >> ( 32 - 8 ) ;
	out[41] |= in[47] << 4 ;

	out[42] = in[48] << 0 ;
	out[42] |= in[49] << 28 ;

	out[43] = in[49] >> ( 32 - 28 ) ;
	out[43] |= in[50] << 24 ;

	out[44] = in[50] >> ( 32 - 24 ) ;
	out[44] |= in[51] << 20 ;

	out[45] = in[51] >> ( 32 - 20 ) ;
	out[45] |= in[52] << 16 ;

	out[46] = in[52] >> ( 32 - 16 ) ;
	out[46] |= in[53] << 12 ;

	out[47] = in[53] >> ( 32 - 12 ) ;
	out[47] |= in[54] << 8 ;

	out[48] = in[54] >> ( 32 - 8 ) ;
	out[48] |= in[55] << 4 ;

	out[49] = in[56] << 0 ;
	out[49] |= in[57] << 28 ;

	out[50] = in[57] >> ( 32 - 28 ) ;
	out[50] |= in[58] << 24 ;

	out[51] = in[58] >> ( 32 - 24 ) ;
	out[51] |= in[59] << 20 ;

	out[52] = in[59] >> ( 32 - 20 ) ;
	out[52] |= in[60] << 16 ;

	out[53] = in[60] >> ( 32 - 16 ) ;
	out[53] |= in[61] << 12 ;

	out[54] = in[61] >> ( 32 - 12 ) ;
	out[54] |= in[62] << 8 ;

	out[55] = in[62] >> ( 32 - 8 ) ;
	out[55] |= in[63] << 4 ;
}


// 29-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c29(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 29 ;

	out[1] = in[1] >> ( 32 - 29 ) ;
	out[1] |= in[2] << 26 ;

	out[2] = in[2] >> ( 32 - 26 ) ;
	out[2] |= in[3] << 23 ;

	out[3] = in[3] >> ( 32 - 23 ) ;
	out[3] |= in[4] << 20 ;

	out[4] = in[4] >> ( 32 - 20 ) ;
	out[4] |= in[5] << 17 ;

	out[5] = in[5] >> ( 32 - 17 ) ;
	out[5] |= in[6] << 14 ;

	out[6] = in[6] >> ( 32 - 14 ) ;
	out[6] |= in[7] << 11 ;

	out[7] = in[7] >> ( 32 - 11 ) ;
	out[7] |= in[8] << 8 ;

	out[8] = in[8] >> ( 32 - 8 ) ;
	out[8] |= in[9] << 5 ;

	out[9] = in[9] >> ( 32 - 5 ) ;
	out[9] |= in[10] << 2 ;
	out[9] |= in[11] << 31 ;

	out[10] = in[11] >> ( 32 - 31 ) ;
	out[10] |= in[12] << 28 ;

	out[11] = in[12] >> ( 32 - 28 ) ;
	out[11] |= in[13] << 25 ;

	out[12] = in[13] >> ( 32 - 25 ) ;
	out[12] |= in[14] << 22 ;

	out[13] = in[14] >> ( 32 - 22 ) ;
	out[13] |= in[15] << 19 ;

	out[14] = in[15] >> ( 32 - 19 ) ;
	out[14] |= in[16] << 16 ;

	out[15] = in[16] >> ( 32 - 16 ) ;
	out[15] |= in[17] << 13 ;

	out[16] = in[17] >> ( 32 - 13 ) ;
	out[16] |= in[18] << 10 ;

	out[17] = in[18] >> ( 32 - 10 ) ;
	out[17] |= in[19] << 7 ;

	out[18] = in[19] >> ( 32 - 7 ) ;
	out[18] |= in[20] << 4 ;

	out[19] = in[20] >> ( 32 - 4 ) ;
	out[19] |= in[21] << 1 ;
	out[19] |= in[22] << 30 ;

	out[20] = in[22] >> ( 32 - 30 ) ;
	out[20] |= in[23] << 27 ;

	out[21] = in[23] >> ( 32 - 27 ) ;
	out[21] |= in[24] << 24 ;

	out[22] = in[24] >> ( 32 - 24 ) ;
	out[22] |= in[25] << 21 ;

	out[23] = in[25] >> ( 32 - 21 ) ;
	out[23] |= in[26] << 18 ;

	out[24] = in[26] >> ( 32 - 18 ) ;
	out[24] |= in[27] << 15 ;

	out[25] = in[27] >> ( 32 - 15 ) ;
	out[25] |= in[28] << 12 ;

	out[26] = in[28] >> ( 32 - 12 ) ;
	out[26] |= in[29] << 9 ;

	out[27] = in[29] >> ( 32 - 9 ) ;
	out[27] |= in[30] << 6 ;

	out[28] = in[30] >> ( 32 - 6 ) ;
	out[28] |= in[31] << 3 ;

	out[29] = in[32] << 0 ;
	out[29] |= in[33] << 29 ;

	out[30] = in[33] >> ( 32 - 29 ) ;
	out[30] |= in[34] << 26 ;

	out[31] = in[34] >> ( 32 - 26 ) ;
	out[31] |= in[35] << 23 ;

	out[32] = in[35] >> ( 32 - 23 ) ;
	out[32] |= in[36] << 20 ;

	out[33] = in[36] >> ( 32 - 20 ) ;
	out[33] |= in[37] << 17 ;

	out[34] = in[37] >> ( 32 - 17 ) ;
	out[34] |= in[38] << 14 ;

	out[35] = in[38] >> ( 32 - 14 ) ;
	out[35] |= in[39] << 11 ;

	out[36] = in[39] >> ( 32 - 11 ) ;
	out[36] |= in[40] << 8 ;

	out[37] = in[40] >> ( 32 - 8 ) ;
	out[37] |= in[41] << 5 ;

	out[38] = in[41] >> ( 32 - 5 ) ;
	out[38] |= in[42] << 2 ;
	out[38] |= in[43] << 31 ;

	out[39] = in[43] >> ( 32 - 31 ) ;
	out[39] |= in[44] << 28 ;

	out[40] = in[44] >> ( 32 - 28 ) ;
	out[40] |= in[45] << 25 ;

	out[41] = in[45] >> ( 32 - 25 ) ;
	out[41] |= in[46] << 22 ;

	out[42] = in[46] >> ( 32 - 22 ) ;
	out[42] |= in[47] << 19 ;

	out[43] = in[47] >> ( 32 - 19 ) ;
	out[43] |= in[48] << 16 ;

	out[44] = in[48] >> ( 32 - 16 ) ;
	out[44] |= in[49] << 13 ;

	out[45] = in[49] >> ( 32 - 13 ) ;
	out[45] |= in[50] << 10 ;

	out[46] = in[50] >> ( 32 - 10 ) ;
	out[46] |= in[51] << 7 ;

	out[47] = in[51] >> ( 32 - 7 ) ;
	out[47] |= in[52] << 4 ;

	out[48] = in[52] >> ( 32 - 4 ) ;
	out[48] |= in[53] << 1 ;
	out[48] |= in[54] << 30 ;

	out[49] = in[54] >> ( 32 - 30 ) ;
	out[49] |= in[55] << 27 ;

	out[50] = in[55] >> ( 32 - 27 ) ;
	out[50] |= in[56] << 24 ;

	out[51] = in[56] >> ( 32 - 24 ) ;
	out[51] |= in[57] << 21 ;

	out[52] = in[57] >> ( 32 - 21 ) ;
	out[52] |= in[58] << 18 ;

	out[53] = in[58] >> ( 32 - 18 ) ;
	out[53] |= in[59] << 15 ;

	out[54] = in[59] >> ( 32 - 15 ) ;
	out[54] |= in[60] << 12 ;

	out[55] = in[60] >> ( 32 - 12 ) ;
	out[55] |= in[61] << 9 ;

	out[56] = in[61] >> ( 32 - 9 ) ;
	out[56] |= in[62] << 6 ;

	out[57] = in[62] >> ( 32 - 6 ) ;
	out[57] |= in[63] << 3 ;
}


// 30-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c30(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 30 ;

	out[1] = in[1] >> ( 32 - 30 ) ;
	out[1] |= in[2] << 28 ;

	out[2] = in[2] >> ( 32 - 28 ) ;
	out[2] |= in[3] << 26 ;

	out[3] = in[3] >> ( 32 - 26 ) ;
	out[3] |= in[4] << 24 ;

	out[4] = in[4] >> ( 32 - 24 ) ;
	out[4] |= in[5] << 22 ;

	out[5] = in[5] >> ( 32 - 22 ) ;
	out[5] |= in[6] << 20 ;

	out[6] = in[6] >> ( 32 - 20 ) ;
	out[6] |= in[7] << 18 ;

	out[7] = in[7] >> ( 32 - 18 ) ;
	out[7] |= in[8] << 16 ;

	out[8] = in[8] >> ( 32 - 16 ) ;
	out[8] |= in[9] << 14 ;

	out[9] = in[9] >> ( 32 - 14 ) ;
	out[9] |= in[10] << 12 ;

	out[10] = in[10] >> ( 32 - 12 ) ;
	out[10] |= in[11] << 10 ;

	out[11] = in[11] >> ( 32 - 10 ) ;
	out[11] |= in[12] << 8 ;

	out[12] = in[12] >> ( 32 - 8 ) ;
	out[12] |= in[13] << 6 ;

	out[13] = in[13] >> ( 32 - 6 ) ;
	out[13] |= in[14] << 4 ;

	out[14] = in[14] >> ( 32 - 4 ) ;
	out[14] |= in[15] << 2 ;

	out[15] = in[16] << 0 ;
	out[15] |= in[17] << 30 ;

	out[16] = in[17] >> ( 32 - 30 ) ;
	out[16] |= in[18] << 28 ;

	out[17] = in[18] >> ( 32 - 28 ) ;
	out[17] |= in[19] << 26 ;

	out[18] = in[19] >> ( 32 - 26 ) ;
	out[18] |= in[20] << 24 ;

	out[19] = in[20] >> ( 32 - 24 ) ;
	out[19] |= in[21] << 22 ;

	out[20] = in[21] >> ( 32 - 22 ) ;
	out[20] |= in[22] << 20 ;

	out[21] = in[22] >> ( 32 - 20 ) ;
	out[21] |= in[23] << 18 ;

	out[22] = in[23] >> ( 32 - 18 ) ;
	out[22] |= in[24] << 16 ;

	out[23] = in[24] >> ( 32 - 16 ) ;
	out[23] |= in[25] << 14 ;

	out[24] = in[25] >> ( 32 - 14 ) ;
	out[24] |= in[26] << 12 ;

	out[25] = in[26] >> ( 32 - 12 ) ;
	out[25] |= in[27] << 10 ;

	out[26] = in[27] >> ( 32 - 10 ) ;
	out[26] |= in[28] << 8 ;

	out[27] = in[28] >> ( 32 - 8 ) ;
	out[27] |= in[29] << 6 ;

	out[28] = in[29] >> ( 32 - 6 ) ;
	out[28] |= in[30] << 4 ;

	out[29] = in[30] >> ( 32 - 4 ) ;
	out[29] |= in[31] << 2 ;

	out[30] = in[32] << 0 ;
	out[30] |= in[33] << 30 ;

	out[31] = in[33] >> ( 32 - 30 ) ;
	out[31] |= in[34] << 28 ;

	out[32] = in[34] >> ( 32 - 28 ) ;
	out[32] |= in[35] << 26 ;

	out[33] = in[35] >> ( 32 - 26 ) ;
	out[33] |= in[36] << 24 ;

	out[34] = in[36] >> ( 32 - 24 ) ;
	out[34] |= in[37] << 22 ;

	out[35] = in[37] >> ( 32 - 22 ) ;
	out[35] |= in[38] << 20 ;

	out[36] = in[38] >> ( 32 - 20 ) ;
	out[36] |= in[39] << 18 ;

	out[37] = in[39] >> ( 32 - 18 ) ;
	out[37] |= in[40] << 16 ;

	out[38] = in[40] >> ( 32 - 16 ) ;
	out[38] |= in[41] << 14 ;

	out[39] = in[41] >> ( 32 - 14 ) ;
	out[39] |= in[42] << 12 ;

	out[40] = in[42] >> ( 32 - 12 ) ;
	out[40] |= in[43] << 10 ;

	out[41] = in[43] >> ( 32 - 10 ) ;
	out[41] |= in[44] << 8 ;

	out[42] = in[44] >> ( 32 - 8 ) ;
	out[42] |= in[45] << 6 ;

	out[43] = in[45] >> ( 32 - 6 ) ;
	out[43] |= in[46] << 4 ;

	out[44] = in[46] >> ( 32 - 4 ) ;
	out[44] |= in[47] << 2 ;

	out[45] = in[48] << 0 ;
	out[45] |= in[49] << 30 ;

	out[46] = in[49] >> ( 32 - 30 ) ;
	out[46] |= in[50] << 28 ;

	out[47] = in[50] >> ( 32 - 28 ) ;
	out[47] |= in[51] << 26 ;

	out[48] = in[51] >> ( 32 - 26 ) ;
	out[48] |= in[52] << 24 ;

	out[49] = in[52] >> ( 32 - 24 ) ;
	out[49] |= in[53] << 22 ;

	out[50] = in[53] >> ( 32 - 22 ) ;
	out[50] |= in[54] << 20 ;

	out[51] = in[54] >> ( 32 - 20 ) ;
	out[51] |= in[55] << 18 ;

	out[52] = in[55] >> ( 32 - 18 ) ;
	out[52] |= in[56] << 16 ;

	out[53] = in[56] >> ( 32 - 16 ) ;
	out[53] |= in[57] << 14 ;

	out[54] = in[57] >> ( 32 - 14 ) ;
	out[54] |= in[58] << 12 ;

	out[55] = in[58] >> ( 32 - 12 ) ;
	out[55] |= in[59] << 10 ;

	out[56] = in[59] >> ( 32 - 10 ) ;
	out[56] |= in[60] << 8 ;

	out[57] = in[60] >> ( 32 - 8 ) ;
	out[57] |= in[61] << 6 ;

	out[58] = in[61] >> ( 32 - 6 ) ;
	out[58] |= in[62] << 4 ;

	out[59] = in[62] >> ( 32 - 4 ) ;
	out[59] |= in[63] << 2 ;
}


// 31-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c31(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = in[0] << 0 ;
	out[0] |= in[1] << 31 ;

	out[1] = in[1] >> ( 32 - 31 ) ;
	out[1] |= in[2] << 30 ;

	out[2] = in[2] >> ( 32 - 30 ) ;
	out[2] |= in[3] << 29 ;

	out[3] = in[3] >> ( 32 - 29 ) ;
	out[3] |= in[4] << 28 ;

	out[4] = in[4] >> ( 32 - 28 ) ;
	out[4] |= in[5] << 27 ;

	out[5] = in[5] >> ( 32 - 27 ) ;
	out[5] |= in[6] << 26 ;

	out[6] = in[6] >> ( 32 - 26 ) ;
	out[6] |= in[7] << 25 ;

	out[7] = in[7] >> ( 32 - 25 ) ;
	out[7] |= in[8] << 24 ;

	out[8] = in[8] >> ( 32 - 24 ) ;
	out[8] |= in[9] << 23 ;

	out[9] = in[9] >> ( 32 - 23 ) ;
	out[9] |= in[10] << 22 ;

	out[10] = in[10] >> ( 32 - 22 ) ;
	out[10] |= in[11] << 21 ;

	out[11] = in[11] >> ( 32 - 21 ) ;
	out[11] |= in[12] << 20 ;

	out[12] = in[12] >> ( 32 - 20 ) ;
	out[12] |= in[13] << 19 ;

	out[13] = in[13] >> ( 32 - 19 ) ;
	out[13] |= in[14] << 18 ;

	out[14] = in[14] >> ( 32 - 18 ) ;
	out[14] |= in[15] << 17 ;

	out[15] = in[15] >> ( 32 - 17 ) ;
	out[15] |= in[16] << 16 ;

	out[16] = in[16] >> ( 32 - 16 ) ;
	out[16] |= in[17] << 15 ;

	out[17] = in[17] >> ( 32 - 15 ) ;
	out[17] |= in[18] << 14 ;

	out[18] = in[18] >> ( 32 - 14 ) ;
	out[18] |= in[19] << 13 ;

	out[19] = in[19] >> ( 32 - 13 ) ;
	out[19] |= in[20] << 12 ;

	out[20] = in[20] >> ( 32 - 12 ) ;
	out[20] |= in[21] << 11 ;

	out[21] = in[21] >> ( 32 - 11 ) ;
	out[21] |= in[22] << 10 ;

	out[22] = in[22] >> ( 32 - 10 ) ;
	out[22] |= in[23] << 9 ;

	out[23] = in[23] >> ( 32 - 9 ) ;
	out[23] |= in[24] << 8 ;

	out[24] = in[24] >> ( 32 - 8 ) ;
	out[24] |= in[25] << 7 ;

	out[25] = in[25] >> ( 32 - 7 ) ;
	out[25] |= in[26] << 6 ;

	out[26] = in[26] >> ( 32 - 6 ) ;
	out[26] |= in[27] << 5 ;

	out[27] = in[27] >> ( 32 - 5 ) ;
	out[27] |= in[28] << 4 ;

	out[28] = in[28] >> ( 32 - 4 ) ;
	out[28] |= in[29] << 3 ;

	out[29] = in[29] >> ( 32 - 3 ) ;
	out[29] |= in[30] << 2 ;

	out[30] = in[30] >> ( 32 - 2 ) ;
	out[30] |= in[31] << 1 ;

	out[31] = in[32] << 0 ;
	out[31] |= in[33] << 31 ;

	out[32] = in[33] >> ( 32 - 31 ) ;
	out[32] |= in[34] << 30 ;

	out[33] = in[34] >> ( 32 - 30 ) ;
	out[33] |= in[35] << 29 ;

	out[34] = in[35] >> ( 32 - 29 ) ;
	out[34] |= in[36] << 28 ;

	out[35] = in[36] >> ( 32 - 28 ) ;
	out[35] |= in[37] << 27 ;

	out[36] = in[37] >> ( 32 - 27 ) ;
	out[36] |= in[38] << 26 ;

	out[37] = in[38] >> ( 32 - 26 ) ;
	out[37] |= in[39] << 25 ;

	out[38] = in[39] >> ( 32 - 25 ) ;
	out[38] |= in[40] << 24 ;

	out[39] = in[40] >> ( 32 - 24 ) ;
	out[39] |= in[41] << 23 ;

	out[40] = in[41] >> ( 32 - 23 ) ;
	out[40] |= in[42] << 22 ;

	out[41] = in[42] >> ( 32 - 22 ) ;
	out[41] |= in[43] << 21 ;

	out[42] = in[43] >> ( 32 - 21 ) ;
	out[42] |= in[44] << 20 ;

	out[43] = in[44] >> ( 32 - 20 ) ;
	out[43] |= in[45] << 19 ;

	out[44] = in[45] >> ( 32 - 19 ) ;
	out[44] |= in[46] << 18 ;

	out[45] = in[46] >> ( 32 - 18 ) ;
	out[45] |= in[47] << 17 ;

	out[46] = in[47] >> ( 32 - 17 ) ;
	out[46] |= in[48] << 16 ;

	out[47] = in[48] >> ( 32 - 16 ) ;
	out[47] |= in[49] << 15 ;

	out[48] = in[49] >> ( 32 - 15 ) ;
	out[48] |= in[50] << 14 ;

	out[49] = in[50] >> ( 32 - 14 ) ;
	out[49] |= in[51] << 13 ;

	out[50] = in[51] >> ( 32 - 13 ) ;
	out[50] |= in[52] << 12 ;

	out[51] = in[52] >> ( 32 - 12 ) ;
	out[51] |= in[53] << 11 ;

	out[52] = in[53] >> ( 32 - 11 ) ;
	out[52] |= in[54] << 10 ;

	out[53] = in[54] >> ( 32 - 10 ) ;
	out[53] |= in[55] << 9 ;

	out[54] = in[55] >> ( 32 - 9 ) ;
	out[54] |= in[56] << 8 ;

	out[55] = in[56] >> ( 32 - 8 ) ;
	out[55] |= in[57] << 7 ;

	out[56] = in[57] >> ( 32 - 7 ) ;
	out[56] |= in[58] << 6 ;

	out[57] = in[58] >> ( 32 - 6 ) ;
	out[57] |= in[59] << 5 ;

	out[58] = in[59] >> ( 32 - 5 ) ;
	out[58] |= in[60] << 4 ;

	out[59] = in[60] >> ( 32 - 4 ) ;
	out[59] |= in[61] << 3 ;

	out[60] = in[61] >> ( 32 - 3 ) ;
	out[60] |= in[62] << 2 ;

	out[61] = in[62] >> ( 32 - 2 ) ;
	out[61] |= in[63] << 1 ;
}


// 32-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_packwithoutmask64_c32(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	memcpy(out, in, 64 * sizeof(uint32_t));
}


// 1-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c1(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x01 ) << 0 ;
	out[0] |= ( in[1] & 0x01 ) << 1 ;
	out[0] |= ( in[2] & 0x01 ) << 2 ;
	out[0] |= ( in[3] & 0x01 ) << 3 ;
	out[0] |= ( in[4] & 0x01 ) << 4 ;
	out[0] |= ( in[5] & 0x01 ) << 5 ;
	out[0] |= ( in[6] & 0x01 ) << 6 ;
	out[0] |= ( in[7] & 0x01 ) << 7 ;
	out[0] |= ( in[8] & 0x01 ) << 8 ;
	out[0] |= ( in[9] & 0x01 ) << 9 ;
	out[0] |= ( in[10] & 0x01 ) << 10 ;
	out[0] |= ( in[11] & 0x01 ) << 11 ;
	out[0] |= ( in[12] & 0x01 ) << 12 ;
	out[0] |= ( in[13] & 0x01 ) << 13 ;
	out[0] |= ( in[14] & 0x01 ) << 14 ;
	out[0] |= ( in[15] & 0x01 ) << 15 ;
	out[0] |= ( in[16] & 0x01 ) << 16 ;
	out[0] |= ( in[17] & 0x01 ) << 17 ;
	out[0] |= ( in[18] & 0x01 ) << 18 ;
	out[0] |= ( in[19] & 0x01 ) << 19 ;
	out[0] |= ( in[20] & 0x01 ) << 20 ;
	out[0] |= ( in[21] & 0x01 ) << 21 ;
	out[0] |= ( in[22] & 0x01 ) << 22 ;
	out[0] |= ( in[23] & 0x01 ) << 23 ;
	out[0] |= ( in[24] & 0x01 ) << 24 ;
	out[0] |= ( in[25] & 0x01 ) << 25 ;
	out[0] |= ( in[26] & 0x01 ) << 26 ;
	out[0] |= ( in[27] & 0x01 ) << 27 ;
	out[0] |= ( in[28] & 0x01 ) << 28 ;
	out[0] |= ( in[29] & 0x01 ) << 29 ;
	out[0] |= ( in[30] & 0x01 ) << 30 ;
	out[0] |= in[31] << 31 ;

	out[1] = ( in[32] & 0x01 ) << 0 ;
	out[1] |= ( in[33] & 0x01 ) << 1 ;
	out[1] |= ( in[34] & 0x01 ) << 2 ;
	out[1] |= ( in[35] & 0x01 ) << 3 ;
	out[1] |= ( in[36] & 0x01 ) << 4 ;
	out[1] |= ( in[37] & 0x01 ) << 5 ;
	out[1] |= ( in[38] & 0x01 ) << 6 ;
	out[1] |= ( in[39] & 0x01 ) << 7 ;
	out[1] |= ( in[40] & 0x01 ) << 8 ;
	out[1] |= ( in[41] & 0x01 ) << 9 ;
	out[1] |= ( in[42] & 0x01 ) << 10 ;
	out[1] |= ( in[43] & 0x01 ) << 11 ;
	out[1] |= ( in[44] & 0x01 ) << 12 ;
	out[1] |= ( in[45] & 0x01 ) << 13 ;
	out[1] |= ( in[46] & 0x01 ) << 14 ;
	out[1] |= ( in[47] & 0x01 ) << 15 ;
	out[1] |= ( in[48] & 0x01 ) << 16 ;
	out[1] |= ( in[49] & 0x01 ) << 17 ;
	out[1] |= ( in[50] & 0x01 ) << 18 ;
	out[1] |= ( in[51] & 0x01 ) << 19 ;
	out[1] |= ( in[52] & 0x01 ) << 20 ;
	out[1] |= ( in[53] & 0x01 ) << 21 ;
	out[1] |= ( in[54] & 0x01 ) << 22 ;
	out[1] |= ( in[55] & 0x01 ) << 23 ;
	out[1] |= ( in[56] & 0x01 ) << 24 ;
	out[1] |= ( in[57] & 0x01 ) << 25 ;
	out[1] |= ( in[58] & 0x01 ) << 26 ;
	out[1] |= ( in[59] & 0x01 ) << 27 ;
	out[1] |= ( in[60] & 0x01 ) << 28 ;
	out[1] |= ( in[61] & 0x01 ) << 29 ;
	out[1] |= ( in[62] & 0x01 ) << 30 ;
	out[1] |= in[63] << 31 ;
}


// 2-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c2(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x03 ) << 0 ;
	out[0] |= ( in[1] & 0x03 ) << 2 ;
	out[0] |= ( in[2] & 0x03 ) << 4 ;
	out[0] |= ( in[3] & 0x03 ) << 6 ;
	out[0] |= ( in[4] & 0x03 ) << 8 ;
	out[0] |= ( in[5] & 0x03 ) << 10 ;
	out[0] |= ( in[6] & 0x03 ) << 12 ;
	out[0] |= ( in[7] & 0x03 ) << 14 ;
	out[0] |= ( in[8] & 0x03 ) << 16 ;
	out[0] |= ( in[9] & 0x03 ) << 18 ;
	out[0] |= ( in[10] & 0x03 ) << 20 ;
	out[0] |= ( in[11] & 0x03 ) << 22 ;
	out[0] |= ( in[12] & 0x03 ) << 24 ;
	out[0] |= ( in[13] & 0x03 ) << 26 ;
	out[0] |= ( in[14] & 0x03 ) << 28 ;
	out[0] |= in[15] << 30 ;

	out[1] = ( in[16] & 0x03 ) << 0 ;
	out[1] |= ( in[17] & 0x03 ) << 2 ;
	out[1] |= ( in[18] & 0x03 ) << 4 ;
	out[1] |= ( in[19] & 0x03 ) << 6 ;
	out[1] |= ( in[20] & 0x03 ) << 8 ;
	out[1] |= ( in[21] & 0x03 ) << 10 ;
	out[1] |= ( in[22] & 0x03 ) << 12 ;
	out[1] |= ( in[23] & 0x03 ) << 14 ;
	out[1] |= ( in[24] & 0x03 ) << 16 ;
	out[1] |= ( in[25] & 0x03 ) << 18 ;
	out[1] |= ( in[26] & 0x03 ) << 20 ;
	out[1] |= ( in[27] & 0x03 ) << 22 ;
	out[1] |= ( in[28] & 0x03 ) << 24 ;
	out[1] |= ( in[29] & 0x03 ) << 26 ;
	out[1] |= ( in[30] & 0x03 ) << 28 ;
	out[1] |= in[31] << 30 ;

	out[2] = ( in[32] & 0x03 ) << 0 ;
	out[2] |= ( in[33] & 0x03 ) << 2 ;
	out[2] |= ( in[34] & 0x03 ) << 4 ;
	out[2] |= ( in[35] & 0x03 ) << 6 ;
	out[2] |= ( in[36] & 0x03 ) << 8 ;
	out[2] |= ( in[37] & 0x03 ) << 10 ;
	out[2] |= ( in[38] & 0x03 ) << 12 ;
	out[2] |= ( in[39] & 0x03 ) << 14 ;
	out[2] |= ( in[40] & 0x03 ) << 16 ;
	out[2] |= ( in[41] & 0x03 ) << 18 ;
	out[2] |= ( in[42] & 0x03 ) << 20 ;
	out[2] |= ( in[43] & 0x03 ) << 22 ;
	out[2] |= ( in[44] & 0x03 ) << 24 ;
	out[2] |= ( in[45] & 0x03 ) << 26 ;
	out[2] |= ( in[46] & 0x03 ) << 28 ;
	out[2] |= in[47] << 30 ;

	out[3] = ( in[48] & 0x03 ) << 0 ;
	out[3] |= ( in[49] & 0x03 ) << 2 ;
	out[3] |= ( in[50] & 0x03 ) << 4 ;
	out[3] |= ( in[51] & 0x03 ) << 6 ;
	out[3] |= ( in[52] & 0x03 ) << 8 ;
	out[3] |= ( in[53] & 0x03 ) << 10 ;
	out[3] |= ( in[54] & 0x03 ) << 12 ;
	out[3] |= ( in[55] & 0x03 ) << 14 ;
	out[3] |= ( in[56] & 0x03 ) << 16 ;
	out[3] |= ( in[57] & 0x03 ) << 18 ;
	out[3] |= ( in[58] & 0x03 ) << 20 ;
	out[3] |= ( in[59] & 0x03 ) << 22 ;
	out[3] |= ( in[60] & 0x03 ) << 24 ;
	out[3] |= ( in[61] & 0x03 ) << 26 ;
	out[3] |= ( in[62] & 0x03 ) << 28 ;
	out[3] |= in[63] << 30 ;
}


// 3-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c3(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x07 ) << 0 ;
	out[0] |= ( in[1] & 0x07 ) << 3 ;
	out[0] |= ( in[2] & 0x07 ) << 6 ;
	out[0] |= ( in[3] & 0x07 ) << 9 ;
	out[0] |= ( in[4] & 0x07 ) << 12 ;
	out[0] |= ( in[5] & 0x07 ) << 15 ;
	out[0] |= ( in[6] & 0x07 ) << 18 ;
	out[0] |= ( in[7] & 0x07 ) << 21 ;
	out[0] |= ( in[8] & 0x07 ) << 24 ;
	out[0] |= ( in[9] & 0x07 ) << 27 ;
	out[0] |= in[10] << 30 ;

	out[1] = ( in[10] & 0x07 ) >> ( 32 - 30 ) ;
	out[1] |= ( in[11] & 0x07 ) << 1 ;
	out[1] |= ( in[12] & 0x07 ) << 4 ;
	out[1] |= ( in[13] & 0x07 ) << 7 ;
	out[1] |= ( in[14] & 0x07 ) << 10 ;
	out[1] |= ( in[15] & 0x07 ) << 13 ;
	out[1] |= ( in[16] & 0x07 ) << 16 ;
	out[1] |= ( in[17] & 0x07 ) << 19 ;
	out[1] |= ( in[18] & 0x07 ) << 22 ;
	out[1] |= ( in[19] & 0x07 ) << 25 ;
	out[1] |= ( in[20] & 0x07 ) << 28 ;
	out[1] |= in[21] << 31 ;

	out[2] = ( in[21] & 0x07 ) >> ( 32 - 31 ) ;
	out[2] |= ( in[22] & 0x07 ) << 2 ;
	out[2] |= ( in[23] & 0x07 ) << 5 ;
	out[2] |= ( in[24] & 0x07 ) << 8 ;
	out[2] |= ( in[25] & 0x07 ) << 11 ;
	out[2] |= ( in[26] & 0x07 ) << 14 ;
	out[2] |= ( in[27] & 0x07 ) << 17 ;
	out[2] |= ( in[28] & 0x07 ) << 20 ;
	out[2] |= ( in[29] & 0x07 ) << 23 ;
	out[2] |= ( in[30] & 0x07 ) << 26 ;
	out[2] |= in[31] << 29 ;

	out[3] = ( in[32] & 0x07 ) << 0 ;
	out[3] |= ( in[33] & 0x07 ) << 3 ;
	out[3] |= ( in[34] & 0x07 ) << 6 ;
	out[3] |= ( in[35] & 0x07 ) << 9 ;
	out[3] |= ( in[36] & 0x07 ) << 12 ;
	out[3] |= ( in[37] & 0x07 ) << 15 ;
	out[3] |= ( in[38] & 0x07 ) << 18 ;
	out[3] |= ( in[39] & 0x07 ) << 21 ;
	out[3] |= ( in[40] & 0x07 ) << 24 ;
	out[3] |= ( in[41] & 0x07 ) << 27 ;
	out[3] |= in[42] << 30 ;

	out[4] = ( in[42] & 0x07 ) >> ( 32 - 30 ) ;
	out[4] |= ( in[43] & 0x07 ) << 1 ;
	out[4] |= ( in[44] & 0x07 ) << 4 ;
	out[4] |= ( in[45] & 0x07 ) << 7 ;
	out[4] |= ( in[46] & 0x07 ) << 10 ;
	out[4] |= ( in[47] & 0x07 ) << 13 ;
	out[4] |= ( in[48] & 0x07 ) << 16 ;
	out[4] |= ( in[49] & 0x07 ) << 19 ;
	out[4] |= ( in[50] & 0x07 ) << 22 ;
	out[4] |= ( in[51] & 0x07 ) << 25 ;
	out[4] |= ( in[52] & 0x07 ) << 28 ;
	out[4] |= in[53] << 31 ;

	out[5] = ( in[53] & 0x07 ) >> ( 32 - 31 ) ;
	out[5] |= ( in[54] & 0x07 ) << 2 ;
	out[5] |= ( in[55] & 0x07 ) << 5 ;
	out[5] |= ( in[56] & 0x07 ) << 8 ;
	out[5] |= ( in[57] & 0x07 ) << 11 ;
	out[5] |= ( in[58] & 0x07 ) << 14 ;
	out[5] |= ( in[59] & 0x07 ) << 17 ;
	out[5] |= ( in[60] & 0x07 ) << 20 ;
	out[5] |= ( in[61] & 0x07 ) << 23 ;
	out[5] |= ( in[62] & 0x07 ) << 26 ;
	out[5] |= in[63] << 29 ;
}


// 4-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c4(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x0f ) << 0 ;
	out[0] |= ( in[1] & 0x0f ) << 4 ;
	out[0] |= ( in[2] & 0x0f ) << 8 ;
	out[0] |= ( in[3] & 0x0f ) << 12 ;
	out[0] |= ( in[4] & 0x0f ) << 16 ;
	out[0] |= ( in[5] & 0x0f ) << 20 ;
	out[0] |= ( in[6] & 0x0f ) << 24 ;
	out[0] |= in[7] << 28 ;

	out[1] = ( in[8] & 0x0f ) << 0 ;
	out[1] |= ( in[9] & 0x0f ) << 4 ;
	out[1] |= ( in[10] & 0x0f ) << 8 ;
	out[1] |= ( in[11] & 0x0f ) << 12 ;
	out[1] |= ( in[12] & 0x0f ) << 16 ;
	out[1] |= ( in[13] & 0x0f ) << 20 ;
	out[1] |= ( in[14] & 0x0f ) << 24 ;
	out[1] |= in[15] << 28 ;

	out[2] = ( in[16] & 0x0f ) << 0 ;
	out[2] |= ( in[17] & 0x0f ) << 4 ;
	out[2] |= ( in[18] & 0x0f ) << 8 ;
	out[2] |= ( in[19] & 0x0f ) << 12 ;
	out[2] |= ( in[20] & 0x0f ) << 16 ;
	out[2] |= ( in[21] & 0x0f ) << 20 ;
	out[2] |= ( in[22] & 0x0f ) << 24 ;
	out[2] |= in[23] << 28 ;

	out[3] = ( in[24] & 0x0f ) << 0 ;
	out[3] |= ( in[25] & 0x0f ) << 4 ;
	out[3] |= ( in[26] & 0x0f ) << 8 ;
	out[3] |= ( in[27] & 0x0f ) << 12 ;
	out[3] |= ( in[28] & 0x0f ) << 16 ;
	out[3] |= ( in[29] & 0x0f ) << 20 ;
	out[3] |= ( in[30] & 0x0f ) << 24 ;
	out[3] |= in[31] << 28 ;

	out[4] = ( in[32] & 0x0f ) << 0 ;
	out[4] |= ( in[33] & 0x0f ) << 4 ;
	out[4] |= ( in[34] & 0x0f ) << 8 ;
	out[4] |= ( in[35] & 0x0f ) << 12 ;
	out[4] |= ( in[36] & 0x0f ) << 16 ;
	out[4] |= ( in[37] & 0x0f ) << 20 ;
	out[4] |= ( in[38] & 0x0f ) << 24 ;
	out[4] |= in[39] << 28 ;

	out[5] = ( in[40] & 0x0f ) << 0 ;
	out[5] |= ( in[41] & 0x0f ) << 4 ;
	out[5] |= ( in[42] & 0x0f ) << 8 ;
	out[5] |= ( in[43] & 0x0f ) << 12 ;
	out[5] |= ( in[44] & 0x0f ) << 16 ;
	out[5] |= ( in[45] & 0x0f ) << 20 ;
	out[5] |= ( in[46] & 0x0f ) << 24 ;
	out[5] |= in[47] << 28 ;

	out[6] = ( in[48] & 0x0f ) << 0 ;
	out[6] |= ( in[49] & 0x0f ) << 4 ;
	out[6] |= ( in[50] & 0x0f ) << 8 ;
	out[6] |= ( in[51] & 0x0f ) << 12 ;
	out[6] |= ( in[52] & 0x0f ) << 16 ;
	out[6] |= ( in[53] & 0x0f ) << 20 ;
	out[6] |= ( in[54] & 0x0f ) << 24 ;
	out[6] |= in[55] << 28 ;

	out[7] = ( in[56] & 0x0f ) << 0 ;
	out[7] |= ( in[57] & 0x0f ) << 4 ;
	out[7] |= ( in[58] & 0x0f ) << 8 ;
	out[7] |= ( in[59] & 0x0f ) << 12 ;
	out[7] |= ( in[60] & 0x0f ) << 16 ;
	out[7] |= ( in[61] & 0x0f ) << 20 ;
	out[7] |= ( in[62] & 0x0f ) << 24 ;
	out[7] |= in[63] << 28 ;
}


// 5-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c5(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x1f ) << 0 ;
	out[0] |= ( in[1] & 0x1f ) << 5 ;
	out[0] |= ( in[2] & 0x1f ) << 10 ;
	out[0] |= ( in[3] & 0x1f ) << 15 ;
	out[0] |= ( in[4] & 0x1f ) << 20 ;
	out[0] |= ( in[5] & 0x1f ) << 25 ;
	out[0] |= in[6] << 30 ;

	out[1] = ( in[6] & 0x1f ) >> ( 32 - 30 ) ;
	out[1] |= ( in[7] & 0x1f ) << 3 ;
	out[1] |= ( in[8] & 0x1f ) << 8 ;
	out[1] |= ( in[9] & 0x1f ) << 13 ;
	out[1] |= ( in[10] & 0x1f ) << 18 ;
	out[1] |= ( in[11] & 0x1f ) << 23 ;
	out[1] |= in[12] << 28 ;

	out[2] = ( in[12] & 0x1f ) >> ( 32 - 28 ) ;
	out[2] |= ( in[13] & 0x1f ) << 1 ;
	out[2] |= ( in[14] & 0x1f ) << 6 ;
	out[2] |= ( in[15] & 0x1f ) << 11 ;
	out[2] |= ( in[16] & 0x1f ) << 16 ;
	out[2] |= ( in[17] & 0x1f ) << 21 ;
	out[2] |= ( in[18] & 0x1f ) << 26 ;
	out[2] |= in[19] << 31 ;

	out[3] = ( in[19] & 0x1f ) >> ( 32 - 31 ) ;
	out[3] |= ( in[20] & 0x1f ) << 4 ;
	out[3] |= ( in[21] & 0x1f ) << 9 ;
	out[3] |= ( in[22] & 0x1f ) << 14 ;
	out[3] |= ( in[23] & 0x1f ) << 19 ;
	out[3] |= ( in[24] & 0x1f ) << 24 ;
	out[3] |= in[25] << 29 ;

	out[4] = ( in[25] & 0x1f ) >> ( 32 - 29 ) ;
	out[4] |= ( in[26] & 0x1f ) << 2 ;
	out[4] |= ( in[27] & 0x1f ) << 7 ;
	out[4] |= ( in[28] & 0x1f ) << 12 ;
	out[4] |= ( in[29] & 0x1f ) << 17 ;
	out[4] |= ( in[30] & 0x1f ) << 22 ;
	out[4] |= in[31] << 27 ;

	out[5] = ( in[32] & 0x1f ) << 0 ;
	out[5] |= ( in[33] & 0x1f ) << 5 ;
	out[5] |= ( in[34] & 0x1f ) << 10 ;
	out[5] |= ( in[35] & 0x1f ) << 15 ;
	out[5] |= ( in[36] & 0x1f ) << 20 ;
	out[5] |= ( in[37] & 0x1f ) << 25 ;
	out[5] |= in[38] << 30 ;

	out[6] = ( in[38] & 0x1f ) >> ( 32 - 30 ) ;
	out[6] |= ( in[39] & 0x1f ) << 3 ;
	out[6] |= ( in[40] & 0x1f ) << 8 ;
	out[6] |= ( in[41] & 0x1f ) << 13 ;
	out[6] |= ( in[42] & 0x1f ) << 18 ;
	out[6] |= ( in[43] & 0x1f ) << 23 ;
	out[6] |= in[44] << 28 ;

	out[7] = ( in[44] & 0x1f ) >> ( 32 - 28 ) ;
	out[7] |= ( in[45] & 0x1f ) << 1 ;
	out[7] |= ( in[46] & 0x1f ) << 6 ;
	out[7] |= ( in[47] & 0x1f ) << 11 ;
	out[7] |= ( in[48] & 0x1f ) << 16 ;
	out[7] |= ( in[49] & 0x1f ) << 21 ;
	out[7] |= ( in[50] & 0x1f ) << 26 ;
	out[7] |= in[51] << 31 ;

	out[8] = ( in[51] & 0x1f ) >> ( 32 - 31 ) ;
	out[8] |= ( in[52] & 0x1f ) << 4 ;
	out[8] |= ( in[53] & 0x1f ) << 9 ;
	out[8] |= ( in[54] & 0x1f ) << 14 ;
	out[8] |= ( in[55] & 0x1f ) << 19 ;
	out[8] |= ( in[56] & 0x1f ) << 24 ;
	out[8] |= in[57] << 29 ;

	out[9] = ( in[57] & 0x1f ) >> ( 32 - 29 ) ;
	out[9] |= ( in[58] & 0x1f ) << 2 ;
	out[9] |= ( in[59] & 0x1f ) << 7 ;
	out[9] |= ( in[60] & 0x1f ) << 12 ;
	out[9] |= ( in[61] & 0x1f ) << 17 ;
	out[9] |= ( in[62] & 0x1f ) << 22 ;
	out[9] |= in[63] << 27 ;
}


// 6-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c6(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x3f ) << 0 ;
	out[0] |= ( in[1] & 0x3f ) << 6 ;
	out[0] |= ( in[2] & 0x3f ) << 12 ;
	out[0] |= ( in[3] & 0x3f ) << 18 ;
	out[0] |= ( in[4] & 0x3f ) << 24 ;
	out[0] |= in[5] << 30 ;

	out[1] = ( in[5] & 0x3f ) >> ( 32 - 30 ) ;
	out[1] |= ( in[6] & 0x3f ) << 4 ;
	out[1] |= ( in[7] & 0x3f ) << 10 ;
	out[1] |= ( in[8] & 0x3f ) << 16 ;
	out[1] |= ( in[9] & 0x3f ) << 22 ;
	out[1] |= in[10] << 28 ;

	out[2] = ( in[10] & 0x3f ) >> ( 32 - 28 ) ;
	out[2] |= ( in[11] & 0x3f ) << 2 ;
	out[2] |= ( in[12] & 0x3f ) << 8 ;
	out[2] |= ( in[13] & 0x3f ) << 14 ;
	out[2] |= ( in[14] & 0x3f ) << 20 ;
	out[2] |= in[15] << 26 ;

	out[3] = ( in[16] & 0x3f ) << 0 ;
	out[3] |= ( in[17] & 0x3f ) << 6 ;
	out[3] |= ( in[18] & 0x3f ) << 12 ;
	out[3] |= ( in[19] & 0x3f ) << 18 ;
	out[3] |= ( in[20] & 0x3f ) << 24 ;
	out[3] |= in[21] << 30 ;

	out[4] = ( in[21] & 0x3f ) >> ( 32 - 30 ) ;
	out[4] |= ( in[22] & 0x3f ) << 4 ;
	out[4] |= ( in[23] & 0x3f ) << 10 ;
	out[4] |= ( in[24] & 0x3f ) << 16 ;
	out[4] |= ( in[25] & 0x3f ) << 22 ;
	out[4] |= in[26] << 28 ;

	out[5] = ( in[26] & 0x3f ) >> ( 32 - 28 ) ;
	out[5] |= ( in[27] & 0x3f ) << 2 ;
	out[5] |= ( in[28] & 0x3f ) << 8 ;
	out[5] |= ( in[29] & 0x3f ) << 14 ;
	out[5] |= ( in[30] & 0x3f ) << 20 ;
	out[5] |= in[31] << 26 ;

	out[6] = ( in[32] & 0x3f ) << 0 ;
	out[6] |= ( in[33] & 0x3f ) << 6 ;
	out[6] |= ( in[34] & 0x3f ) << 12 ;
	out[6] |= ( in[35] & 0x3f ) << 18 ;
	out[6] |= ( in[36] & 0x3f ) << 24 ;
	out[6] |= in[37] << 30 ;

	out[7] = ( in[37] & 0x3f ) >> ( 32 - 30 ) ;
	out[7] |= ( in[38] & 0x3f ) << 4 ;
	out[7] |= ( in[39] & 0x3f ) << 10 ;
	out[7] |= ( in[40] & 0x3f ) << 16 ;
	out[7] |= ( in[41] & 0x3f ) << 22 ;
	out[7] |= in[42] << 28 ;

	out[8] = ( in[42] & 0x3f ) >> ( 32 - 28 ) ;
	out[8] |= ( in[43] & 0x3f ) << 2 ;
	out[8] |= ( in[44] & 0x3f ) << 8 ;
	out[8] |= ( in[45] & 0x3f ) << 14 ;
	out[8] |= ( in[46] & 0x3f ) << 20 ;
	out[8] |= in[47] << 26 ;

	out[9] = ( in[48] & 0x3f ) << 0 ;
	out[9] |= ( in[49] & 0x3f ) << 6 ;
	out[9] |= ( in[50] & 0x3f ) << 12 ;
	out[9] |= ( in[51] & 0x3f ) << 18 ;
	out[9] |= ( in[52] & 0x3f ) << 24 ;
	out[9] |= in[53] << 30 ;

	out[10] = ( in[53] & 0x3f ) >> ( 32 - 30 ) ;
	out[10] |= ( in[54] & 0x3f ) << 4 ;
	out[10] |= ( in[55] & 0x3f ) << 10 ;
	out[10] |= ( in[56] & 0x3f ) << 16 ;
	out[10] |= ( in[57] & 0x3f ) << 22 ;
	out[10] |= in[58] << 28 ;

	out[11] = ( in[58] & 0x3f ) >> ( 32 - 28 ) ;
	out[11] |= ( in[59] & 0x3f ) << 2 ;
	out[11] |= ( in[60] & 0x3f ) << 8 ;
	out[11] |= ( in[61] & 0x3f ) << 14 ;
	out[11] |= ( in[62] & 0x3f ) << 20 ;
	out[11] |= in[63] << 26 ;
}


// 7-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c7(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x7f ) << 0 ;
	out[0] |= ( in[1] & 0x7f ) << 7 ;
	out[0] |= ( in[2] & 0x7f ) << 14 ;
	out[0] |= ( in[3] & 0x7f ) << 21 ;
	out[0] |= in[4] << 28 ;

	out[1] = ( in[4] & 0x7f ) >> ( 32 - 28 ) ;
	out[1] |= ( in[5] & 0x7f ) << 3 ;
	out[1] |= ( in[6] & 0x7f ) << 10 ;
	out[1] |= ( in[7] & 0x7f ) << 17 ;
	out[1] |= ( in[8] & 0x7f ) << 24 ;
	out[1] |= in[9] << 31 ;

	out[2] = ( in[9] & 0x7f ) >> ( 32 - 31 ) ;
	out[2] |= ( in[10] & 0x7f ) << 6 ;
	out[2] |= ( in[11] & 0x7f ) << 13 ;
	out[2] |= ( in[12] & 0x7f ) << 20 ;
	out[2] |= in[13] << 27 ;

	out[3] = ( in[13] & 0x7f ) >> ( 32 - 27 ) ;
	out[3] |= ( in[14] & 0x7f ) << 2 ;
	out[3] |= ( in[15] & 0x7f ) << 9 ;
	out[3] |= ( in[16] & 0x7f ) << 16 ;
	out[3] |= ( in[17] & 0x7f ) << 23 ;
	out[3] |= in[18] << 30 ;

	out[4] = ( in[18] & 0x7f ) >> ( 32 - 30 ) ;
	out[4] |= ( in[19] & 0x7f ) << 5 ;
	out[4] |= ( in[20] & 0x7f ) << 12 ;
	out[4] |= ( in[21] & 0x7f ) << 19 ;
	out[4] |= in[22] << 26 ;

	out[5] = ( in[22] & 0x7f ) >> ( 32 - 26 ) ;
	out[5] |= ( in[23] & 0x7f ) << 1 ;
	out[5] |= ( in[24] & 0x7f ) << 8 ;
	out[5] |= ( in[25] & 0x7f ) << 15 ;
	out[5] |= ( in[26] & 0x7f ) << 22 ;
	out[5] |= in[27] << 29 ;

	out[6] = ( in[27] & 0x7f ) >> ( 32 - 29 ) ;
	out[6] |= ( in[28] & 0x7f ) << 4 ;
	out[6] |= ( in[29] & 0x7f ) << 11 ;
	out[6] |= ( in[30] & 0x7f ) << 18 ;
	out[6] |= in[31] << 25 ;

	out[7] = ( in[32] & 0x7f ) << 0 ;
	out[7] |= ( in[33] & 0x7f ) << 7 ;
	out[7] |= ( in[34] & 0x7f ) << 14 ;
	out[7] |= ( in[35] & 0x7f ) << 21 ;
	out[7] |= in[36] << 28 ;

	out[8] = ( in[36] & 0x7f ) >> ( 32 - 28 ) ;
	out[8] |= ( in[37] & 0x7f ) << 3 ;
	out[8] |= ( in[38] & 0x7f ) << 10 ;
	out[8] |= ( in[39] & 0x7f ) << 17 ;
	out[8] |= ( in[40] & 0x7f ) << 24 ;
	out[8] |= in[41] << 31 ;

	out[9] = ( in[41] & 0x7f ) >> ( 32 - 31 ) ;
	out[9] |= ( in[42] & 0x7f ) << 6 ;
	out[9] |= ( in[43] & 0x7f ) << 13 ;
	out[9] |= ( in[44] & 0x7f ) << 20 ;
	out[9] |= in[45] << 27 ;

	out[10] = ( in[45] & 0x7f ) >> ( 32 - 27 ) ;
	out[10] |= ( in[46] & 0x7f ) << 2 ;
	out[10] |= ( in[47] & 0x7f ) << 9 ;
	out[10] |= ( in[48] & 0x7f ) << 16 ;
	out[10] |= ( in[49] & 0x7f ) << 23 ;
	out[10] |= in[50] << 30 ;

	out[11] = ( in[50] & 0x7f ) >> ( 32 - 30 ) ;
	out[11] |= ( in[51] & 0x7f ) << 5 ;
	out[11] |= ( in[52] & 0x7f ) << 12 ;
	out[11] |= ( in[53] & 0x7f ) << 19 ;
	out[11] |= in[54] << 26 ;

	out[12] = ( in[54] & 0x7f ) >> ( 32 - 26 ) ;
	out[12] |= ( in[55] & 0x7f ) << 1 ;
	out[12] |= ( in[56] & 0x7f ) << 8 ;
	out[12] |= ( in[57] & 0x7f ) << 15 ;
	out[12] |= ( in[58] & 0x7f ) << 22 ;
	out[12] |= in[59] << 29 ;

	out[13] = ( in[59] & 0x7f ) >> ( 32 - 29 ) ;
	out[13] |= ( in[60] & 0x7f ) << 4 ;
	out[13] |= ( in[61] & 0x7f ) << 11 ;
	out[13] |= ( in[62] & 0x7f ) << 18 ;
	out[13] |= in[63] << 25 ;
}


// 8-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c8(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0xff ) << 0 ;
	out[0] |= ( in[1] & 0xff ) << 8 ;
	out[0] |= ( in[2] & 0xff ) << 16 ;
	out[0] |= in[3] << 24 ;

	out[1] = ( in[4] & 0xff ) << 0 ;
	out[1] |= ( in[5] & 0xff ) << 8 ;
	out[1] |= ( in[6] & 0xff ) << 16 ;
	out[1] |= in[7] << 24 ;

	out[2] = ( in[8] & 0xff ) << 0 ;
	out[2] |= ( in[9] & 0xff ) << 8 ;
	out[2] |= ( in[10] & 0xff ) << 16 ;
	out[2] |= in[11] << 24 ;

	out[3] = ( in[12] & 0xff ) << 0 ;
	out[3] |= ( in[13] & 0xff ) << 8 ;
	out[3] |= ( in[14] & 0xff ) << 16 ;
	out[3] |= in[15] << 24 ;

	out[4] = ( in[16] & 0xff ) << 0 ;
	out[4] |= ( in[17] & 0xff ) << 8 ;
	out[4] |= ( in[18] & 0xff ) << 16 ;
	out[4] |= in[19] << 24 ;

	out[5] = ( in[20] & 0xff ) << 0 ;
	out[5] |= ( in[21] & 0xff ) << 8 ;
	out[5] |= ( in[22] & 0xff ) << 16 ;
	out[5] |= in[23] << 24 ;

	out[6] = ( in[24] & 0xff ) << 0 ;
	out[6] |= ( in[25] & 0xff ) << 8 ;
	out[6] |= ( in[26] & 0xff ) << 16 ;
	out[6] |= in[27] << 24 ;

	out[7] = ( in[28] & 0xff ) << 0 ;
	out[7] |= ( in[29] & 0xff ) << 8 ;
	out[7] |= ( in[30] & 0xff ) << 16 ;
	out[7] |= in[31] << 24 ;

	out[8] = ( in[32] & 0xff ) << 0 ;
	out[8] |= ( in[33] & 0xff ) << 8 ;
	out[8] |= ( in[34] & 0xff ) << 16 ;
	out[8] |= in[35] << 24 ;

	out[9] = ( in[36] & 0xff ) << 0 ;
	out[9] |= ( in[37] & 0xff ) << 8 ;
	out[9] |= ( in[38] & 0xff ) << 16 ;
	out[9] |= in[39] << 24 ;

	out[10] = ( in[40] & 0xff ) << 0 ;
	out[10] |= ( in[41] & 0xff ) << 8 ;
	out[10] |= ( in[42] & 0xff ) << 16 ;
	out[10] |= in[43] << 24 ;

	out[11] = ( in[44] & 0xff ) << 0 ;
	out[11] |= ( in[45] & 0xff ) << 8 ;
	out[11] |= ( in[46] & 0xff ) << 16 ;
	out[11] |= in[47] << 24 ;

	out[12] = ( in[48] & 0xff ) << 0 ;
	out[12] |= ( in[49] & 0xff ) << 8 ;
	out[12] |= ( in[50] & 0xff ) << 16 ;
	out[12] |= in[51] << 24 ;

	out[13] = ( in[52] & 0xff ) << 0 ;
	out[13] |= ( in[53] & 0xff ) << 8 ;
	out[13] |= ( in[54] & 0xff ) << 16 ;
	out[13] |= in[55] << 24 ;

	out[14] = ( in[56] & 0xff ) << 0 ;
	out[14] |= ( in[57] & 0xff ) << 8 ;
	out[14] |= ( in[58] & 0xff ) << 16 ;
	out[14] |= in[59] << 24 ;

	out[15] = ( in[60] & 0xff ) << 0 ;
	out[15] |= ( in[61] & 0xff ) << 8 ;
	out[15] |= ( in[62] & 0xff ) << 16 ;
	out[15] |= in[63] << 24 ;
}


// 9-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c9(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x01ff ) << 0 ;
	out[0] |= ( in[1] & 0x01ff ) << 9 ;
	out[0] |= ( in[2] & 0x01ff ) << 18 ;
	out[0] |= in[3] << 27 ;

	out[1] = ( in[3] & 0x01ff ) >> ( 32 - 27 ) ;
	out[1] |= ( in[4] & 0x01ff ) << 4 ;
	out[1] |= ( in[5] & 0x01ff ) << 13 ;
	out[1] |= ( in[6] & 0x01ff ) << 22 ;
	out[1] |= in[7] << 31 ;

	out[2] = ( in[7] & 0x01ff ) >> ( 32 - 31 ) ;
	out[2] |= ( in[8] & 0x01ff ) << 8 ;
	out[2] |= ( in[9] & 0x01ff ) << 17 ;
	out[2] |= in[10] << 26 ;

	out[3] = ( in[10] & 0x01ff ) >> ( 32 - 26 ) ;
	out[3] |= ( in[11] & 0x01ff ) << 3 ;
	out[3] |= ( in[12] & 0x01ff ) << 12 ;
	out[3] |= ( in[13] & 0x01ff ) << 21 ;
	out[3] |= in[14] << 30 ;

	out[4] = ( in[14] & 0x01ff ) >> ( 32 - 30 ) ;
	out[4] |= ( in[15] & 0x01ff ) << 7 ;
	out[4] |= ( in[16] & 0x01ff ) << 16 ;
	out[4] |= in[17] << 25 ;

	out[5] = ( in[17] & 0x01ff ) >> ( 32 - 25 ) ;
	out[5] |= ( in[18] & 0x01ff ) << 2 ;
	out[5] |= ( in[19] & 0x01ff ) << 11 ;
	out[5] |= ( in[20] & 0x01ff ) << 20 ;
	out[5] |= in[21] << 29 ;

	out[6] = ( in[21] & 0x01ff ) >> ( 32 - 29 ) ;
	out[6] |= ( in[22] & 0x01ff ) << 6 ;
	out[6] |= ( in[23] & 0x01ff ) << 15 ;
	out[6] |= in[24] << 24 ;

	out[7] = ( in[24] & 0x01ff ) >> ( 32 - 24 ) ;
	out[7] |= ( in[25] & 0x01ff ) << 1 ;
	out[7] |= ( in[26] & 0x01ff ) << 10 ;
	out[7] |= ( in[27] & 0x01ff ) << 19 ;
	out[7] |= in[28] << 28 ;

	out[8] = ( in[28] & 0x01ff ) >> ( 32 - 28 ) ;
	out[8] |= ( in[29] & 0x01ff ) << 5 ;
	out[8] |= ( in[30] & 0x01ff ) << 14 ;
	out[8] |= in[31] << 23 ;

	out[9] = ( in[32] & 0x01ff ) << 0 ;
	out[9] |= ( in[33] & 0x01ff ) << 9 ;
	out[9] |= ( in[34] & 0x01ff ) << 18 ;
	out[9] |= in[35] << 27 ;

	out[10] = ( in[35] & 0x01ff ) >> ( 32 - 27 ) ;
	out[10] |= ( in[36] & 0x01ff ) << 4 ;
	out[10] |= ( in[37] & 0x01ff ) << 13 ;
	out[10] |= ( in[38] & 0x01ff ) << 22 ;
	out[10] |= in[39] << 31 ;

	out[11] = ( in[39] & 0x01ff ) >> ( 32 - 31 ) ;
	out[11] |= ( in[40] & 0x01ff ) << 8 ;
	out[11] |= ( in[41] & 0x01ff ) << 17 ;
	out[11] |= in[42] << 26 ;

	out[12] = ( in[42] & 0x01ff ) >> ( 32 - 26 ) ;
	out[12] |= ( in[43] & 0x01ff ) << 3 ;
	out[12] |= ( in[44] & 0x01ff ) << 12 ;
	out[12] |= ( in[45] & 0x01ff ) << 21 ;
	out[12] |= in[46] << 30 ;

	out[13] = ( in[46] & 0x01ff ) >> ( 32 - 30 ) ;
	out[13] |= ( in[47] & 0x01ff ) << 7 ;
	out[13] |= ( in[48] & 0x01ff ) << 16 ;
	out[13] |= in[49] << 25 ;

	out[14] = ( in[49] & 0x01ff ) >> ( 32 - 25 ) ;
	out[14] |= ( in[50] & 0x01ff ) << 2 ;
	out[14] |= ( in[51] & 0x01ff ) << 11 ;
	out[14] |= ( in[52] & 0x01ff ) << 20 ;
	out[14] |= in[53] << 29 ;

	out[15] = ( in[53] & 0x01ff ) >> ( 32 - 29 ) ;
	out[15] |= ( in[54] & 0x01ff ) << 6 ;
	out[15] |= ( in[55] & 0x01ff ) << 15 ;
	out[15] |= in[56] << 24 ;

	out[16] = ( in[56] & 0x01ff ) >> ( 32 - 24 ) ;
	out[16] |= ( in[57] & 0x01ff ) << 1 ;
	out[16] |= ( in[58] & 0x01ff ) << 10 ;
	out[16] |= ( in[59] & 0x01ff ) << 19 ;
	out[16] |= in[60] << 28 ;

	out[17] = ( in[60] & 0x01ff ) >> ( 32 - 28 ) ;
	out[17] |= ( in[61] & 0x01ff ) << 5 ;
	out[17] |= ( in[62] & 0x01ff ) << 14 ;
	out[17] |= in[63] << 23 ;
}


// 10-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c10(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x03ff ) << 0 ;
	out[0] |= ( in[1] & 0x03ff ) << 10 ;
	out[0] |= ( in[2] & 0x03ff ) << 20 ;
	out[0] |= in[3] << 30 ;

	out[1] = ( in[3] & 0x03ff ) >> ( 32 - 30 ) ;
	out[1] |= ( in[4] & 0x03ff ) << 8 ;
	out[1] |= ( in[5] & 0x03ff ) << 18 ;
	out[1] |= in[6] << 28 ;

	out[2] = ( in[6] & 0x03ff ) >> ( 32 - 28 ) ;
	out[2] |= ( in[7] & 0x03ff ) << 6 ;
	out[2] |= ( in[8] & 0x03ff ) << 16 ;
	out[2] |= in[9] << 26 ;

	out[3] = ( in[9] & 0x03ff ) >> ( 32 - 26 ) ;
	out[3] |= ( in[10] & 0x03ff ) << 4 ;
	out[3] |= ( in[11] & 0x03ff ) << 14 ;
	out[3] |= in[12] << 24 ;

	out[4] = ( in[12] & 0x03ff ) >> ( 32 - 24 ) ;
	out[4] |= ( in[13] & 0x03ff ) << 2 ;
	out[4] |= ( in[14] & 0x03ff ) << 12 ;
	out[4] |= in[15] << 22 ;

	out[5] = ( in[16] & 0x03ff ) << 0 ;
	out[5] |= ( in[17] & 0x03ff ) << 10 ;
	out[5] |= ( in[18] & 0x03ff ) << 20 ;
	out[5] |= in[19] << 30 ;

	out[6] = ( in[19] & 0x03ff ) >> ( 32 - 30 ) ;
	out[6] |= ( in[20] & 0x03ff ) << 8 ;
	out[6] |= ( in[21] & 0x03ff ) << 18 ;
	out[6] |= in[22] << 28 ;

	out[7] = ( in[22] & 0x03ff ) >> ( 32 - 28 ) ;
	out[7] |= ( in[23] & 0x03ff ) << 6 ;
	out[7] |= ( in[24] & 0x03ff ) << 16 ;
	out[7] |= in[25] << 26 ;

	out[8] = ( in[25] & 0x03ff ) >> ( 32 - 26 ) ;
	out[8] |= ( in[26] & 0x03ff ) << 4 ;
	out[8] |= ( in[27] & 0x03ff ) << 14 ;
	out[8] |= in[28] << 24 ;

	out[9] = ( in[28] & 0x03ff ) >> ( 32 - 24 ) ;
	out[9] |= ( in[29] & 0x03ff ) << 2 ;
	out[9] |= ( in[30] & 0x03ff ) << 12 ;
	out[9] |= in[31] << 22 ;

	out[10] = ( in[32] & 0x03ff ) << 0 ;
	out[10] |= ( in[33] & 0x03ff ) << 10 ;
	out[10] |= ( in[34] & 0x03ff ) << 20 ;
	out[10] |= in[35] << 30 ;

	out[11] = ( in[35] & 0x03ff ) >> ( 32 - 30 ) ;
	out[11] |= ( in[36] & 0x03ff ) << 8 ;
	out[11] |= ( in[37] & 0x03ff ) << 18 ;
	out[11] |= in[38] << 28 ;

	out[12] = ( in[38] & 0x03ff ) >> ( 32 - 28 ) ;
	out[12] |= ( in[39] & 0x03ff ) << 6 ;
	out[12] |= ( in[40] & 0x03ff ) << 16 ;
	out[12] |= in[41] << 26 ;

	out[13] = ( in[41] & 0x03ff ) >> ( 32 - 26 ) ;
	out[13] |= ( in[42] & 0x03ff ) << 4 ;
	out[13] |= ( in[43] & 0x03ff ) << 14 ;
	out[13] |= in[44] << 24 ;

	out[14] = ( in[44] & 0x03ff ) >> ( 32 - 24 ) ;
	out[14] |= ( in[45] & 0x03ff ) << 2 ;
	out[14] |= ( in[46] & 0x03ff ) << 12 ;
	out[14] |= in[47] << 22 ;

	out[15] = ( in[48] & 0x03ff ) << 0 ;
	out[15] |= ( in[49] & 0x03ff ) << 10 ;
	out[15] |= ( in[50] & 0x03ff ) << 20 ;
	out[15] |= in[51] << 30 ;

	out[16] = ( in[51] & 0x03ff ) >> ( 32 - 30 ) ;
	out[16] |= ( in[52] & 0x03ff ) << 8 ;
	out[16] |= ( in[53] & 0x03ff ) << 18 ;
	out[16] |= in[54] << 28 ;

	out[17] = ( in[54] & 0x03ff ) >> ( 32 - 28 ) ;
	out[17] |= ( in[55] & 0x03ff ) << 6 ;
	out[17] |= ( in[56] & 0x03ff ) << 16 ;
	out[17] |= in[57] << 26 ;

	out[18] = ( in[57] & 0x03ff ) >> ( 32 - 26 ) ;
	out[18] |= ( in[58] & 0x03ff ) << 4 ;
	out[18] |= ( in[59] & 0x03ff ) << 14 ;
	out[18] |= in[60] << 24 ;

	out[19] = ( in[60] & 0x03ff ) >> ( 32 - 24 ) ;
	out[19] |= ( in[61] & 0x03ff ) << 2 ;
	out[19] |= ( in[62] & 0x03ff ) << 12 ;
	out[19] |= in[63] << 22 ;
}


// 11-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c11(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x07ff ) << 0 ;
	out[0] |= ( in[1] & 0x07ff ) << 11 ;
	out[0] |= in[2] << 22 ;

	out[1] = ( in[2] & 0x07ff ) >> ( 32 - 22 ) ;
	out[1] |= ( in[3] & 0x07ff ) << 1 ;
	out[1] |= ( in[4] & 0x07ff ) << 12 ;
	out[1] |= in[5] << 23 ;

	out[2] = ( in[5] & 0x07ff ) >> ( 32 - 23 ) ;
	out[2] |= ( in[6] & 0x07ff ) << 2 ;
	out[2] |= ( in[7] & 0x07ff ) << 13 ;
	out[2] |= in[8] << 24 ;

	out[3] = ( in[8] & 0x07ff ) >> ( 32 - 24 ) ;
	out[3] |= ( in[9] & 0x07ff ) << 3 ;
	out[3] |= ( in[10] & 0x07ff ) << 14 ;
	out[3] |= in[11] << 25 ;

	out[4] = ( in[11] & 0x07ff ) >> ( 32 - 25 ) ;
	out[4] |= ( in[12] & 0x07ff ) << 4 ;
	out[4] |= ( in[13] & 0x07ff ) << 15 ;
	out[4] |= in[14] << 26 ;

	out[5] = ( in[14] & 0x07ff ) >> ( 32 - 26 ) ;
	out[5] |= ( in[15] & 0x07ff ) << 5 ;
	out[5] |= ( in[16] & 0x07ff ) << 16 ;
	out[5] |= in[17] << 27 ;

	out[6] = ( in[17] & 0x07ff ) >> ( 32 - 27 ) ;
	out[6] |= ( in[18] & 0x07ff ) << 6 ;
	out[6] |= ( in[19] & 0x07ff ) << 17 ;
	out[6] |= in[20] << 28 ;

	out[7] = ( in[20] & 0x07ff ) >> ( 32 - 28 ) ;
	out[7] |= ( in[21] & 0x07ff ) << 7 ;
	out[7] |= ( in[22] & 0x07ff ) << 18 ;
	out[7] |= in[23] << 29 ;

	out[8] = ( in[23] & 0x07ff ) >> ( 32 - 29 ) ;
	out[8] |= ( in[24] & 0x07ff ) << 8 ;
	out[8] |= ( in[25] & 0x07ff ) << 19 ;
	out[8] |= in[26] << 30 ;

	out[9] = ( in[26] & 0x07ff ) >> ( 32 - 30 ) ;
	out[9] |= ( in[27] & 0x07ff ) << 9 ;
	out[9] |= ( in[28] & 0x07ff ) << 20 ;
	out[9] |= in[29] << 31 ;

	out[10] = ( in[29] & 0x07ff ) >> ( 32 - 31 ) ;
	out[10] |= ( in[30] & 0x07ff ) << 10 ;
	out[10] |= in[31] << 21 ;

	out[11] = ( in[32] & 0x07ff ) << 0 ;
	out[11] |= ( in[33] & 0x07ff ) << 11 ;
	out[11] |= in[34] << 22 ;

	out[12] = ( in[34] & 0x07ff ) >> ( 32 - 22 ) ;
	out[12] |= ( in[35] & 0x07ff ) << 1 ;
	out[12] |= ( in[36] & 0x07ff ) << 12 ;
	out[12] |= in[37] << 23 ;

	out[13] = ( in[37] & 0x07ff ) >> ( 32 - 23 ) ;
	out[13] |= ( in[38] & 0x07ff ) << 2 ;
	out[13] |= ( in[39] & 0x07ff ) << 13 ;
	out[13] |= in[40] << 24 ;

	out[14] = ( in[40] & 0x07ff ) >> ( 32 - 24 ) ;
	out[14] |= ( in[41] & 0x07ff ) << 3 ;
	out[14] |= ( in[42] & 0x07ff ) << 14 ;
	out[14] |= in[43] << 25 ;

	out[15] = ( in[43] & 0x07ff ) >> ( 32 - 25 ) ;
	out[15] |= ( in[44] & 0x07ff ) << 4 ;
	out[15] |= ( in[45] & 0x07ff ) << 15 ;
	out[15] |= in[46] << 26 ;

	out[16] = ( in[46] & 0x07ff ) >> ( 32 - 26 ) ;
	out[16] |= ( in[47] & 0x07ff ) << 5 ;
	out[16] |= ( in[48] & 0x07ff ) << 16 ;
	out[16] |= in[49] << 27 ;

	out[17] = ( in[49] & 0x07ff ) >> ( 32 - 27 ) ;
	out[17] |= ( in[50] & 0x07ff ) << 6 ;
	out[17] |= ( in[51] & 0x07ff ) << 17 ;
	out[17] |= in[52] << 28 ;

	out[18] = ( in[52] & 0x07ff ) >> ( 32 - 28 ) ;
	out[18] |= ( in[53] & 0x07ff ) << 7 ;
	out[18] |= ( in[54] & 0x07ff ) << 18 ;
	out[18] |= in[55] << 29 ;

	out[19] = ( in[55] & 0x07ff ) >> ( 32 - 29 ) ;
	out[19] |= ( in[56] & 0x07ff ) << 8 ;
	out[19] |= ( in[57] & 0x07ff ) << 19 ;
	out[19] |= in[58] << 30 ;

	out[20] = ( in[58] & 0x07ff ) >> ( 32 - 30 ) ;
	out[20] |= ( in[59] & 0x07ff ) << 9 ;
	out[20] |= ( in[60] & 0x07ff ) << 20 ;
	out[20] |= in[61] << 31 ;

	out[21] = ( in[61] & 0x07ff ) >> ( 32 - 31 ) ;
	out[21] |= ( in[62] & 0x07ff ) << 10 ;
	out[21] |= in[63] << 21 ;
}


// 12-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c12(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x0fff ) << 0 ;
	out[0] |= ( in[1] & 0x0fff ) << 12 ;
	out[0] |= in[2] << 24 ;

	out[1] = ( in[2] & 0x0fff ) >> ( 32 - 24 ) ;
	out[1] |= ( in[3] & 0x0fff ) << 4 ;
	out[1] |= ( in[4] & 0x0fff ) << 16 ;
	out[1] |= in[5] << 28 ;

	out[2] = ( in[5] & 0x0fff ) >> ( 32 - 28 ) ;
	out[2] |= ( in[6] & 0x0fff ) << 8 ;
	out[2] |= in[7] << 20 ;

	out[3] = ( in[8] & 0x0fff ) << 0 ;
	out[3] |= ( in[9] & 0x0fff ) << 12 ;
	out[3] |= in[10] << 24 ;

	out[4] = ( in[10] & 0x0fff ) >> ( 32 - 24 ) ;
	out[4] |= ( in[11] & 0x0fff ) << 4 ;
	out[4] |= ( in[12] & 0x0fff ) << 16 ;
	out[4] |= in[13] << 28 ;

	out[5] = ( in[13] & 0x0fff ) >> ( 32 - 28 ) ;
	out[5] |= ( in[14] & 0x0fff ) << 8 ;
	out[5] |= in[15] << 20 ;

	out[6] = ( in[16] & 0x0fff ) << 0 ;
	out[6] |= ( in[17] & 0x0fff ) << 12 ;
	out[6] |= in[18] << 24 ;

	out[7] = ( in[18] & 0x0fff ) >> ( 32 - 24 ) ;
	out[7] |= ( in[19] & 0x0fff ) << 4 ;
	out[7] |= ( in[20] & 0x0fff ) << 16 ;
	out[7] |= in[21] << 28 ;

	out[8] = ( in[21] & 0x0fff ) >> ( 32 - 28 ) ;
	out[8] |= ( in[22] & 0x0fff ) << 8 ;
	out[8] |= in[23] << 20 ;

	out[9] = ( in[24] & 0x0fff ) << 0 ;
	out[9] |= ( in[25] & 0x0fff ) << 12 ;
	out[9] |= in[26] << 24 ;

	out[10] = ( in[26] & 0x0fff ) >> ( 32 - 24 ) ;
	out[10] |= ( in[27] & 0x0fff ) << 4 ;
	out[10] |= ( in[28] & 0x0fff ) << 16 ;
	out[10] |= in[29] << 28 ;

	out[11] = ( in[29] & 0x0fff ) >> ( 32 - 28 ) ;
	out[11] |= ( in[30] & 0x0fff ) << 8 ;
	out[11] |= in[31] << 20 ;

	out[12] = ( in[32] & 0x0fff ) << 0 ;
	out[12] |= ( in[33] & 0x0fff ) << 12 ;
	out[12] |= in[34] << 24 ;

	out[13] = ( in[34] & 0x0fff ) >> ( 32 - 24 ) ;
	out[13] |= ( in[35] & 0x0fff ) << 4 ;
	out[13] |= ( in[36] & 0x0fff ) << 16 ;
	out[13] |= in[37] << 28 ;

	out[14] = ( in[37] & 0x0fff ) >> ( 32 - 28 ) ;
	out[14] |= ( in[38] & 0x0fff ) << 8 ;
	out[14] |= in[39] << 20 ;

	out[15] = ( in[40] & 0x0fff ) << 0 ;
	out[15] |= ( in[41] & 0x0fff ) << 12 ;
	out[15] |= in[42] << 24 ;

	out[16] = ( in[42] & 0x0fff ) >> ( 32 - 24 ) ;
	out[16] |= ( in[43] & 0x0fff ) << 4 ;
	out[16] |= ( in[44] & 0x0fff ) << 16 ;
	out[16] |= in[45] << 28 ;

	out[17] = ( in[45] & 0x0fff ) >> ( 32 - 28 ) ;
	out[17] |= ( in[46] & 0x0fff ) << 8 ;
	out[17] |= in[47] << 20 ;

	out[18] = ( in[48] & 0x0fff ) << 0 ;
	out[18] |= ( in[49] & 0x0fff ) << 12 ;
	out[18] |= in[50] << 24 ;

	out[19] = ( in[50] & 0x0fff ) >> ( 32 - 24 ) ;
	out[19] |= ( in[51] & 0x0fff ) << 4 ;
	out[19] |= ( in[52] & 0x0fff ) << 16 ;
	out[19] |= in[53] << 28 ;

	out[20] = ( in[53] & 0x0fff ) >> ( 32 - 28 ) ;
	out[20] |= ( in[54] & 0x0fff ) << 8 ;
	out[20] |= in[55] << 20 ;

	out[21] = ( in[56] & 0x0fff ) << 0 ;
	out[21] |= ( in[57] & 0x0fff ) << 12 ;
	out[21] |= in[58] << 24 ;

	out[22] = ( in[58] & 0x0fff ) >> ( 32 - 24 ) ;
	out[22] |= ( in[59] & 0x0fff ) << 4 ;
	out[22] |= ( in[60] & 0x0fff ) << 16 ;
	out[22] |= in[61] << 28 ;

	out[23] = ( in[61] & 0x0fff ) >> ( 32 - 28 ) ;
	out[23] |= ( in[62] & 0x0fff ) << 8 ;
	out[23] |= in[63] << 20 ;
}


// 13-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c13(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x1fff ) << 0 ;
	out[0] |= ( in[1] & 0x1fff ) << 13 ;
	out[0] |= in[2] << 26 ;

	out[1] = ( in[2] & 0x1fff ) >> ( 32 - 26 ) ;
	out[1] |= ( in[3] & 0x1fff ) << 7 ;
	out[1] |= in[4] << 20 ;

	out[2] = ( in[4] & 0x1fff ) >> ( 32 - 20 ) ;
	out[2] |= ( in[5] & 0x1fff ) << 1 ;
	out[2] |= ( in[6] & 0x1fff ) << 14 ;
	out[2] |= in[7] << 27 ;

	out[3] = ( in[7] & 0x1fff ) >> ( 32 - 27 ) ;
	out[3] |= ( in[8] & 0x1fff ) << 8 ;
	out[3] |= in[9] << 21 ;

	out[4] = ( in[9] & 0x1fff ) >> ( 32 - 21 ) ;
	out[4] |= ( in[10] & 0x1fff ) << 2 ;
	out[4] |= ( in[11] & 0x1fff ) << 15 ;
	out[4] |= in[12] << 28 ;

	out[5] = ( in[12] & 0x1fff ) >> ( 32 - 28 ) ;
	out[5] |= ( in[13] & 0x1fff ) << 9 ;
	out[5] |= in[14] << 22 ;

	out[6] = ( in[14] & 0x1fff ) >> ( 32 - 22 ) ;
	out[6] |= ( in[15] & 0x1fff ) << 3 ;
	out[6] |= ( in[16] & 0x1fff ) << 16 ;
	out[6] |= in[17] << 29 ;

	out[7] = ( in[17] & 0x1fff ) >> ( 32 - 29 ) ;
	out[7] |= ( in[18] & 0x1fff ) << 10 ;
	out[7] |= in[19] << 23 ;

	out[8] = ( in[19] & 0x1fff ) >> ( 32 - 23 ) ;
	out[8] |= ( in[20] & 0x1fff ) << 4 ;
	out[8] |= ( in[21] & 0x1fff ) << 17 ;
	out[8] |= in[22] << 30 ;

	out[9] = ( in[22] & 0x1fff ) >> ( 32 - 30 ) ;
	out[9] |= ( in[23] & 0x1fff ) << 11 ;
	out[9] |= in[24] << 24 ;

	out[10] = ( in[24] & 0x1fff ) >> ( 32 - 24 ) ;
	out[10] |= ( in[25] & 0x1fff ) << 5 ;
	out[10] |= ( in[26] & 0x1fff ) << 18 ;
	out[10] |= in[27] << 31 ;

	out[11] = ( in[27] & 0x1fff ) >> ( 32 - 31 ) ;
	out[11] |= ( in[28] & 0x1fff ) << 12 ;
	out[11] |= in[29] << 25 ;

	out[12] = ( in[29] & 0x1fff ) >> ( 32 - 25 ) ;
	out[12] |= ( in[30] & 0x1fff ) << 6 ;
	out[12] |= in[31] << 19 ;

	out[13] = ( in[32] & 0x1fff ) << 0 ;
	out[13] |= ( in[33] & 0x1fff ) << 13 ;
	out[13] |= in[34] << 26 ;

	out[14] = ( in[34] & 0x1fff ) >> ( 32 - 26 ) ;
	out[14] |= ( in[35] & 0x1fff ) << 7 ;
	out[14] |= in[36] << 20 ;

	out[15] = ( in[36] & 0x1fff ) >> ( 32 - 20 ) ;
	out[15] |= ( in[37] & 0x1fff ) << 1 ;
	out[15] |= ( in[38] & 0x1fff ) << 14 ;
	out[15] |= in[39] << 27 ;

	out[16] = ( in[39] & 0x1fff ) >> ( 32 - 27 ) ;
	out[16] |= ( in[40] & 0x1fff ) << 8 ;
	out[16] |= in[41] << 21 ;

	out[17] = ( in[41] & 0x1fff ) >> ( 32 - 21 ) ;
	out[17] |= ( in[42] & 0x1fff ) << 2 ;
	out[17] |= ( in[43] & 0x1fff ) << 15 ;
	out[17] |= in[44] << 28 ;

	out[18] = ( in[44] & 0x1fff ) >> ( 32 - 28 ) ;
	out[18] |= ( in[45] & 0x1fff ) << 9 ;
	out[18] |= in[46] << 22 ;

	out[19] = ( in[46] & 0x1fff ) >> ( 32 - 22 ) ;
	out[19] |= ( in[47] & 0x1fff ) << 3 ;
	out[19] |= ( in[48] & 0x1fff ) << 16 ;
	out[19] |= in[49] << 29 ;

	out[20] = ( in[49] & 0x1fff ) >> ( 32 - 29 ) ;
	out[20] |= ( in[50] & 0x1fff ) << 10 ;
	out[20] |= in[51] << 23 ;

	out[21] = ( in[51] & 0x1fff ) >> ( 32 - 23 ) ;
	out[21] |= ( in[52] & 0x1fff ) << 4 ;
	out[21] |= ( in[53] & 0x1fff ) << 17 ;
	out[21] |= in[54] << 30 ;

	out[22] = ( in[54] & 0x1fff ) >> ( 32 - 30 ) ;
	out[22] |= ( in[55] & 0x1fff ) << 11 ;
	out[22] |= in[56] << 24 ;

	out[23] = ( in[56] & 0x1fff ) >> ( 32 - 24 ) ;
	out[23] |= ( in[57] & 0x1fff ) << 5 ;
	out[23] |= ( in[58] & 0x1fff ) << 18 ;
	out[23] |= in[59] << 31 ;

	out[24] = ( in[59] & 0x1fff ) >> ( 32 - 31 ) ;
	out[24] |= ( in[60] & 0x1fff ) << 12 ;
	out[24] |= in[61] << 25 ;

	out[25] = ( in[61] & 0x1fff ) >> ( 32 - 25 ) ;
	out[25] |= ( in[62] & 0x1fff ) << 6 ;
	out[25] |= in[63] << 19 ;
}


// 14-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c14(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x3fff ) << 0 ;
	out[0] |= ( in[1] & 0x3fff ) << 14 ;
	out[0] |= in[2] << 28 ;

	out[1] = ( in[2] & 0x3fff ) >> ( 32 - 28 ) ;
	out[1] |= ( in[3] & 0x3fff ) << 10 ;
	out[1] |= in[4] << 24 ;

	out[2] = ( in[4] & 0x3fff ) >> ( 32 - 24 ) ;
	out[2] |= ( in[5] & 0x3fff ) << 6 ;
	out[2] |= in[6] << 20 ;

	out[3] = ( in[6] & 0x3fff ) >> ( 32 - 20 ) ;
	out[3] |= ( in[7] & 0x3fff ) << 2 ;
	out[3] |= ( in[8] & 0x3fff ) << 16 ;
	out[3] |= in[9] << 30 ;

	out[4] = ( in[9] & 0x3fff ) >> ( 32 - 30 ) ;
	out[4] |= ( in[10] & 0x3fff ) << 12 ;
	out[4] |= in[11] << 26 ;

	out[5] = ( in[11] & 0x3fff ) >> ( 32 - 26 ) ;
	out[5] |= ( in[12] & 0x3fff ) << 8 ;
	out[5] |= in[13] << 22 ;

	out[6] = ( in[13] & 0x3fff ) >> ( 32 - 22 ) ;
	out[6] |= ( in[14] & 0x3fff ) << 4 ;
	out[6] |= in[15] << 18 ;

	out[7] = ( in[16] & 0x3fff ) << 0 ;
	out[7] |= ( in[17] & 0x3fff ) << 14 ;
	out[7] |= in[18] << 28 ;

	out[8] = ( in[18] & 0x3fff ) >> ( 32 - 28 ) ;
	out[8] |= ( in[19] & 0x3fff ) << 10 ;
	out[8] |= in[20] << 24 ;

	out[9] = ( in[20] & 0x3fff ) >> ( 32 - 24 ) ;
	out[9] |= ( in[21] & 0x3fff ) << 6 ;
	out[9] |= in[22] << 20 ;

	out[10] = ( in[22] & 0x3fff ) >> ( 32 - 20 ) ;
	out[10] |= ( in[23] & 0x3fff ) << 2 ;
	out[10] |= ( in[24] & 0x3fff ) << 16 ;
	out[10] |= in[25] << 30 ;

	out[11] = ( in[25] & 0x3fff ) >> ( 32 - 30 ) ;
	out[11] |= ( in[26] & 0x3fff ) << 12 ;
	out[11] |= in[27] << 26 ;

	out[12] = ( in[27] & 0x3fff ) >> ( 32 - 26 ) ;
	out[12] |= ( in[28] & 0x3fff ) << 8 ;
	out[12] |= in[29] << 22 ;

	out[13] = ( in[29] & 0x3fff ) >> ( 32 - 22 ) ;
	out[13] |= ( in[30] & 0x3fff ) << 4 ;
	out[13] |= in[31] << 18 ;

	out[14] = ( in[32] & 0x3fff ) << 0 ;
	out[14] |= ( in[33] & 0x3fff ) << 14 ;
	out[14] |= in[34] << 28 ;

	out[15] = ( in[34] & 0x3fff ) >> ( 32 - 28 ) ;
	out[15] |= ( in[35] & 0x3fff ) << 10 ;
	out[15] |= in[36] << 24 ;

	out[16] = ( in[36] & 0x3fff ) >> ( 32 - 24 ) ;
	out[16] |= ( in[37] & 0x3fff ) << 6 ;
	out[16] |= in[38] << 20 ;

	out[17] = ( in[38] & 0x3fff ) >> ( 32 - 20 ) ;
	out[17] |= ( in[39] & 0x3fff ) << 2 ;
	out[17] |= ( in[40] & 0x3fff ) << 16 ;
	out[17] |= in[41] << 30 ;

	out[18] = ( in[41] & 0x3fff ) >> ( 32 - 30 ) ;
	out[18] |= ( in[42] & 0x3fff ) << 12 ;
	out[18] |= in[43] << 26 ;

	out[19] = ( in[43] & 0x3fff ) >> ( 32 - 26 ) ;
	out[19] |= ( in[44] & 0x3fff ) << 8 ;
	out[19] |= in[45] << 22 ;

	out[20] = ( in[45] & 0x3fff ) >> ( 32 - 22 ) ;
	out[20] |= ( in[46] & 0x3fff ) << 4 ;
	out[20] |= in[47] << 18 ;

	out[21] = ( in[48] & 0x3fff ) << 0 ;
	out[21] |= ( in[49] & 0x3fff ) << 14 ;
	out[21] |= in[50] << 28 ;

	out[22] = ( in[50] & 0x3fff ) >> ( 32 - 28 ) ;
	out[22] |= ( in[51] & 0x3fff ) << 10 ;
	out[22] |= in[52] << 24 ;

	out[23] = ( in[52] & 0x3fff ) >> ( 32 - 24 ) ;
	out[23] |= ( in[53] & 0x3fff ) << 6 ;
	out[23] |= in[54] << 20 ;

	out[24] = ( in[54] & 0x3fff ) >> ( 32 - 20 ) ;
	out[24] |= ( in[55] & 0x3fff ) << 2 ;
	out[24] |= ( in[56] & 0x3fff ) << 16 ;
	out[24] |= in[57] << 30 ;

	out[25] = ( in[57] & 0x3fff ) >> ( 32 - 30 ) ;
	out[25] |= ( in[58] & 0x3fff ) << 12 ;
	out[25] |= in[59] << 26 ;

	out[26] = ( in[59] & 0x3fff ) >> ( 32 - 26 ) ;
	out[26] |= ( in[60] & 0x3fff ) << 8 ;
	out[26] |= in[61] << 22 ;

	out[27] = ( in[61] & 0x3fff ) >> ( 32 - 22 ) ;
	out[27] |= ( in[62] & 0x3fff ) << 4 ;
	out[27] |= in[63] << 18 ;
}


// 15-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c15(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x7fff ) << 0 ;
	out[0] |= ( in[1] & 0x7fff ) << 15 ;
	out[0] |= in[2] << 30 ;

	out[1] = ( in[2] & 0x7fff ) >> ( 32 - 30 ) ;
	out[1] |= ( in[3] & 0x7fff ) << 13 ;
	out[1] |= in[4] << 28 ;

	out[2] = ( in[4] & 0x7fff ) >> ( 32 - 28 ) ;
	out[2] |= ( in[5] & 0x7fff ) << 11 ;
	out[2] |= in[6] << 26 ;

	out[3] = ( in[6] & 0x7fff ) >> ( 32 - 26 ) ;
	out[3] |= ( in[7] & 0x7fff ) << 9 ;
	out[3] |= in[8] << 24 ;

	out[4] = ( in[8] & 0x7fff ) >> ( 32 - 24 ) ;
	out[4] |= ( in[9] & 0x7fff ) << 7 ;
	out[4] |= in[10] << 22 ;

	out[5] = ( in[10] & 0x7fff ) >> ( 32 - 22 ) ;
	out[5] |= ( in[11] & 0x7fff ) << 5 ;
	out[5] |= in[12] << 20 ;

	out[6] = ( in[12] & 0x7fff ) >> ( 32 - 20 ) ;
	out[6] |= ( in[13] & 0x7fff ) << 3 ;
	out[6] |= in[14] << 18 ;

	out[7] = ( in[14] & 0x7fff ) >> ( 32 - 18 ) ;
	out[7] |= ( in[15] & 0x7fff ) << 1 ;
	out[7] |= ( in[16] & 0x7fff ) << 16 ;
	out[7] |= in[17] << 31 ;

	out[8] = ( in[17] & 0x7fff ) >> ( 32 - 31 ) ;
	out[8] |= ( in[18] & 0x7fff ) << 14 ;
	out[8] |= in[19] << 29 ;

	out[9] = ( in[19] & 0x7fff ) >> ( 32 - 29 ) ;
	out[9] |= ( in[20] & 0x7fff ) << 12 ;
	out[9] |= in[21] << 27 ;

	out[10] = ( in[21] & 0x7fff ) >> ( 32 - 27 ) ;
	out[10] |= ( in[22] & 0x7fff ) << 10 ;
	out[10] |= in[23] << 25 ;

	out[11] = ( in[23] & 0x7fff ) >> ( 32 - 25 ) ;
	out[11] |= ( in[24] & 0x7fff ) << 8 ;
	out[11] |= in[25] << 23 ;

	out[12] = ( in[25] & 0x7fff ) >> ( 32 - 23 ) ;
	out[12] |= ( in[26] & 0x7fff ) << 6 ;
	out[12] |= in[27] << 21 ;

	out[13] = ( in[27] & 0x7fff ) >> ( 32 - 21 ) ;
	out[13] |= ( in[28] & 0x7fff ) << 4 ;
	out[13] |= in[29] << 19 ;

	out[14] = ( in[29] & 0x7fff ) >> ( 32 - 19 ) ;
	out[14] |= ( in[30] & 0x7fff ) << 2 ;
	out[14] |= in[31] << 17 ;

	out[15] = ( in[32] & 0x7fff ) << 0 ;
	out[15] |= ( in[33] & 0x7fff ) << 15 ;
	out[15] |= in[34] << 30 ;

	out[16] = ( in[34] & 0x7fff ) >> ( 32 - 30 ) ;
	out[16] |= ( in[35] & 0x7fff ) << 13 ;
	out[16] |= in[36] << 28 ;

	out[17] = ( in[36] & 0x7fff ) >> ( 32 - 28 ) ;
	out[17] |= ( in[37] & 0x7fff ) << 11 ;
	out[17] |= in[38] << 26 ;

	out[18] = ( in[38] & 0x7fff ) >> ( 32 - 26 ) ;
	out[18] |= ( in[39] & 0x7fff ) << 9 ;
	out[18] |= in[40] << 24 ;

	out[19] = ( in[40] & 0x7fff ) >> ( 32 - 24 ) ;
	out[19] |= ( in[41] & 0x7fff ) << 7 ;
	out[19] |= in[42] << 22 ;

	out[20] = ( in[42] & 0x7fff ) >> ( 32 - 22 ) ;
	out[20] |= ( in[43] & 0x7fff ) << 5 ;
	out[20] |= in[44] << 20 ;

	out[21] = ( in[44] & 0x7fff ) >> ( 32 - 20 ) ;
	out[21] |= ( in[45] & 0x7fff ) << 3 ;
	out[21] |= in[46] << 18 ;

	out[22] = ( in[46] & 0x7fff ) >> ( 32 - 18 ) ;
	out[22] |= ( in[47] & 0x7fff ) << 1 ;
	out[22] |= ( in[48] & 0x7fff ) << 16 ;
	out[22] |= in[49] << 31 ;

	out[23] = ( in[49] & 0x7fff ) >> ( 32 - 31 ) ;
	out[23] |= ( in[50] & 0x7fff ) << 14 ;
	out[23] |= in[51] << 29 ;

	out[24] = ( in[51] & 0x7fff ) >> ( 32 - 29 ) ;
	out[24] |= ( in[52] & 0x7fff ) << 12 ;
	out[24] |= in[53] << 27 ;

	out[25] = ( in[53] & 0x7fff ) >> ( 32 - 27 ) ;
	out[25] |= ( in[54] & 0x7fff ) << 10 ;
	out[25] |= in[55] << 25 ;

	out[26] = ( in[55] & 0x7fff ) >> ( 32 - 25 ) ;
	out[26] |= ( in[56] & 0x7fff ) << 8 ;
	out[26] |= in[57] << 23 ;

	out[27] = ( in[57] & 0x7fff ) >> ( 32 - 23 ) ;
	out[27] |= ( in[58] & 0x7fff ) << 6 ;
	out[27] |= in[59] << 21 ;

	out[28] = ( in[59] & 0x7fff ) >> ( 32 - 21 ) ;
	out[28] |= ( in[60] & 0x7fff ) << 4 ;
	out[28] |= in[61] << 19 ;

	out[29] = ( in[61] & 0x7fff ) >> ( 32 - 19 ) ;
	out[29] |= ( in[62] & 0x7fff ) << 2 ;
	out[29] |= in[63] << 17 ;
}


// 16-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c16(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0xffff ) << 0 ;
	out[0] |= in[1] << 16 ;

	out[1] = ( in[2] & 0xffff ) << 0 ;
	out[1] |= in[3] << 16 ;

	out[2] = ( in[4] & 0xffff ) << 0 ;
	out[2] |= in[5] << 16 ;

	out[3] = ( in[6] & 0xffff ) << 0 ;
	out[3] |= in[7] << 16 ;

	out[4] = ( in[8] & 0xffff ) << 0 ;
	out[4] |= in[9] << 16 ;

	out[5] = ( in[10] & 0xffff ) << 0 ;
	out[5] |= in[11] << 16 ;

	out[6] = ( in[12] & 0xffff ) << 0 ;
	out[6] |= in[13] << 16 ;

	out[7] = ( in[14] & 0xffff ) << 0 ;
	out[7] |= in[15] << 16 ;

	out[8] = ( in[16] & 0xffff ) << 0 ;
	out[8] |= in[17] << 16 ;

	out[9] = ( in[18] & 0xffff ) << 0 ;
	out[9] |= in[19] << 16 ;

	out[10] = ( in[20] & 0xffff ) << 0 ;
	out[10] |= in[21] << 16 ;

	out[11] = ( in[22] & 0xffff ) << 0 ;
	out[11] |= in[23] << 16 ;

	out[12] = ( in[24] & 0xffff ) << 0 ;
	out[12] |= in[25] << 16 ;

	out[13] = ( in[26] & 0xffff ) << 0 ;
	out[13] |= in[27] << 16 ;

	out[14] = ( in[28] & 0xffff ) << 0 ;
	out[14] |= in[29] << 16 ;

	out[15] = ( in[30] & 0xffff ) << 0 ;
	out[15] |= in[31] << 16 ;

	out[16] = ( in[32] & 0xffff ) << 0 ;
	out[16] |= in[33] << 16 ;

	out[17] = ( in[34] & 0xffff ) << 0 ;
	out[17] |= in[35] << 16 ;

	out[18] = ( in[36] & 0xffff ) << 0 ;
	out[18] |= in[37] << 16 ;

	out[19] = ( in[38] & 0xffff ) << 0 ;
	out[19] |= in[39] << 16 ;

	out[20] = ( in[40] & 0xffff ) << 0 ;
	out[20] |= in[41] << 16 ;

	out[21] = ( in[42] & 0xffff ) << 0 ;
	out[21] |= in[43] << 16 ;

	out[22] = ( in[44] & 0xffff ) << 0 ;
	out[22] |= in[45] << 16 ;

	out[23] = ( in[46] & 0xffff ) << 0 ;
	out[23] |= in[47] << 16 ;

	out[24] = ( in[48] & 0xffff ) << 0 ;
	out[24] |= in[49] << 16 ;

	out[25] = ( in[50] & 0xffff ) << 0 ;
	out[25] |= in[51] << 16 ;

	out[26] = ( in[52] & 0xffff ) << 0 ;
	out[26] |= in[53] << 16 ;

	out[27] = ( in[54] & 0xffff ) << 0 ;
	out[27] |= in[55] << 16 ;

	out[28] = ( in[56] & 0xffff ) << 0 ;
	out[28] |= in[57] << 16 ;

	out[29] = ( in[58] & 0xffff ) << 0 ;
	out[29] |= in[59] << 16 ;

	out[30] = ( in[60] & 0xffff ) << 0 ;
	out[30] |= in[61] << 16 ;

	out[31] = ( in[62] & 0xffff ) << 0 ;
	out[31] |= in[63] << 16 ;
}


// 17-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c17(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x01ffff ) << 0 ;
	out[0] |= in[1] << 17 ;

	out[1] = ( in[1] & 0x01ffff ) >> ( 32 - 17 ) ;
	out[1] |= ( in[2] & 0x01ffff ) << 2 ;
	out[1] |= in[3] << 19 ;

	out[2] = ( in[3] & 0x01ffff ) >> ( 32 - 19 ) ;
	out[2] |= ( in[4] & 0x01ffff ) << 4 ;
	out[2] |= in[5] << 21 ;

	out[3] = ( in[5] & 0x01ffff ) >> ( 32 - 21 ) ;
	out[3] |= ( in[6] & 0x01ffff ) << 6 ;
	out[3] |= in[7] << 23 ;

	out[4] = ( in[7] & 0x01ffff ) >> ( 32 - 23 ) ;
	out[4] |= ( in[8] & 0x01ffff ) << 8 ;
	out[4] |= in[9] << 25 ;

	out[5] = ( in[9] & 0x01ffff ) >> ( 32 - 25 ) ;
	out[5] |= ( in[10] & 0x01ffff ) << 10 ;
	out[5] |= in[11] << 27 ;

	out[6] = ( in[11] & 0x01ffff ) >> ( 32 - 27 ) ;
	out[6] |= ( in[12] & 0x01ffff ) << 12 ;
	out[6] |= in[13] << 29 ;

	out[7] = ( in[13] & 0x01ffff ) >> ( 32 - 29 ) ;
	out[7] |= ( in[14] & 0x01ffff ) << 14 ;
	out[7] |= in[15] << 31 ;

	out[8] = ( in[15] & 0x01ffff ) >> ( 32 - 31 ) ;
	out[8] |= in[16] << 16 ;

	out[9] = ( in[16] & 0x01ffff ) >> ( 32 - 16 ) ;
	out[9] |= ( in[17] & 0x01ffff ) << 1 ;
	out[9] |= in[18] << 18 ;

	out[10] = ( in[18] & 0x01ffff ) >> ( 32 - 18 ) ;
	out[10] |= ( in[19] & 0x01ffff ) << 3 ;
	out[10] |= in[20] << 20 ;

	out[11] = ( in[20] & 0x01ffff ) >> ( 32 - 20 ) ;
	out[11] |= ( in[21] & 0x01ffff ) << 5 ;
	out[11] |= in[22] << 22 ;

	out[12] = ( in[22] & 0x01ffff ) >> ( 32 - 22 ) ;
	out[12] |= ( in[23] & 0x01ffff ) << 7 ;
	out[12] |= in[24] << 24 ;

	out[13] = ( in[24] & 0x01ffff ) >> ( 32 - 24 ) ;
	out[13] |= ( in[25] & 0x01ffff ) << 9 ;
	out[13] |= in[26] << 26 ;

	out[14] = ( in[26] & 0x01ffff ) >> ( 32 - 26 ) ;
	out[14] |= ( in[27] & 0x01ffff ) << 11 ;
	out[14] |= in[28] << 28 ;

	out[15] = ( in[28] & 0x01ffff ) >> ( 32 - 28 ) ;
	out[15] |= ( in[29] & 0x01ffff ) << 13 ;
	out[15] |= in[30] << 30 ;

	out[16] = ( in[30] & 0x01ffff ) >> ( 32 - 30 ) ;
	out[16] |= in[31] << 15 ;

	out[17] = ( in[32] & 0x01ffff ) << 0 ;
	out[17] |= in[33] << 17 ;

	out[18] = ( in[33] & 0x01ffff ) >> ( 32 - 17 ) ;
	out[18] |= ( in[34] & 0x01ffff ) << 2 ;
	out[18] |= in[35] << 19 ;

	out[19] = ( in[35] & 0x01ffff ) >> ( 32 - 19 ) ;
	out[19] |= ( in[36] & 0x01ffff ) << 4 ;
	out[19] |= in[37] << 21 ;

	out[20] = ( in[37] & 0x01ffff ) >> ( 32 - 21 ) ;
	out[20] |= ( in[38] & 0x01ffff ) << 6 ;
	out[20] |= in[39] << 23 ;

	out[21] = ( in[39] & 0x01ffff ) >> ( 32 - 23 ) ;
	out[21] |= ( in[40] & 0x01ffff ) << 8 ;
	out[21] |= in[41] << 25 ;

	out[22] = ( in[41] & 0x01ffff ) >> ( 32 - 25 ) ;
	out[22] |= ( in[42] & 0x01ffff ) << 10 ;
	out[22] |= in[43] << 27 ;

	out[23] = ( in[43] & 0x01ffff ) >> ( 32 - 27 ) ;
	out[23] |= ( in[44] & 0x01ffff ) << 12 ;
	out[23] |= in[45] << 29 ;

	out[24] = ( in[45] & 0x01ffff ) >> ( 32 - 29 ) ;
	out[24] |= ( in[46] & 0x01ffff ) << 14 ;
	out[24] |= in[47] << 31 ;

	out[25] = ( in[47] & 0x01ffff ) >> ( 32 - 31 ) ;
	out[25] |= in[48] << 16 ;

	out[26] = ( in[48] & 0x01ffff ) >> ( 32 - 16 ) ;
	out[26] |= ( in[49] & 0x01ffff ) << 1 ;
	out[26] |= in[50] << 18 ;

	out[27] = ( in[50] & 0x01ffff ) >> ( 32 - 18 ) ;
	out[27] |= ( in[51] & 0x01ffff ) << 3 ;
	out[27] |= in[52] << 20 ;

	out[28] = ( in[52] & 0x01ffff ) >> ( 32 - 20 ) ;
	out[28] |= ( in[53] & 0x01ffff ) << 5 ;
	out[28] |= in[54] << 22 ;

	out[29] = ( in[54] & 0x01ffff ) >> ( 32 - 22 ) ;
	out[29] |= ( in[55] & 0x01ffff ) << 7 ;
	out[29] |= in[56] << 24 ;

	out[30] = ( in[56] & 0x01ffff ) >> ( 32 - 24 ) ;
	out[30] |= ( in[57] & 0x01ffff ) << 9 ;
	out[30] |= in[58] << 26 ;

	out[31] = ( in[58] & 0x01ffff ) >> ( 32 - 26 ) ;
	out[31] |= ( in[59] & 0x01ffff ) << 11 ;
	out[31] |= in[60] << 28 ;

	out[32] = ( in[60] & 0x01ffff ) >> ( 32 - 28 ) ;
	out[32] |= ( in[61] & 0x01ffff ) << 13 ;
	out[32] |= in[62] << 30 ;

	out[33] = ( in[62] & 0x01ffff ) >> ( 32 - 30 ) ;
	out[33] |= in[63] << 15 ;
}


// 18-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c18(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x03ffff ) << 0 ;
	out[0] |= in[1] << 18 ;

	out[1] = ( in[1] & 0x03ffff ) >> ( 32 - 18 ) ;
	out[1] |= ( in[2] & 0x03ffff ) << 4 ;
	out[1] |= in[3] << 22 ;

	out[2] = ( in[3] & 0x03ffff ) >> ( 32 - 22 ) ;
	out[2] |= ( in[4] & 0x03ffff ) << 8 ;
	out[2] |= in[5] << 26 ;

	out[3] = ( in[5] & 0x03ffff ) >> ( 32 - 26 ) ;
	out[3] |= ( in[6] & 0x03ffff ) << 12 ;
	out[3] |= in[7] << 30 ;

	out[4] = ( in[7] & 0x03ffff ) >> ( 32 - 30 ) ;
	out[4] |= in[8] << 16 ;

	out[5] = ( in[8] & 0x03ffff ) >> ( 32 - 16 ) ;
	out[5] |= ( in[9] & 0x03ffff ) << 2 ;
	out[5] |= in[10] << 20 ;

	out[6] = ( in[10] & 0x03ffff ) >> ( 32 - 20 ) ;
	out[6] |= ( in[11] & 0x03ffff ) << 6 ;
	out[6] |= in[12] << 24 ;

	out[7] = ( in[12] & 0x03ffff ) >> ( 32 - 24 ) ;
	out[7] |= ( in[13] & 0x03ffff ) << 10 ;
	out[7] |= in[14] << 28 ;

	out[8] = ( in[14] & 0x03ffff ) >> ( 32 - 28 ) ;
	out[8] |= in[15] << 14 ;

	out[9] = ( in[16] & 0x03ffff ) << 0 ;
	out[9] |= in[17] << 18 ;

	out[10] = ( in[17] & 0x03ffff ) >> ( 32 - 18 ) ;
	out[10] |= ( in[18] & 0x03ffff ) << 4 ;
	out[10] |= in[19] << 22 ;

	out[11] = ( in[19] & 0x03ffff ) >> ( 32 - 22 ) ;
	out[11] |= ( in[20] & 0x03ffff ) << 8 ;
	out[11] |= in[21] << 26 ;

	out[12] = ( in[21] & 0x03ffff ) >> ( 32 - 26 ) ;
	out[12] |= ( in[22] & 0x03ffff ) << 12 ;
	out[12] |= in[23] << 30 ;

	out[13] = ( in[23] & 0x03ffff ) >> ( 32 - 30 ) ;
	out[13] |= in[24] << 16 ;

	out[14] = ( in[24] & 0x03ffff ) >> ( 32 - 16 ) ;
	out[14] |= ( in[25] & 0x03ffff ) << 2 ;
	out[14] |= in[26] << 20 ;

	out[15] = ( in[26] & 0x03ffff ) >> ( 32 - 20 ) ;
	out[15] |= ( in[27] & 0x03ffff ) << 6 ;
	out[15] |= in[28] << 24 ;

	out[16] = ( in[28] & 0x03ffff ) >> ( 32 - 24 ) ;
	out[16] |= ( in[29] & 0x03ffff ) << 10 ;
	out[16] |= in[30] << 28 ;

	out[17] = ( in[30] & 0x03ffff ) >> ( 32 - 28 ) ;
	out[17] |= in[31] << 14 ;

	out[18] = ( in[32] & 0x03ffff ) << 0 ;
	out[18] |= in[33] << 18 ;

	out[19] = ( in[33] & 0x03ffff ) >> ( 32 - 18 ) ;
	out[19] |= ( in[34] & 0x03ffff ) << 4 ;
	out[19] |= in[35] << 22 ;

	out[20] = ( in[35] & 0x03ffff ) >> ( 32 - 22 ) ;
	out[20] |= ( in[36] & 0x03ffff ) << 8 ;
	out[20] |= in[37] << 26 ;

	out[21] = ( in[37] & 0x03ffff ) >> ( 32 - 26 ) ;
	out[21] |= ( in[38] & 0x03ffff ) << 12 ;
	out[21] |= in[39] << 30 ;

	out[22] = ( in[39] & 0x03ffff ) >> ( 32 - 30 ) ;
	out[22] |= in[40] << 16 ;

	out[23] = ( in[40] & 0x03ffff ) >> ( 32 - 16 ) ;
	out[23] |= ( in[41] & 0x03ffff ) << 2 ;
	out[23] |= in[42] << 20 ;

	out[24] = ( in[42] & 0x03ffff ) >> ( 32 - 20 ) ;
	out[24] |= ( in[43] & 0x03ffff ) << 6 ;
	out[24] |= in[44] << 24 ;

	out[25] = ( in[44] & 0x03ffff ) >> ( 32 - 24 ) ;
	out[25] |= ( in[45] & 0x03ffff ) << 10 ;
	out[25] |= in[46] << 28 ;

	out[26] = ( in[46] & 0x03ffff ) >> ( 32 - 28 ) ;
	out[26] |= in[47] << 14 ;

	out[27] = ( in[48] & 0x03ffff ) << 0 ;
	out[27] |= in[49] << 18 ;

	out[28] = ( in[49] & 0x03ffff ) >> ( 32 - 18 ) ;
	out[28] |= ( in[50] & 0x03ffff ) << 4 ;
	out[28] |= in[51] << 22 ;

	out[29] = ( in[51] & 0x03ffff ) >> ( 32 - 22 ) ;
	out[29] |= ( in[52] & 0x03ffff ) << 8 ;
	out[29] |= in[53] << 26 ;

	out[30] = ( in[53] & 0x03ffff ) >> ( 32 - 26 ) ;
	out[30] |= ( in[54] & 0x03ffff ) << 12 ;
	out[30] |= in[55] << 30 ;

	out[31] = ( in[55] & 0x03ffff ) >> ( 32 - 30 ) ;
	out[31] |= in[56] << 16 ;

	out[32] = ( in[56] & 0x03ffff ) >> ( 32 - 16 ) ;
	out[32] |= ( in[57] & 0x03ffff ) << 2 ;
	out[32] |= in[58] << 20 ;

	out[33] = ( in[58] & 0x03ffff ) >> ( 32 - 20 ) ;
	out[33] |= ( in[59] & 0x03ffff ) << 6 ;
	out[33] |= in[60] << 24 ;

	out[34] = ( in[60] & 0x03ffff ) >> ( 32 - 24 ) ;
	out[34] |= ( in[61] & 0x03ffff ) << 10 ;
	out[34] |= in[62] << 28 ;

	out[35] = ( in[62] & 0x03ffff ) >> ( 32 - 28 ) ;
	out[35] |= in[63] << 14 ;
}


// 19-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c19(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x07ffff ) << 0 ;
	out[0] |= in[1] << 19 ;

	out[1] = ( in[1] & 0x07ffff ) >> ( 32 - 19 ) ;
	out[1] |= ( in[2] & 0x07ffff ) << 6 ;
	out[1] |= in[3] << 25 ;

	out[2] = ( in[3] & 0x07ffff ) >> ( 32 - 25 ) ;
	out[2] |= ( in[4] & 0x07ffff ) << 12 ;
	out[2] |= in[5] << 31 ;

	out[3] = ( in[5] & 0x07ffff ) >> ( 32 - 31 ) ;
	out[3] |= in[6] << 18 ;

	out[4] = ( in[6] & 0x07ffff ) >> ( 32 - 18 ) ;
	out[4] |= ( in[7] & 0x07ffff ) << 5 ;
	out[4] |= in[8] << 24 ;

	out[5] = ( in[8] & 0x07ffff ) >> ( 32 - 24 ) ;
	out[5] |= ( in[9] & 0x07ffff ) << 11 ;
	out[5] |= in[10] << 30 ;

	out[6] = ( in[10] & 0x07ffff ) >> ( 32 - 30 ) ;
	out[6] |= in[11] << 17 ;

	out[7] = ( in[11] & 0x07ffff ) >> ( 32 - 17 ) ;
	out[7] |= ( in[12] & 0x07ffff ) << 4 ;
	out[7] |= in[13] << 23 ;

	out[8] = ( in[13] & 0x07ffff ) >> ( 32 - 23 ) ;
	out[8] |= ( in[14] & 0x07ffff ) << 10 ;
	out[8] |= in[15] << 29 ;

	out[9] = ( in[15] & 0x07ffff ) >> ( 32 - 29 ) ;
	out[9] |= in[16] << 16 ;

	out[10] = ( in[16] & 0x07ffff ) >> ( 32 - 16 ) ;
	out[10] |= ( in[17] & 0x07ffff ) << 3 ;
	out[10] |= in[18] << 22 ;

	out[11] = ( in[18] & 0x07ffff ) >> ( 32 - 22 ) ;
	out[11] |= ( in[19] & 0x07ffff ) << 9 ;
	out[11] |= in[20] << 28 ;

	out[12] = ( in[20] & 0x07ffff ) >> ( 32 - 28 ) ;
	out[12] |= in[21] << 15 ;

	out[13] = ( in[21] & 0x07ffff ) >> ( 32 - 15 ) ;
	out[13] |= ( in[22] & 0x07ffff ) << 2 ;
	out[13] |= in[23] << 21 ;

	out[14] = ( in[23] & 0x07ffff ) >> ( 32 - 21 ) ;
	out[14] |= ( in[24] & 0x07ffff ) << 8 ;
	out[14] |= in[25] << 27 ;

	out[15] = ( in[25] & 0x07ffff ) >> ( 32 - 27 ) ;
	out[15] |= in[26] << 14 ;

	out[16] = ( in[26] & 0x07ffff ) >> ( 32 - 14 ) ;
	out[16] |= ( in[27] & 0x07ffff ) << 1 ;
	out[16] |= in[28] << 20 ;

	out[17] = ( in[28] & 0x07ffff ) >> ( 32 - 20 ) ;
	out[17] |= ( in[29] & 0x07ffff ) << 7 ;
	out[17] |= in[30] << 26 ;

	out[18] = ( in[30] & 0x07ffff ) >> ( 32 - 26 ) ;
	out[18] |= in[31] << 13 ;

	out[19] = ( in[32] & 0x07ffff ) << 0 ;
	out[19] |= in[33] << 19 ;

	out[20] = ( in[33] & 0x07ffff ) >> ( 32 - 19 ) ;
	out[20] |= ( in[34] & 0x07ffff ) << 6 ;
	out[20] |= in[35] << 25 ;

	out[21] = ( in[35] & 0x07ffff ) >> ( 32 - 25 ) ;
	out[21] |= ( in[36] & 0x07ffff ) << 12 ;
	out[21] |= in[37] << 31 ;

	out[22] = ( in[37] & 0x07ffff ) >> ( 32 - 31 ) ;
	out[22] |= in[38] << 18 ;

	out[23] = ( in[38] & 0x07ffff ) >> ( 32 - 18 ) ;
	out[23] |= ( in[39] & 0x07ffff ) << 5 ;
	out[23] |= in[40] << 24 ;

	out[24] = ( in[40] & 0x07ffff ) >> ( 32 - 24 ) ;
	out[24] |= ( in[41] & 0x07ffff ) << 11 ;
	out[24] |= in[42] << 30 ;

	out[25] = ( in[42] & 0x07ffff ) >> ( 32 - 30 ) ;
	out[25] |= in[43] << 17 ;

	out[26] = ( in[43] & 0x07ffff ) >> ( 32 - 17 ) ;
	out[26] |= ( in[44] & 0x07ffff ) << 4 ;
	out[26] |= in[45] << 23 ;

	out[27] = ( in[45] & 0x07ffff ) >> ( 32 - 23 ) ;
	out[27] |= ( in[46] & 0x07ffff ) << 10 ;
	out[27] |= in[47] << 29 ;

	out[28] = ( in[47] & 0x07ffff ) >> ( 32 - 29 ) ;
	out[28] |= in[48] << 16 ;

	out[29] = ( in[48] & 0x07ffff ) >> ( 32 - 16 ) ;
	out[29] |= ( in[49] & 0x07ffff ) << 3 ;
	out[29] |= in[50] << 22 ;

	out[30] = ( in[50] & 0x07ffff ) >> ( 32 - 22 ) ;
	out[30] |= ( in[51] & 0x07ffff ) << 9 ;
	out[30] |= in[52] << 28 ;

	out[31] = ( in[52] & 0x07ffff ) >> ( 32 - 28 ) ;
	out[31] |= in[53] << 15 ;

	out[32] = ( in[53] & 0x07ffff ) >> ( 32 - 15 ) ;
	out[32] |= ( in[54] & 0x07ffff ) << 2 ;
	out[32] |= in[55] << 21 ;

	out[33] = ( in[55] & 0x07ffff ) >> ( 32 - 21 ) ;
	out[33] |= ( in[56] & 0x07ffff ) << 8 ;
	out[33] |= in[57] << 27 ;

	out[34] = ( in[57] & 0x07ffff ) >> ( 32 - 27 ) ;
	out[34] |= in[58] << 14 ;

	out[35] = ( in[58] & 0x07ffff ) >> ( 32 - 14 ) ;
	out[35] |= ( in[59] & 0x07ffff ) << 1 ;
	out[35] |= in[60] << 20 ;

	out[36] = ( in[60] & 0x07ffff ) >> ( 32 - 20 ) ;
	out[36] |= ( in[61] & 0x07ffff ) << 7 ;
	out[36] |= in[62] << 26 ;

	out[37] = ( in[62] & 0x07ffff ) >> ( 32 - 26 ) ;
	out[37] |= in[63] << 13 ;
}


// 20-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c20(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x0fffff ) << 0 ;
	out[0] |= in[1] << 20 ;

	out[1] = ( in[1] & 0x0fffff ) >> ( 32 - 20 ) ;
	out[1] |= ( in[2] & 0x0fffff ) << 8 ;
	out[1] |= in[3] << 28 ;

	out[2] = ( in[3] & 0x0fffff ) >> ( 32 - 28 ) ;
	out[2] |= in[4] << 16 ;

	out[3] = ( in[4] & 0x0fffff ) >> ( 32 - 16 ) ;
	out[3] |= ( in[5] & 0x0fffff ) << 4 ;
	out[3] |= in[6] << 24 ;

	out[4] = ( in[6] & 0x0fffff ) >> ( 32 - 24 ) ;
	out[4] |= in[7] << 12 ;

	out[5] = ( in[8] & 0x0fffff ) << 0 ;
	out[5] |= in[9] << 20 ;

	out[6] = ( in[9] & 0x0fffff ) >> ( 32 - 20 ) ;
	out[6] |= ( in[10] & 0x0fffff ) << 8 ;
	out[6] |= in[11] << 28 ;

	out[7] = ( in[11] & 0x0fffff ) >> ( 32 - 28 ) ;
	out[7] |= in[12] << 16 ;

	out[8] = ( in[12] & 0x0fffff ) >> ( 32 - 16 ) ;
	out[8] |= ( in[13] & 0x0fffff ) << 4 ;
	out[8] |= in[14] << 24 ;

	out[9] = ( in[14] & 0x0fffff ) >> ( 32 - 24 ) ;
	out[9] |= in[15] << 12 ;

	out[10] = ( in[16] & 0x0fffff ) << 0 ;
	out[10] |= in[17] << 20 ;

	out[11] = ( in[17] & 0x0fffff ) >> ( 32 - 20 ) ;
	out[11] |= ( in[18] & 0x0fffff ) << 8 ;
	out[11] |= in[19] << 28 ;

	out[12] = ( in[19] & 0x0fffff ) >> ( 32 - 28 ) ;
	out[12] |= in[20] << 16 ;

	out[13] = ( in[20] & 0x0fffff ) >> ( 32 - 16 ) ;
	out[13] |= ( in[21] & 0x0fffff ) << 4 ;
	out[13] |= in[22] << 24 ;

	out[14] = ( in[22] & 0x0fffff ) >> ( 32 - 24 ) ;
	out[14] |= in[23] << 12 ;

	out[15] = ( in[24] & 0x0fffff ) << 0 ;
	out[15] |= in[25] << 20 ;

	out[16] = ( in[25] & 0x0fffff ) >> ( 32 - 20 ) ;
	out[16] |= ( in[26] & 0x0fffff ) << 8 ;
	out[16] |= in[27] << 28 ;

	out[17] = ( in[27] & 0x0fffff ) >> ( 32 - 28 ) ;
	out[17] |= in[28] << 16 ;

	out[18] = ( in[28] & 0x0fffff ) >> ( 32 - 16 ) ;
	out[18] |= ( in[29] & 0x0fffff ) << 4 ;
	out[18] |= in[30] << 24 ;

	out[19] = ( in[30] & 0x0fffff ) >> ( 32 - 24 ) ;
	out[19] |= in[31] << 12 ;

	out[20] = ( in[32] & 0x0fffff ) << 0 ;
	out[20] |= in[33] << 20 ;

	out[21] = ( in[33] & 0x0fffff ) >> ( 32 - 20 ) ;
	out[21] |= ( in[34] & 0x0fffff ) << 8 ;
	out[21] |= in[35] << 28 ;

	out[22] = ( in[35] & 0x0fffff ) >> ( 32 - 28 ) ;
	out[22] |= in[36] << 16 ;

	out[23] = ( in[36] & 0x0fffff ) >> ( 32 - 16 ) ;
	out[23] |= ( in[37] & 0x0fffff ) << 4 ;
	out[23] |= in[38] << 24 ;

	out[24] = ( in[38] & 0x0fffff ) >> ( 32 - 24 ) ;
	out[24] |= in[39] << 12 ;

	out[25] = ( in[40] & 0x0fffff ) << 0 ;
	out[25] |= in[41] << 20 ;

	out[26] = ( in[41] & 0x0fffff ) >> ( 32 - 20 ) ;
	out[26] |= ( in[42] & 0x0fffff ) << 8 ;
	out[26] |= in[43] << 28 ;

	out[27] = ( in[43] & 0x0fffff ) >> ( 32 - 28 ) ;
	out[27] |= in[44] << 16 ;

	out[28] = ( in[44] & 0x0fffff ) >> ( 32 - 16 ) ;
	out[28] |= ( in[45] & 0x0fffff ) << 4 ;
	out[28] |= in[46] << 24 ;

	out[29] = ( in[46] & 0x0fffff ) >> ( 32 - 24 ) ;
	out[29] |= in[47] << 12 ;

	out[30] = ( in[48] & 0x0fffff ) << 0 ;
	out[30] |= in[49] << 20 ;

	out[31] = ( in[49] & 0x0fffff ) >> ( 32 - 20 ) ;
	out[31] |= ( in[50] & 0x0fffff ) << 8 ;
	out[31] |= in[51] << 28 ;

	out[32] = ( in[51] & 0x0fffff ) >> ( 32 - 28 ) ;
	out[32] |= in[52] << 16 ;

	out[33] = ( in[52] & 0x0fffff ) >> ( 32 - 16 ) ;
	out[33] |= ( in[53] & 0x0fffff ) << 4 ;
	out[33] |= in[54] << 24 ;

	out[34] = ( in[54] & 0x0fffff ) >> ( 32 - 24 ) ;
	out[34] |= in[55] << 12 ;

	out[35] = ( in[56] & 0x0fffff ) << 0 ;
	out[35] |= in[57] << 20 ;

	out[36] = ( in[57] & 0x0fffff ) >> ( 32 - 20 ) ;
	out[36] |= ( in[58] & 0x0fffff ) << 8 ;
	out[36] |= in[59] << 28 ;

	out[37] = ( in[59] & 0x0fffff ) >> ( 32 - 28 ) ;
	out[37] |= in[60] << 16 ;

	out[38] = ( in[60] & 0x0fffff ) >> ( 32 - 16 ) ;
	out[38] |= ( in[61] & 0x0fffff ) << 4 ;
	out[38] |= in[62] << 24 ;

	out[39] = ( in[62] & 0x0fffff ) >> ( 32 - 24 ) ;
	out[39] |= in[63] << 12 ;
}


// 21-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c21(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x1fffff ) << 0 ;
	out[0] |= in[1] << 21 ;

	out[1] = ( in[1] & 0x1fffff ) >> ( 32 - 21 ) ;
	out[1] |= ( in[2] & 0x1fffff ) << 10 ;
	out[1] |= in[3] << 31 ;

	out[2] = ( in[3] & 0x1fffff ) >> ( 32 - 31 ) ;
	out[2] |= in[4] << 20 ;

	out[3] = ( in[4] & 0x1fffff ) >> ( 32 - 20 ) ;
	out[3] |= ( in[5] & 0x1fffff ) << 9 ;
	out[3] |= in[6] << 30 ;

	out[4] = ( in[6] & 0x1fffff ) >> ( 32 - 30 ) ;
	out[4] |= in[7] << 19 ;

	out[5] = ( in[7] & 0x1fffff ) >> ( 32 - 19 ) ;
	out[5] |= ( in[8] & 0x1fffff ) << 8 ;
	out[5] |= in[9] << 29 ;

	out[6] = ( in[9] & 0x1fffff ) >> ( 32 - 29 ) ;
	out[6] |= in[10] << 18 ;

	out[7] = ( in[10] & 0x1fffff ) >> ( 32 - 18 ) ;
	out[7] |= ( in[11] & 0x1fffff ) << 7 ;
	out[7] |= in[12] << 28 ;

	out[8] = ( in[12] & 0x1fffff ) >> ( 32 - 28 ) ;
	out[8] |= in[13] << 17 ;

	out[9] = ( in[13] & 0x1fffff ) >> ( 32 - 17 ) ;
	out[9] |= ( in[14] & 0x1fffff ) << 6 ;
	out[9] |= in[15] << 27 ;

	out[10] = ( in[15] & 0x1fffff ) >> ( 32 - 27 ) ;
	out[10] |= in[16] << 16 ;

	out[11] = ( in[16] & 0x1fffff ) >> ( 32 - 16 ) ;
	out[11] |= ( in[17] & 0x1fffff ) << 5 ;
	out[11] |= in[18] << 26 ;

	out[12] = ( in[18] & 0x1fffff ) >> ( 32 - 26 ) ;
	out[12] |= in[19] << 15 ;

	out[13] = ( in[19] & 0x1fffff ) >> ( 32 - 15 ) ;
	out[13] |= ( in[20] & 0x1fffff ) << 4 ;
	out[13] |= in[21] << 25 ;

	out[14] = ( in[21] & 0x1fffff ) >> ( 32 - 25 ) ;
	out[14] |= in[22] << 14 ;

	out[15] = ( in[22] & 0x1fffff ) >> ( 32 - 14 ) ;
	out[15] |= ( in[23] & 0x1fffff ) << 3 ;
	out[15] |= in[24] << 24 ;

	out[16] = ( in[24] & 0x1fffff ) >> ( 32 - 24 ) ;
	out[16] |= in[25] << 13 ;

	out[17] = ( in[25] & 0x1fffff ) >> ( 32 - 13 ) ;
	out[17] |= ( in[26] & 0x1fffff ) << 2 ;
	out[17] |= in[27] << 23 ;

	out[18] = ( in[27] & 0x1fffff ) >> ( 32 - 23 ) ;
	out[18] |= in[28] << 12 ;

	out[19] = ( in[28] & 0x1fffff ) >> ( 32 - 12 ) ;
	out[19] |= ( in[29] & 0x1fffff ) << 1 ;
	out[19] |= in[30] << 22 ;

	out[20] = ( in[30] & 0x1fffff ) >> ( 32 - 22 ) ;
	out[20] |= in[31] << 11 ;

	out[21] = ( in[32] & 0x1fffff ) << 0 ;
	out[21] |= in[33] << 21 ;

	out[22] = ( in[33] & 0x1fffff ) >> ( 32 - 21 ) ;
	out[22] |= ( in[34] & 0x1fffff ) << 10 ;
	out[22] |= in[35] << 31 ;

	out[23] = ( in[35] & 0x1fffff ) >> ( 32 - 31 ) ;
	out[23] |= in[36] << 20 ;

	out[24] = ( in[36] & 0x1fffff ) >> ( 32 - 20 ) ;
	out[24] |= ( in[37] & 0x1fffff ) << 9 ;
	out[24] |= in[38] << 30 ;

	out[25] = ( in[38] & 0x1fffff ) >> ( 32 - 30 ) ;
	out[25] |= in[39] << 19 ;

	out[26] = ( in[39] & 0x1fffff ) >> ( 32 - 19 ) ;
	out[26] |= ( in[40] & 0x1fffff ) << 8 ;
	out[26] |= in[41] << 29 ;

	out[27] = ( in[41] & 0x1fffff ) >> ( 32 - 29 ) ;
	out[27] |= in[42] << 18 ;

	out[28] = ( in[42] & 0x1fffff ) >> ( 32 - 18 ) ;
	out[28] |= ( in[43] & 0x1fffff ) << 7 ;
	out[28] |= in[44] << 28 ;

	out[29] = ( in[44] & 0x1fffff ) >> ( 32 - 28 ) ;
	out[29] |= in[45] << 17 ;

	out[30] = ( in[45] & 0x1fffff ) >> ( 32 - 17 ) ;
	out[30] |= ( in[46] & 0x1fffff ) << 6 ;
	out[30] |= in[47] << 27 ;

	out[31] = ( in[47] & 0x1fffff ) >> ( 32 - 27 ) ;
	out[31] |= in[48] << 16 ;

	out[32] = ( in[48] & 0x1fffff ) >> ( 32 - 16 ) ;
	out[32] |= ( in[49] & 0x1fffff ) << 5 ;
	out[32] |= in[50] << 26 ;

	out[33] = ( in[50] & 0x1fffff ) >> ( 32 - 26 ) ;
	out[33] |= in[51] << 15 ;

	out[34] = ( in[51] & 0x1fffff ) >> ( 32 - 15 ) ;
	out[34] |= ( in[52] & 0x1fffff ) << 4 ;
	out[34] |= in[53] << 25 ;

	out[35] = ( in[53] & 0x1fffff ) >> ( 32 - 25 ) ;
	out[35] |= in[54] << 14 ;

	out[36] = ( in[54] & 0x1fffff ) >> ( 32 - 14 ) ;
	out[36] |= ( in[55] & 0x1fffff ) << 3 ;
	out[36] |= in[56] << 24 ;

	out[37] = ( in[56] & 0x1fffff ) >> ( 32 - 24 ) ;
	out[37] |= in[57] << 13 ;

	out[38] = ( in[57] & 0x1fffff ) >> ( 32 - 13 ) ;
	out[38] |= ( in[58] & 0x1fffff ) << 2 ;
	out[38] |= in[59] << 23 ;

	out[39] = ( in[59] & 0x1fffff ) >> ( 32 - 23 ) ;
	out[39] |= in[60] << 12 ;

	out[40] = ( in[60] & 0x1fffff ) >> ( 32 - 12 ) ;
	out[40] |= ( in[61] & 0x1fffff ) << 1 ;
	out[40] |= in[62] << 22 ;

	out[41] = ( in[62] & 0x1fffff ) >> ( 32 - 22 ) ;
	out[41] |= in[63] << 11 ;
}


// 22-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c22(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x3fffff ) << 0 ;
	out[0] |= in[1] << 22 ;

	out[1] = ( in[1] & 0x3fffff ) >> ( 32 - 22 ) ;
	out[1] |= in[2] << 12 ;

	out[2] = ( in[2] & 0x3fffff ) >> ( 32 - 12 ) ;
	out[2] |= ( in[3] & 0x3fffff ) << 2 ;
	out[2] |= in[4] << 24 ;

	out[3] = ( in[4] & 0x3fffff ) >> ( 32 - 24 ) ;
	out[3] |= in[5] << 14 ;

	out[4] = ( in[5] & 0x3fffff ) >> ( 32 - 14 ) ;
	out[4] |= ( in[6] & 0x3fffff ) << 4 ;
	out[4] |= in[7] << 26 ;

	out[5] = ( in[7] & 0x3fffff ) >> ( 32 - 26 ) ;
	out[5] |= in[8] << 16 ;

	out[6] = ( in[8] & 0x3fffff ) >> ( 32 - 16 ) ;
	out[6] |= ( in[9] & 0x3fffff ) << 6 ;
	out[6] |= in[10] << 28 ;

	out[7] = ( in[10] & 0x3fffff ) >> ( 32 - 28 ) ;
	out[7] |= in[11] << 18 ;

	out[8] = ( in[11] & 0x3fffff ) >> ( 32 - 18 ) ;
	out[8] |= ( in[12] & 0x3fffff ) << 8 ;
	out[8] |= in[13] << 30 ;

	out[9] = ( in[13] & 0x3fffff ) >> ( 32 - 30 ) ;
	out[9] |= in[14] << 20 ;

	out[10] = ( in[14] & 0x3fffff ) >> ( 32 - 20 ) ;
	out[10] |= in[15] << 10 ;

	out[11] = ( in[16] & 0x3fffff ) << 0 ;
	out[11] |= in[17] << 22 ;

	out[12] = ( in[17] & 0x3fffff ) >> ( 32 - 22 ) ;
	out[12] |= in[18] << 12 ;

	out[13] = ( in[18] & 0x3fffff ) >> ( 32 - 12 ) ;
	out[13] |= ( in[19] & 0x3fffff ) << 2 ;
	out[13] |= in[20] << 24 ;

	out[14] = ( in[20] & 0x3fffff ) >> ( 32 - 24 ) ;
	out[14] |= in[21] << 14 ;

	out[15] = ( in[21] & 0x3fffff ) >> ( 32 - 14 ) ;
	out[15] |= ( in[22] & 0x3fffff ) << 4 ;
	out[15] |= in[23] << 26 ;

	out[16] = ( in[23] & 0x3fffff ) >> ( 32 - 26 ) ;
	out[16] |= in[24] << 16 ;

	out[17] = ( in[24] & 0x3fffff ) >> ( 32 - 16 ) ;
	out[17] |= ( in[25] & 0x3fffff ) << 6 ;
	out[17] |= in[26] << 28 ;

	out[18] = ( in[26] & 0x3fffff ) >> ( 32 - 28 ) ;
	out[18] |= in[27] << 18 ;

	out[19] = ( in[27] & 0x3fffff ) >> ( 32 - 18 ) ;
	out[19] |= ( in[28] & 0x3fffff ) << 8 ;
	out[19] |= in[29] << 30 ;

	out[20] = ( in[29] & 0x3fffff ) >> ( 32 - 30 ) ;
	out[20] |= in[30] << 20 ;

	out[21] = ( in[30] & 0x3fffff ) >> ( 32 - 20 ) ;
	out[21] |= in[31] << 10 ;

	out[22] = ( in[32] & 0x3fffff ) << 0 ;
	out[22] |= in[33] << 22 ;

	out[23] = ( in[33] & 0x3fffff ) >> ( 32 - 22 ) ;
	out[23] |= in[34] << 12 ;

	out[24] = ( in[34] & 0x3fffff ) >> ( 32 - 12 ) ;
	out[24] |= ( in[35] & 0x3fffff ) << 2 ;
	out[24] |= in[36] << 24 ;

	out[25] = ( in[36] & 0x3fffff ) >> ( 32 - 24 ) ;
	out[25] |= in[37] << 14 ;

	out[26] = ( in[37] & 0x3fffff ) >> ( 32 - 14 ) ;
	out[26] |= ( in[38] & 0x3fffff ) << 4 ;
	out[26] |= in[39] << 26 ;

	out[27] = ( in[39] & 0x3fffff ) >> ( 32 - 26 ) ;
	out[27] |= in[40] << 16 ;

	out[28] = ( in[40] & 0x3fffff ) >> ( 32 - 16 ) ;
	out[28] |= ( in[41] & 0x3fffff ) << 6 ;
	out[28] |= in[42] << 28 ;

	out[29] = ( in[42] & 0x3fffff ) >> ( 32 - 28 ) ;
	out[29] |= in[43] << 18 ;

	out[30] = ( in[43] & 0x3fffff ) >> ( 32 - 18 ) ;
	out[30] |= ( in[44] & 0x3fffff ) << 8 ;
	out[30] |= in[45] << 30 ;

	out[31] = ( in[45] & 0x3fffff ) >> ( 32 - 30 ) ;
	out[31] |= in[46] << 20 ;

	out[32] = ( in[46] & 0x3fffff ) >> ( 32 - 20 ) ;
	out[32] |= in[47] << 10 ;

	out[33] = ( in[48] & 0x3fffff ) << 0 ;
	out[33] |= in[49] << 22 ;

	out[34] = ( in[49] & 0x3fffff ) >> ( 32 - 22 ) ;
	out[34] |= in[50] << 12 ;

	out[35] = ( in[50] & 0x3fffff ) >> ( 32 - 12 ) ;
	out[35] |= ( in[51] & 0x3fffff ) << 2 ;
	out[35] |= in[52] << 24 ;

	out[36] = ( in[52] & 0x3fffff ) >> ( 32 - 24 ) ;
	out[36] |= in[53] << 14 ;

	out[37] = ( in[53] & 0x3fffff ) >> ( 32 - 14 ) ;
	out[37] |= ( in[54] & 0x3fffff ) << 4 ;
	out[37] |= in[55] << 26 ;

	out[38] = ( in[55] & 0x3fffff ) >> ( 32 - 26 ) ;
	out[38] |= in[56] << 16 ;

	out[39] = ( in[56] & 0x3fffff ) >> ( 32 - 16 ) ;
	out[39] |= ( in[57] & 0x3fffff ) << 6 ;
	out[39] |= in[58] << 28 ;

	out[40] = ( in[58] & 0x3fffff ) >> ( 32 - 28 ) ;
	out[40] |= in[59] << 18 ;

	out[41] = ( in[59] & 0x3fffff ) >> ( 32 - 18 ) ;
	out[41] |= ( in[60] & 0x3fffff ) << 8 ;
	out[41] |= in[61] << 30 ;

	out[42] = ( in[61] & 0x3fffff ) >> ( 32 - 30 ) ;
	out[42] |= in[62] << 20 ;

	out[43] = ( in[62] & 0x3fffff ) >> ( 32 - 20 ) ;
	out[43] |= in[63] << 10 ;
}


// 23-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c23(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x7fffff ) << 0 ;
	out[0] |= in[1] << 23 ;

	out[1] = ( in[1] & 0x7fffff ) >> ( 32 - 23 ) ;
	out[1] |= in[2] << 14 ;

	out[2] = ( in[2] & 0x7fffff ) >> ( 32 - 14 ) ;
	out[2] |= ( in[3] & 0x7fffff ) << 5 ;
	out[2] |= in[4] << 28 ;

	out[3] = ( in[4] & 0x7fffff ) >> ( 32 - 28 ) ;
	out[3] |= in[5] << 19 ;

	out[4] = ( in[5] & 0x7fffff ) >> ( 32 - 19 ) ;
	out[4] |= in[6] << 10 ;

	out[5] = ( in[6] & 0x7fffff ) >> ( 32 - 10 ) ;
	out[5] |= ( in[7] & 0x7fffff ) << 1 ;
	out[5] |= in[8] << 24 ;

	out[6] = ( in[8] & 0x7fffff ) >> ( 32 - 24 ) ;
	out[6] |= in[9] << 15 ;

	out[7] = ( in[9] & 0x7fffff ) >> ( 32 - 15 ) ;
	out[7] |= ( in[10] & 0x7fffff ) << 6 ;
	out[7] |= in[11] << 29 ;

	out[8] = ( in[11] & 0x7fffff ) >> ( 32 - 29 ) ;
	out[8] |= in[12] << 20 ;

	out[9] = ( in[12] & 0x7fffff ) >> ( 32 - 20 ) ;
	out[9] |= in[13] << 11 ;

	out[10] = ( in[13] & 0x7fffff ) >> ( 32 - 11 ) ;
	out[10] |= ( in[14] & 0x7fffff ) << 2 ;
	out[10] |= in[15] << 25 ;

	out[11] = ( in[15] & 0x7fffff ) >> ( 32 - 25 ) ;
	out[11] |= in[16] << 16 ;

	out[12] = ( in[16] & 0x7fffff ) >> ( 32 - 16 ) ;
	out[12] |= ( in[17] & 0x7fffff ) << 7 ;
	out[12] |= in[18] << 30 ;

	out[13] = ( in[18] & 0x7fffff ) >> ( 32 - 30 ) ;
	out[13] |= in[19] << 21 ;

	out[14] = ( in[19] & 0x7fffff ) >> ( 32 - 21 ) ;
	out[14] |= in[20] << 12 ;

	out[15] = ( in[20] & 0x7fffff ) >> ( 32 - 12 ) ;
	out[15] |= ( in[21] & 0x7fffff ) << 3 ;
	out[15] |= in[22] << 26 ;

	out[16] = ( in[22] & 0x7fffff ) >> ( 32 - 26 ) ;
	out[16] |= in[23] << 17 ;

	out[17] = ( in[23] & 0x7fffff ) >> ( 32 - 17 ) ;
	out[17] |= ( in[24] & 0x7fffff ) << 8 ;
	out[17] |= in[25] << 31 ;

	out[18] = ( in[25] & 0x7fffff ) >> ( 32 - 31 ) ;
	out[18] |= in[26] << 22 ;

	out[19] = ( in[26] & 0x7fffff ) >> ( 32 - 22 ) ;
	out[19] |= in[27] << 13 ;

	out[20] = ( in[27] & 0x7fffff ) >> ( 32 - 13 ) ;
	out[20] |= ( in[28] & 0x7fffff ) << 4 ;
	out[20] |= in[29] << 27 ;

	out[21] = ( in[29] & 0x7fffff ) >> ( 32 - 27 ) ;
	out[21] |= in[30] << 18 ;

	out[22] = ( in[30] & 0x7fffff ) >> ( 32 - 18 ) ;
	out[22] |= in[31] << 9 ;

	out[23] = ( in[32] & 0x7fffff ) << 0 ;
	out[23] |= in[33] << 23 ;

	out[24] = ( in[33] & 0x7fffff ) >> ( 32 - 23 ) ;
	out[24] |= in[34] << 14 ;

	out[25] = ( in[34] & 0x7fffff ) >> ( 32 - 14 ) ;
	out[25] |= ( in[35] & 0x7fffff ) << 5 ;
	out[25] |= in[36] << 28 ;

	out[26] = ( in[36] & 0x7fffff ) >> ( 32 - 28 ) ;
	out[26] |= in[37] << 19 ;

	out[27] = ( in[37] & 0x7fffff ) >> ( 32 - 19 ) ;
	out[27] |= in[38] << 10 ;

	out[28] = ( in[38] & 0x7fffff ) >> ( 32 - 10 ) ;
	out[28] |= ( in[39] & 0x7fffff ) << 1 ;
	out[28] |= in[40] << 24 ;

	out[29] = ( in[40] & 0x7fffff ) >> ( 32 - 24 ) ;
	out[29] |= in[41] << 15 ;

	out[30] = ( in[41] & 0x7fffff ) >> ( 32 - 15 ) ;
	out[30] |= ( in[42] & 0x7fffff ) << 6 ;
	out[30] |= in[43] << 29 ;

	out[31] = ( in[43] & 0x7fffff ) >> ( 32 - 29 ) ;
	out[31] |= in[44] << 20 ;

	out[32] = ( in[44] & 0x7fffff ) >> ( 32 - 20 ) ;
	out[32] |= in[45] << 11 ;

	out[33] = ( in[45] & 0x7fffff ) >> ( 32 - 11 ) ;
	out[33] |= ( in[46] & 0x7fffff ) << 2 ;
	out[33] |= in[47] << 25 ;

	out[34] = ( in[47] & 0x7fffff ) >> ( 32 - 25 ) ;
	out[34] |= in[48] << 16 ;

	out[35] = ( in[48] & 0x7fffff ) >> ( 32 - 16 ) ;
	out[35] |= ( in[49] & 0x7fffff ) << 7 ;
	out[35] |= in[50] << 30 ;

	out[36] = ( in[50] & 0x7fffff ) >> ( 32 - 30 ) ;
	out[36] |= in[51] << 21 ;

	out[37] = ( in[51] & 0x7fffff ) >> ( 32 - 21 ) ;
	out[37] |= in[52] << 12 ;

	out[38] = ( in[52] & 0x7fffff ) >> ( 32 - 12 ) ;
	out[38] |= ( in[53] & 0x7fffff ) << 3 ;
	out[38] |= in[54] << 26 ;

	out[39] = ( in[54] & 0x7fffff ) >> ( 32 - 26 ) ;
	out[39] |= in[55] << 17 ;

	out[40] = ( in[55] & 0x7fffff ) >> ( 32 - 17 ) ;
	out[40] |= ( in[56] & 0x7fffff ) << 8 ;
	out[40] |= in[57] << 31 ;

	out[41] = ( in[57] & 0x7fffff ) >> ( 32 - 31 ) ;
	out[41] |= in[58] << 22 ;

	out[42] = ( in[58] & 0x7fffff ) >> ( 32 - 22 ) ;
	out[42] |= in[59] << 13 ;

	out[43] = ( in[59] & 0x7fffff ) >> ( 32 - 13 ) ;
	out[43] |= ( in[60] & 0x7fffff ) << 4 ;
	out[43] |= in[61] << 27 ;

	out[44] = ( in[61] & 0x7fffff ) >> ( 32 - 27 ) ;
	out[44] |= in[62] << 18 ;

	out[45] = ( in[62] & 0x7fffff ) >> ( 32 - 18 ) ;
	out[45] |= in[63] << 9 ;
}


// 24-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c24(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0xffffff ) << 0 ;
	out[0] |= in[1] << 24 ;

	out[1] = ( in[1] & 0xffffff ) >> ( 32 - 24 ) ;
	out[1] |= in[2] << 16 ;

	out[2] = ( in[2] & 0xffffff ) >> ( 32 - 16 ) ;
	out[2] |= in[3] << 8 ;

	out[3] = ( in[4] & 0xffffff ) << 0 ;
	out[3] |= in[5] << 24 ;

	out[4] = ( in[5] & 0xffffff ) >> ( 32 - 24 ) ;
	out[4] |= in[6] << 16 ;

	out[5] = ( in[6] & 0xffffff ) >> ( 32 - 16 ) ;
	out[5] |= in[7] << 8 ;

	out[6] = ( in[8] & 0xffffff ) << 0 ;
	out[6] |= in[9] << 24 ;

	out[7] = ( in[9] & 0xffffff ) >> ( 32 - 24 ) ;
	out[7] |= in[10] << 16 ;

	out[8] = ( in[10] & 0xffffff ) >> ( 32 - 16 ) ;
	out[8] |= in[11] << 8 ;

	out[9] = ( in[12] & 0xffffff ) << 0 ;
	out[9] |= in[13] << 24 ;

	out[10] = ( in[13] & 0xffffff ) >> ( 32 - 24 ) ;
	out[10] |= in[14] << 16 ;

	out[11] = ( in[14] & 0xffffff ) >> ( 32 - 16 ) ;
	out[11] |= in[15] << 8 ;

	out[12] = ( in[16] & 0xffffff ) << 0 ;
	out[12] |= in[17] << 24 ;

	out[13] = ( in[17] & 0xffffff ) >> ( 32 - 24 ) ;
	out[13] |= in[18] << 16 ;

	out[14] = ( in[18] & 0xffffff ) >> ( 32 - 16 ) ;
	out[14] |= in[19] << 8 ;

	out[15] = ( in[20] & 0xffffff ) << 0 ;
	out[15] |= in[21] << 24 ;

	out[16] = ( in[21] & 0xffffff ) >> ( 32 - 24 ) ;
	out[16] |= in[22] << 16 ;

	out[17] = ( in[22] & 0xffffff ) >> ( 32 - 16 ) ;
	out[17] |= in[23] << 8 ;

	out[18] = ( in[24] & 0xffffff ) << 0 ;
	out[18] |= in[25] << 24 ;

	out[19] = ( in[25] & 0xffffff ) >> ( 32 - 24 ) ;
	out[19] |= in[26] << 16 ;

	out[20] = ( in[26] & 0xffffff ) >> ( 32 - 16 ) ;
	out[20] |= in[27] << 8 ;

	out[21] = ( in[28] & 0xffffff ) << 0 ;
	out[21] |= in[29] << 24 ;

	out[22] = ( in[29] & 0xffffff ) >> ( 32 - 24 ) ;
	out[22] |= in[30] << 16 ;

	out[23] = ( in[30] & 0xffffff ) >> ( 32 - 16 ) ;
	out[23] |= in[31] << 8 ;

	out[24] = ( in[32] & 0xffffff ) << 0 ;
	out[24] |= in[33] << 24 ;

	out[25] = ( in[33] & 0xffffff ) >> ( 32 - 24 ) ;
	out[25] |= in[34] << 16 ;

	out[26] = ( in[34] & 0xffffff ) >> ( 32 - 16 ) ;
	out[26] |= in[35] << 8 ;

	out[27] = ( in[36] & 0xffffff ) << 0 ;
	out[27] |= in[37] << 24 ;

	out[28] = ( in[37] & 0xffffff ) >> ( 32 - 24 ) ;
	out[28] |= in[38] << 16 ;

	out[29] = ( in[38] & 0xffffff ) >> ( 32 - 16 ) ;
	out[29] |= in[39] << 8 ;

	out[30] = ( in[40] & 0xffffff ) << 0 ;
	out[30] |= in[41] << 24 ;

	out[31] = ( in[41] & 0xffffff ) >> ( 32 - 24 ) ;
	out[31] |= in[42] << 16 ;

	out[32] = ( in[42] & 0xffffff ) >> ( 32 - 16 ) ;
	out[32] |= in[43] << 8 ;

	out[33] = ( in[44] & 0xffffff ) << 0 ;
	out[33] |= in[45] << 24 ;

	out[34] = ( in[45] & 0xffffff ) >> ( 32 - 24 ) ;
	out[34] |= in[46] << 16 ;

	out[35] = ( in[46] & 0xffffff ) >> ( 32 - 16 ) ;
	out[35] |= in[47] << 8 ;

	out[36] = ( in[48] & 0xffffff ) << 0 ;
	out[36] |= in[49] << 24 ;

	out[37] = ( in[49] & 0xffffff ) >> ( 32 - 24 ) ;
	out[37] |= in[50] << 16 ;

	out[38] = ( in[50] & 0xffffff ) >> ( 32 - 16 ) ;
	out[38] |= in[51] << 8 ;

	out[39] = ( in[52] & 0xffffff ) << 0 ;
	out[39] |= in[53] << 24 ;

	out[40] = ( in[53] & 0xffffff ) >> ( 32 - 24 ) ;
	out[40] |= in[54] << 16 ;

	out[41] = ( in[54] & 0xffffff ) >> ( 32 - 16 ) ;
	out[41] |= in[55] << 8 ;

	out[42] = ( in[56] & 0xffffff ) << 0 ;
	out[42] |= in[57] << 24 ;

	out[43] = ( in[57] & 0xffffff ) >> ( 32 - 24 ) ;
	out[43] |= in[58] << 16 ;

	out[44] = ( in[58] & 0xffffff ) >> ( 32 - 16 ) ;
	out[44] |= in[59] << 8 ;

	out[45] = ( in[60] & 0xffffff ) << 0 ;
	out[45] |= in[61] << 24 ;

	out[46] = ( in[61] & 0xffffff ) >> ( 32 - 24 ) ;
	out[46] |= in[62] << 16 ;

	out[47] = ( in[62] & 0xffffff ) >> ( 32 - 16 ) ;
	out[47] |= in[63] << 8 ;
}


// 25-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c25(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x01ffffff ) << 0 ;
	out[0] |= in[1] << 25 ;

	out[1] = ( in[1] & 0x01ffffff ) >> ( 32 - 25 ) ;
	out[1] |= in[2] << 18 ;

	out[2] = ( in[2] & 0x01ffffff ) >> ( 32 - 18 ) ;
	out[2] |= in[3] << 11 ;

	out[3] = ( in[3] & 0x01ffffff ) >> ( 32 - 11 ) ;
	out[3] |= ( in[4] & 0x01ffffff ) << 4 ;
	out[3] |= in[5] << 29 ;

	out[4] = ( in[5] & 0x01ffffff ) >> ( 32 - 29 ) ;
	out[4] |= in[6] << 22 ;

	out[5] = ( in[6] & 0x01ffffff ) >> ( 32 - 22 ) ;
	out[5] |= in[7] << 15 ;

	out[6] = ( in[7] & 0x01ffffff ) >> ( 32 - 15 ) ;
	out[6] |= in[8] << 8 ;

	out[7] = ( in[8] & 0x01ffffff ) >> ( 32 - 8 ) ;
	out[7] |= ( in[9] & 0x01ffffff ) << 1 ;
	out[7] |= in[10] << 26 ;

	out[8] = ( in[10] & 0x01ffffff ) >> ( 32 - 26 ) ;
	out[8] |= in[11] << 19 ;

	out[9] = ( in[11] & 0x01ffffff ) >> ( 32 - 19 ) ;
	out[9] |= in[12] << 12 ;

	out[10] = ( in[12] & 0x01ffffff ) >> ( 32 - 12 ) ;
	out[10] |= ( in[13] & 0x01ffffff ) << 5 ;
	out[10] |= in[14] << 30 ;

	out[11] = ( in[14] & 0x01ffffff ) >> ( 32 - 30 ) ;
	out[11] |= in[15] << 23 ;

	out[12] = ( in[15] & 0x01ffffff ) >> ( 32 - 23 ) ;
	out[12] |= in[16] << 16 ;

	out[13] = ( in[16] & 0x01ffffff ) >> ( 32 - 16 ) ;
	out[13] |= in[17] << 9 ;

	out[14] = ( in[17] & 0x01ffffff ) >> ( 32 - 9 ) ;
	out[14] |= ( in[18] & 0x01ffffff ) << 2 ;
	out[14] |= in[19] << 27 ;

	out[15] = ( in[19] & 0x01ffffff ) >> ( 32 - 27 ) ;
	out[15] |= in[20] << 20 ;

	out[16] = ( in[20] & 0x01ffffff ) >> ( 32 - 20 ) ;
	out[16] |= in[21] << 13 ;

	out[17] = ( in[21] & 0x01ffffff ) >> ( 32 - 13 ) ;
	out[17] |= ( in[22] & 0x01ffffff ) << 6 ;
	out[17] |= in[23] << 31 ;

	out[18] = ( in[23] & 0x01ffffff ) >> ( 32 - 31 ) ;
	out[18] |= in[24] << 24 ;

	out[19] = ( in[24] & 0x01ffffff ) >> ( 32 - 24 ) ;
	out[19] |= in[25] << 17 ;

	out[20] = ( in[25] & 0x01ffffff ) >> ( 32 - 17 ) ;
	out[20] |= in[26] << 10 ;

	out[21] = ( in[26] & 0x01ffffff ) >> ( 32 - 10 ) ;
	out[21] |= ( in[27] & 0x01ffffff ) << 3 ;
	out[21] |= in[28] << 28 ;

	out[22] = ( in[28] & 0x01ffffff ) >> ( 32 - 28 ) ;
	out[22] |= in[29] << 21 ;

	out[23] = ( in[29] & 0x01ffffff ) >> ( 32 - 21 ) ;
	out[23] |= in[30] << 14 ;

	out[24] = ( in[30] & 0x01ffffff ) >> ( 32 - 14 ) ;
	out[24] |= in[31] << 7 ;

	out[25] = ( in[32] & 0x01ffffff ) << 0 ;
	out[25] |= in[33] << 25 ;

	out[26] = ( in[33] & 0x01ffffff ) >> ( 32 - 25 ) ;
	out[26] |= in[34] << 18 ;

	out[27] = ( in[34] & 0x01ffffff ) >> ( 32 - 18 ) ;
	out[27] |= in[35] << 11 ;

	out[28] = ( in[35] & 0x01ffffff ) >> ( 32 - 11 ) ;
	out[28] |= ( in[36] & 0x01ffffff ) << 4 ;
	out[28] |= in[37] << 29 ;

	out[29] = ( in[37] & 0x01ffffff ) >> ( 32 - 29 ) ;
	out[29] |= in[38] << 22 ;

	out[30] = ( in[38] & 0x01ffffff ) >> ( 32 - 22 ) ;
	out[30] |= in[39] << 15 ;

	out[31] = ( in[39] & 0x01ffffff ) >> ( 32 - 15 ) ;
	out[31] |= in[40] << 8 ;

	out[32] = ( in[40] & 0x01ffffff ) >> ( 32 - 8 ) ;
	out[32] |= ( in[41] & 0x01ffffff ) << 1 ;
	out[32] |= in[42] << 26 ;

	out[33] = ( in[42] & 0x01ffffff ) >> ( 32 - 26 ) ;
	out[33] |= in[43] << 19 ;

	out[34] = ( in[43] & 0x01ffffff ) >> ( 32 - 19 ) ;
	out[34] |= in[44] << 12 ;

	out[35] = ( in[44] & 0x01ffffff ) >> ( 32 - 12 ) ;
	out[35] |= ( in[45] & 0x01ffffff ) << 5 ;
	out[35] |= in[46] << 30 ;

	out[36] = ( in[46] & 0x01ffffff ) >> ( 32 - 30 ) ;
	out[36] |= in[47] << 23 ;

	out[37] = ( in[47] & 0x01ffffff ) >> ( 32 - 23 ) ;
	out[37] |= in[48] << 16 ;

	out[38] = ( in[48] & 0x01ffffff ) >> ( 32 - 16 ) ;
	out[38] |= in[49] << 9 ;

	out[39] = ( in[49] & 0x01ffffff ) >> ( 32 - 9 ) ;
	out[39] |= ( in[50] & 0x01ffffff ) << 2 ;
	out[39] |= in[51] << 27 ;

	out[40] = ( in[51] & 0x01ffffff ) >> ( 32 - 27 ) ;
	out[40] |= in[52] << 20 ;

	out[41] = ( in[52] & 0x01ffffff ) >> ( 32 - 20 ) ;
	out[41] |= in[53] << 13 ;

	out[42] = ( in[53] & 0x01ffffff ) >> ( 32 - 13 ) ;
	out[42] |= ( in[54] & 0x01ffffff ) << 6 ;
	out[42] |= in[55] << 31 ;

	out[43] = ( in[55] & 0x01ffffff ) >> ( 32 - 31 ) ;
	out[43] |= in[56] << 24 ;

	out[44] = ( in[56] & 0x01ffffff ) >> ( 32 - 24 ) ;
	out[44] |= in[57] << 17 ;

	out[45] = ( in[57] & 0x01ffffff ) >> ( 32 - 17 ) ;
	out[45] |= in[58] << 10 ;

	out[46] = ( in[58] & 0x01ffffff ) >> ( 32 - 10 ) ;
	out[46] |= ( in[59] & 0x01ffffff ) << 3 ;
	out[46] |= in[60] << 28 ;

	out[47] = ( in[60] & 0x01ffffff ) >> ( 32 - 28 ) ;
	out[47] |= in[61] << 21 ;

	out[48] = ( in[61] & 0x01ffffff ) >> ( 32 - 21 ) ;
	out[48] |= in[62] << 14 ;

	out[49] = ( in[62] & 0x01ffffff ) >> ( 32 - 14 ) ;
	out[49] |= in[63] << 7 ;
}


// 26-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c26(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x03ffffff ) << 0 ;
	out[0] |= in[1] << 26 ;

	out[1] = ( in[1] & 0x03ffffff ) >> ( 32 - 26 ) ;
	out[1] |= in[2] << 20 ;

	out[2] = ( in[2] & 0x03ffffff ) >> ( 32 - 20 ) ;
	out[2] |= in[3] << 14 ;

	out[3] = ( in[3] & 0x03ffffff ) >> ( 32 - 14 ) ;
	out[3] |= in[4] << 8 ;

	out[4] = ( in[4] & 0x03ffffff ) >> ( 32 - 8 ) ;
	out[4] |= ( in[5] & 0x03ffffff ) << 2 ;
	out[4] |= in[6] << 28 ;

	out[5] = ( in[6] & 0x03ffffff ) >> ( 32 - 28 ) ;
	out[5] |= in[7] << 22 ;

	out[6] = ( in[7] & 0x03ffffff ) >> ( 32 - 22 ) ;
	out[6] |= in[8] << 16 ;

	out[7] = ( in[8] & 0x03ffffff ) >> ( 32 - 16 ) ;
	out[7] |= in[9] << 10 ;

	out[8] = ( in[9] & 0x03ffffff ) >> ( 32 - 10 ) ;
	out[8] |= ( in[10] & 0x03ffffff ) << 4 ;
	out[8] |= in[11] << 30 ;

	out[9] = ( in[11] & 0x03ffffff ) >> ( 32 - 30 ) ;
	out[9] |= in[12] << 24 ;

	out[10] = ( in[12] & 0x03ffffff ) >> ( 32 - 24 ) ;
	out[10] |= in[13] << 18 ;

	out[11] = ( in[13] & 0x03ffffff ) >> ( 32 - 18 ) ;
	out[11] |= in[14] << 12 ;

	out[12] = ( in[14] & 0x03ffffff ) >> ( 32 - 12 ) ;
	out[12] |= in[15] << 6 ;

	out[13] = ( in[16] & 0x03ffffff ) << 0 ;
	out[13] |= in[17] << 26 ;

	out[14] = ( in[17] & 0x03ffffff ) >> ( 32 - 26 ) ;
	out[14] |= in[18] << 20 ;

	out[15] = ( in[18] & 0x03ffffff ) >> ( 32 - 20 ) ;
	out[15] |= in[19] << 14 ;

	out[16] = ( in[19] & 0x03ffffff ) >> ( 32 - 14 ) ;
	out[16] |= in[20] << 8 ;

	out[17] = ( in[20] & 0x03ffffff ) >> ( 32 - 8 ) ;
	out[17] |= ( in[21] & 0x03ffffff ) << 2 ;
	out[17] |= in[22] << 28 ;

	out[18] = ( in[22] & 0x03ffffff ) >> ( 32 - 28 ) ;
	out[18] |= in[23] << 22 ;

	out[19] = ( in[23] & 0x03ffffff ) >> ( 32 - 22 ) ;
	out[19] |= in[24] << 16 ;

	out[20] = ( in[24] & 0x03ffffff ) >> ( 32 - 16 ) ;
	out[20] |= in[25] << 10 ;

	out[21] = ( in[25] & 0x03ffffff ) >> ( 32 - 10 ) ;
	out[21] |= ( in[26] & 0x03ffffff ) << 4 ;
	out[21] |= in[27] << 30 ;

	out[22] = ( in[27] & 0x03ffffff ) >> ( 32 - 30 ) ;
	out[22] |= in[28] << 24 ;

	out[23] = ( in[28] & 0x03ffffff ) >> ( 32 - 24 ) ;
	out[23] |= in[29] << 18 ;

	out[24] = ( in[29] & 0x03ffffff ) >> ( 32 - 18 ) ;
	out[24] |= in[30] << 12 ;

	out[25] = ( in[30] & 0x03ffffff ) >> ( 32 - 12 ) ;
	out[25] |= in[31] << 6 ;

	out[26] = ( in[32] & 0x03ffffff ) << 0 ;
	out[26] |= in[33] << 26 ;

	out[27] = ( in[33] & 0x03ffffff ) >> ( 32 - 26 ) ;
	out[27] |= in[34] << 20 ;

	out[28] = ( in[34] & 0x03ffffff ) >> ( 32 - 20 ) ;
	out[28] |= in[35] << 14 ;

	out[29] = ( in[35] & 0x03ffffff ) >> ( 32 - 14 ) ;
	out[29] |= in[36] << 8 ;

	out[30] = ( in[36] & 0x03ffffff ) >> ( 32 - 8 ) ;
	out[30] |= ( in[37] & 0x03ffffff ) << 2 ;
	out[30] |= in[38] << 28 ;

	out[31] = ( in[38] & 0x03ffffff ) >> ( 32 - 28 ) ;
	out[31] |= in[39] << 22 ;

	out[32] = ( in[39] & 0x03ffffff ) >> ( 32 - 22 ) ;
	out[32] |= in[40] << 16 ;

	out[33] = ( in[40] & 0x03ffffff ) >> ( 32 - 16 ) ;
	out[33] |= in[41] << 10 ;

	out[34] = ( in[41] & 0x03ffffff ) >> ( 32 - 10 ) ;
	out[34] |= ( in[42] & 0x03ffffff ) << 4 ;
	out[34] |= in[43] << 30 ;

	out[35] = ( in[43] & 0x03ffffff ) >> ( 32 - 30 ) ;
	out[35] |= in[44] << 24 ;

	out[36] = ( in[44] & 0x03ffffff ) >> ( 32 - 24 ) ;
	out[36] |= in[45] << 18 ;

	out[37] = ( in[45] & 0x03ffffff ) >> ( 32 - 18 ) ;
	out[37] |= in[46] << 12 ;

	out[38] = ( in[46] & 0x03ffffff ) >> ( 32 - 12 ) ;
	out[38] |= in[47] << 6 ;

	out[39] = ( in[48] & 0x03ffffff ) << 0 ;
	out[39] |= in[49] << 26 ;

	out[40] = ( in[49] & 0x03ffffff ) >> ( 32 - 26 ) ;
	out[40] |= in[50] << 20 ;

	out[41] = ( in[50] & 0x03ffffff ) >> ( 32 - 20 ) ;
	out[41] |= in[51] << 14 ;

	out[42] = ( in[51] & 0x03ffffff ) >> ( 32 - 14 ) ;
	out[42] |= in[52] << 8 ;

	out[43] = ( in[52] & 0x03ffffff ) >> ( 32 - 8 ) ;
	out[43] |= ( in[53] & 0x03ffffff ) << 2 ;
	out[43] |= in[54] << 28 ;

	out[44] = ( in[54] & 0x03ffffff ) >> ( 32 - 28 ) ;
	out[44] |= in[55] << 22 ;

	out[45] = ( in[55] & 0x03ffffff ) >> ( 32 - 22 ) ;
	out[45] |= in[56] << 16 ;

	out[46] = ( in[56] & 0x03ffffff ) >> ( 32 - 16 ) ;
	out[46] |= in[57] << 10 ;

	out[47] = ( in[57] & 0x03ffffff ) >> ( 32 - 10 ) ;
	out[47] |= ( in[58] & 0x03ffffff ) << 4 ;
	out[47] |= in[59] << 30 ;

	out[48] = ( in[59] & 0x03ffffff ) >> ( 32 - 30 ) ;
	out[48] |= in[60] << 24 ;

	out[49] = ( in[60] & 0x03ffffff ) >> ( 32 - 24 ) ;
	out[49] |= in[61] << 18 ;

	out[50] = ( in[61] & 0x03ffffff ) >> ( 32 - 18 ) ;
	out[50] |= in[62] << 12 ;

	out[51] = ( in[62] & 0x03ffffff ) >> ( 32 - 12 ) ;
	out[51] |= in[63] << 6 ;
}


// 27-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c27(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x07ffffff ) << 0 ;
	out[0] |= in[1] << 27 ;

	out[1] = ( in[1] & 0x07ffffff ) >> ( 32 - 27 ) ;
	out[1] |= in[2] << 22 ;

	out[2] = ( in[2] & 0x07ffffff ) >> ( 32 - 22 ) ;
	out[2] |= in[3] << 17 ;

	out[3] = ( in[3] & 0x07ffffff ) >> ( 32 - 17 ) ;
	out[3] |= in[4] << 12 ;

	out[4] = ( in[4] & 0x07ffffff ) >> ( 32 - 12 ) ;
	out[4] |= in[5] << 7 ;

	out[5] = ( in[5] & 0x07ffffff ) >> ( 32 - 7 ) ;
	out[5] |= ( in[6] & 0x07ffffff ) << 2 ;
	out[5] |= in[7] << 29 ;

	out[6] = ( in[7] & 0x07ffffff ) >> ( 32 - 29 ) ;
	out[6] |= in[8] << 24 ;

	out[7] = ( in[8] & 0x07ffffff ) >> ( 32 - 24 ) ;
	out[7] |= in[9] << 19 ;

	out[8] = ( in[9] & 0x07ffffff ) >> ( 32 - 19 ) ;
	out[8] |= in[10] << 14 ;

	out[9] = ( in[10] & 0x07ffffff ) >> ( 32 - 14 ) ;
	out[9] |= in[11] << 9 ;

	out[10] = ( in[11] & 0x07ffffff ) >> ( 32 - 9 ) ;
	out[10] |= ( in[12] & 0x07ffffff ) << 4 ;
	out[10] |= in[13] << 31 ;

	out[11] = ( in[13] & 0x07ffffff ) >> ( 32 - 31 ) ;
	out[11] |= in[14] << 26 ;

	out[12] = ( in[14] & 0x07ffffff ) >> ( 32 - 26 ) ;
	out[12] |= in[15] << 21 ;

	out[13] = ( in[15] & 0x07ffffff ) >> ( 32 - 21 ) ;
	out[13] |= in[16] << 16 ;

	out[14] = ( in[16] & 0x07ffffff ) >> ( 32 - 16 ) ;
	out[14] |= in[17] << 11 ;

	out[15] = ( in[17] & 0x07ffffff ) >> ( 32 - 11 ) ;
	out[15] |= in[18] << 6 ;

	out[16] = ( in[18] & 0x07ffffff ) >> ( 32 - 6 ) ;
	out[16] |= ( in[19] & 0x07ffffff ) << 1 ;
	out[16] |= in[20] << 28 ;

	out[17] = ( in[20] & 0x07ffffff ) >> ( 32 - 28 ) ;
	out[17] |= in[21] << 23 ;

	out[18] = ( in[21] & 0x07ffffff ) >> ( 32 - 23 ) ;
	out[18] |= in[22] << 18 ;

	out[19] = ( in[22] & 0x07ffffff ) >> ( 32 - 18 ) ;
	out[19] |= in[23] << 13 ;

	out[20] = ( in[23] & 0x07ffffff ) >> ( 32 - 13 ) ;
	out[20] |= in[24] << 8 ;

	out[21] = ( in[24] & 0x07ffffff ) >> ( 32 - 8 ) ;
	out[21] |= ( in[25] & 0x07ffffff ) << 3 ;
	out[21] |= in[26] << 30 ;

	out[22] = ( in[26] & 0x07ffffff ) >> ( 32 - 30 ) ;
	out[22] |= in[27] << 25 ;

	out[23] = ( in[27] & 0x07ffffff ) >> ( 32 - 25 ) ;
	out[23] |= in[28] << 20 ;

	out[24] = ( in[28] & 0x07ffffff ) >> ( 32 - 20 ) ;
	out[24] |= in[29] << 15 ;

	out[25] = ( in[29] & 0x07ffffff ) >> ( 32 - 15 ) ;
	out[25] |= in[30] << 10 ;

	out[26] = ( in[30] & 0x07ffffff ) >> ( 32 - 10 ) ;
	out[26] |= in[31] << 5 ;

	out[27] = ( in[32] & 0x07ffffff ) << 0 ;
	out[27] |= in[33] << 27 ;

	out[28] = ( in[33] & 0x07ffffff ) >> ( 32 - 27 ) ;
	out[28] |= in[34] << 22 ;

	out[29] = ( in[34] & 0x07ffffff ) >> ( 32 - 22 ) ;
	out[29] |= in[35] << 17 ;

	out[30] = ( in[35] & 0x07ffffff ) >> ( 32 - 17 ) ;
	out[30] |= in[36] << 12 ;

	out[31] = ( in[36] & 0x07ffffff ) >> ( 32 - 12 ) ;
	out[31] |= in[37] << 7 ;

	out[32] = ( in[37] & 0x07ffffff ) >> ( 32 - 7 ) ;
	out[32] |= ( in[38] & 0x07ffffff ) << 2 ;
	out[32] |= in[39] << 29 ;

	out[33] = ( in[39] & 0x07ffffff ) >> ( 32 - 29 ) ;
	out[33] |= in[40] << 24 ;

	out[34] = ( in[40] & 0x07ffffff ) >> ( 32 - 24 ) ;
	out[34] |= in[41] << 19 ;

	out[35] = ( in[41] & 0x07ffffff ) >> ( 32 - 19 ) ;
	out[35] |= in[42] << 14 ;

	out[36] = ( in[42] & 0x07ffffff ) >> ( 32 - 14 ) ;
	out[36] |= in[43] << 9 ;

	out[37] = ( in[43] & 0x07ffffff ) >> ( 32 - 9 ) ;
	out[37] |= ( in[44] & 0x07ffffff ) << 4 ;
	out[37] |= in[45] << 31 ;

	out[38] = ( in[45] & 0x07ffffff ) >> ( 32 - 31 ) ;
	out[38] |= in[46] << 26 ;

	out[39] = ( in[46] & 0x07ffffff ) >> ( 32 - 26 ) ;
	out[39] |= in[47] << 21 ;

	out[40] = ( in[47] & 0x07ffffff ) >> ( 32 - 21 ) ;
	out[40] |= in[48] << 16 ;

	out[41] = ( in[48] & 0x07ffffff ) >> ( 32 - 16 ) ;
	out[41] |= in[49] << 11 ;

	out[42] = ( in[49] & 0x07ffffff ) >> ( 32 - 11 ) ;
	out[42] |= in[50] << 6 ;

	out[43] = ( in[50] & 0x07ffffff ) >> ( 32 - 6 ) ;
	out[43] |= ( in[51] & 0x07ffffff ) << 1 ;
	out[43] |= in[52] << 28 ;

	out[44] = ( in[52] & 0x07ffffff ) >> ( 32 - 28 ) ;
	out[44] |= in[53] << 23 ;

	out[45] = ( in[53] & 0x07ffffff ) >> ( 32 - 23 ) ;
	out[45] |= in[54] << 18 ;

	out[46] = ( in[54] & 0x07ffffff ) >> ( 32 - 18 ) ;
	out[46] |= in[55] << 13 ;

	out[47] = ( in[55] & 0x07ffffff ) >> ( 32 - 13 ) ;
	out[47] |= in[56] << 8 ;

	out[48] = ( in[56] & 0x07ffffff ) >> ( 32 - 8 ) ;
	out[48] |= ( in[57] & 0x07ffffff ) << 3 ;
	out[48] |= in[58] << 30 ;

	out[49] = ( in[58] & 0x07ffffff ) >> ( 32 - 30 ) ;
	out[49] |= in[59] << 25 ;

	out[50] = ( in[59] & 0x07ffffff ) >> ( 32 - 25 ) ;
	out[50] |= in[60] << 20 ;

	out[51] = ( in[60] & 0x07ffffff ) >> ( 32 - 20 ) ;
	out[51] |= in[61] << 15 ;

	out[52] = ( in[61] & 0x07ffffff ) >> ( 32 - 15 ) ;
	out[52] |= in[62] << 10 ;

	out[53] = ( in[62] & 0x07ffffff ) >> ( 32 - 10 ) ;
	out[53] |= in[63] << 5 ;
}


// 28-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c28(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x0fffffff ) << 0 ;
	out[0] |= in[1] << 28 ;

	out[1] = ( in[1] & 0x0fffffff ) >> ( 32 - 28 ) ;
	out[1] |= in[2] << 24 ;

	out[2] = ( in[2] & 0x0fffffff ) >> ( 32 - 24 ) ;
	out[2] |= in[3] << 20 ;

	out[3] = ( in[3] & 0x0fffffff ) >> ( 32 - 20 ) ;
	out[3] |= in[4] << 16 ;

	out[4] = ( in[4] & 0x0fffffff ) >> ( 32 - 16 ) ;
	out[4] |= in[5] << 12 ;

	out[5] = ( in[5] & 0x0fffffff ) >> ( 32 - 12 ) ;
	out[5] |= in[6] << 8 ;

	out[6] = ( in[6] & 0x0fffffff ) >> ( 32 - 8 ) ;
	out[6] |= in[7] << 4 ;

	out[7] = ( in[8] & 0x0fffffff ) << 0 ;
	out[7] |= in[9] << 28 ;

	out[8] = ( in[9] & 0x0fffffff ) >> ( 32 - 28 ) ;
	out[8] |= in[10] << 24 ;

	out[9] = ( in[10] & 0x0fffffff ) >> ( 32 - 24 ) ;
	out[9] |= in[11] << 20 ;

	out[10] = ( in[11] & 0x0fffffff ) >> ( 32 - 20 ) ;
	out[10] |= in[12] << 16 ;

	out[11] = ( in[12] & 0x0fffffff ) >> ( 32 - 16 ) ;
	out[11] |= in[13] << 12 ;

	out[12] = ( in[13] & 0x0fffffff ) >> ( 32 - 12 ) ;
	out[12] |= in[14] << 8 ;

	out[13] = ( in[14] & 0x0fffffff ) >> ( 32 - 8 ) ;
	out[13] |= in[15] << 4 ;

	out[14] = ( in[16] & 0x0fffffff ) << 0 ;
	out[14] |= in[17] << 28 ;

	out[15] = ( in[17] & 0x0fffffff ) >> ( 32 - 28 ) ;
	out[15] |= in[18] << 24 ;

	out[16] = ( in[18] & 0x0fffffff ) >> ( 32 - 24 ) ;
	out[16] |= in[19] << 20 ;

	out[17] = ( in[19] & 0x0fffffff ) >> ( 32 - 20 ) ;
	out[17] |= in[20] << 16 ;

	out[18] = ( in[20] & 0x0fffffff ) >> ( 32 - 16 ) ;
	out[18] |= in[21] << 12 ;

	out[19] = ( in[21] & 0x0fffffff ) >> ( 32 - 12 ) ;
	out[19] |= in[22] << 8 ;

	out[20] = ( in[22] & 0x0fffffff ) >> ( 32 - 8 ) ;
	out[20] |= in[23] << 4 ;

	out[21] = ( in[24] & 0x0fffffff ) << 0 ;
	out[21] |= in[25] << 28 ;

	out[22] = ( in[25] & 0x0fffffff ) >> ( 32 - 28 ) ;
	out[22] |= in[26] << 24 ;

	out[23] = ( in[26] & 0x0fffffff ) >> ( 32 - 24 ) ;
	out[23] |= in[27] << 20 ;

	out[24] = ( in[27] & 0x0fffffff ) >> ( 32 - 20 ) ;
	out[24] |= in[28] << 16 ;

	out[25] = ( in[28] & 0x0fffffff ) >> ( 32 - 16 ) ;
	out[25] |= in[29] << 12 ;

	out[26] = ( in[29] & 0x0fffffff ) >> ( 32 - 12 ) ;
	out[26] |= in[30] << 8 ;

	out[27] = ( in[30] & 0x0fffffff ) >> ( 32 - 8 ) ;
	out[27] |= in[31] << 4 ;

	out[28] = ( in[32] & 0x0fffffff ) << 0 ;
	out[28] |= in[33] << 28 ;

	out[29] = ( in[33] & 0x0fffffff ) >> ( 32 - 28 ) ;
	out[29] |= in[34] << 24 ;

	out[30] = ( in[34] & 0x0fffffff ) >> ( 32 - 24 ) ;
	out[30] |= in[35] << 20 ;

	out[31] = ( in[35] & 0x0fffffff ) >> ( 32 - 20 ) ;
	out[31] |= in[36] << 16 ;

	out[32] = ( in[36] & 0x0fffffff ) >> ( 32 - 16 ) ;
	out[32] |= in[37] << 12 ;

	out[33] = ( in[37] & 0x0fffffff ) >> ( 32 - 12 ) ;
	out[33] |= in[38] << 8 ;

	out[34] = ( in[38] & 0x0fffffff ) >> ( 32 - 8 ) ;
	out[34] |= in[39] << 4 ;

	out[35] = ( in[40] & 0x0fffffff ) << 0 ;
	out[35] |= in[41] << 28 ;

	out[36] = ( in[41] & 0x0fffffff ) >> ( 32 - 28 ) ;
	out[36] |= in[42] << 24 ;

	out[37] = ( in[42] & 0x0fffffff ) >> ( 32 - 24 ) ;
	out[37] |= in[43] << 20 ;

	out[38] = ( in[43] & 0x0fffffff ) >> ( 32 - 20 ) ;
	out[38] |= in[44] << 16 ;

	out[39] = ( in[44] & 0x0fffffff ) >> ( 32 - 16 ) ;
	out[39] |= in[45] << 12 ;

	out[40] = ( in[45] & 0x0fffffff ) >> ( 32 - 12 ) ;
	out[40] |= in[46] << 8 ;

	out[41] = ( in[46] & 0x0fffffff ) >> ( 32 - 8 ) ;
	out[41] |= in[47] << 4 ;

	out[42] = ( in[48] & 0x0fffffff ) << 0 ;
	out[42] |= in[49] << 28 ;

	out[43] = ( in[49] & 0x0fffffff ) >> ( 32 - 28 ) ;
	out[43] |= in[50] << 24 ;

	out[44] = ( in[50] & 0x0fffffff ) >> ( 32 - 24 ) ;
	out[44] |= in[51] << 20 ;

	out[45] = ( in[51] & 0x0fffffff ) >> ( 32 - 20 ) ;
	out[45] |= in[52] << 16 ;

	out[46] = ( in[52] & 0x0fffffff ) >> ( 32 - 16 ) ;
	out[46] |= in[53] << 12 ;

	out[47] = ( in[53] & 0x0fffffff ) >> ( 32 - 12 ) ;
	out[47] |= in[54] << 8 ;

	out[48] = ( in[54] & 0x0fffffff ) >> ( 32 - 8 ) ;
	out[48] |= in[55] << 4 ;

	out[49] = ( in[56] & 0x0fffffff ) << 0 ;
	out[49] |= in[57] << 28 ;

	out[50] = ( in[57] & 0x0fffffff ) >> ( 32 - 28 ) ;
	out[50] |= in[58] << 24 ;

	out[51] = ( in[58] & 0x0fffffff ) >> ( 32 - 24 ) ;
	out[51] |= in[59] << 20 ;

	out[52] = ( in[59] & 0x0fffffff ) >> ( 32 - 20 ) ;
	out[52] |= in[60] << 16 ;

	out[53] = ( in[60] & 0x0fffffff ) >> ( 32 - 16 ) ;
	out[53] |= in[61] << 12 ;

	out[54] = ( in[61] & 0x0fffffff ) >> ( 32 - 12 ) ;
	out[54] |= in[62] << 8 ;

	out[55] = ( in[62] & 0x0fffffff ) >> ( 32 - 8 ) ;
	out[55] |= in[63] << 4 ;
}


// 29-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c29(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x1fffffff ) << 0 ;
	out[0] |= in[1] << 29 ;

	out[1] = ( in[1] & 0x1fffffff ) >> ( 32 - 29 ) ;
	out[1] |= in[2] << 26 ;

	out[2] = ( in[2] & 0x1fffffff ) >> ( 32 - 26 ) ;
	out[2] |= in[3] << 23 ;

	out[3] = ( in[3] & 0x1fffffff ) >> ( 32 - 23 ) ;
	out[3] |= in[4] << 20 ;

	out[4] = ( in[4] & 0x1fffffff ) >> ( 32 - 20 ) ;
	out[4] |= in[5] << 17 ;

	out[5] = ( in[5] & 0x1fffffff ) >> ( 32 - 17 ) ;
	out[5] |= in[6] << 14 ;

	out[6] = ( in[6] & 0x1fffffff ) >> ( 32 - 14 ) ;
	out[6] |= in[7] << 11 ;

	out[7] = ( in[7] & 0x1fffffff ) >> ( 32 - 11 ) ;
	out[7] |= in[8] << 8 ;

	out[8] = ( in[8] & 0x1fffffff ) >> ( 32 - 8 ) ;
	out[8] |= in[9] << 5 ;

	out[9] = ( in[9] & 0x1fffffff ) >> ( 32 - 5 ) ;
	out[9] |= ( in[10] & 0x1fffffff ) << 2 ;
	out[9] |= in[11] << 31 ;

	out[10] = ( in[11] & 0x1fffffff ) >> ( 32 - 31 ) ;
	out[10] |= in[12] << 28 ;

	out[11] = ( in[12] & 0x1fffffff ) >> ( 32 - 28 ) ;
	out[11] |= in[13] << 25 ;

	out[12] = ( in[13] & 0x1fffffff ) >> ( 32 - 25 ) ;
	out[12] |= in[14] << 22 ;

	out[13] = ( in[14] & 0x1fffffff ) >> ( 32 - 22 ) ;
	out[13] |= in[15] << 19 ;

	out[14] = ( in[15] & 0x1fffffff ) >> ( 32 - 19 ) ;
	out[14] |= in[16] << 16 ;

	out[15] = ( in[16] & 0x1fffffff ) >> ( 32 - 16 ) ;
	out[15] |= in[17] << 13 ;

	out[16] = ( in[17] & 0x1fffffff ) >> ( 32 - 13 ) ;
	out[16] |= in[18] << 10 ;

	out[17] = ( in[18] & 0x1fffffff ) >> ( 32 - 10 ) ;
	out[17] |= in[19] << 7 ;

	out[18] = ( in[19] & 0x1fffffff ) >> ( 32 - 7 ) ;
	out[18] |= in[20] << 4 ;

	out[19] = ( in[20] & 0x1fffffff ) >> ( 32 - 4 ) ;
	out[19] |= ( in[21] & 0x1fffffff ) << 1 ;
	out[19] |= in[22] << 30 ;

	out[20] = ( in[22] & 0x1fffffff ) >> ( 32 - 30 ) ;
	out[20] |= in[23] << 27 ;

	out[21] = ( in[23] & 0x1fffffff ) >> ( 32 - 27 ) ;
	out[21] |= in[24] << 24 ;

	out[22] = ( in[24] & 0x1fffffff ) >> ( 32 - 24 ) ;
	out[22] |= in[25] << 21 ;

	out[23] = ( in[25] & 0x1fffffff ) >> ( 32 - 21 ) ;
	out[23] |= in[26] << 18 ;

	out[24] = ( in[26] & 0x1fffffff ) >> ( 32 - 18 ) ;
	out[24] |= in[27] << 15 ;

	out[25] = ( in[27] & 0x1fffffff ) >> ( 32 - 15 ) ;
	out[25] |= in[28] << 12 ;

	out[26] = ( in[28] & 0x1fffffff ) >> ( 32 - 12 ) ;
	out[26] |= in[29] << 9 ;

	out[27] = ( in[29] & 0x1fffffff ) >> ( 32 - 9 ) ;
	out[27] |= in[30] << 6 ;

	out[28] = ( in[30] & 0x1fffffff ) >> ( 32 - 6 ) ;
	out[28] |= in[31] << 3 ;

	out[29] = ( in[32] & 0x1fffffff ) << 0 ;
	out[29] |= in[33] << 29 ;

	out[30] = ( in[33] & 0x1fffffff ) >> ( 32 - 29 ) ;
	out[30] |= in[34] << 26 ;

	out[31] = ( in[34] & 0x1fffffff ) >> ( 32 - 26 ) ;
	out[31] |= in[35] << 23 ;

	out[32] = ( in[35] & 0x1fffffff ) >> ( 32 - 23 ) ;
	out[32] |= in[36] << 20 ;

	out[33] = ( in[36] & 0x1fffffff ) >> ( 32 - 20 ) ;
	out[33] |= in[37] << 17 ;

	out[34] = ( in[37] & 0x1fffffff ) >> ( 32 - 17 ) ;
	out[34] |= in[38] << 14 ;

	out[35] = ( in[38] & 0x1fffffff ) >> ( 32 - 14 ) ;
	out[35] |= in[39] << 11 ;

	out[36] = ( in[39] & 0x1fffffff ) >> ( 32 - 11 ) ;
	out[36] |= in[40] << 8 ;

	out[37] = ( in[40] & 0x1fffffff ) >> ( 32 - 8 ) ;
	out[37] |= in[41] << 5 ;

	out[38] = ( in[41] & 0x1fffffff ) >> ( 32 - 5 ) ;
	out[38] |= ( in[42] & 0x1fffffff ) << 2 ;
	out[38] |= in[43] << 31 ;

	out[39] = ( in[43] & 0x1fffffff ) >> ( 32 - 31 ) ;
	out[39] |= in[44] << 28 ;

	out[40] = ( in[44] & 0x1fffffff ) >> ( 32 - 28 ) ;
	out[40] |= in[45] << 25 ;

	out[41] = ( in[45] & 0x1fffffff ) >> ( 32 - 25 ) ;
	out[41] |= in[46] << 22 ;

	out[42] = ( in[46] & 0x1fffffff ) >> ( 32 - 22 ) ;
	out[42] |= in[47] << 19 ;

	out[43] = ( in[47] & 0x1fffffff ) >> ( 32 - 19 ) ;
	out[43] |= in[48] << 16 ;

	out[44] = ( in[48] & 0x1fffffff ) >> ( 32 - 16 ) ;
	out[44] |= in[49] << 13 ;

	out[45] = ( in[49] & 0x1fffffff ) >> ( 32 - 13 ) ;
	out[45] |= in[50] << 10 ;

	out[46] = ( in[50] & 0x1fffffff ) >> ( 32 - 10 ) ;
	out[46] |= in[51] << 7 ;

	out[47] = ( in[51] & 0x1fffffff ) >> ( 32 - 7 ) ;
	out[47] |= in[52] << 4 ;

	out[48] = ( in[52] & 0x1fffffff ) >> ( 32 - 4 ) ;
	out[48] |= ( in[53] & 0x1fffffff ) << 1 ;
	out[48] |= in[54] << 30 ;

	out[49] = ( in[54] & 0x1fffffff ) >> ( 32 - 30 ) ;
	out[49] |= in[55] << 27 ;

	out[50] = ( in[55] & 0x1fffffff ) >> ( 32 - 27 ) ;
	out[50] |= in[56] << 24 ;

	out[51] = ( in[56] & 0x1fffffff ) >> ( 32 - 24 ) ;
	out[51] |= in[57] << 21 ;

	out[52] = ( in[57] & 0x1fffffff ) >> ( 32 - 21 ) ;
	out[52] |= in[58] << 18 ;

	out[53] = ( in[58] & 0x1fffffff ) >> ( 32 - 18 ) ;
	out[53] |= in[59] << 15 ;

	out[54] = ( in[59] & 0x1fffffff ) >> ( 32 - 15 ) ;
	out[54] |= in[60] << 12 ;

	out[55] = ( in[60] & 0x1fffffff ) >> ( 32 - 12 ) ;
	out[55] |= in[61] << 9 ;

	out[56] = ( in[61] & 0x1fffffff ) >> ( 32 - 9 ) ;
	out[56] |= in[62] << 6 ;

	out[57] = ( in[62] & 0x1fffffff ) >> ( 32 - 6 ) ;
	out[57] |= in[63] << 3 ;
}


// 30-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c30(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x3fffffff ) << 0 ;
	out[0] |= in[1] << 30 ;

	out[1] = ( in[1] & 0x3fffffff ) >> ( 32 - 30 ) ;
	out[1] |= in[2] << 28 ;

	out[2] = ( in[2] & 0x3fffffff ) >> ( 32 - 28 ) ;
	out[2] |= in[3] << 26 ;

	out[3] = ( in[3] & 0x3fffffff ) >> ( 32 - 26 ) ;
	out[3] |= in[4] << 24 ;

	out[4] = ( in[4] & 0x3fffffff ) >> ( 32 - 24 ) ;
	out[4] |= in[5] << 22 ;

	out[5] = ( in[5] & 0x3fffffff ) >> ( 32 - 22 ) ;
	out[5] |= in[6] << 20 ;

	out[6] = ( in[6] & 0x3fffffff ) >> ( 32 - 20 ) ;
	out[6] |= in[7] << 18 ;

	out[7] = ( in[7] & 0x3fffffff ) >> ( 32 - 18 ) ;
	out[7] |= in[8] << 16 ;

	out[8] = ( in[8] & 0x3fffffff ) >> ( 32 - 16 ) ;
	out[8] |= in[9] << 14 ;

	out[9] = ( in[9] & 0x3fffffff ) >> ( 32 - 14 ) ;
	out[9] |= in[10] << 12 ;

	out[10] = ( in[10] & 0x3fffffff ) >> ( 32 - 12 ) ;
	out[10] |= in[11] << 10 ;

	out[11] = ( in[11] & 0x3fffffff ) >> ( 32 - 10 ) ;
	out[11] |= in[12] << 8 ;

	out[12] = ( in[12] & 0x3fffffff ) >> ( 32 - 8 ) ;
	out[12] |= in[13] << 6 ;

	out[13] = ( in[13] & 0x3fffffff ) >> ( 32 - 6 ) ;
	out[13] |= in[14] << 4 ;

	out[14] = ( in[14] & 0x3fffffff ) >> ( 32 - 4 ) ;
	out[14] |= in[15] << 2 ;

	out[15] = ( in[16] & 0x3fffffff ) << 0 ;
	out[15] |= in[17] << 30 ;

	out[16] = ( in[17] & 0x3fffffff ) >> ( 32 - 30 ) ;
	out[16] |= in[18] << 28 ;

	out[17] = ( in[18] & 0x3fffffff ) >> ( 32 - 28 ) ;
	out[17] |= in[19] << 26 ;

	out[18] = ( in[19] & 0x3fffffff ) >> ( 32 - 26 ) ;
	out[18] |= in[20] << 24 ;

	out[19] = ( in[20] & 0x3fffffff ) >> ( 32 - 24 ) ;
	out[19] |= in[21] << 22 ;

	out[20] = ( in[21] & 0x3fffffff ) >> ( 32 - 22 ) ;
	out[20] |= in[22] << 20 ;

	out[21] = ( in[22] & 0x3fffffff ) >> ( 32 - 20 ) ;
	out[21] |= in[23] << 18 ;

	out[22] = ( in[23] & 0x3fffffff ) >> ( 32 - 18 ) ;
	out[22] |= in[24] << 16 ;

	out[23] = ( in[24] & 0x3fffffff ) >> ( 32 - 16 ) ;
	out[23] |= in[25] << 14 ;

	out[24] = ( in[25] & 0x3fffffff ) >> ( 32 - 14 ) ;
	out[24] |= in[26] << 12 ;

	out[25] = ( in[26] & 0x3fffffff ) >> ( 32 - 12 ) ;
	out[25] |= in[27] << 10 ;

	out[26] = ( in[27] & 0x3fffffff ) >> ( 32 - 10 ) ;
	out[26] |= in[28] << 8 ;

	out[27] = ( in[28] & 0x3fffffff ) >> ( 32 - 8 ) ;
	out[27] |= in[29] << 6 ;

	out[28] = ( in[29] & 0x3fffffff ) >> ( 32 - 6 ) ;
	out[28] |= in[30] << 4 ;

	out[29] = ( in[30] & 0x3fffffff ) >> ( 32 - 4 ) ;
	out[29] |= in[31] << 2 ;

	out[30] = ( in[32] & 0x3fffffff ) << 0 ;
	out[30] |= in[33] << 30 ;

	out[31] = ( in[33] & 0x3fffffff ) >> ( 32 - 30 ) ;
	out[31] |= in[34] << 28 ;

	out[32] = ( in[34] & 0x3fffffff ) >> ( 32 - 28 ) ;
	out[32] |= in[35] << 26 ;

	out[33] = ( in[35] & 0x3fffffff ) >> ( 32 - 26 ) ;
	out[33] |= in[36] << 24 ;

	out[34] = ( in[36] & 0x3fffffff ) >> ( 32 - 24 ) ;
	out[34] |= in[37] << 22 ;

	out[35] = ( in[37] & 0x3fffffff ) >> ( 32 - 22 ) ;
	out[35] |= in[38] << 20 ;

	out[36] = ( in[38] & 0x3fffffff ) >> ( 32 - 20 ) ;
	out[36] |= in[39] << 18 ;

	out[37] = ( in[39] & 0x3fffffff ) >> ( 32 - 18 ) ;
	out[37] |= in[40] << 16 ;

	out[38] = ( in[40] & 0x3fffffff ) >> ( 32 - 16 ) ;
	out[38] |= in[41] << 14 ;

	out[39] = ( in[41] & 0x3fffffff ) >> ( 32 - 14 ) ;
	out[39] |= in[42] << 12 ;

	out[40] = ( in[42] & 0x3fffffff ) >> ( 32 - 12 ) ;
	out[40] |= in[43] << 10 ;

	out[41] = ( in[43] & 0x3fffffff ) >> ( 32 - 10 ) ;
	out[41] |= in[44] << 8 ;

	out[42] = ( in[44] & 0x3fffffff ) >> ( 32 - 8 ) ;
	out[42] |= in[45] << 6 ;

	out[43] = ( in[45] & 0x3fffffff ) >> ( 32 - 6 ) ;
	out[43] |= in[46] << 4 ;

	out[44] = ( in[46] & 0x3fffffff ) >> ( 32 - 4 ) ;
	out[44] |= in[47] << 2 ;

	out[45] = ( in[48] & 0x3fffffff ) << 0 ;
	out[45] |= in[49] << 30 ;

	out[46] = ( in[49] & 0x3fffffff ) >> ( 32 - 30 ) ;
	out[46] |= in[50] << 28 ;

	out[47] = ( in[50] & 0x3fffffff ) >> ( 32 - 28 ) ;
	out[47] |= in[51] << 26 ;

	out[48] = ( in[51] & 0x3fffffff ) >> ( 32 - 26 ) ;
	out[48] |= in[52] << 24 ;

	out[49] = ( in[52] & 0x3fffffff ) >> ( 32 - 24 ) ;
	out[49] |= in[53] << 22 ;

	out[50] = ( in[53] & 0x3fffffff ) >> ( 32 - 22 ) ;
	out[50] |= in[54] << 20 ;

	out[51] = ( in[54] & 0x3fffffff ) >> ( 32 - 20 ) ;
	out[51] |= in[55] << 18 ;

	out[52] = ( in[55] & 0x3fffffff ) >> ( 32 - 18 ) ;
	out[52] |= in[56] << 16 ;

	out[53] = ( in[56] & 0x3fffffff ) >> ( 32 - 16 ) ;
	out[53] |= in[57] << 14 ;

	out[54] = ( in[57] & 0x3fffffff ) >> ( 32 - 14 ) ;
	out[54] |= in[58] << 12 ;

	out[55] = ( in[58] & 0x3fffffff ) >> ( 32 - 12 ) ;
	out[55] |= in[59] << 10 ;

	out[56] = ( in[59] & 0x3fffffff ) >> ( 32 - 10 ) ;
	out[56] |= in[60] << 8 ;

	out[57] = ( in[60] & 0x3fffffff ) >> ( 32 - 8 ) ;
	out[57] |= in[61] << 6 ;

	out[58] = ( in[61] & 0x3fffffff ) >> ( 32 - 6 ) ;
	out[58] |= in[62] << 4 ;

	out[59] = ( in[62] & 0x3fffffff ) >> ( 32 - 4 ) ;
	out[59] |= in[63] << 2 ;
}


// 31-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c31(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	out[0] = ( in[0] & 0x7fffffff ) << 0 ;
	out[0] |= in[1] << 31 ;

	out[1] = ( in[1] & 0x7fffffff ) >> ( 32 - 31 ) ;
	out[1] |= in[2] << 30 ;

	out[2] = ( in[2] & 0x7fffffff ) >> ( 32 - 30 ) ;
	out[2] |= in[3] << 29 ;

	out[3] = ( in[3] & 0x7fffffff ) >> ( 32 - 29 ) ;
	out[3] |= in[4] << 28 ;

	out[4] = ( in[4] & 0x7fffffff ) >> ( 32 - 28 ) ;
	out[4] |= in[5] << 27 ;

	out[5] = ( in[5] & 0x7fffffff ) >> ( 32 - 27 ) ;
	out[5] |= in[6] << 26 ;

	out[6] = ( in[6] & 0x7fffffff ) >> ( 32 - 26 ) ;
	out[6] |= in[7] << 25 ;

	out[7] = ( in[7] & 0x7fffffff ) >> ( 32 - 25 ) ;
	out[7] |= in[8] << 24 ;

	out[8] = ( in[8] & 0x7fffffff ) >> ( 32 - 24 ) ;
	out[8] |= in[9] << 23 ;

	out[9] = ( in[9] & 0x7fffffff ) >> ( 32 - 23 ) ;
	out[9] |= in[10] << 22 ;

	out[10] = ( in[10] & 0x7fffffff ) >> ( 32 - 22 ) ;
	out[10] |= in[11] << 21 ;

	out[11] = ( in[11] & 0x7fffffff ) >> ( 32 - 21 ) ;
	out[11] |= in[12] << 20 ;

	out[12] = ( in[12] & 0x7fffffff ) >> ( 32 - 20 ) ;
	out[12] |= in[13] << 19 ;

	out[13] = ( in[13] & 0x7fffffff ) >> ( 32 - 19 ) ;
	out[13] |= in[14] << 18 ;

	out[14] = ( in[14] & 0x7fffffff ) >> ( 32 - 18 ) ;
	out[14] |= in[15] << 17 ;

	out[15] = ( in[15] & 0x7fffffff ) >> ( 32 - 17 ) ;
	out[15] |= in[16] << 16 ;

	out[16] = ( in[16] & 0x7fffffff ) >> ( 32 - 16 ) ;
	out[16] |= in[17] << 15 ;

	out[17] = ( in[17] & 0x7fffffff ) >> ( 32 - 15 ) ;
	out[17] |= in[18] << 14 ;

	out[18] = ( in[18] & 0x7fffffff ) >> ( 32 - 14 ) ;
	out[18] |= in[19] << 13 ;

	out[19] = ( in[19] & 0x7fffffff ) >> ( 32 - 13 ) ;
	out[19] |= in[20] << 12 ;

	out[20] = ( in[20] & 0x7fffffff ) >> ( 32 - 12 ) ;
	out[20] |= in[21] << 11 ;

	out[21] = ( in[21] & 0x7fffffff ) >> ( 32 - 11 ) ;
	out[21] |= in[22] << 10 ;

	out[22] = ( in[22] & 0x7fffffff ) >> ( 32 - 10 ) ;
	out[22] |= in[23] << 9 ;

	out[23] = ( in[23] & 0x7fffffff ) >> ( 32 - 9 ) ;
	out[23] |= in[24] << 8 ;

	out[24] = ( in[24] & 0x7fffffff ) >> ( 32 - 8 ) ;
	out[24] |= in[25] << 7 ;

	out[25] = ( in[25] & 0x7fffffff ) >> ( 32 - 7 ) ;
	out[25] |= in[26] << 6 ;

	out[26] = ( in[26] & 0x7fffffff ) >> ( 32 - 6 ) ;
	out[26] |= in[27] << 5 ;

	out[27] = ( in[27] & 0x7fffffff ) >> ( 32 - 5 ) ;
	out[27] |= in[28] << 4 ;

	out[28] = ( in[28] & 0x7fffffff ) >> ( 32 - 4 ) ;
	out[28] |= in[29] << 3 ;

	out[29] = ( in[29] & 0x7fffffff ) >> ( 32 - 3 ) ;
	out[29] |= in[30] << 2 ;

	out[30] = ( in[30] & 0x7fffffff ) >> ( 32 - 2 ) ;
	out[30] |= in[31] << 1 ;

	out[31] = ( in[32] & 0x7fffffff ) << 0 ;
	out[31] |= in[33] << 31 ;

	out[32] = ( in[33] & 0x7fffffff ) >> ( 32 - 31 ) ;
	out[32] |= in[34] << 30 ;

	out[33] = ( in[34] & 0x7fffffff ) >> ( 32 - 30 ) ;
	out[33] |= in[35] << 29 ;

	out[34] = ( in[35] & 0x7fffffff ) >> ( 32 - 29 ) ;
	out[34] |= in[36] << 28 ;

	out[35] = ( in[36] & 0x7fffffff ) >> ( 32 - 28 ) ;
	out[35] |= in[37] << 27 ;

	out[36] = ( in[37] & 0x7fffffff ) >> ( 32 - 27 ) ;
	out[36] |= in[38] << 26 ;

	out[37] = ( in[38] & 0x7fffffff ) >> ( 32 - 26 ) ;
	out[37] |= in[39] << 25 ;

	out[38] = ( in[39] & 0x7fffffff ) >> ( 32 - 25 ) ;
	out[38] |= in[40] << 24 ;

	out[39] = ( in[40] & 0x7fffffff ) >> ( 32 - 24 ) ;
	out[39] |= in[41] << 23 ;

	out[40] = ( in[41] & 0x7fffffff ) >> ( 32 - 23 ) ;
	out[40] |= in[42] << 22 ;

	out[41] = ( in[42] & 0x7fffffff ) >> ( 32 - 22 ) ;
	out[41] |= in[43] << 21 ;

	out[42] = ( in[43] & 0x7fffffff ) >> ( 32 - 21 ) ;
	out[42] |= in[44] << 20 ;

	out[43] = ( in[44] & 0x7fffffff ) >> ( 32 - 20 ) ;
	out[43] |= in[45] << 19 ;

	out[44] = ( in[45] & 0x7fffffff ) >> ( 32 - 19 ) ;
	out[44] |= in[46] << 18 ;

	out[45] = ( in[46] & 0x7fffffff ) >> ( 32 - 18 ) ;
	out[45] |= in[47] << 17 ;

	out[46] = ( in[47] & 0x7fffffff ) >> ( 32 - 17 ) ;
	out[46] |= in[48] << 16 ;

	out[47] = ( in[48] & 0x7fffffff ) >> ( 32 - 16 ) ;
	out[47] |= in[49] << 15 ;

	out[48] = ( in[49] & 0x7fffffff ) >> ( 32 - 15 ) ;
	out[48] |= in[50] << 14 ;

	out[49] = ( in[50] & 0x7fffffff ) >> ( 32 - 14 ) ;
	out[49] |= in[51] << 13 ;

	out[50] = ( in[51] & 0x7fffffff ) >> ( 32 - 13 ) ;
	out[50] |= in[52] << 12 ;

	out[51] = ( in[52] & 0x7fffffff ) >> ( 32 - 12 ) ;
	out[51] |= in[53] << 11 ;

	out[52] = ( in[53] & 0x7fffffff ) >> ( 32 - 11 ) ;
	out[52] |= in[54] << 10 ;

	out[53] = ( in[54] & 0x7fffffff ) >> ( 32 - 10 ) ;
	out[53] |= in[55] << 9 ;

	out[54] = ( in[55] & 0x7fffffff ) >> ( 32 - 9 ) ;
	out[54] |= in[56] << 8 ;

	out[55] = ( in[56] & 0x7fffffff ) >> ( 32 - 8 ) ;
	out[55] |= in[57] << 7 ;

	out[56] = ( in[57] & 0x7fffffff ) >> ( 32 - 7 ) ;
	out[56] |= in[58] << 6 ;

	out[57] = ( in[58] & 0x7fffffff ) >> ( 32 - 6 ) ;
	out[57] |= in[59] << 5 ;

	out[58] = ( in[59] & 0x7fffffff ) >> ( 32 - 5 ) ;
	out[58] |= in[60] << 4 ;

	out[59] = ( in[60] & 0x7fffffff ) >> ( 32 - 4 ) ;
	out[59] |= in[61] << 3 ;

	out[60] = ( in[61] & 0x7fffffff ) >> ( 32 - 3 ) ;
	out[60] |= in[62] << 2 ;

	out[61] = ( in[62] & 0x7fffffff ) >> ( 32 - 2 ) ;
	out[61] |= in[63] << 1 ;
}


// 32-bit
template <bool IsRiceCoding>
void HorizontalScalarUnpacker<IsRiceCoding>::__horizontal_scalar_pack64_c32(const uint32_t * __restrict__  in,
		uint32_t *  __restrict__  out) {
	memcpy(out, in, 64 * sizeof(uint32_t));
}


#endif /* HORIZONTALSCALARUNPACKERIMP_H_ */
