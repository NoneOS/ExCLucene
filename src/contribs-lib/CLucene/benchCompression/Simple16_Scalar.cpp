/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#include "Simple16_Scalar.h"

void Simple16_Scalar::encodeArray(const uint32_t *in, size_t nvalue,
		uint32_t *out, size_t &csize) {
    const uint32_t *const initout(out);
    size_t valuesRemaining(nvalue), numberOfValuesCoded(0);

    // complete codewords
    while (valuesRemaining >= 28) {
    	uint32_t &codeword = out[0];
    	if (trymefull<28, 1>(in)) {
    		descriptor_writer(0, codeword);

    		numberOfValuesCoded = 28;
            data_writer<28, 1>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<7, 2, 14, 1>(in)) {
    		descriptor_writer(1, codeword);

    		numberOfValuesCoded = 21;
            data_writer<7, 2, 14, 1>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<7, 1, 7, 2, 7, 1>(in)) {
    		descriptor_writer(2, codeword);

    		numberOfValuesCoded = 21;
            data_writer<7, 1, 7, 2, 7, 1>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<14, 1, 7, 2>(in)) {
        	descriptor_writer(3, codeword);

    		numberOfValuesCoded = 21;
            data_writer<14, 1, 7, 2>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<14, 2>(in)) {
        	descriptor_writer(4, codeword);

            numberOfValuesCoded = 14;
            data_writer<14, 2>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<1, 4, 8, 3>(in)) {
        	descriptor_writer(5, codeword);

            numberOfValuesCoded = 9;
            data_writer<1, 4, 8, 3>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<1, 3, 4, 4, 3, 3>(in)) {
        	descriptor_writer(6, codeword);

            numberOfValuesCoded = 8;
            data_writer<1, 3, 4, 4, 3, 3>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<7, 4>(in)) {
        	descriptor_writer(7, codeword);

            numberOfValuesCoded = 7;
            data_writer<7, 4>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<4, 5, 2, 4>(in)) {
        	descriptor_writer(8, codeword);

            numberOfValuesCoded = 6;
            data_writer<4, 5, 2, 4>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<2, 4, 4, 5>(in)) {
        	descriptor_writer(9, codeword);

            numberOfValuesCoded = 6;
            data_writer<2, 4, 4, 5>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<3, 6, 2, 5>(in)) {
        	descriptor_writer(10, codeword);

            numberOfValuesCoded = 5;
            data_writer<3, 6, 2, 5>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<2, 5, 3, 6>(in)) {
        	descriptor_writer(11, codeword);

        	numberOfValuesCoded = 5;
        	data_writer<2, 5, 3, 6>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<4, 7>(in)) {
        	descriptor_writer(12, codeword);

            numberOfValuesCoded = 4;
            data_writer<4, 7>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<1, 10, 2, 9>(in)) {
        	descriptor_writer(13, codeword);

            numberOfValuesCoded = 3;
            data_writer<1, 10, 2, 9>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<2, 14>(in)) {
        	descriptor_writer(14, codeword);

            numberOfValuesCoded = 2;
            data_writer<2, 14>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<1, 28>(in)) {
        	descriptor_writer(15, codeword);

            numberOfValuesCoded = 1;
            data_writer<1, 28>(in, numberOfValuesCoded, codeword);
        }
        else {
        	std::cerr << "Input's out of range: " << *in << std::endl;
        	throw std::runtime_error("You tried to apply " + name() + " to an incompatible set of integers.");
        }

        ++out;
        in += numberOfValuesCoded;
        valuesRemaining -= numberOfValuesCoded;
    }

    // possibly incomplete codewords
    while (valuesRemaining > 0) {
    	uint32_t &codeword = out[0];
    	if (tryme<28, 1>(in, valuesRemaining)) {
    		descriptor_writer(0, codeword);

    		numberOfValuesCoded = (valuesRemaining < 28) ? valuesRemaining : 28;
    		data_writer<28, 1>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<7, 2, 14, 1>(in, valuesRemaining)) {
    		descriptor_writer(1, codeword);

            numberOfValuesCoded = (valuesRemaining < 21) ? valuesRemaining : 21;
            data_writer<7, 2, 14, 1>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<7, 1, 7, 2, 7, 1>(in, valuesRemaining)) {
        	descriptor_writer(2, codeword);

        	numberOfValuesCoded = (valuesRemaining < 21) ? valuesRemaining : 21;
        	data_writer<7, 1, 7, 2, 7, 1>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<14, 1, 7, 2>(in, valuesRemaining)) {
        	descriptor_writer(3, codeword);

        	numberOfValuesCoded = (valuesRemaining < 21) ? valuesRemaining : 21;
        	data_writer<14, 1, 7, 2>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<14, 2>(in, valuesRemaining)) {
        	descriptor_writer(4, codeword);

            numberOfValuesCoded = (valuesRemaining < 14) ? valuesRemaining : 14;
            data_writer<14, 2>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<1, 4, 8, 3>(in, valuesRemaining)) {
        	descriptor_writer(5, codeword);

            numberOfValuesCoded = (valuesRemaining < 9) ? valuesRemaining : 9;
            data_writer<1, 4, 8, 3>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<1, 3, 4, 4, 3, 3>(in, valuesRemaining)) {
        	descriptor_writer(6, codeword);

            numberOfValuesCoded = (valuesRemaining < 8) ? valuesRemaining : 8;
            data_writer<1, 3, 4, 4, 3, 3>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<7, 4>(in, valuesRemaining)) {
        	descriptor_writer(7, codeword);

            numberOfValuesCoded = (valuesRemaining < 7) ? valuesRemaining : 7;
            data_writer<7, 4>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<4, 5, 2, 4>(in, valuesRemaining)) {
        	descriptor_writer(8, codeword);

            numberOfValuesCoded = (valuesRemaining < 6) ? valuesRemaining : 6;
            data_writer<4, 5, 2, 4>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<2, 4, 4, 5>(in, valuesRemaining)) {
        	descriptor_writer(9, codeword);

        	numberOfValuesCoded = (valuesRemaining < 6) ? valuesRemaining : 6;
        	data_writer<2, 4, 4, 5>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<3, 6, 2, 5>(in, valuesRemaining)) {
        	descriptor_writer(10, codeword);

            numberOfValuesCoded = (valuesRemaining < 5) ? valuesRemaining : 5;
            data_writer<3, 6, 2, 5>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<2, 5, 3, 6>(in, valuesRemaining)) {
        	descriptor_writer(11, codeword);

        	numberOfValuesCoded = (valuesRemaining < 5) ? valuesRemaining : 5;
        	data_writer<2, 5, 3, 6>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<4, 7>(in, valuesRemaining)) {
        	descriptor_writer(12, codeword);

            numberOfValuesCoded = (valuesRemaining < 4) ? valuesRemaining : 4;
            data_writer<4, 7>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<1, 10, 2, 9>(in, valuesRemaining)) {
        	descriptor_writer(13, codeword);

            numberOfValuesCoded = (valuesRemaining < 3) ? valuesRemaining : 3;
            data_writer<1, 10, 2, 9>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<2, 14>(in, valuesRemaining)) {
        	descriptor_writer(14, codeword);

            numberOfValuesCoded = (valuesRemaining < 2) ? valuesRemaining : 2;
            data_writer<2, 14>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<1, 28>(in, valuesRemaining)) {
        	descriptor_writer(15, codeword);

            numberOfValuesCoded = 1;
            data_writer<1, 28>(in, numberOfValuesCoded, codeword);
        }
        else {
        	std::cerr << "Input's out of range: " << *in << std::endl;
        	throw std::runtime_error("You tried to apply " + name() + " to an incompatible set of integers.");
        }

        ++out;
        in += numberOfValuesCoded;
        valuesRemaining -= numberOfValuesCoded;
    }

    csize = out - initout;
}

void Simple16_Scalar::fakeencodeArray(const uint32_t *in, size_t nvalue,
		size_t &csize) {
	csize = 0;
    size_t valuesRemaining(nvalue), numberOfValuesCoded(0);

    // complete codewords
    while (valuesRemaining >= 28) {
    	if (trymefull<28, 1>(in)) {
    		numberOfValuesCoded = 28;
        }
        else if (trymefull<7, 2, 14, 1>(in)) {
    		numberOfValuesCoded = 21;
        }
        else if (trymefull<7, 1, 7, 2, 7, 1>(in)) {
    		numberOfValuesCoded = 21;
        }
        else if (trymefull<14, 1, 7, 2>(in)) {
    		numberOfValuesCoded = 21;
        }
        else if (trymefull<14, 2>(in)) {
            numberOfValuesCoded = 14;
        }
        else if (trymefull<1, 4, 8, 3>(in)) {
            numberOfValuesCoded = 9;
        }
        else if (trymefull<1, 3, 4, 4, 3, 3>(in)) {
            numberOfValuesCoded = 8;
        }
        else if (trymefull<7, 4>(in)) {
            numberOfValuesCoded = 7;
        }
        else if (trymefull<4, 5, 2, 4>(in)) {
            numberOfValuesCoded = 6;
        }
        else if (trymefull<2, 4, 4, 5>(in)) {
            numberOfValuesCoded = 6;
        }
        else if (trymefull<3, 6, 2, 5>(in)) {
            numberOfValuesCoded = 5;
        }
        else if (trymefull<2, 5, 3, 6>(in)) {
        	numberOfValuesCoded = 5;
        }
        else if (trymefull<4, 7>(in)) {
            numberOfValuesCoded = 4;
        }
        else if (trymefull<1, 10, 2, 9>(in)) {
            numberOfValuesCoded = 3;
        }
        else if (trymefull<2, 14>(in)) {
            numberOfValuesCoded = 2;
        }
        else if (trymefull<1, 28>(in)) {
            numberOfValuesCoded = 1;
        }
        else {
        	std::cerr << "Input's out of range: " << *in << std::endl;
        	throw std::runtime_error("You tried to apply " + name() + " to an incompatible set of integers.");
        }

        ++csize;
        in += numberOfValuesCoded;
        valuesRemaining -= numberOfValuesCoded;
    }

    // possibly incomplete codewords
    while (valuesRemaining > 0) {
    	if (tryme<28, 1>(in, valuesRemaining)) {
    		numberOfValuesCoded = (valuesRemaining < 28) ? valuesRemaining : 28;
        }
        else if (tryme<7, 2, 14, 1>(in, valuesRemaining)) {
            numberOfValuesCoded = (valuesRemaining < 21) ? valuesRemaining : 21;
        }
        else if (tryme<7, 1, 7, 2, 7, 1>(in, valuesRemaining)) {
        	numberOfValuesCoded = (valuesRemaining < 21) ? valuesRemaining : 21;
        }
        else if (tryme<14, 1, 7, 2>(in, valuesRemaining)) {
        	numberOfValuesCoded = (valuesRemaining < 21) ? valuesRemaining : 21;
        }
        else if (tryme<14, 2>(in, valuesRemaining)) {
            numberOfValuesCoded = (valuesRemaining < 14) ? valuesRemaining : 14;
        }
        else if (tryme<1, 4, 8, 3>(in, valuesRemaining)) {
            numberOfValuesCoded = (valuesRemaining < 9) ? valuesRemaining : 9;
        }
        else if (tryme<1, 3, 4, 4, 3, 3>(in, valuesRemaining)) {
            numberOfValuesCoded = (valuesRemaining < 8) ? valuesRemaining : 8;
        }
        else if (tryme<7, 4>(in, valuesRemaining)) {
            numberOfValuesCoded = (valuesRemaining < 7) ? valuesRemaining : 7;
        }
        else if (tryme<4, 5, 2, 4>(in, valuesRemaining)) {
            numberOfValuesCoded = (valuesRemaining < 6) ? valuesRemaining : 6;
        }
        else if (tryme<2, 4, 4, 5>(in, valuesRemaining)) {
        	numberOfValuesCoded = (valuesRemaining < 6) ? valuesRemaining : 6;
        }
        else if (tryme<3, 6, 2, 5>(in, valuesRemaining)) {
            numberOfValuesCoded = (valuesRemaining < 5) ? valuesRemaining : 5;
        }
        else if (tryme<2, 5, 3, 6>(in, valuesRemaining)) {
        	numberOfValuesCoded = (valuesRemaining < 5) ? valuesRemaining : 5;
        }
        else if (tryme<4, 7>(in, valuesRemaining)) {
            numberOfValuesCoded = (valuesRemaining < 4) ? valuesRemaining : 4;
        }
        else if (tryme<1, 10, 2, 9>(in, valuesRemaining)) {
            numberOfValuesCoded = (valuesRemaining < 3) ? valuesRemaining : 3;
        }
        else if (tryme<2, 14>(in, valuesRemaining)) {
            numberOfValuesCoded = (valuesRemaining < 2) ? valuesRemaining : 2;
        }
        else if (tryme<1, 28>(in, valuesRemaining)) {
            numberOfValuesCoded = 1;
        }
        else {
        	std::cerr << "Input's out of range: " << *in << std::endl;
        	throw std::runtime_error("You tried to apply " + name() + " to an incompatible set of integers.");
        }

    	++csize;
        in += numberOfValuesCoded;
        valuesRemaining -= numberOfValuesCoded;
    }
}



const uint32_t * Simple16_Scalar::decodeArray(const uint32_t *in, size_t csize,
			uint32_t *out, size_t nvalue) {
    const uint32_t *const endout(out + nvalue);
    while (endout > out) {
    	const uint32_t codeword = in[0];
    	++in;
    	const uint32_t descriptor = codeword >> (32 - SIMPLE16_LOGDESC);
        switch (descriptor) {
        case 0: // 28 * 1-bit
        	out[0] = (codeword >> 27) & 0x01;
            out[1] = (codeword >> 26) & 0x01;
            out[2] = (codeword >> 25) & 0x01;
            out[3] = (codeword >> 24) & 0x01;
            out[4] = (codeword >> 23) & 0x01;
            out[5] = (codeword >> 22) & 0x01;
            out[6] = (codeword >> 21) & 0x01;
            out[7] = (codeword >> 20) & 0x01;
            out[8] = (codeword >> 19) & 0x01;
            out[9] = (codeword >> 18) & 0x01;
            out[10] = (codeword >> 17) & 0x01;
            out[11] = (codeword >> 16) & 0x01;
            out[12] = (codeword >> 15) & 0x01;
            out[13] = (codeword >> 14) & 0x01;
            out[14] = (codeword >> 13) & 0x01;
            out[15] = (codeword >> 12) & 0x01;
            out[16] = (codeword >> 11) & 0x01;
            out[17] = (codeword >> 10) & 0x01;
            out[18] = (codeword >> 9) & 0x01;
            out[19] = (codeword >> 8) & 0x01;
            out[20] = (codeword >> 7) & 0x01;
            out[21] = (codeword >> 6) & 0x01;
            out[22] = (codeword >> 5) & 0x01;
            out[23] = (codeword >> 4) & 0x01;
            out[24] = (codeword >> 3) & 0x01;
            out[25] = (codeword >> 2) & 0x01;
            out[26] = (codeword >> 1) & 0x01;
            out[27] = codeword & 0x01;

            out += 28;

            break;
        case 1: // 7 * 2-bit + 14 * 1-bit
            out[0] = (codeword >> 26) & 0x03;
            out[1] = (codeword >> 24) & 0x03;
            out[2] = (codeword >> 22) & 0x03;
            out[3] = (codeword >> 20) & 0x03;
            out[4] = (codeword >> 18) & 0x03;
            out[5] = (codeword >> 16) & 0x03;
            out[6] = (codeword >> 14) & 0x03;

            out[7] = (codeword >> 13) & 0x01;
            out[8] = (codeword >> 12) & 0x01;
            out[9] = (codeword >> 11) & 0x01;
            out[10] = (codeword >> 10) & 0x01;
            out[11] = (codeword >> 9) & 0x01;
            out[12] = (codeword >> 8) & 0x01;
            out[13] = (codeword >> 7) & 0x01;
            out[14] = (codeword >> 6) & 0x01;
            out[15] = (codeword >> 5) & 0x01;
            out[16] = (codeword >> 4) & 0x01;
            out[17] = (codeword >> 3) & 0x01;
            out[18] = (codeword >> 2) & 0x01;
            out[19] = (codeword >> 1) & 0x01;
            out[20] = codeword & 0x01;

            out += 21;

            break;
        case 2: // 7 * 1-bit + 7 * 2-bit + 7 * 1-bit
            out[0] = (codeword >> 27) & 0x01;
            out[1] = (codeword >> 26) & 0x01;
            out[2] = (codeword >> 25) & 0x01;
            out[3] = (codeword >> 24) & 0x01;
            out[4] = (codeword >> 23) & 0x01;
            out[5] = (codeword >> 22) & 0x01;
            out[6] = (codeword >> 21) & 0x01;

            out[7] = (codeword >> 19) & 0x03;
            out[8] = (codeword >> 17) & 0x03;
            out[9] = (codeword >> 15) & 0x03;
            out[10] = (codeword >> 13) & 0x03;
            out[11] = (codeword >> 11) & 0x03;
            out[12] = (codeword >> 9) & 0x03;
            out[13] = (codeword >> 7) & 0x03;

            out[14] = (codeword >> 6) & 0x01;
            out[15] = (codeword >> 5) & 0x01;
            out[16] = (codeword >> 4) & 0x01;
            out[17] = (codeword >> 3) & 0x01;
            out[18] = (codeword >> 2) & 0x01;
            out[19] = (codeword >> 1) & 0x01;
            out[20] = codeword & 0x01;

            out += 21;

            break;
        case 3: // 14 * 1-bit + 7 * 2-bit
            out[0] = (codeword >> 27) & 0x01;
            out[1] = (codeword >> 26) & 0x01;
            out[2] = (codeword >> 25) & 0x01;
            out[3] = (codeword >> 24) & 0x01;
            out[4] = (codeword >> 23) & 0x01;
            out[5] = (codeword >> 22) & 0x01;
            out[6] = (codeword >> 21) & 0x01;
            out[7] = (codeword >> 20) & 0x01;
            out[8] = (codeword >> 19) & 0x01;
            out[9] = (codeword >> 18) & 0x01;
            out[10] = (codeword >> 17) & 0x01;
            out[11] = (codeword >> 16) & 0x01;
            out[12] = (codeword >> 15) & 0x01;
            out[13] = (codeword >> 14) & 0x01;

            out[14] = (codeword >> 12) & 0x03;
            out[15] = (codeword >> 10) & 0x03;
            out[16] = (codeword >> 8) & 0x03;
            out[17] = (codeword >> 6) & 0x03;
            out[18] = (codeword >> 4) & 0x03;
            out[19] = (codeword >> 2) & 0x03;
            out[20] = codeword & 0x03;

            out += 21;

            break;
        case 4: // 14 * 2-bit
            out[0] = (codeword >> 26) & 0x03;
            out[1] = (codeword >> 24) & 0x03;
            out[2] = (codeword >> 22) & 0x03;
            out[3] = (codeword >> 20) & 0x03;
            out[4] = (codeword >> 18) & 0x03;
            out[5] = (codeword >> 16) & 0x03;
            out[6] = (codeword >> 14) & 0x03;
            out[7] = (codeword >> 12) & 0x03;
            out[8] = (codeword >> 10) & 0x03;
            out[9] = (codeword >> 8) & 0x03;
            out[10] = (codeword >> 6) & 0x03;
            out[11] = (codeword >> 4) & 0x03;
            out[12] = (codeword >> 2) & 0x03;
            out[13] = codeword & 0x03;

            out += 14;

            break;
        case 5: // 1 * 4-bit + 8 * 3-bit
            out[0] = (codeword >> 24) & 0x0f;

            out[1] = (codeword >> 21) & 0x07;
            out[2] = (codeword >> 18) & 0x07;
            out[3] = (codeword >> 15) & 0x07;
            out[4] = (codeword >> 12) & 0x07;
            out[5] = (codeword >> 9) & 0x07;
            out[6] = (codeword >> 6) & 0x07;
            out[7] = (codeword >> 3) & 0x07;
            out[8] = codeword & 0x07;

            out += 9;

            break;
        case 6: // 1 * 3-bit + 4 * 4-bit + 3 * 3-bit
            out[0] = (codeword >> 25) & 0x07;

            out[1] = (codeword >> 21) & 0x0f;
            out[2] = (codeword >> 17) & 0x0f;
            out[3] = (codeword >> 13) & 0x0f;
            out[4] = (codeword >> 9) & 0x0f;

            out[5] = (codeword >> 6) & 0x07;
            out[6] = (codeword >> 3) & 0x07;
            out[7] = codeword & 0x07;

            out += 8;

            break;
        case 7: // 7 * 4-bit
            out[0] = (codeword >> 24) & 0x0f;
            out[1] = (codeword >> 20) & 0x0f;
            out[2] = (codeword >> 16) & 0x0f;
            out[3] = (codeword >> 12) & 0x0f;
            out[4] = (codeword >> 8) & 0x0f;
            out[5] = (codeword >> 4) & 0x0f;
            out[6] = codeword & 0x0f;

            out += 7;

            break;
        case 8: // 4 * 5-bit + 2 * 4-bit
            out[0] = (codeword >> 23) & 0x1f;
            out[1] = (codeword >> 18) & 0x1f;
            out[2] = (codeword >> 13) & 0x1f;
            out[3] = (codeword >> 8) & 0x1f;

            out[4] = (codeword >> 4) & 0x0f;
            out[5] = codeword & 0x0f;

            out += 6;

            break;
        case 9: // 2 * 4-bit + 4 * 5-bit
            out[0] = (codeword >> 24) & 0x0f;
            out[1] = (codeword >> 20) & 0x0f;

            out[2] = (codeword >> 15) & 0x1f;
            out[3] = (codeword >> 10) & 0x1f;
            out[4] = (codeword >> 5) & 0x1f;
            out[5] = codeword & 0x1f;

            out += 6;

            break;
        case 10: // 3 * 6-bit + 2 * 5-bit
            out[0] = (codeword >> 22) & 0x3f;
            out[1] = (codeword >> 16) & 0x3f;
            out[2] = (codeword >> 10) & 0x3f;

            out[3] = (codeword >> 5) & 0x1f;
            out[4] = codeword & 0x1f;

            out += 5;

            break;
        case 11: // 2 * 5-bit + 3 * 6-bit
            out[0] = (codeword >> 23) & 0x1f;
            out[1] = (codeword >> 18) & 0x1f;

            out[2] = (codeword >> 12) & 0x3f;
            out[3] = (codeword >> 6) & 0x3f;
            out[4] = codeword & 0x3f;

            out += 5;

            break;
        case 12: // 4 * 7-bit
            out[0] = (codeword >> 21) & 0x7f;
            out[1] = (codeword >> 14) & 0x7f;
            out[2] = (codeword >> 7) & 0x7f;
            out[3] = codeword & 0x7f;

            out += 4;

            break;
        case 13: // 1 * 10-bit + 2 * 9-bit
            out[0] = (codeword >> 18) & 0x03ff;

            out[1] = (codeword >> 9) & 0x01ff;
            out[2] = codeword & 0x01ff;

            out += 3;

            break;
        case 14: // 2 * 14-bit
            out[0] = (codeword >> 14) & 0x3fff;
            out[1] = codeword & 0x3fff;

            out += 2;

            break;
        case 15: // 1 * 28-bit
            out[0] = codeword & 0x0fffffff;

            ++out;

            break;
		default: // You won't actually get here
        	break;
        }
    }

    return in;
}


