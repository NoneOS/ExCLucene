/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#include "Simple9_Scalar.h"

void Simple9_Scalar::encodeArray(const uint32_t *in, size_t nvalue,
		uint32_t *out, size_t &csize) {
    const uint32_t *const initout(out);
    size_t valuesRemaining(nvalue), numberOfValuesCoded(0);

    // complete codewords
    while (valuesRemaining >= 28) {
    	uint32_t &codeword(out[0]);
        if (trymefull<28, 1>(in)) {
        	descriptor_writer(0, codeword);

        	numberOfValuesCoded = 28;
        	data_writer<28, 1>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<14, 2>(in)) {
        	descriptor_writer(1, codeword);

        	numberOfValuesCoded = 14;
        	data_writer<14, 2>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<9, 3>(in)) {
        	descriptor_writer(2, codeword);

        	numberOfValuesCoded = 9;
        	data_writer<9, 3>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<7, 4>(in)) {
        	descriptor_writer(3, codeword);

        	numberOfValuesCoded = 7;
        	data_writer<7, 4>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<5, 5>(in)) {
        	descriptor_writer(4, codeword);

        	numberOfValuesCoded = 5;
        	data_writer<5, 5>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<4, 7>(in)) {
        	descriptor_writer(5, codeword);

        	numberOfValuesCoded = 4;
        	data_writer<4, 7>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<3, 9>(in)) {
        	descriptor_writer(6, codeword);

        	numberOfValuesCoded = 3;
        	data_writer<3, 9>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<2, 14>(in)) {
        	descriptor_writer(7, codeword);

        	numberOfValuesCoded = 2;
        	data_writer<2, 14>(in, numberOfValuesCoded, codeword);
        }
        else if (trymefull<1, 28>(in)) {
        	descriptor_writer(8, codeword);

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
    	uint32_t &codeword(out[0]);
        if (tryme<28, 1>(in, valuesRemaining)) {
        	descriptor_writer(0, codeword);

            numberOfValuesCoded = (valuesRemaining < 28) ? valuesRemaining : 28;
        	data_writer<28, 1>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<14, 2>(in, valuesRemaining)) {
        	descriptor_writer(1, codeword);

        	numberOfValuesCoded = (valuesRemaining < 14) ? valuesRemaining : 14;
        	data_writer<14, 2>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<9, 3>(in, valuesRemaining)) {
        	descriptor_writer(2, codeword);

        	numberOfValuesCoded = (valuesRemaining < 9) ? valuesRemaining : 9;
        	data_writer<9, 3>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<7, 4>(in, valuesRemaining)) {
        	descriptor_writer(3, codeword);

            numberOfValuesCoded = (valuesRemaining < 7) ? valuesRemaining : 7;
        	data_writer<7, 4>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<5, 5>(in, valuesRemaining)) {
        	descriptor_writer(4, codeword);

        	numberOfValuesCoded = (valuesRemaining < 5) ? valuesRemaining : 5;
        	data_writer<5, 5>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<4, 7>(in, valuesRemaining)) {
        	descriptor_writer(5, codeword);

            numberOfValuesCoded = (valuesRemaining < 4) ? valuesRemaining : 4;
        	data_writer<4, 7>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<3, 9>(in, valuesRemaining)) {
        	descriptor_writer(6, codeword);

        	numberOfValuesCoded = (valuesRemaining < 3) ? valuesRemaining : 3;
        	data_writer<3, 9>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<2, 14>(in, valuesRemaining)) {
        	descriptor_writer(7, codeword);

        	numberOfValuesCoded = (valuesRemaining < 2) ? valuesRemaining : 2;
        	data_writer<2, 14>(in, numberOfValuesCoded, codeword);
        }
        else if (tryme<1, 28>(in, valuesRemaining)) {
        	descriptor_writer(8, codeword);

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

const uint32_t * Simple9_Scalar::decodeArray(const uint32_t *in, size_t csize,
		uint32_t *out, size_t nvalue) {
	const uint32_t *const endout(out + nvalue);
	while (endout > out) {
		const uint32_t codeword = in[0];
		++in;
		const uint32_t descriptor = codeword >> (32 - SIMPLE9_LOGDESC);
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
		case 1: // 14 * 2-bit
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
		case 2: // 9 * 3-bit
			out[0] = (codeword >> 25) & 0x07;
			out[1] = (codeword >> 22) & 0x07;
			out[2] = (codeword >> 19) & 0x07;
			out[3] = (codeword >> 16) & 0x07;
			out[4] = (codeword >> 13) & 0x07;
			out[5] = (codeword >> 10) & 0x07;
			out[6] = (codeword >> 7) & 0x07;
			out[7] = (codeword >> 4) & 0x07;
			out[8] = (codeword >> 1) & 0x07;

			out += 9;

			break;
		case 3: // 7 * 4-bit
			out[0] = (codeword >> 24) & 0x0f;
			out[1] = (codeword >> 20) & 0x0f;
			out[2] = (codeword >> 16) & 0x0f;
			out[3] = (codeword >> 12) & 0x0f;
			out[4] = (codeword >> 8) & 0x0f;
			out[5] = (codeword >> 4) & 0x0f;
			out[6] = codeword & 0x0f;

			out += 7;

			break;
		case 4: // 5 * 5-bit
			out[0] = (codeword >> 23) & 0x1f;
			out[1] = (codeword >> 18) & 0x1f;
			out[2] = (codeword >> 13) & 0x1f;
			out[3] = (codeword >> 8) & 0x1f;
			out[4] = (codeword >> 3) & 0x1f;

			out += 5;

			break;
		case 5: // 4 * 7-bit
			out[0] = (codeword >> 21) & 0x7f;
			out[1] = (codeword >> 14) & 0x7f;
			out[2] = (codeword >> 7) & 0x7f;
			out[3] = codeword & 0x7f;

			out += 4;

			break;
		case 6: // 3 * 9-bit
			out[0] = (codeword >> 19) & 0x01ff;
			out[1] = (codeword >> 10) & 0x01ff;
			out[2] = (codeword >> 1) & 0x01ff;

			out += 3;

			break;
		case 7: // 2 * 14-bit
			out[0] = (codeword >> 14) & 0x3fff;
			out[1] = codeword & 0x3fff;

			out += 2;

			break;
		case 8: // 1 * 28-bit
			out[0] = codeword & 0x0fffffff;

			++out;

			break;
		default: // invalid descriptor
			std::cerr << "Invalid descriptor: " << descriptor << std::endl;
			throw std::runtime_error("Invalid descriptor for " + name() + ".");
		}
	}

	return in;
}


