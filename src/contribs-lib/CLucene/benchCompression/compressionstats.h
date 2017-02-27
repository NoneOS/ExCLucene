/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef COMPRESSIONSTATS_H_
#define COMPRESSIONSTATS_H_

#include "common.h"
#include "IntegerCodec.h"
#include "Delta.h"

class compressionstats {
public:
	compressionstats(const std::string &prep = "") :
		preprocessor(prep), codecs(), processednvalue(0), skippednvalue(0), totalcsize(), bitsperint(),
		prepTime(0), postpTime(0), encodeTime(), decodeTime(), compTime(), decompTime(),
        prepSpeed(0), postpSpeed(0), encodeSpeed(), decodeSpeed(), compSpeed(), decompSpeed(),
		unitOfCompression("bits/int"), unitOfTime("us"), unitOfSpeed("million ints/s"),
		isSummarized(false), isNormalized(false) {
	}

	void summarize() {
		if (isSummarized)
			return;

		prepSpeed = processednvalue / prepTime;
		postpSpeed = processednvalue / postpTime;

		auto n = codecs.size();
		for (decltype(n) i = 0; i != n; ++i) {
			double cr = static_cast<double>(processednvalue + skippednvalue) / (totalcsize[i] + skippednvalue);
			bitsperint.push_back(32 / cr);

			compTime.push_back(prepTime + encodeTime[i]);
			decompTime.push_back(decodeTime[i] + postpTime);

			encodeSpeed.push_back(processednvalue / encodeTime[i]);
			decodeSpeed.push_back(processednvalue / decodeTime[i]);
			compSpeed.push_back(processednvalue / compTime[i]);
			decompSpeed.push_back(processednvalue / decompTime[i]);
		}

		isSummarized = true;
	}

	// us -> ms
	static void timeNormalizer(double &time) {
		time /= 1000;
	}

	void normalize() {
		if (isNormalized)
			return;

		if (!isSummarized)
			summarize();

		timeNormalizer(prepTime);
		timeNormalizer(postpTime);
		auto n = codecs.size();
		for (decltype(n) i = 0; i != n; ++i) {
			timeNormalizer(encodeTime[i]);
			timeNormalizer(decodeTime[i]);
			timeNormalizer(compTime[i]);
			timeNormalizer(decompTime[i]);
		}

		unitOfTime = "ms";
		isNormalized = true;
	}

	std::ostream &display(std::ostream &os = std::cout, uint32_t precision = 3) {
		if (!isNormalized) {
			normalize();
		}

		os << "preprocessor = " << preprocessor << std::endl << std::endl;

		auto n = codecs.size();
		os << "codec" << "\t" << unitOfCompression << std::endl;
		for (decltype(n) i = 0; i != n; ++i) {
			os << codecs[i] << "\t" << bitsperint[i] << std::endl;
		}
		os << std::endl;

		os << "unitOfTime: " << unitOfTime << std::endl;
		os << "codec\t"
		   << "prepTime\t" << "encodeTime\t" << "compTime\t"
		   << "decodeTime\t" << "postpTime\t" << "decompTime"
		   << std::endl;
		for (decltype(n) i = 0; i != n; ++i) {
			os << codecs[i] << "\t"
			   << prepTime << "\t" << encodeTime[i] << "\t" << compTime[i] << "\t"
			   << decodeTime[i] << "\t" << postpTime << "\t" << decompTime[i]
			   << std::endl;
		}
		os << std::endl;

		os << "unitOfSpeed: " << unitOfSpeed << std::endl;
		os << "codec\t"
		   << "prepSpeed\t" << "encodeSpeed\t" << "compSpeed\t"
		   << "decodeSpeed\t" << "postpSpeed\t" << "decompSpeed"
		   << std::endl;
		for (decltype(n) i = 0; i != n; ++i) {
			os << codecs[i] << "\t"
			   << prepSpeed << "\t" << encodeSpeed[i] << "\t" << compSpeed[i] << "\t"
			   << decodeSpeed[i] << "\t" << postpSpeed << "\t" << decompSpeed[i]
		       << std::endl;
		}
		os << std::endl;

		return os;
	}


	std::string preprocessor;  // name of preprocessor (e.g. "RegularDeltaSSE", "SegLR<256>")
	double prepTime;   // time of converting docIDs to dgaps or VDs
	double postpTime;  // time of reconstructing docIDs from dgaps or VDs 
	double prepSpeed;  
	double postpSpeed;


	std::vector<std::string> codecs;
	// how many integers
	uint64_t processednvalue;
	uint64_t skippednvalue;           // inputsize = processednvalue + skippednvalue
	std::vector<uint64_t> totalcsize; // outputsize = totalcsize + skippednvalue
	std::vector<double> bitsperint;   // := (32 * outputsize) / inputsize

	std::vector<double> encodeTime;
	std::vector<double> decodeTime;
	std::vector<double> compTime;    // := prepTime + encodeTime
	std::vector<double> decompTime;  // := decodeTime + postpTime
	std::vector<double> encodeSpeed;
	std::vector<double> decodeSpeed;
	std::vector<double> compSpeed;
	std::vector<double> decompSpeed;

	std::string unitOfCompression;
	std::string unitOfTime;
	std::string unitOfSpeed;

	bool isSummarized;
	bool isNormalized;
};

#endif /* COMPRESSIONSTATS_H_ */
