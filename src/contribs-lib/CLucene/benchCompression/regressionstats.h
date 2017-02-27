/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef REGRESSIONSTATS_H_
#define REGRESSIONSTATS_H_

#include "common.h"
#include "IndexInfo.h"
#include "LinearRegression.h"

template <uint32_t IntervalSize>
class regressionstats {
public:
	regressionstats() : kIntervalNum(MAXLEN / IntervalSize),
						count(kIntervalNum, 0), totalListLen(kIntervalNum, 0), statInfo(kIntervalNum) {
	}

	void accumulate(const size_t listLen, const std::vector<RegressionStatInfo_t> &segStatInfo) {
		const size_t idx = listLen / IntervalSize;
		++count[idx];
		totalListLen[idx] += listLen;

		auto kSegNum = segStatInfo.size();
		double dSumOfRSquare = 0, dSumOfContractionRatio = 0, dSumOfBitsNeeded = 0;
		for (decltype(kSegNum) i = 0; i < kSegNum; ++i) {
			dSumOfRSquare += segStatInfo[i].dRSquare;
			dSumOfContractionRatio += segStatInfo[i].dContractionRatio;
			dSumOfBitsNeeded += segStatInfo[i].dBitsNeeded;
		}
		statInfo[idx].dRSquare += dSumOfRSquare / kSegNum;
		statInfo[idx].dContractionRatio += dSumOfContractionRatio / kSegNum;
		statInfo[idx].dBitsNeeded += dSumOfBitsNeeded;
	}

	std::ostream &display(std::ostream &os = std::cout, const std::string &prefix = "") {
		// display title
		os << prefix << std::endl;

		os << "length" << "\t"
		   << "count" << "\t"
		   << "totalListLen" << "\t"
		   << "R^2" << "\t"
		   << "ratio" << "\t"
		   << "bitsPerInt" << std::endl;

		for (size_t i = 0; i < kIntervalNum; ++i) {
			if (count[i]) {
				// get averages
				statInfo[i].dRSquare /= count[i];
				statInfo[i].dContractionRatio /= count[i];
				statInfo[i].dBitsNeeded /= totalListLen[i];

				// display results
				os << "[" << (i * IntervalSize) / 1000 << "K, "
				   << ((i + 1) * IntervalSize) / 1000 << "K)" << "\t"
				   << count[i] << "\t"
				   << totalListLen[i] << "\t"
				   << statInfo[i].dRSquare << "\t"
				   << statInfo[i].dContractionRatio << "\t"
				   << statInfo[i].dBitsNeeded << std::endl;
			}
		}

		return os;
	}

	size_t kIntervalNum;          // number of intervals
	std::vector<uint64_t> count;  // number of lists falling into each interval
	std::vector<uint64_t> totalListLen;
	std::vector<RegressionStatInfo_t> statInfo;
};



#endif /* REGRESSIONSTATS_H_ */
