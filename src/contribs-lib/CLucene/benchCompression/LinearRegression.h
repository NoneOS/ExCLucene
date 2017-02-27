/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef LINEARREGRESSION_H_
#define LINEARREGRESSION_H_

#include "common.h"
#include "util.h"


/**
 * To reduce both computation and space cost, we don't keep the minimum.
 * To that end, we have to represent intercept as an int. For the reason,
 * let's consider an input array (or block) with only one value. If that
 * value has too many digits to be represented precisely as a float, we
 * have to keep the minimum, since otherwise the vertical deviation 
 * might be -1.
 */
struct RegressionInfo_t {
	float fSlope;
	int iIntercept;

	RegressionInfo_t() : fSlope(0), iIntercept(0) { }
};

struct RegressionStatInfo_t {
	double dRSquare;
	double dContractionRatio;
	uint32_t uSearchRange;
	double dBitsNeeded;

	RegressionStatInfo_t() : dRSquare(0), dContractionRatio(0), uSearchRange(0), dBitsNeeded(0) { }

	std::ostream &display(std::ostream &os = std::cout) {
		os << dRSquare << "\t" << dContractionRatio << "\t" << dBitsNeeded << std::endl;
		return os;
	}
};


/**
 * Since we represent the intercept as an int, our input values should 
 * fit in int, which is true for both baidu and gov2 datasets.
 */
class LR {

//#undef __AVX2__
//#undef __SSE2__

public:
	std::vector<RegressionInfo_t> regressionInfo;
	std::vector<RegressionStatInfo_t> statInfo;

	LR(): regressionInfo(1), statInfo(1) {
		regressionInfo.shrink_to_fit();
		statInfo.shrink_to_fit();
	}
	virtual ~LR() = default;


	virtual std::string name() const {
        std::ostringstream lrname;
        std::string platform;
#ifdef __AVX2__
        platform = "AVX";
#else  /* !__AVX2__ */
#ifdef __SSE2__
        platform = "SSE";
#else  /* !__SSE2__ */
        platform = "Scalar";
#endif  /* __SSE2__ */
#endif  /* __AVX2__ */
        lrname << "LR_" << platform;
		return lrname.str();
	}

	/**
	 * We take into account size of auxiliary info when computing compression ratio.
	 * For LR and SegLR, this function returns size of regression info.
	 */
	virtual uint64_t sizeOfAuxiliaryInfo() const {
		return regressionInfo.size() * sizeof(RegressionInfo_t);
	}



	static inline void getRegressionInfo(const uint32_t *data, size_t size,
			RegressionInfo_t &regInfo, RegressionStatInfo_t &stInfo);

	/**
	 * For x = 0...size-1, calculate vertical deviation (possibly negative):
	 * iVDs[x] = data[x] - static_cast<int>(fSlope * x) - iIntercept.
	 */
	template <bool IsTailBlock>
	static void getiVDs(const uint32_t *data, size_t size, 
			const RegressionInfo_t &regInfo, int *iVDs);

	/**
	 * For x = 0...size-1, calculate non-negative vertical deviation:
	 * uVDs[x] = VDs[x] - minimum, where minimum = min{VDs[x]|x=0...size-1}.
	 * Note that this can be performed either globally or locally, which is
	 * reflected through T (int - globally, uint32_t - locally).
	 */
	template <bool IsTailBlock, typename T>
	static void getuVDs(const T *VDs, size_t size,
			T &minimum, uint32_t *uVDs, double &dBitsNeeded);

	/**
	 * This is to reduce the space cost and computation cost
	 * of reconstructing docIDs from the corresponding VDs.
	 */
	static void translateLine(int &iIntercept, int iMin) {
		iIntercept += iMin;
	}


	/**
	 * Convert input data to non-negative vertical deviations.
	 * This happens in place.
	 */
	template <bool IsTailBlock>
	static void convert(uint32_t *data, size_t size, 
			RegressionInfo_t &regInfo, RegressionStatInfo_t &stInfo);

	/**
	 * Reconstruct original input data from corresponding vertical deviations.
	 * This also happens in place.
	 */
	template <bool IsTailBlock>
	static void reconstruct(uint32_t *data, size_t size, 
			const RegressionInfo_t &regInfo);


	virtual void runConversion(uint32_t *data, size_t size) {
		convert<true>(data, size, regressionInfo[0], statInfo[0]);
	}

	virtual void runReconstruction(uint32_t *data, size_t size) {
		reconstruct<true>(data, size, regressionInfo[0]);
	}
};


template <uint32_t BlockSize>
class LRSeg : public LR {
public:
	std::vector<uint32_t> localMins;

	LRSeg(): localMins() { }
	virtual ~LRSeg() = default;


	virtual std::string name() const {
        std::ostringstream lrname;
        std::string platform;
#ifdef __AVX2__
        platform = "AVX";
#else  /* !__AVX2__ */
#ifdef __SSE2__
        platform = "SSE";
#else  /* !__SSE2__ */
        platform = "Scalar";
#endif  /* __SSE2__ */
#endif  /* __AVX2__ */
        lrname << "LRSeg<" << BlockSize << ">_" << platform;
		return lrname.str();
	}

	/**
	 * For LRSeg, returns size of regression info plus local minimums.
	 */
	virtual uint64_t sizeOfAuxiliaryInfo() const {
		return regressionInfo.size() * sizeof(RegressionInfo_t) +
			   localMins.size() * sizeof(uint32_t);
	}


	template <bool IsTailBlock>
	static void reconstruct(uint32_t *data, size_t startx, size_t endx,
			const RegressionInfo_t &regInfo);

	
	virtual void runConversion(uint32_t *data, size_t size) {
		convert<true>(data, size, regressionInfo[0], statInfo[0]);

		statInfo[0].dBitsNeeded = 0; // clear since we're gonna recompute it

		// get local minimums of uVDs and reduce each uVDs 
		// from corresponding local minimum
		uint32_t *uVDs = data;

		size_t kBlockNum = div_roundup(size, BlockSize);
		localMins.resize(kBlockNum);
	    localMins.shrink_to_fit();
		for (size_t i = 0; i < kBlockNum - 1; ++i) {
			getuVDs<false>(uVDs, BlockSize, localMins[i], uVDs, statInfo[0].dBitsNeeded);
			uVDs += BlockSize;
		}
		size_t tailBlockSize = size - (kBlockNum - 1) * BlockSize;
		getuVDs<true>(uVDs, tailBlockSize, localMins[kBlockNum - 1], uVDs, statInfo[0].dBitsNeeded);
	}

	virtual void runReconstruction(uint32_t *data, size_t size) {
		RegressionInfo_t regInfo(regressionInfo[0]);
		int iIntercept = regressionInfo[0].iIntercept;

		size_t kBlockNum = div_roundup(size, BlockSize);
		size_t thisBlockOffset = 0;
		for (size_t i = 0; i < kBlockNum - 1; ++i) {
			regInfo.iIntercept = iIntercept + localMins[i];
			reconstruct<false>(data, thisBlockOffset, thisBlockOffset + BlockSize, regInfo);
			thisBlockOffset += BlockSize;
		}
	
		regInfo.iIntercept = iIntercept + localMins[kBlockNum - 1]; 
		reconstruct<true>(data, thisBlockOffset, size, regInfo);
	}
};


template <uint32_t BlockSize>
class SegLR : public LR {
public:
	virtual ~SegLR() = default;

	virtual std::string name() const {
        std::ostringstream lrname;
        std::string platform;
#ifdef __AVX2__
        platform = "AVX";
#else  /* !__AVX2__ */
#ifdef __SSE2__
        platform = "SSE";
#else  /* !__SSE2__ */
        platform = "Scalar";
#endif  /* __SSE2__ */
#endif  /* __AVX2__ */
        lrname << "SegLR<" << BlockSize << ">_" << platform;
		return lrname.str();
	}

	virtual void runConversion(uint32_t *data, size_t size) {
		size_t kBlockNum = div_roundup(size, BlockSize);
		regressionInfo.resize(kBlockNum);
		statInfo.resize(kBlockNum);
		regressionInfo.shrink_to_fit();
		statInfo.shrink_to_fit();

		for (size_t i = 0; i < kBlockNum - 1; ++i) {
			convert<false>(data, BlockSize, regressionInfo[i], statInfo[i]);
			data += BlockSize;
		}

		size_t tailBlockSize = size - (kBlockNum - 1) * BlockSize;
		convert<true>(data, tailBlockSize, regressionInfo[kBlockNum - 1], statInfo[kBlockNum - 1]);
	}

	virtual void runReconstruction(uint32_t *data, size_t size) {
		size_t kBlockNum = div_roundup(size, BlockSize);
		for (size_t i = 0; i < kBlockNum - 1; ++i) {
			reconstruct<false>(data, BlockSize, regressionInfo[i]);
			data += BlockSize;
		}

		size_t tailBlockSize = size - (kBlockNum - 1) * BlockSize;
		reconstruct<true>(data, tailBlockSize, regressionInfo[kBlockNum - 1]);
	}
};

#include "LinearRegressionIMP.h"

#endif /* LINEARREGRESSION_H_ */
