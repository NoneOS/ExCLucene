/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 */

#ifndef LINEARREGRESSIONIMP_H_
#define LINEARREGRESSIONIMP_H_

void LR::getRegressionInfo(const uint32_t *data, size_t size, RegressionInfo_t &regInfo, RegressionStatInfo_t &stInfo) {
	if (size == 0) {
		regInfo.fSlope = 1;
		regInfo.iIntercept = 0;

		stInfo.dRSquare = 1;
		stInfo.dContractionRatio = 0;
		stInfo.uSearchRange = 0;

		return;
	}

	if (size == 1) {
		regInfo.fSlope = 1;
		regInfo.iIntercept = static_cast<int>(data[0]);

		stInfo.dRSquare = 1;
		stInfo.dContractionRatio = 0;
		stInfo.uSearchRange = 0;

		return;
	}

	// get averages
	double dAx = 0, dAy = 0;
	for (size_t x = 0; x < size; ++x) {
		dAx += x;
		dAy += data[x];
	}
	dAx /= size;
	dAy /= size;

	// get sums of squares
	double dSSxx = 0, dSSyy = 0, dSSxy = 0;
	for (size_t x = 0; x < size; ++x) {
		dSSxx += (x - dAx) * (x - dAx);
		dSSyy += (data[x] - dAy) * (data[x] - dAy);
		dSSxy += (x - dAx) * (data[x] - dAy);
	}

	// get slope and intercept
	double dSlope = dSSxy / dSSxx;
	double dIntercept = dAy - dSlope * dAx;

	// get safe search ranges
	double dLeftSearchRange = 0, dRightSearchRange = 0;
	for (size_t x = 0; x < size; ++x) {
		double dInverseX = (data[x] - dIntercept) / dSlope;
		if (dInverseX - x > dLeftSearchRange) {
			dLeftSearchRange = dInverseX - x;
		}
		else if (dInverseX - x < dRightSearchRange) {
			dRightSearchRange = dInverseX - x;
		}
	}
	uint32_t uSearchRange = static_cast<uint32_t>(dLeftSearchRange - dRightSearchRange) + 1;


	// write regression info
	regInfo.fSlope = static_cast<float>(dSlope);
	regInfo.iIntercept = static_cast<int>(dIntercept);

	// write regression stat info
	stInfo.dRSquare = (dSSxy * dSSxy) / (dSSxx * dSSyy);
	stInfo.dContractionRatio = static_cast<double>(uSearchRange) / size;
	stInfo.uSearchRange = uSearchRange;
}


template <bool IsTailBlock>
void LR::getiVDs(const uint32_t *data, size_t size, const RegressionInfo_t &regInfo, int *iVDs) {
	float fSlope = regInfo.fSlope;
	int iIntercept = regInfo.iIntercept;
#ifdef __AVX2__
	__m256 slope = _mm256_set1_ps(fSlope);
	__m256i intercept = _mm256_set1_epi32(iIntercept);

	const size_t sz8 = size / 8 * 8;
	const __m256i *pData = reinterpret_cast<const __m256i *>(data);
	__m256i *piVDs = reinterpret_cast<__m256i *>(iVDs);
	for (size_t x = 0; x < sz8; x += 8) {
		__m256 index = _mm256_set_ps(x + 7, x + 6, x + 5, x + 4,
				                     x + 3, x + 2, x + 1, x + 0);
		__m256 y0 = _mm256_mul_ps(slope, index);
		__m256i y1 = _mm256_add_epi32(_mm256_cvtps_epi32(y0), intercept);
		__m256i y2 = _mm256_sub_epi32(_mm256_loadu_si256(pData++), y1);
		_mm256_storeu_si256(piVDs++, y2);
	}

	if (IsTailBlock) {
		for (size_t x = sz8; x < size; ++x) {
			int y = static_cast<int>(fSlope * x) + iIntercept;
			iVDs[x] = static_cast<int>(data[x]) - y;
		}
	}
#else /* !__AVX2__ */
#ifdef __SSE2__
	__m128 slope = _mm_set1_ps(fSlope);
	__m128i intercept = _mm_set1_epi32(iIntercept);

	const size_t sz4 = size / 4 * 4;
	const __m128i *pData = reinterpret_cast<const __m128i *>(data);
	__m128i *piVDs = reinterpret_cast<__m128i *>(iVDs);
	for (size_t x = 0; x < sz4; x += 4) {
		__m128 index = _mm_set_ps(x + 3, x + 2, x + 1, x + 0);
		__m128 y0 = _mm_mul_ps(slope, index);
		__m128i y1 = _mm_add_epi32(_mm_cvtps_epi32(y0), intercept);
		__m128i y2 = _mm_sub_epi32(_mm_loadu_si128(pData++), y1);
		_mm_storeu_si128(piVDs++, y2);
	}

	if (IsTailBlock) {
		for (size_t x = sz4; x < size; ++x) {
			int y = static_cast<int>(fSlope * x) + iIntercept;
			iVDs[x] = static_cast<int>(data[x]) - y;
		}
	}
#else /* !__AVX2__ && !__SSE2__ */
	for (size_t x = 0; x < size; ++x) {
		int y = static_cast<int>(fSlope * x) + iIntercept;
		iVDs[x] = static_cast<int>(data[x]) - y;
	}
#endif /* __SSE2__ */
#endif /* __AVX2__ */
}


template <bool IsTailBlock, typename T>
void LR::getuVDs(const T *VDs, size_t size, T &minimum, uint32_t *uVDs, double &dBitsNeeded) {
	// get minimum of VDs[0...size-1]
	minimum = VDs[0];
	for (size_t x = 1; x < size; ++x) {
		if (VDs[x] < minimum)
			minimum = VDs[x];
	}

	// get uVDs by reduce minimum from each of VDs[0...size-1]
#ifdef __AVX2__
	__m256i min = _mm256_set1_epi32(minimum);

	const size_t sz8 = size / 8;
	const __m256i *pVDs = reinterpret_cast<const __m256i *>(VDs);
	__m256i *puVDs = reinterpret_cast<__m256i *>(uVDs);
	for (size_t i = 0; i < sz8; ++i) {
		__m256i VD = _mm256_loadu_si256(pVDs++);
		_mm256_storeu_si256(puVDs++, _mm256_sub_epi32(VD, min));
	}

	if (IsTailBlock) {
		for (size_t x = sz8 * 8; x < size; ++x) 
			uVDs[x] = VDs[x] - minimum;
	}
#else /* !__AVX2__ */
#ifdef __SSE2__
	__m128i min = _mm_set1_epi32(minimum);

	const size_t sz4 = size / 4;
	const __m128i *pVDs = reinterpret_cast<const __m128i *>(VDs);
	__m128i *puVDs = reinterpret_cast<__m128i *>(uVDs);
	for (size_t i = 0; i < sz4; ++i) {
		__m128i VD = _mm_loadu_si128(pVDs++);
		_mm_storeu_si128(puVDs++, _mm_sub_epi32(VD, min));
	}

	if (IsTailBlock) {
		for (size_t x = sz4 * 4; x < size; ++x) 
			uVDs[x] = VDs[x] - minimum;
	}
#else /* !__AVX2__ && !__SSE2__ */
	for (size_t x = 0; x < size; ++x) {
		uVDs[x] = VDs[x] - minimum;
	}
#endif /* __SSE2__ */
#endif /* __AVX2__ */

	// compute total number of bits needed if we encode 
	// every VDs using bit width of the maximum VD
	uint32_t mb = maxbits(uVDs, uVDs + size);
	uint64_t nwords = div_roundup(size * mb, 32);
	dBitsNeeded += nwords * 32;
}


template <bool IsTailBlock>
void LR::convert(uint32_t *data, size_t size, RegressionInfo_t &regInfo, RegressionStatInfo_t &stInfo) {
	getRegressionInfo(data, size, regInfo, stInfo);

	int *iVDs = reinterpret_cast<int *>(data);
	getiVDs<IsTailBlock>(data, size, regInfo, iVDs);

	int globalMin = 0;
	uint32_t *uVDs = data;
	getuVDs<IsTailBlock>(iVDs, size, globalMin, uVDs, stInfo.dBitsNeeded);

	translateLine(regInfo.iIntercept, globalMin);
}


template <bool IsTailBlock>
void LR::reconstruct(uint32_t *data, size_t size, const RegressionInfo_t &regInfo) {
	float fSlope = regInfo.fSlope;
	int iIntercept = regInfo.iIntercept;
#ifdef __AVX2__
	__m256 slope = _mm256_set1_ps(fSlope);
	__m256i intercept = _mm256_set1_epi32(iIntercept);

	const size_t sz8 = size / 8 * 8;
	__m256i *pData = reinterpret_cast<__m256i *>(data);
	for (size_t x = 0; x < sz8; x += 8) {
		__m256 index = _mm256_set_ps(x + 7, x + 6, x + 5, x + 4,
				                     x + 3, x + 2, x + 1, x + 0);
		__m256 y0 = _mm256_mul_ps(slope, index);
		__m256i y1 = _mm256_add_epi32(_mm256_cvtps_epi32(y0), intercept);
		__m256i y2 = _mm256_add_epi32(_mm256_loadu_si256(pData), y1);
		_mm256_storeu_si256(pData++, y2);
	}

	if (IsTailBlock) {
		for (size_t x = sz8; x < size; ++x) {
			int y = static_cast<int>(fSlope * x) + iIntercept;
			data[x] += y;
		}
	}
#else /* !__AVX2__ */
#ifdef __SSE2__
	__m128 slope = _mm_set1_ps(fSlope);
	__m128i intercept = _mm_set1_epi32(iIntercept);

	const size_t  sz4 = size / 4 * 4;
	__m128i *pData = reinterpret_cast<__m128i *>(data);
	for (size_t x = 0; x < sz4; x += 4) {
		__m128 index = _mm_set_ps(x + 3, x + 2, x + 1, x + 0);

		__m128 y0 = _mm_mul_ps(slope, index);
		__m128i y1 = _mm_add_epi32(_mm_cvtps_epi32(y0), intercept);
		__m128i y2 = _mm_add_epi32(_mm_loadu_si128(pData), y1);
		_mm_storeu_si128(pData++, y2);
	}

	if (IsTailBlock) {
		for (size_t x = sz4; x < size; ++x) {
			int y = static_cast<int>(fSlope * x) + iIntercept;
			data[x] += y;
		}
	}
#else /* !__AVX2__ && !__SSE2__ */
	for (size_t x = 0; x < size; ++x) {
		int y = static_cast<int>(fSlope * x) + iIntercept;
		data[x] += y;
	}
#endif /* __SSE2__ */
#endif /* __AVX2__ */
}


template <uint32_t BlockSize>
template <bool IsTailBlock>
void LRSeg<BlockSize>::reconstruct(uint32_t *data, size_t startx, size_t endx, 
		const RegressionInfo_t &regInfo) {
	float fSlope = regInfo.fSlope;
	int iIntercept = regInfo.iIntercept;
#ifdef __AVX2__
	__m256 slope = _mm256_set1_ps(fSlope);
	__m256i intercept = _mm256_set1_epi32(iIntercept);

	const size_t endx8 = startx + (endx - startx) / 8 * 8;
	__m256i *pData = reinterpret_cast<__m256i *>(data + startx);
	for (size_t x = startx; x < endx8; x += 8) {
		__m256 index = _mm256_set_ps(x + 7, x + 6, x + 5, x + 4,
				                     x + 3, x + 2, x + 1, x + 0);
		__m256 y0 = _mm256_mul_ps(slope, index);
		__m256i y1 = _mm256_add_epi32(_mm256_cvtps_epi32(y0), intercept);
		__m256i y2 = _mm256_add_epi32(_mm256_loadu_si256(pData), y1);
		_mm256_storeu_si256(pData++, y2);
	}

	if (IsTailBlock) {
		for (size_t x = endx8; x < endx; ++x) {
			int y = static_cast<int>(fSlope * x) + iIntercept;
			data[x] += y;
		}
	}
#else /* !__AVX2__ */
#ifdef __SSE2__
	__m128 slope = _mm_set1_ps(fSlope);
	__m128i intercept = _mm_set1_epi32(iIntercept);

	const size_t endx4 = startx + (endx - startx) / 4 * 4;
	__m128i *pData = reinterpret_cast<__m128i *>(data + startx);
	for (size_t x = startx; x < endx4; x += 4) {
		__m128 index = _mm_set_ps(x + 3, x + 2, x + 1, x + 0);

		__m128 y0 = _mm_mul_ps(slope, index);
		__m128i y1 = _mm_add_epi32(_mm_cvtps_epi32(y0), intercept);
		__m128i y2 = _mm_add_epi32(_mm_loadu_si128(pData), y1);
		_mm_storeu_si128(pData++, y2);
	}

	if (IsTailBlock) {
		for (size_t x = endx4; x < endx; ++x) {
			int y = static_cast<int>(fSlope * x) + iIntercept;
			data[x] += y;
		}
	}
#else /* !__AVX2__ && !__SSE2__ */
	for (size_t x = startx; x < endx; ++x) {
		int y = static_cast<int>(fSlope * x) + iIntercept;
		data[x] += y;
	}
#endif /* __SSE2__ */
#endif /* __AVX2__ */
}

#endif /* LINEARREGRESSIONIMP_H_ */
