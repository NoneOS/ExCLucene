/***************************************************************************
 * 
 * Copyright (c) 2010 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file lr_diff_block.cpp
 * @author wudi01(com@baidu.com)
 * @date 2010/09/21 15:18:09
 * @brief implementation for _LR_DIFF_BLOCK class
 *  
 **/

#include "lr_diff_block.h"

_LR_DIFF_BLOCK::_LR_DIFF_BLOCK():pnUncompressedList(NULL), pnCompressedList(NULL), pnSegHeads(NULL), pnSegHeadsOffset(NULL), pnSegLRs(NULL), nUcnt(0), BS(SEGLEN), EXRATIO(0), nCompressedSizeInInt(0), nSegNum(0), nCurSegIdx(0), nStartPointInSeg(0), MASK(0), nSeekTimesInSeg(0), nBitPosition(0), fLRResult(0), dLRResult(0)
{
    bzero(&sign, sizeof(term_sign_t));
    nCurSegSize = BS;
}

unsigned int _LR_DIFF_BLOCK::select(const unsigned int nIndex)
{
    //locate
    register const unsigned int nSegIdx = nIndex / BS;
    register const unsigned int nInSegIdx = nIndex & (BS - 1);
    const unsigned int *pnTmpCompressedList = pnCompressedList;
    pnTmpCompressedList += pnSegHeadsOffset[nSegIdx];
    //locating ends

    //decompression
    const uint32_t lb = (pnTmpCompressedList[0] >> 26) & 31;
    const int32_t bp = nInSegIdx * lb;
    pnTmpCompressedList += 1;  //only flag is in front of compressed docIDs 
    const int32_t MASK = (1 << lb) - 1;
    uint64_t lTmp = *((uint64_t*)(pnTmpCompressedList + (bp >> 5)));  //bp >> 5 is the index of uint32 it exits
    int32_t iVD = (lTmp >> (bp & 31)) & MASK;
    
    if (1 & iVD)
    {
        iVD >>= 1;
        iVD = -iVD;
    }
    else
    {
        iVD >>= 1;
    }
    //decompression ends
    

    //get real docID
    const regression_info_t * const pTmpRegression = pnSegLRs + nSegIdx;
    return ((uint32_t)(pTmpRegression->fSlope * nInSegIdx + pTmpRegression->fIntercept)) + iVD;
}

inline unsigned int _LR_DIFF_BLOCK::selectInSeg(const unsigned int nIndexInSeg)
{
    int bp = nBitPosition = nIndexInSeg * nCurBitsNum;
    int wp = (bp >> 5);

    uint64_t lTmp = *((uint64_t*)(pnCurSeg + wp));
    int iVD = (lTmp >> (bp & 31)) & MASK;

    /*
    //debug stat
    nSeekTimesInSeg++;
    //debug ends
    */

    //restore the real vd
    if (1 & iVD)
    {
        iVD >>= 1;
        iVD = -iVD;
    }
    else
    {
        iVD >>= 1;
    }

//    fLRResult = fSlope * nIndexInSeg + fIntercept;
//    dLRResult = fSlope * nIndexInSeg + fIntercept;
    return (unsigned int)((int)(fSlope * nIndexInSeg + fIntercept) + iVD);
//    return (unsigned int)((int)dLRResult + iVD);
}

inline unsigned int _LR_DIFF_BLOCK::selectInSegSub()
{
    nBitPosition -= nCurBitsNum;
    int wp = (nBitPosition >> 5);

    uint64_t lTmp = *((uint64_t*)(pnCurSeg + wp));
    int iVD = (lTmp >> (nBitPosition & 31)) & MASK;

    //restore the real vd
    if (1 & iVD)
    {
        iVD >>= 1;
        iVD = -iVD;
    }
    else
    {
        iVD >>= 1;
    }

//    fLRResult -= fSlope;
    dLRResult -= fSlope;
    return (unsigned int)((int)dLRResult + iVD);
}


inline unsigned int _LR_DIFF_BLOCK::selectInSegAdd()
{
    nBitPosition += nCurBitsNum;
    int wp = (nBitPosition >> 5);

    uint64_t lTmp = *((uint64_t*)(pnCurSeg + wp));
    int iVD = (lTmp >> (nBitPosition & 31)) & MASK;

    //restore the real vd
    if (1 & iVD)
    {
        iVD >>= 1;
        iVD = -iVD;
    }
    else
    {
        iVD >>= 1;
    }

//    fLRResult += fSlope;
    dLRResult += fSlope;
    return (unsigned int)((int)dLRResult + iVD);
}



#define SEGHEAD_STEP 16 
bool _LR_DIFF_BLOCK::seek(unsigned int uno)
{
    /*
    //locate the segment
    while (nCurSegIdx + 1 < nSegNum && uno >= pnSegHeads[nCurSegIdx + 1])
    {
        ++nCurSegIdx;

        //init cursors in the segment
        nStartPointInSeg = 0;
    }

    if (uno == pnSegHeads[nCurSegIdx])
    {
        return true;
    }
    //locating ends
    */

    //locate the segment
    //gallop with serial; correct
    unsigned int count = 0;
    register bool isNewSeg = false;
    while (nCurSegIdx + SEGHEAD_STEP < nSegNum && uno >= pnSegHeads[nCurSegIdx + SEGHEAD_STEP])
    {
        nCurSegIdx += SEGHEAD_STEP;
        if (!isNewSeg) isNewSeg = true;
    }
    count = 0;
    while (count < SEGHEAD_STEP && nCurSegIdx + 1 < nSegNum && uno >= pnSegHeads[nCurSegIdx + 1])
    {
        ++nCurSegIdx;
        count++;
        if (!isNewSeg) isNewSeg = true;
    }
    if (isNewSeg)
    {
        nStartPointInSeg = 0;
    }
    if (uno == pnSegHeads[nCurSegIdx])
    {
        return true;
    }
    //locating ends

    /*
    //gallop with binary; with a little error
    unsigned int step = 1;
    register bool isNewSeg = false;
    while (nCurSegIdx + step < nSegNum && uno >= pnSegHeads[nCurSegIdx + step])
    {
        nCurSegIdx += step;
        step <<= 1;
        if (!isNewSeg) isNewSeg = true;
    }
    if (nCurSegIdx + step >= nSegNum)  //exceeds the end of list
    {
        step = nSegNum - nCurSegIdx - 1;
    }
    //step = (step >> 1) + 1;
    step >>= 1;
    while (step)
    {
        if (pnSegHeads[nCurSegIdx + step] < uno)
        {
            nCurSegIdx += step;
        }
        step >>= 1;
    }
    if (uno == pnSegHeads[nCurSegIdx + 1] || uno == pnSegHeads[nCurSegIdx])
    {
        return true;
    }
    */


    /*
    while (nCurSegIdx + 1 < nSegNum && uno >= pnSegHeads[nCurSegIdx + 1])
    {
        ++nCurSegIdx;
        if (!isNewSeg) isNewSeg = true;
    }
    */
    if (isNewSeg)
    {
        nStartPointInSeg = 0;
    }

    //search in the segment
    //a new segment
    if (!nStartPointInSeg)
    {
        pnCurSeg = pnCompressedList + pnSegHeadsOffset[nCurSegIdx];

        //get lb
        unsigned int flag = pnCurSeg[0];
        nCurBitsNum = (flag >> 26) & 31;
        MASK = (1 << nCurBitsNum) - 1;
        pnCurSeg += 1;

        //get segSize; segSize has been initialized to BS
        if (nCurSegIdx == nSegNum - 1)
        {
            nCurSegSize = (nUcnt & (BS - 1)) ? (nUcnt & (BS - 1)) : (BS);
        }

        //get regression formular
        fSlope = pnSegLRs[nCurSegIdx].fSlope;
        fIntercept = pnSegLRs[nCurSegIdx].fIntercept;
    }

    //pivot
    int iPivot = (int)((uno - fIntercept) / fSlope);
    if (iPivot < (int)nStartPointInSeg)
    {
        iPivot = (int)nStartPointInSeg;
    }
    else if (iPivot >= (int)nCurSegSize)
    {
        iPivot = (int)nCurSegSize - 1;
    }
    unsigned int nPivot = (unsigned int)iPivot;
    unsigned int q = selectInSeg(nPivot);
    if (q == uno)
    {
        nStartPointInSeg = (nPivot + 1 < nCurSegSize) ? (nPivot + 1) : (nCurSegSize - 1);
        return true;
    }

    //search
    if (uno < q)  //search in left
    {
        /*
        nPivot = nStartPointInSeg;
        while (1)
        {
            q = selectInSeg(nPivot);

            //hit
            if (q == uno)
            {
//                nStartPointInSeg = (nPivot + 1 < nCurSegSize) ? (nPivot + 1) : (nCurSegSize - 1);
                nStartPointInSeg = nPivot + 1;  //as previous nPivot is the right bound, which is smaller than nCurSegSize absolutely
                return true;
            }

            //miss
            if (q > uno)
            {
                nStartPointInSeg = nPivot;
                return false;
            }

            nPivot++;
        }
        */

        iPivot--;
        while (iPivot >= (int)nStartPointInSeg)
        {
            q = selectInSeg(iPivot);

            //hit
            if (q == uno)
            {
                nStartPointInSeg = iPivot + 1;
                return true;
            }

            //miss
            if (q < uno)
            {
                nStartPointInSeg = iPivot + 1;
                return false;
            }

            iPivot--;
        }

    }
    else  //to right
    {
        nPivot++;
        while (nPivot < nCurSegSize)
        {
            q = selectInSeg(nPivot);

            //hit
            if (q == uno)
            {
                nStartPointInSeg = (nPivot + 1 < nCurSegSize) ? (nPivot + 1) : (nCurSegSize - 1);
                return true;
            }

            //miss
            if (q > uno)
            {
                nStartPointInSeg = nPivot;
                return false;
            }

            //for next loop
            nPivot++;
        }

        //exhausted the segment
        nStartPointInSeg = nCurSegSize - 1;
        return false;
    }
    
    

    return false;
}

/*
bool _LR_DIFF_BLOCK::seek(unsigned int uno)
{
    return (uno & 1);
}
*/

/*
 *
//  simple binary search
bool _LR_DIFF_BLOCK::seek(unsigned int uno)
{
    unsigned int q = 0, p = uno;
    int left = 0, right = nUcnt - 1, middle = 0;

    while (left <= right)
    {
        middle = (left + right) / 2;
        
        q = select((unsigned int)middle);

        if (p == q)
        {
            return true;
        }

        if (p > q)
        {
            left = middle + 1;
        }
        else
        {
            right = middle - 1;
        }
    }

    return false;

}
*/

void _LR_DIFF_BLOCK::LRCompressInMemory(indfmt_index_diff_wei_looper *looper)
{
    //get unCompressed docIDs set
//    printf("filling uncompressed list...\n");
    unsigned int nWei;
    unsigned int *pnBuf = (unsigned int *)malloc(2000 * 10000 * sizeof(unsigned int));
    if (!pnBuf)
    {
        printErr();
    }
    while (looper->next(pnBuf + nUcnt, &nWei))
    {
        nUcnt++;
    }
    nSegNum = (nUcnt & (BS - 1)) ? (nUcnt / BS + 1) : (nUcnt / BS);
    pnUncompressedList = (unsigned int*)malloc(nUcnt * sizeof(unsigned int));
    if (!pnUncompressedList)
    {
        printErr();
    }
//    printf("nUcnt is %u\n");
    memcpy(pnUncompressedList, pnBuf, nUcnt * sizeof(unsigned int));
    free(pnBuf); pnBuf = NULL;
//    printf("filling ends.\n");
    
    //compression
    pnBuf = (unsigned int *)malloc(2000 * 10000 * sizeof(unsigned int));
    pnSegLRs = (regression_info_t*)calloc((nUcnt / BS + 1) , sizeof(regression_info_t));
    if (!pnSegLRs)
    {
        printErr();
    }
    pnSegHeads = (unsigned int*)calloc((nUcnt / BS + 1) , sizeof(unsigned int));
    if (!pnSegHeads)
    {
        printErr();
    }
    pnSegHeadsOffset = (unsigned int*)calloc((nUcnt / BS + 1) , sizeof(unsigned int));
    if (!pnSegHeadsOffset)
    {
        printErr();
    }
    unsigned int *pnTmpSegDecompressed = (unsigned int*)malloc(BS * sizeof(unsigned int));
    int *piBuf = (int *)calloc((BS + 1) , sizeof(unsigned int));
    unsigned int nSegIdx = 0, nHeadsOffset = 0, nSegSize = 0, nCompressedSize = 0;
    unsigned int *pnTmpUncompressed = pnUncompressedList, *pnTmpCompressed = pnBuf;
    statInfo_t SStatInfo;

//    printf("compressing big guys...\n");

    for (; nSegIdx < nUcnt / BS; ++nSegIdx)
    {
        pnSegHeads[nSegIdx] = *pnTmpUncompressed;
        //regression
        getRegressionInfo(pnTmpUncompressed, BS, pnSegLRs + nSegIdx);

        //VD
        getVD(pnTmpUncompressed, BS, (pnSegLRs + nSegIdx)->fSlope, (pnSegLRs + nSegIdx)->fIntercept, piBuf);

        //compress the segment
        nCompressedSize += (nSegSize = VB_Compression((uint32_t*)piBuf, pnTmpCompressed, BS, &SStatInfo));
        pnSegHeadsOffset[nSegIdx] = pnTmpCompressed - pnBuf;


        /*
        //debug decompression
        //decompress
        uint32_t flag = pnTmpCompressed[0];
        uint32_t lb = (flag >> 26) & 31; //printf("lb:%u\n", lb);
        uint32_t eb = (flag >> 21) & 31;
        uint32_t hb = (flag >> 16) & 31;
        uint32_t en = flag & 65535;
        VB_decode(pnTmpCompressed + 1, pnTmpSegDecompressed, lb, eb, hb, en, BS);
        for (uint32_t k = 0; k < BS; ++k)
        {
            if (pnTmpSegDecompressed[k] != (uint32_t)piBuf[k])
            {
                printf("decompression error occurs! old: %u\tnew: %u\n", piBuf[k], pnTmpSegDecompressed[k]);
                exit(1);
            }
        }
        //debug ends
        */

        //for next loop
        pnTmpUncompressed += BS;
        pnTmpCompressed += nSegSize;
    }

//    printf("compressing little guys...\n");
    //deal with little guy
    unsigned int nLittleBS = nUcnt & (BS - 1);
    if (nLittleBS)
    {
        pnSegHeads[nSegIdx] = *pnTmpUncompressed;
        //regression
        if (nLittleBS > 1)
        {
            getRegressionInfo(pnTmpUncompressed, nLittleBS, pnSegLRs + nSegIdx);
        }
        else
        {
            (pnSegLRs + nSegIdx)->fSlope = 1;
            (pnSegLRs + nSegIdx)->fIntercept = 0;
        }

        //VD
        getVD(pnTmpUncompressed, nLittleBS, (pnSegLRs + nSegIdx)->fSlope, (pnSegLRs + nSegIdx)->fIntercept, piBuf);

        //compress
        nCompressedSize += (nSegSize = VB_Compression((uint32_t*)piBuf, pnTmpCompressed, nLittleBS, &SStatInfo));
        pnSegHeadsOffset[nSegIdx] = pnTmpCompressed - pnBuf;
        //debug decompression

        //decompress
        uint32_t flag = pnTmpCompressed[0];
        uint32_t lb = (flag >> 26) & 31; //printf("lb: %u\n", lb);
        uint32_t eb = (flag >> 21) & 31;
        uint32_t hb = (flag >> 16) & 31;
        uint32_t en = flag & 65535;
        VB_decode(pnTmpCompressed + 1, pnTmpSegDecompressed, lb, eb, hb, en, BS);
        for (uint32_t k = 0; k < nLittleBS; ++k)
        {
            if (pnTmpSegDecompressed[k] != (uint32_t)piBuf[k])
            {
                printf("decompression error occurs! old: %u\tnew: %u k:%u\n", piBuf[k], pnTmpSegDecompressed[k], k);
                exit(1);
            }
        }
        //debug ends
    }

    pnCompressedList = (unsigned int *)malloc(nCompressedSize * sizeof(unsigned int));
    if (!pnCompressedList)
    {
        printErr();
    }
    memcpy(pnCompressedList, pnBuf, nCompressedSize * sizeof(unsigned int));

    nCompressedSizeInInt = nCompressedSize;



    free(piBuf); piBuf = NULL;
    free(pnBuf); pnBuf = NULL;
    free(pnTmpSegDecompressed); pnTmpSegDecompressed = NULL;
}


void _LR_DIFF_BLOCK::getRegressionInfo(const uint32_t * const pnList, const uint32_t nListLen, regression_info_t * const pSRegressionInfo)
{
    double dXA = 0, dYA = 0;
    double dDiffSumX = 0, dDiffSumXY = 0, dDiffSumY = 0;
    double dValueY = 0;
    float fRangeLeft = 0, fRangeRight = 0;
    float fPrivateX = 0;

    if (0 == nListLen)
    {
        pSRegressionInfo->fSlope = 1;
        pSRegressionInfo->fIntercept = 0;
        pSRegressionInfo->nRangeLeft = 0;
        pSRegressionInfo->nRangeRight = 0;
        return;
    }

    if (1 == nListLen)
    {
        pSRegressionInfo->fSlope = 1;
        pSRegressionInfo->fIntercept = pnList[0];
        pSRegressionInfo->nRangeLeft = 0;
        pSRegressionInfo->nRangeRight = 0;
        return;
    }

    //average
    for (unsigned int i = 0; i < nListLen; ++i)
    {
        dXA += i / (double)nListLen;
        dYA += pnList[i] / (double)nListLen;
    }

    //diffSum
    for (unsigned int i = 0; i < nListLen; ++i)
    {
        dValueY = (double)(pnList[i]);
        dDiffSumX += (double)((i - dXA) * (i - dXA));
        dDiffSumXY += (double)((i - dXA) * (dValueY - dYA));
        dDiffSumY += (double)((dValueY - dYA) * (dValueY - dYA));
    }

    //regression formula
    pSRegressionInfo->fSlope = (float)(dDiffSumXY / dDiffSumX);
    pSRegressionInfo->fIntercept = (float)(dYA - pSRegressionInfo->fSlope * dXA);
    if (nListLen != 2)
    {
        pSRegressionInfo->fRSquare = (float)(dDiffSumXY / sqrt(dDiffSumX * dDiffSumY));
    }
    else
    {
        pSRegressionInfo->fRSquare = 1;
    }

    //furthest points
    for (unsigned int i = 0; i < nListLen; ++i)
    {
        fPrivateX = (pnList[i] - pSRegressionInfo->fIntercept) / pSRegressionInfo->fSlope;

        if (fPrivateX - i > fRangeLeft)
		{
			fRangeLeft = fPrivateX - i;
		}
		else if (fPrivateX - i < fRangeRight)
		{
			fRangeRight = fPrivateX - i;
		}
	}

	//safe range
	pSRegressionInfo->nRangeLeft = (uint32_t)(fRangeLeft) + 1;
	pSRegressionInfo->nRangeRight = (uint32_t)(0 - fRangeRight) + 1;
}



void _LR_DIFF_BLOCK::getVD(uint32_t *pnList, const uint32_t nNum, const float fSlope, const float fIntercept, int32_t *piVD)
{
	uint32_t nCheckSumOri = 0;
	for (uint32_t nDocIdx = 0; nDocIdx < nNum; ++nDocIdx)
	{
		nCheckSumOri ^= pnList[nDocIdx];
		
		int32_t iVD = (int32_t)(pnList[nDocIdx] - (nDocIdx * fSlope + fIntercept));  //can be converted into an arithmetic sequence, if it is too slow here
		int32_t iYTheory = (int32_t)(nDocIdx * fSlope + fIntercept); //similarly
		register uint32_t nDocID = pnList[nDocIdx];

		//above the line
		if (iVD >= 0)
		{
			iVD -= 2;
			//attempts
			if (iYTheory + iVD == (int32_t)nDocID)
			{
				piVD[nDocIdx] = iVD;
			}
			else if (iYTheory + (++iVD) == (int32_t)nDocID)
			{
				piVD[nDocIdx] = iVD;
			}
			else if (iYTheory + (++iVD) == (int32_t)nDocID)
			{
				piVD[nDocIdx] = iVD;
			}
			else if (iYTheory + (++iVD) == (int32_t)nDocID)
			{
				piVD[nDocIdx] = iVD;
			}
			else
			{
				printf("miss!! nDocID:%u\tiYTheory:%d\tiVD:%d\tslope:%f\tintercept:%flistLen:%u\n", nDocID, iYTheory, iVD, fSlope, fIntercept, nNum);
				exit(1);
			}
		}
		else  //below
		{
			iVD += 2;
			if (iYTheory + iVD == (int32_t)nDocID)
			{
				piVD[nDocIdx] = iVD;
			}
			else if (iYTheory + (--iVD) == (int32_t)nDocID)
			{
				piVD[nDocIdx] = iVD;
			}
			else if (iYTheory + (--iVD) == (int32_t)nDocID)
			{
				piVD[nDocIdx] = iVD;
			}
			else if (iYTheory + (--iVD) == (int32_t)nDocID)
			{
				piVD[nDocIdx] = iVD;
			}
			else
			{
				printf("miss!! nDocID:%u\tiYTheory:%d\tiVD:%d\tslope:%f\tintercept:%flistLen:%u\n", nDocID, iYTheory, iVD, fSlope, fIntercept, nNum);
				exit(1);
			}
		}
	}  //end of nDocIdx

	//sign bit
	for (uint32_t nDocIdx = 0; nDocIdx < nNum; ++nDocIdx)
	{
		if (piVD[nDocIdx] < 0)
		{
			piVD[nDocIdx] = -(piVD[nDocIdx]);
			piVD[nDocIdx] <<= 1;
			piVD[nDocIdx] |= 1;
		}
		else
		{
			piVD[nDocIdx] <<= 1;
		}
	}

	//debug restore
	uint32_t nCheckSumNew = 0;
	for (unsigned int nDocIdx = 0; nDocIdx < nNum; ++nDocIdx)
	{
		int32_t iTmpVD = piVD[nDocIdx];
		if (1 & iTmpVD)  //negative
		{
			iTmpVD >>= 1;
			iTmpVD = -iTmpVD;
		}
		else  //positive
		{
			iTmpVD >>= 1;
		}
		uint32_t nRestoredDocID = ((int32_t)(fSlope * nDocIdx + fIntercept)) + iTmpVD;
		nCheckSumNew ^= nRestoredDocID;
	}

	if (nCheckSumNew != nCheckSumOri)
	{
		printf("checksum err!!\n");
		exit(1);
	}
	else
	{
//		printf("pass:) ori:%X\tnew:%X\n", nCheckSumOri, nCheckSumNew);
	}
	//debug ends
}


template <class type>
uint32_t _LR_DIFF_BLOCK::VB_Compression(const type * const input, type * const output, uint32_t block_size, statInfo_t *pSStatInfo)
{
	int32_t flag = -1;
	unsigned int nSize = 0;  //segment size after being compressed
	for (unsigned int lb = 0; flag < 0; ++lb)
	{
		flag = VB_encode(input, output + 1, lb, block_size, &nSize, pSStatInfo);
	}
	*(output) = flag;
	return nSize;
}




template <class type>
int32_t _LR_DIFF_BLOCK::VB_encode(const type  * const input, type *output, const uint32_t lb, const uint32_t block_size, uint32_t * const nSize, statInfo_t *pSStatInfo)
{
	//three parts allocation
	uint32_t *lowBits, *highBits, *exPos;
	if (!(lowBits = (uint32_t*)malloc(sizeof(uint32_t) * block_size * 2)))
	{
		printErr(); exit(1);
	}
	if (!(highBits = (uint32_t*)malloc(sizeof(uint32_t) * block_size * 2)))
	{
		printErr(); exit(1);
	}
	if (!(exPos = (uint32_t*)malloc(sizeof(uint32_t) * block_size * 2)))
	{
		printErr(); exit(1);
	}
	//allocation ends
	
	//local
	uint32_t en = 0;  //exceeding elements number
	int32_t flag = 0;
	uint32_t eb = 0, hb = 0;  //exceeding pos bits and high bits, after compression per ele
	uint32_t nTmpSize = 0;
	//local ends

	//travers all ints in the segment
	for (unsigned int i = 0; i < block_size; ++i)
	{
		/*
		if (globalListIdx == 601 && globalSegIdx == 7358)
		{
			printf("%u ", input[i]);
		}
		*/

		if (input[i] >= (uint32_t)(1 << lb))  //exceeding docID
		{
			lowBits[i] = input[i] & ((1 << lb) - 1);
			highBits[en] = input[i] >> lb;
			exPos[en++] = i;  //pos in the segment
		}
		else
		{
			lowBits[i] = input[i];
		}
	}

	if (en <= EXRATIO * block_size)
	{
		//lowBits part
		uint32_t nWordNum = ((lb * block_size) >> 5) + ((((lb * block_size) & 31) > 0) ? 1: 0);
		bzero(output, sizeof(uint32_t) * nWordNum);
		pack(lowBits, lb, block_size, output);
		output += nWordNum;
		nTmpSize += nWordNum;


		//the following two parts are compressed with the biggest element
		//exPos part
		eb = maxBitsNum(exPos, en);
		nWordNum = ((eb * en) >> 5) + ((((eb * en) & 31) > 0) ? 1: 0);
		bzero(output, sizeof(uint32_t) * nWordNum);
		pack(exPos, eb, en, output);
		output += nWordNum;
		nTmpSize += nWordNum;

		//highBits part
		hb = maxBitsNum(input, block_size) - lb; 
		nWordNum = ((hb * en) >> 5) + ((((hb * en) & 31) > 0) ? 1: 0); 
		bzero(output, sizeof(uint32_t) * nWordNum);
		pack(highBits, hb, en, output);
		nTmpSize += nWordNum;

		//return the size (ints number) of the segment
		*nSize = nTmpSize + 1;  //1 is the flag

		//assemble the flag
		flag = (lb << 26) + (eb << 21) + (hb << 16) + en;

		//stat
		pSStatInfo->lbA += lb;
		pSStatInfo->ebA += eb;
		pSStatInfo->hbA += hb;
		pSStatInfo->enA += en;
	}
	else
	{
		flag = -1;
	}

	free(lowBits); lowBits = NULL;
	free(highBits); highBits = NULL;
	free(exPos); exPos = NULL;
	return flag;

}

template <class type>
uint32_t _LR_DIFF_BLOCK::VB_decode(const type * input, type * const output, const uint32_t lb, const uint32_t eb, const uint32_t hb, const uint32_t en, const uint32_t block_size)
{
	//local
	uint32_t *exPos = (uint32_t*)malloc(sizeof(uint32_t) * block_size * 4);
	uint32_t *highBits = (uint32_t*)malloc(sizeof(uint32_t) * block_size * 4);
	const type *inputOri = input;
	//local ends
	
	//lowBits
	unpack(input, lb, block_size, output);
	input += ((lb * block_size) >> 5) + ((lb * block_size & 31) ? 1 : 0);

	//exPos
	unpack(input, eb, en, exPos);
	input += ((eb * en) >> 5) + ((eb * en & 31) ? 1 : 0);

	//highBits
	unpack(input, hb, en, highBits);
	input += ((hb * en) >> 5) + ((hb * en & 31) ? 1 : 0);
	
	//connect
	for (unsigned int i = 0; i < en; ++i)
	{
		highBits[i] <<= lb;
		output[exPos[i]] |= highBits[i];
	}

	/*
	//debug
	for (unsigned int k = 0; k < block_size; ++k)
		printf("restored: %u\n", output[k]);
	//debug ends
	*/

	//release
	free(exPos);  exPos = NULL;
	free(highBits);  highBits = NULL;
	return input - inputOri;
}


/*
 * make sure that the type can be compared with built-in operators
 */
template <class type>
uint32_t _LR_DIFF_BLOCK::maxBitsNum(const type * const input, const uint32_t nEleNum)
{
	//find the max one
	type max = 0;
	uint32_t nBitsNum = 0;
	for (uint32_t i = 0; i < nEleNum; ++i)
	{
		if (max < input[i])
		{
			max = input[i];
		}
	}


	//determine the bits num needed
	for (; max > 0; max = max >> 1)
	{
		nBitsNum++;
	}

	return nBitsNum;
}

//implementation of _HS_LR_DIFF_BLOCK class
void _HS_LR_DIFF_BLOCK::generateHashSegmentInfo()
{
    nMaxDocIDBitsNum = getBitsNum(pnUncompressedList[nUcnt - 1]);

    if (nUcnt < EXPECTED_SEGLEN)
    {
        HS_SHIFT = nMaxDocIDBitsNum;
        nHashSegNum = 1;
        pnHSSegLen = (unsigned int *)malloc(sizeof(unsigned int));
        *pnHSSegLen = nUcnt;
        pnSegHeadsOffset = (unsigned int*)malloc(sizeof(unsigned int));
        *pnSegHeadsOffset = 0;

        return;
    }

    double dLog = log(((double)nUcnt) / EXPECTED_SEGLEN) / log((double)2);
    unsigned int nHighBitsNum = 0;
    if (dLog - (unsigned int)dLog > 0)
    {
        nHighBitsNum = ((unsigned int)dLog) + 1;
    }
    else
    {
        nHighBitsNum = (unsigned int)dLog;
    }
    unsigned int nVirtualSegNum = (unsigned int)pow(2.0, (int)nHighBitsNum);
    HS_SHIFT = nMaxDocIDBitsNum - nHighBitsNum;

    //allocate tmp headsOffset and segLenArray
    unsigned int *pnTmpHashSegLen = (unsigned int*)malloc(nVirtualSegNum * sizeof(unsigned int));

    //traverse the list
    unsigned int nCurSegID = 0, nSegID = 0;
    unsigned int nPrevSegHeadOffset = 0;
    unsigned int docIndex = 0;
    for (; docIndex < nUcnt; ++docIndex)
    {
        nSegID = pnUncompressedList[docIndex] >> HS_SHIFT;
        if (nSegID > nCurSegID)  //a new seg is found
        {
            while (nCurSegID < nSegID)
            {
                pnTmpHashSegLen[nCurSegID] = docIndex - nPrevSegHeadOffset;
                nPrevSegHeadOffset = docIndex;
                ++nCurSegID;
            }
        }
    }
    if (docIndex > nPrevSegHeadOffset)
    {
        pnTmpHashSegLen[nCurSegID] = docIndex - nPrevSegHeadOffset;
        ++nCurSegID;
    }

    nHashSegNum = nCurSegID;

    //allocate real segLenArray
    pnHSSegLen = (unsigned int *)malloc(nHashSegNum * sizeof(unsigned int));
    if (!pnHSSegLen)
    {
        printErr();
    }
    memcpy(pnHSSegLen, pnTmpHashSegLen, nHashSegNum * sizeof(unsigned int));

    //free local resource
    free(pnTmpHashSegLen); pnTmpHashSegLen = NULL;
}

void _HS_LR_DIFF_BLOCK::LRCompressInMemory(indfmt_index_diff_wei_looper *looper)
{
    //build uncompressed list
    printf("building uncompressed...\n");
    unsigned int nWei;
    unsigned int *pnBuf = (unsigned int *)malloc(2000 * 10000 * sizeof(unsigned int));
    if (!pnBuf)
    {
        printErr();
    }
    while (looper->next(pnBuf + nUcnt, &nWei))
    {
        nUcnt++;
    }
    pnUncompressedList = (unsigned int*)malloc(nUcnt * sizeof(unsigned int));
    if (!pnUncompressedList)
    {
        printErr();
    }
    memcpy(pnUncompressedList, pnBuf, nUcnt * sizeof(unsigned int));
    free(pnBuf); pnBuf = NULL;

    //generate hash infomation; mainly fill the segment len array
    printf("generating hash...\n");
    generateHashSegmentInfo();

    //prepare for compression
    pnBuf = (unsigned int *)malloc(2000 * 10000 * sizeof(unsigned int));
    pnSegLRs = (regression_info_t*)calloc((nHashSegNum + 1) , sizeof(regression_info_t));
    if (!pnSegLRs)
    {
        printErr();
    }
    pnSegHeadsOffset = (unsigned int*)calloc((nHashSegNum + 1) , sizeof(unsigned int));
    if (!pnSegHeadsOffset)
    {
        printErr();
    }
    unsigned int *pnTmpSegDecompressed = (unsigned int*)malloc(EXPECTED_SEGLEN * 1024 * sizeof(unsigned int));
    int *piBuf = (int *)calloc(EXPECTED_SEGLEN * 1024 , sizeof(unsigned int));
    unsigned int nSegIdx = 0, nHeadsOffset = 0, nSegSize = 0, nCompressedSize = 0;
    unsigned int *pnTmpUncompressed = pnUncompressedList, *pnTmpCompressed = pnBuf;
    statInfo_t SStatInfo;

    printf("compressing...\n");
    //traverse all hash segments
    for (; nSegIdx < nHashSegNum; ++nSegIdx)
    {
        //regression
        getRegressionInfo(pnTmpUncompressed, pnHSSegLen[nSegIdx], pnSegLRs + nSegIdx);
        if (pnHSSegLen[nSegIdx])
        {
            //VD
            getVD(pnTmpUncompressed, pnHSSegLen[nSegIdx], (pnSegLRs + nSegIdx)->fSlope, (pnSegLRs + nSegIdx)->fIntercept, piBuf);

            //compress the segment
            nCompressedSize += (nSegSize = VB_Compression((uint32_t*)piBuf, pnTmpCompressed, pnHSSegLen[nSegIdx], &SStatInfo));
        }
        pnSegHeadsOffset[nSegIdx] = pnTmpCompressed - pnBuf;

        if (pnHSSegLen[nSegIdx])
        {
            //debug decompression
            //decompress
            uint32_t flag = pnTmpCompressed[0];
            uint32_t lb = (flag >> 26) & 31; //printf("lb:%u\n", lb);
            uint32_t eb = (flag >> 21) & 31;
            uint32_t hb = (flag >> 16) & 31;
            uint32_t en = flag & 65535;
            VB_decode(pnTmpCompressed + 1, pnTmpSegDecompressed, lb, eb, hb, en, pnHSSegLen[nSegIdx]);
            for (uint32_t k = 0; k < pnHSSegLen[nSegIdx]; ++k)
            {
                if (pnTmpSegDecompressed[k] != (uint32_t)piBuf[k])
                {
                    printf("decompression error occurs! old: %u\tnew: %u\n", piBuf[k], pnTmpSegDecompressed[k]);
                    exit(1);
                }
            }
            //debug ends
        }

        //for next loop
        pnTmpUncompressed += pnHSSegLen[nSegIdx];
        pnTmpCompressed += nSegSize;
    }

    pnCompressedList = (unsigned int *)malloc(nCompressedSize * sizeof(unsigned int));
    if (!pnCompressedList)
    {
        printErr();
    }
    memcpy(pnCompressedList, pnBuf, nCompressedSize * sizeof(unsigned int));

    nCompressedSizeInInt = nCompressedSize;



    free(piBuf); piBuf = NULL;
    free(pnBuf); pnBuf = NULL;
    free(pnTmpSegDecompressed); pnTmpSegDecompressed = NULL;
}

unsigned int _HS_LR_DIFF_BLOCK::getBitsNum(unsigned int p)
{
    unsigned int i = 0;

    for (; p > 0; p >>= 1, ++i);

    return i;
}


bool _HS_LR_DIFF_BLOCK::seek(unsigned int uno)
{
    //hash locate
    register unsigned int nSegIdx = (uno >> HS_SHIFT);
    //locating ends

    //a new segment
    if (nSegIdx != nPrevSegIdx)
    {
        nStartPointInSeg = 0;
        pnCurSeg = pnCompressedList + pnSegHeadsOffset[nSegIdx];

        //get lb
        unsigned int flag = pnCurSeg[0];
        nCurBitsNum = (flag >> 26) & 31;
        MASK = (1 << nCurBitsNum) - 1;
        pnCurSeg += 1;

        //get segSize
        if(!(nCurSegSize = pnHSSegLen[nSegIdx]))
        {
            return false;
        }

        //get regression 
        fSlope = pnSegLRs[nSegIdx].fSlope;
        fIntercept = pnSegLRs[nSegIdx].fIntercept;

        //record prev seg idx
        nPrevSegIdx = nSegIdx;
    }

    //pivot
    int iPivot = (int)((uno - fIntercept) / fSlope);
    if (iPivot < (int)nStartPointInSeg)
    {
        iPivot = (int)nStartPointInSeg;
    }
    else if (iPivot >= (int)nCurSegSize)
    {
        iPivot = (int)nCurSegSize - 1;
    }
    unsigned int nPivot = (unsigned int)iPivot;
    unsigned int q = selectInSeg(nPivot);
    if (q == uno)
    {
        nStartPointInSeg = (nPivot + 1 < nCurSegSize) ? (nPivot + 1) : (nCurSegSize - 1);
        return true;
    }

    //search
    if (uno < q)  //search in left
    {
        iPivot--;
        while (iPivot >= (int)nStartPointInSeg)
        {
            q = selectInSeg(iPivot);
//            q = selectInSegSub();

            //hit
            if (q == uno)
            {
                nStartPointInSeg = iPivot + 1;
                return true;
            }

            //miss
            if (q < uno)
            {
                nStartPointInSeg = iPivot + 1;
                return false;
            }

            iPivot--;
        }

        return false;
    }
    else  //search in right
    {
        nPivot++;
        while (nPivot < nCurSegSize)
        {
            q = selectInSeg(nPivot);
//            q = selectInSegAdd();

            //hit
            if (q == uno)
            {
                nStartPointInSeg = (nPivot + 1 < nCurSegSize) ? (nPivot + 1) : (nCurSegSize - 1);
                return true;
            }

            //miss
            if (q > uno)
            {
                nStartPointInSeg = nPivot;
                return false;
            }

            //for next loop
            nPivot++;
        }

        //exhausted the segment
        nStartPointInSeg = nCurSegSize - 1;
        return false;
    }

    return false;
}


/*
void _PFD_LR_DIFF_BLOCK::LRCompressInMemory(indfmt_index_diff_wei_looper *looper)
{
    //build uncompressed list
    printf("building uncompressed...\n");
    unsigned int nWei;
    unsigned int *pnBuf = (unsigned int *)malloc(2000 * 10000 * sizeof(unsigned int));
    if (!pnBuf)
    {
        printErr();
    }
    while (looper->next(pnBuf + nUcnt, &nWei))
    {
        nUcnt++;
    }
    pnUncompressedList = (unsigned int*)malloc(nUcnt * sizeof(unsigned int));
    if (!pnUncompressedList)
    {
        printErr();
    }
    memcpy(pnUncompressedList, pnBuf, nUcnt * sizeof(unsigned int));
    free(pnBuf); pnBuf = NULL;

    //generate hash infomation; mainly fill the segment len array
    printf("generating hash...\n");
    generateHashSegmentInfo();

    //pnSegHeads
    pnSegHeads = (unsigned int*)calloc(nHashSegNum + 1, sizeof(unsigned int));
    if (!pnSegHeads)
    {
        printErr();
    }

    //pnDGap
    unsigned int *pnDGap = (unsigned int*)malloc(10000000 * sizeof(unsigned int));
    if (!pnDGap)
    {
        printErr();
    }

    //SegLR
    pnSegLRs = (regression_info_t*)calloc((nHashSegNum + 1) , sizeof(regression_info_t));
    if (!pnSegLRs)
    {
        printErr();
    }

    //headOffset
    pnSegHeadsOffset = (unsigned int*)calloc((nHashSegNum + 1) , sizeof(unsigned int));
    if (!pnSegHeadsOffset)
    {
        printErr();
    }

    //others
    unsigned int *pnTmpSegDecompressed = (unsigned int*)malloc(EXPECTED_SEGLEN * 1024 * sizeof(unsigned int));
    
    pnBuf = (unsigned int *)malloc(EXPECTED_SEGLEN * 1024 * sizeof(unsigned int));
    unsigned int nSegIdx = 0, nHeadsOffset = 0, nSegSize = 0, nCompressedSize = 0;
    unsigned int *pnTmpUncompressed = pnUncompressedList, *pnTmpCompressed = pnBuf;
    statInfo_t SStatInfo;

    printf("compressing...\n");
    //traverse all hash segments
    for (; nSegIdx < nHashSegNum; ++nSegIdx)
    {
        printf("CMP begin\n");
        //regression
//        getRegressionInfo(pnTmpUncompressed, pnHSSegLen[nSegIdx], pnSegLRs + nSegIdx);
        //seg heads value
        pnSegHeads[nSegIdx] = pnTmpUncompressed[0];

        if (pnHSSegLen[nSegIdx])
        {
            //VD
            printf("VD...\n");
            getDGap(pnTmpUncompressed, pnHSSegLen[nSegIdx], pnDGap);

            //compress the segment
            printf("compression...\n");
            printf("hashSegNum: %u\tlen: %u\t\n", nHashSegNum, pnHSSegLen[nSegIdx]);
            nCompressedSize += (nSegSize = VB_Compression((uint32_t*)pnDGap, pnTmpCompressed, pnHSSegLen[nSegIdx], &SStatInfo));
        }
        pnSegHeadsOffset[nSegIdx] = pnTmpCompressed - pnBuf;
        printf("CMP ends\n");

        if (pnHSSegLen[nSegIdx])
        {
            //debug decompression
            //decompress
            uint32_t flag = pnTmpCompressed[0];
            uint32_t lb = (flag >> 26) & 31; //printf("lb:%u\n", lb);
            uint32_t eb = (flag >> 21) & 31;
            uint32_t hb = (flag >> 16) & 31;
            uint32_t en = flag & 65535;
            VB_decode(pnTmpCompressed + 1, pnTmpSegDecompressed, lb, eb, hb, en, pnHSSegLen[nSegIdx]);
            for (uint32_t k = 0; k < pnHSSegLen[nSegIdx]; ++k)
            {
                if (pnTmpSegDecompressed[k] != (uint32_t)pnDGap[k])
                {
                    printf("decompression error occurs! old: %u\tnew: %u\n", pnDGap[k], pnTmpSegDecompressed[k]);
                    exit(1);
                }
            }
            //debug ends
        }

        //for next loop
        pnTmpUncompressed += pnHSSegLen[nSegIdx];
        pnTmpCompressed += nSegSize;
    }
    printf("compression and check finished===\n");

    pnCompressedList = (unsigned int *)malloc(nCompressedSize * sizeof(unsigned int));
    if (!pnCompressedList)
    {
        printErr();
    }
    memcpy(pnCompressedList, pnBuf, nCompressedSize * sizeof(unsigned int));

    nCompressedSizeInInt = nCompressedSize;



    free(pnBuf); pnBuf = NULL;
    if (pnDGap)
    {
        free(pnDGap);
        pnDGap = NULL;
    }
    free(pnTmpSegDecompressed); pnTmpSegDecompressed = NULL;
}
*/



void _PFD_LR_DIFF_BLOCK::LRCompressInMemory(indfmt_index_diff_wei_looper *looper)
{
    //build uncompressed list
    printf("building uncompressed...\n");
    unsigned int nWei;
    unsigned int *pnBuf = (unsigned int *)malloc(2000 * 10000 * sizeof(unsigned int));
    if (!pnBuf)
    {
        printErr();
    }
    while (looper->next(pnBuf + nUcnt, &nWei))
    {
        nUcnt++;
    }
    pnUncompressedList = (unsigned int*)malloc(nUcnt * sizeof(unsigned int));
    if (!pnUncompressedList)
    {
        printErr();
    }
    memcpy(pnUncompressedList, pnBuf, nUcnt * sizeof(unsigned int));
    free(pnBuf); pnBuf = NULL;

    //generate hash infomation; mainly fill the segment len array
    printf("generating hash...\n");
    generateHashSegmentInfo();

    //prepare for compression
    pnBuf = (unsigned int *)malloc(2000 * 10000 * sizeof(unsigned int));
    pnSegLRs = (regression_info_t*)calloc((nHashSegNum + 1) , sizeof(regression_info_t));
    if (!pnSegLRs)
    {
        printErr();
    }
    pnSegHeadsOffset = (unsigned int*)calloc((nHashSegNum + 1) , sizeof(unsigned int));
    if (!pnSegHeadsOffset)
    {
        printErr();
    }
    pnSegHeads = (unsigned int*)calloc((nHashSegNum + 1) , sizeof(unsigned int));
    if (!pnSegHeads)
    {
        printErr();
    }
    unsigned int *pnTmpSegDecompressed = (unsigned int*)malloc(EXPECTED_SEGLEN * 1024 * sizeof(unsigned int));
    int *piBuf = (int *)calloc(EXPECTED_SEGLEN * 1024 , sizeof(unsigned int));
    unsigned int nSegIdx = 0, nHeadsOffset = 0, nSegSize = 0, nCompressedSize = 0;
    unsigned int *pnTmpUncompressed = pnUncompressedList, *pnTmpCompressed = pnBuf;
    statInfo_t SStatInfo;

    printf("compressing...\n");
    //traverse all hash segments
    for (; nSegIdx < nHashSegNum; ++nSegIdx)
    {
        pnSegHeads[nSegIdx] = *pnTmpUncompressed;
        //regression
        getRegressionInfo(pnTmpUncompressed, pnHSSegLen[nSegIdx], pnSegLRs + nSegIdx);
        if (pnHSSegLen[nSegIdx])
        {
            //VD
            //getVD(pnTmpUncompressed, pnHSSegLen[nSegIdx], (pnSegLRs + nSegIdx)->fSlope, (pnSegLRs + nSegIdx)->fIntercept, piBuf);
            getDGap(pnTmpUncompressed, pnHSSegLen[nSegIdx], (unsigned int*)piBuf);

            //compress the segment
            nCompressedSize += (nSegSize = VB_Compression((uint32_t*)piBuf, pnTmpCompressed, pnHSSegLen[nSegIdx], &SStatInfo));
        }
        pnSegHeadsOffset[nSegIdx] = pnTmpCompressed - pnBuf;

        if (pnHSSegLen[nSegIdx])
        {
            //debug decompression
            //decompress
            uint32_t flag = pnTmpCompressed[0];
            uint32_t lb = (flag >> 26) & 31; //printf("lb:%u\n", lb);
            uint32_t eb = (flag >> 21) & 31;
            uint32_t hb = (flag >> 16) & 31;
            uint32_t en = flag & 65535;
            VB_decode(pnTmpCompressed + 1, pnTmpSegDecompressed, lb, eb, hb, en, pnHSSegLen[nSegIdx]);
            for (uint32_t k = 0; k < pnHSSegLen[nSegIdx]; ++k)
            {
                if (pnTmpSegDecompressed[k] != (uint32_t)piBuf[k])
                {
                    printf("decompression error occurs! old: %u\tnew: %u\n", piBuf[k], pnTmpSegDecompressed[k]);
                    exit(1);
                }
            }
            //debug ends
        }

        //for next loop
        pnTmpUncompressed += pnHSSegLen[nSegIdx];
        pnTmpCompressed += nSegSize;
    }

    pnCompressedList = (unsigned int *)malloc(nCompressedSize * sizeof(unsigned int));
    if (!pnCompressedList)
    {
        printErr();
    }
    memcpy(pnCompressedList, pnBuf, nCompressedSize * sizeof(unsigned int));

    nCompressedSizeInInt = nCompressedSize;



    free(piBuf); piBuf = NULL;
    free(pnBuf); pnBuf = NULL;
    free(pnTmpSegDecompressed); pnTmpSegDecompressed = NULL;
}

void _PFD_LR_DIFF_BLOCK::getDGap(unsigned int *pnList, const unsigned int nNum, unsigned int *pnDGap)
{
    pnDGap[0] = 0;
    for (unsigned nDocIdx = 1; nDocIdx < nNum; ++nDocIdx)
    {
        pnDGap[nDocIdx] = pnList[nDocIdx] - pnList[nDocIdx - 1];
    }

    //debug restore
    unsigned int nRestoreCur = pnList[0];
    for (unsigned int nDocIdx = 1; nDocIdx < nNum; ++nDocIdx)
    {
        nRestoreCur += pnDGap[nDocIdx];
        if (nRestoreCur != pnList[nDocIdx])
        {
            printf("faint, serial restore failed on earth!!\n");
            exit(1);
        }
    }
}


bool _PFD_LR_DIFF_BLOCK::seek(unsigned int uno)
{
    register unsigned int nSegIdx = (uno >> HS_SHIFT);

    //a new segment
    if (nSegIdx != nPrevSegIdx)
    {
        nStartPointInSeg = 0;
        pnCurSeg = pnCompressedList + pnSegHeadsOffset[nSegIdx];

        //get lb
        unsigned int flag = pnCurSeg[0];
        nCurBitsNum = (flag >> 26) & 31;
        MASK = (1 << nCurBitsNum) - 1;
        pnCurSeg += 1;

        //get segSize
        if(!(nCurSegSize = pnHSSegLen[nSegIdx]))
        {
            return false;
        }

        //record prev seg idx
        nPrevSegIdx = nSegIdx;

        //get headValue
        nCurDocID = pnSegHeads[nSegIdx];
    }

    //compare current
    if (uno == nCurDocID)  //hit
    {
        return true;
    }
    else if (uno < nCurDocID)  //miss
    {
        return false;
    }

    //serial search
    while (1)
    {
        selectInSegNext();
        if (uno == nCurDocID)
        {
            return true;
        }
        else if (uno < nCurDocID)
        {
            return false;
        }
    }
}


void _PFD_LR_DIFF_BLOCK::selectInSegNext()
{
    if (nStartPointInSeg >= nCurSegSize)
    {
        nCurDocID = INT_MAX;
        return;
    }

    ++nStartPointInSeg;
    int bp = nStartPointInSeg * nCurBitsNum;
    int wp = (bp >> 5);

    uint64_t lTmp = *((uint64_t*)(pnCurSeg + wp));
    unsigned int nDGap = (lTmp >> (bp & 31)) & MASK;

    nCurDocID += nDGap;
}



/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
