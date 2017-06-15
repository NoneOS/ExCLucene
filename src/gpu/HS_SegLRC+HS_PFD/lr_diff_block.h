/***************************************************************************
 * 
 * Copyright (c) 2010 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file lr_diff_block.h
 * @author wudi01(com@baidu.com)
 * @date 2010/09/21 14:07:02
 * @brief lr_diff_block_t
 *  
 **/




#ifndef  __LR_DIFF_BLOCK_H_
#define  __LR_DIFF_BLOCK_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <atct_search.h>
#include <atct_index.h>
#include <indsearch.interface.h>
#include <at_search.h>
#include <ct_search.h>
#include <diff_search.h>
#include <cindex.h>
#include <ccol.h>
#include <pool.h>
#include <sys/types.h>
#include <linux/unistd.h>
#include "bsearch.h"
#include "interface.h"
#include <sys/types.h>
#include <linux/unistd.h>
#include "diff_search.h"
#include "lr_diff_block.h"

#define printErr() \
{\
    char buf[256];\
    sprintf(buf, "line:%u\tfunc:%s\tfile:%s", __LINE__, __FUNCTION__, __FILE__);\
    perror(buf);\
    exit(1);\
}

const unsigned int SEGLEN = 64;
const unsigned int EXPECTED_SEGLEN = 64;

inline void unpack(const uint32_t * const input, const uint32_t lb, const uint32_t BS, uint32_t * const output)
{
    uint32_t i = 0;
    uint32_t bp = 0;
    uint32_t MASK = (1 << lb) - 1;
    uint64_t lTmp = 0;

    for (bp = 0, i = 0; i < BS; ++i, bp += lb)
    {
        //locate uint32
        /*
        lTmp = ((uint64_t*)(input + (bp >> 5)))[0];
        output[i] = (lTmp >> (bp & 31)) & MASK;
        */
        lTmp = ((uint64_t*)(((unsigned char*)input) + (bp >> 3)))[0];
        output[i] = (lTmp >> (bp & 7)) & MASK;
    }
}
inline void pack(const uint32_t * const input, const uint32_t lb, const uint32_t BS, uint32_t * const output)
{
    uint32_t i = 0;
    uint32_t bp = 0;
    uint64_t *plOutput= NULL;

    for (bp = 0, i = 0; i < BS; ++i, bp += lb)
    {
        /*
        plOutput = (uint64_t*)(output + (bp >> 5));
        *plOutput |= (input[i] << (bp & 31));
        */
        plOutput = (uint64_t*)(((unsigned char *)output) + (bp >> 3));
        *plOutput |= (input[i] << (bp & 7));
    }      
}

typedef struct _REGRESSION_INFO
{
    float fSlope;
    float fIntercept;

    //safe search range
    unsigned int nRangeLeft, nRangeRight;

    //stat info
    float fRSquare;

} regression_info_t;

typedef struct _statInfo
{
    float lbA, ebA, hbA, enA;
    float fCompressionRatio;
    uint32_t pnRange[25];
} statInfo_t;

typedef class _LR_DIFF_BLOCK
{
    //attribute
    protected:
        unsigned int nSegNum;
        unsigned int *pnSegHeads;
        unsigned int BS;

    protected:
        unsigned int *pnUncompressedList;
        unsigned int *pnCompressedList;
        regression_info_t *pnSegLRs;
        unsigned int *pnSegHeadsOffset;

    protected:
        unsigned int nCurSegSize;
        unsigned int nCurBitsNum;
        float fSlope, fIntercept;
        unsigned int *pnCurSeg;
        unsigned int EXRATIO;
        unsigned int nCompressedSizeInInt;
        unsigned int nCurSegIdx;  //current segment idx
        unsigned int nStartPointInSeg;
        unsigned int MASK;

    protected:
        unsigned int nBitPosition;
        float fLRResult;
        double dLRResult;

    public:
        term_sign_t sign;
        unsigned int nUcnt;
        unsigned int m_size; //compatible with gehao, mainly for qsort; PS: m_size is quite different from nUcnt
        unsigned int nSeekTimesInSeg;



    //method
    public:
        _LR_DIFF_BLOCK();
        ~_LR_DIFF_BLOCK()
        {
            finalize();
        }
        void finalize()
        {
            if (pnUncompressedList)
            {
                free(pnUncompressedList);
                pnUncompressedList = NULL;
            }
            if (pnCompressedList)
            {
                free(pnCompressedList);
                pnCompressedList = NULL;
            }
            if (pnSegLRs)
            {
                free(pnSegLRs);
                pnSegLRs = NULL;
            }
            if (pnSegHeads)
            {
                free(pnSegHeads);
                pnSegHeads = NULL;
            }
            if (pnSegHeadsOffset)
            {
                free(pnSegHeadsOffset);
                pnSegHeadsOffset = NULL;
            }
        }
        void LRCompressInMemory(indfmt_index_diff_wei_looper *looper);
        bool seek(unsigned int uno);
        unsigned int select(const unsigned int nIndex);
        inline unsigned int selectInSeg(const unsigned int nIndexInSeg);
        inline unsigned int _LR_DIFF_BLOCK::selectInSegSub();
        inline unsigned int _LR_DIFF_BLOCK::selectInSegAdd();

    protected:
        void getRegressionInfo(const uint32_t * const pnList, const uint32_t nListLen, regression_info_t * const pSRegressionInfo);
        void getVD(uint32_t *pnList, const uint32_t nNum, const float fSlope, const float fIntercept, int32_t *piVD);
        template <class type>
            uint32_t VB_Compression(const type * const input, type * const output, uint32_t block_size, statInfo_t *pSStatInfo);
        template <class type>
            int32_t VB_encode(const type  * const input, type *output, const uint32_t lb, const uint32_t block_size, uint32_t * const nSize, statInfo_t *pSStatInfo);
        template <class type>
            uint32_t VB_decode(const type * input, type * const output, const uint32_t lb, const uint32_t eb, const uint32_t hb, const uint32_t en, const uint32_t block_size);
        template <class type>
            uint32_t maxBitsNum(const type * const input, const uint32_t nEleNum);
} ori_lr_df_block_t;

typedef class _HS_LR_DIFF_BLOCK : public _LR_DIFF_BLOCK
{
    //attribute
    protected:
       unsigned int *pnHSSegLen; 

    protected:
       unsigned int nExpectedHashSegLen;
       unsigned int HS_SHIFT;
       unsigned int nMaxDocIDBitsNum;
       unsigned int nHashSegNum;
       unsigned int nPrevSegIdx;

    //method
    public:
        _HS_LR_DIFF_BLOCK():_LR_DIFF_BLOCK()
        {
            pnHSSegLen = NULL;
            nExpectedHashSegLen = EXPECTED_SEGLEN;
            HS_SHIFT = 0;
            nMaxDocIDBitsNum = 0;
            nHashSegNum = 0;
            nPrevSegIdx = UINT_MAX;
        }

        void finalize()
        {
            if (pnHSSegLen)
            {
                free(pnHSSegLen);
                pnHSSegLen = NULL;
            }
        }

        ~_HS_LR_DIFF_BLOCK()
        {
            finalize();
        }    

        bool seek(unsigned int uno);
        void LRCompressInMemory(indfmt_index_diff_wei_looper *looper);

    protected:
        void generateHashSegmentInfo();
        unsigned int getBitsNum(unsigned int);

} hs_lr_df_block_t;


//d-gap version
typedef class _PFD_LR_DIFF_BLOCK : public _HS_LR_DIFF_BLOCK 
{
    //attr
    unsigned int nCurDocID;

    //method
    public:
        _PFD_LR_DIFF_BLOCK():_HS_LR_DIFF_BLOCK()
        {
            nCurDocID = 0;
        }
        
        ~_PFD_LR_DIFF_BLOCK()
        {
        }

        bool seek(unsigned int uno);
        void LRCompressInMemory(indfmt_index_diff_wei_looper *looper);  //pfd compression indeed

    protected:
        void getDGap(unsigned int *pnList, const unsigned int nNum, unsigned int *pnDGap);
        void selectInSegNext();

} lr_df_block_t;


















#endif  //__LR_DIFF_BLOCK_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
