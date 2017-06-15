#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdint.h>
#include <stdio.h>
#include <cudpp.h>

#define SDATA( index)      cutilBankChecker(sdata, index)

// ind1
typedef struct _at_term_ind1 {
	uint32_t m_urlcount;  // length 
	uint64_t m_off;       // offset (in bytes)
} at_term_ind1_t;


typedef struct _at_search_ind {
	FILE* fd_ind1;				//一级索引文件句柄
	FILE* fd_ind2;				//二级索引文件句柄
	uint64_t sz_ind1;			//一级索引文件大小
	uint64_t sz_ind2;			//二级索引文件大小
	at_term_ind1_t *m_pind1;	//一级索引指针
	unsigned char *m_pind2;		//二级索引指针
	int m_tcount;				//AT索引中的term数量
} at_search_ind_t;


#define	MAX_QUERY_LEN	80							//public

// query
typedef struct _query_input {
	uint32_t tnum;
	uint32_t tno[MAX_QUERY_LEN];
} query_input_t;


// note: we assume that ind2 is less than 16GB
//       so that offset can fit in 32-bit
struct testCell {
	uint32_t tcount;                // number of terms
	uint32_t ucnt[MAX_QUERY_LEN];   // lengths
	uint32_t uwlist[MAX_QUERY_LEN]; // offsets (in 4bytes) 
	uint32_t tno[MAX_QUERY_LEN];    // term ids
}; 


// cudpp scan
CUDPPConfiguration config;
CUDPPHandle scanplan = 0;;
CUDPPHandle theCudpp = 0;




//filled by CPUPreprocess
struct batchInfo
{
	//index info in constant memory
	//unsigned int startOffset_queriesOffset;
	//unsigned int startOffset_queries;
	//index info in constant memory ends

	unsigned int blockNum;	//block number needed by kernel
	unsigned int queryNum;	//query number contained by the batch
	unsigned int constantUsedInByte;	//capacity used in constant memory
	unsigned int ucntShortest_sum;	//length sum of shortest lists, which is the actual docID number in d_isCommon and d_scan_odata
	unsigned int nTotalQueryNum;  //including those queries have only one keyword
	unsigned int nLinearInfoInByte;  //bytes d_fLinearInfo needs to transfer
};

struct checkSum
{
	unsigned int check_sum1;	//number of queries with empty result
	unsigned int check_sum2;	//number of docIDs in valid result
	unsigned int check_sum3;
};

//global constants
#define QUERY_NUM 16184  //一批多少个查询
#define	TOTAL_QUERY 33337  //比33337多一些
#define THREAD_NUM 256 
#define ROUTE_BINARY 0
#define ROUTE_BLOOM 1
#define GOV_RESNUM 183430377 
#define BAIDU_RESNUM 2743515
unsigned int DOCID_LIMIT = 30 * 2 * THREAD_NUM;
const unsigned int at_trunc_count = 300000;  //百度约束的结果上限个数
const unsigned int baseSize = sizeof(unsigned int) * at_trunc_count * 16;  //最短桶长和的上限；因为现在是静态分配资源
const unsigned int batchInfoElementNum = sizeof(struct batchInfo) / sizeof(unsigned int);	//element number in batchInfo structure
//global constants end


// Definintion

#define	MAX_PATH_LEN	1024
#define MAX_SUBQUERY_TERM 80
#define MAX_QUERY_NUM 512

typedef struct _singleKeyword
{
	unsigned int nLen, nOffset, nQueryIDInBatch;
} singleKeyword_t;

typedef struct _regression_info
{
	//regression formula
	float fSlope;
	float fIntercept;
	float fRSquare;
	float fMultiple;  //the multiple of distance and VDistance; equals to sqrt(1 + fSlope * fSlope)
	//two parallel lines
	float fInterceptHigh;
	float fInterceptLow;
	//horizontal range
	unsigned int nRangeLeft;
	unsigned int nRangeRight;
	//x-axis multiple
	unsigned int nMagnification;
} regresstion_info_t;

//global device variables
__constant__ unsigned int d_constantOld[4];  //constant space
unsigned int *d_lists;  //equivalent to d_ind2
unsigned int *d_isCommon;  //0-1 array
unsigned int *d_scan_odata;  //for scan operation
unsigned int *d_result;  //storage results, used in compact operation
unsigned int *d_ucntResult;	//number of results per query
uint16_t *d_queryID_perBlock;	//corresponding to queryID for each block 
unsigned char *d_bloom;
unsigned int *d_batchInfo;
float *d_fLinearInfo;  //slope..., intercept..., lRange..., rRange... 
//global device variables end

//global host variables
//unsigned int h_constant[163800];  //corresponding to d_constant
unsigned int *h_constant;  //corresponding to d_constant
unsigned int *h_result;  //corresponding to d_result
unsigned int *h_lists;	//corresponding to d_lists
unsigned int *h_ucntResult;	//corresponding to d_ucntResult
uint16_t *h_queryID_perBlock;	//corresponding to queryID for each block
unsigned char *h_bloom;
float *h_fLinearInfo;  //slope..., intercept..., lRange..., rRange... 
regresstion_info_t *h_pSLinearInfoDetail;  //stores at_regression_info file

unsigned int h_queries[16380];	//these three arrays compose the entire content of constant memory
unsigned int h_startBlockId[16380];
unsigned int h_queriesOffset[16380];
unsigned int h_baseOffset[16380];

//global host variables end


__global__ void zero(int* isCommon,int size)
{
	const unsigned int tid = threadIdx.x;
	const unsigned int subscript = blockDim.x * blockIdx.x + tid;
	if(subscript >= size) 	return ;

	isCommon[subscript] = 0;
}


__global__ void mqSearch(unsigned int *d_constant, unsigned int *d_lists, unsigned int *d_isCommon, uint16_t *d_queryID_perBlock, float *d_fLinearInfo)
{
	//shared
	__shared__ unsigned int s_array[128];
	__shared__ float s_linearInfo[160];
	__shared__ unsigned int activeThreadNum;	//thread number needed in the block
	__shared__ unsigned int startIdx;	//start index of the shortest list in the query for this block
	__shared__ unsigned int s_tcount;	//tcount in shared
	__shared__ unsigned int constantOffset;	//segment offset from the head of constant memory
	//shared ends

	//registers
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int i, j;	//loop control variables
	__shared__ unsigned int queryNum;
	unsigned int startBlockId = 0;	//the first block ID who deals with the query
	unsigned int queryID = 0;	//my queryID
	unsigned int queriesOffset = 0;
	unsigned int tcount;	//number of terms
	unsigned int *isCommon;
	unsigned int *ucnts, *uwlists;
	//registers end


	if (tid == 0)
	{
		//thread0, fetch queryNum from the first segment of constant memory
		queryNum = ((batchInfo*)d_constant)->queryNum;

		//search in startBlockId segment, confirm which query I will deal with
		constantOffset = batchInfoElementNum;

		//fetch the result of reduction; and set the queryID and startBlockId
		queryID = d_queryID_perBlock[bid]; //queryNum - s_array[0] - 1;
		startBlockId = (d_constant + constantOffset)[queryID];

		//calculate startIdx
		startIdx = (blockIdx.x - startBlockId) * THREAD_NUM;

		//get the query offset in queries segment
		constantOffset += queryNum;
		queriesOffset = (d_constant + constantOffset)[queryID];
		constantOffset += queryNum + queriesOffset;

		//get tcount
		s_tcount = (d_constant + constantOffset)[0];
		constantOffset += 1;

	}
	__syncthreads();


	tcount = s_tcount;

	//retrieve baseOffset, ucnts and uwlists
	//s_array[0] is baseOffset
	if (tid < (tcount << 1) + 3)
	{
		s_array[tid] = (d_constant + constantOffset)[tid];
	}
	__syncthreads();
	
	//set my own ucnts and wulists
	ucnts = s_array + 3;
	uwlists = ucnts + tcount;
	isCommon = d_isCommon + s_array[0];

	//calculate activeThreadNum
	//set shared baseOffset
	if (threadIdx.x == 0)
	{
		activeThreadNum = ucnts[0] - startIdx >= blockDim.x ? blockDim.x : ucnts[0] - startIdx;
	}
	__syncthreads();

	//retrieve linearInfo
	if (tid < tcount * 4)
	{
		s_linearInfo[tid] = d_fLinearInfo[s_array[1] + tid];
	}
	__syncthreads();


	//binary search
	int eleIndex = startIdx + threadIdx.x;
	int shortestStart = uwlists[0];
	int found = 0;	//identify whether has been found
	int listIdx = 0;
	int leftNew = 0, rightNew = 0;
	int leftNewTmp = 0, rightNewTmp = 0;
	int iPivotTmp = 0, nListIdxTmp = 0;

	
	
	//get left and right borders
	__shared__ unsigned int s_nPHead , s_nPTail ;
	__shared__ unsigned int s_Border[80];  //as the max number of lists per query is 36
	if (0 == tid)
	{
		s_nPHead = (d_lists[shortestStart + eleIndex]);
		s_nPTail = (d_lists[shortestStart + eleIndex + activeThreadNum - 1]);
	}
	__syncthreads();

	if (tid < (tcount << 1))
	{
		nListIdxTmp = tid >> 1;
//		if (ucnts[nListIdxTmp] > 1)
		{
			/*if (tid % 2)  //right border*/
			if (tid & 1)  //right border
			{
				/*iPivotTmp = ((float)s_nPTail - s_linearInfo[tcount + nListIdxTmp]) / s_linearInfo[nListIdxTmp];*/
				iPivotTmp = __fdividef(((float)s_nPTail - s_linearInfo[tcount + nListIdxTmp]), s_linearInfo[nListIdxTmp]);
				/*iPivotTmp = __fdiv_rn(((float)s_nPTail - s_linearInfo[tcount + nListIdxTmp]), s_linearInfo[nListIdxTmp]);*/
				/*if (iPivotTmp < 0) iPivotTmp = 0;*/
				/*rightNewTmp = iPivotTmp + (int)s_linearInfo[tcount * 3 + nListIdxTmp];*/
				rightNewTmp = iPivotTmp + (int)s_linearInfo[__umul24(tcount, 3) + nListIdxTmp];
				if (rightNewTmp < ucnts[nListIdxTmp])
				{
					s_Border[tid] = rightNewTmp;
				}
				else
				{
					s_Border[tid] = ucnts[nListIdxTmp] - 1;
				}
			}
			else  //left border
			{
				/*iPivotTmp = ((float)s_nPHead- s_linearInfo[tcount + nListIdxTmp]) / s_linearInfo[nListIdxTmp];*/
				iPivotTmp = __fdividef(((float)s_nPHead- s_linearInfo[tcount + nListIdxTmp]), s_linearInfo[nListIdxTmp]);
				/*iPivotTmp = __fdiv_rn(((float)s_nPHead- s_linearInfo[tcount + nListIdxTmp]), s_linearInfo[nListIdxTmp]);*/
				/*if (iPivotTmp < 0) iPivotTmp = 0;*/
				/*leftNewTmp = iPivotTmp - (int)s_linearInfo[tcount * 2 + nListIdxTmp];*/
				leftNewTmp = iPivotTmp - (int)s_linearInfo[__umul24(tcount , 2) + nListIdxTmp];
				if (leftNewTmp > 0)
				{
					s_Border[tid] = leftNewTmp;
				}
				else
				{
					s_Border[tid] = 0;
				}
			}
		}
	}
	__syncthreads();
	//get ends



	if (threadIdx.x < activeThreadNum)
	{
		int middle, left, right;
		unsigned int p, q;
		unsigned int listLen = 0;
		int nPivot = 0;	//the index calculated from regression formula
		p = (d_lists[shortestStart + eleIndex]);
		isCommon[eleIndex] = 1;	//set to 1 first

		for (listIdx = 1; listIdx < tcount; ++listIdx)
		{
			listLen = ucnts[listIdx];
			left = 0; right = listLen - 1;
			found = 0;

			//for less formula calculation
			if (ucnts[listIdx] > 1)
			{
				left = s_Border[listIdx << 1];
				right = s_Border[(listIdx <<  1) + 1];
			}



			/*
			if (ucnts[listIdx] > 1)
			{
				nPivot = __fdividef((((float)p) - s_linearInfo[tcount + listIdx]) , s_linearInfo[listIdx]);
				leftNew = nPivot - (int)(s_linearInfo[__umul24(tcount , 2) + listIdx]); 
				rightNew = nPivot + s_linearInfo[__umul24(tcount , 3) + listIdx]; 
				if (leftNew > left)
				{
					left = leftNew;
				}
				if (rightNew < right)
				{
					right = rightNew;
				}
			}
			*/
			
			/*
			if (ucnts[listIdx] > 1)
			{
				nPivot = (((float)p) - s_linearInfo[tcount + listIdx]) / s_linearInfo[listIdx];

				if (nPivot <= 0 || nPivot >= listLen)
				{
					leftNew = nPivot - (int)(s_linearInfo[tcount * 2 + listIdx]); 
					rightNew = nPivot + s_linearInfo[tcount * 3 + listIdx]; 
					if (leftNew > left)
					{
						left = leftNew;
					}
					if (rightNew < right)
					{
						right = rightNew;
					}
				}
				else
				{
					if (p > (d_lists[uwlists[listIdx] + nPivot]))
					{
						rightNew = nPivot + s_linearInfo[tcount * 3 + listIdx]; 
						if (rightNew < right)
						{
							right = rightNew;
						}
						leftNew = nPivot  > 0 ? nPivot  : 0;
						left = leftNew;
					}
					else
					{
						leftNew = nPivot - (int)(s_linearInfo[tcount * 2 + listIdx]); 
						if (leftNew > left)
						{
							left = leftNew;
						}
						//rightNew = nPivot < right ? nPivot : right;
						//right = rightNew;
						rightNew = nPivot > 0 ? nPivot : right;
						right = rightNew;
					}
				}
			}
		*/
//			isCommon[eleIndex] = right;//s_linearInfo[listIdx + tcount];//(d_lists[uwlists[listIdx] + 7995]);


			while (left <= right)
			{
				middle = (left + right) >> 1;
				q = (d_lists[uwlists[listIdx] + middle]);

				if (p == q)
				{
					found = 1;
					break;
				}
				if (p > q)
				{
					left = middle + 1;
				}
				else
				{
					right = middle - 1;
				}
			};

			if (!found)
			{
				isCommon[eleIndex] = 0;
				break;
			}
		}
	}



	__syncthreads();
}


__global__ void saCompact(unsigned int *d_constant, unsigned int *d_lists, unsigned int* d_isCommon, unsigned int *d_scan_odata, unsigned int *d_result, uint16_t *d_queryID_perBlock)
{
	//shared
	__shared__ unsigned int uwlist; 
	__shared__ unsigned int ucnt;
	__shared__ unsigned int activeThreadNum;	//thread number needed in the block
	__shared__ unsigned int startIdx;	//start index of the shortest list in the query for this block
	__shared__ unsigned int constantOffset;	//offset to the head of constant memory, as pointer to constant memory is not allowed
	__shared__ unsigned int baseOffset;
	__shared__ unsigned int s_tcount;
	//shared ends

	//registers
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int i, j;	//loop control variables
	__shared__ unsigned int queryNum;
	unsigned int startBlockId = 0;	//the first block ID who deals with the query
	unsigned int queryID = 0;	//my queryID
	unsigned int queriesOffset = 0;
	unsigned int *isCommon, *scan_odata;
	//registers end


	if (threadIdx.x == 0)
	{
		//thread0, fetch queryNum from the first segment of constant memory
		queryNum = ((batchInfo*)d_constant)->queryNum;

		//search in startBlockId segment, confirm which query I will deal with
		constantOffset = batchInfoElementNum;

		//fetch the queryID from d_queryID_per_block, and set startBlockId
		queryID = d_queryID_perBlock[bid];
		startBlockId = (d_constant + constantOffset)[queryID];

		//calculate startIdx
		startIdx = (blockIdx.x - startBlockId) * THREAD_NUM;

		//get the query offset in queries segment
		constantOffset += queryNum;
		queriesOffset = (d_constant + constantOffset)[queryID];
		constantOffset += queryNum + queriesOffset;

		s_tcount = *(d_constant + constantOffset);
		baseOffset = *(d_constant + constantOffset + 1);
		constantOffset += 4;
		
		//retieve detailed lists info
		ucnt = ((d_constant + constantOffset))[0];
		uwlist = ((d_constant + constantOffset + s_tcount))[0];
	}
	__syncthreads();


	//calculate activeThreadNum
	if (threadIdx.x == 0)
	{
		activeThreadNum = ucnt - startIdx >= blockDim.x ? blockDim.x : ucnt - startIdx;
	}
	__syncthreads();


	//compact
	unsigned int eleIndex = startIdx + threadIdx.x;
	unsigned int shortestStart = uwlist;
	isCommon = d_isCommon + baseOffset;
	scan_odata = d_scan_odata + baseOffset;



	if (threadIdx.x < activeThreadNum)
	{
		if (isCommon[eleIndex])
		{
			d_result[scan_odata[eleIndex] - 1] = d_lists[shortestStart + eleIndex];
		}
	}
	__syncthreads();
}

__global__ void ucntResult(unsigned int *d_constant, unsigned int *d_scan_odata, unsigned int *d_ucntResult)
{
	//shared
	__shared__ unsigned int queryNum;
	//shared ends

	//registers
	unsigned int queryID = blockDim.x * blockIdx.x + threadIdx.x;	//each thread deals with one query
	unsigned int baseOffset;
	unsigned int *segPointer = d_constant;	//for baseOffset segment in constant memory
	//registers end

	if (threadIdx.x == 0)
	{
		queryNum = ((batchInfo*)d_constant)->queryNum;
	}
	__syncthreads();

	if (queryID < queryNum - 1)
	{
		//segPointer += batchInfoElementNum + queryNum;
		//baseOffset = segPointer[queryID + 1];
		segPointer += batchInfoElementNum + queryNum;
		segPointer += queryNum + segPointer[queryID + 1];
		baseOffset = segPointer[1];
		d_ucntResult[queryID] = d_scan_odata[baseOffset - 1];
	}
	else if (queryNum - 1 == queryID)
	{
		baseOffset = ((batchInfo*)d_constant)->ucntShortest_sum;
		d_ucntResult[queryID] = d_scan_odata[baseOffset - 1];
	}
	__syncthreads();
}


#endif // #ifndef _TEMPLATE_KERNEL_H_


