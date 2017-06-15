#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_


#include <stdint.h>


#define CUDA_SAFE_CALL(call) {                                         \
    cudaError err = call;                                              \
    if (cudaSuccess != err) {                                          \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",  \
                __FILE__, __LINE__, cudaGetErrorString( err) );        \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
}


// ind1
typedef struct _at_term_ind1 {
	uint32_t m_urlcount;  // length 
	uint64_t m_off;       // offset (in bytes)
} at_term_ind1_t;

typedef struct _at_search_ind {
	FILE *fd_ind1;	
	FILE *fd_ind2;
	uint64_t sz_ind1;
	uint64_t sz_ind2;
	at_term_ind1_t *m_pind1;
	unsigned char *m_pind2;	
	uint32_t m_tcount;		
} at_search_ind_t;


#define	MAX_QUERY_LEN	80	

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
	uint32_t offset[MAX_QUERY_LEN];
	uint32_t hashoffset[MAX_QUERY_LEN];
}; 


// cudpp scan
CUDPPConfiguration config;
CUDPPHandle scanplan = 0;;
CUDPPHandle theCudpp = 0;


#define MAX_BLOCK_NUM 65535 
#define THREAD_NUM 256 
#define THREAD_NUM_BIT 8
const uint64_t baseSize = MAX_BLOCK_NUM * THREAD_NUM * sizeof(uint32_t);
uint64_t Threshold = 0;


// filled by CPUPreprocess
struct batchInfo {
	uint32_t blockNum;		 	 // block number needed by kernel
	uint32_t constantUsedInByte; // capacity used in constant memory
	uint32_t ucntShortest_sum;	 // length sum of shortest lists, which is the actual docID number in d_isCommon and d_scan_odata
	uint32_t nTotalQueryNum;	 // query number in this batch, including those who have only keyword
	uint32_t nLinearInfoInByte;	 // bytes d_fLinearInfo needs to transfer
};

const uint32_t batchInfoElementNum = sizeof(struct batchInfo) / sizeof(uint32_t);


struct checkSum {
	uint64_t checkSum1;	// number of queries with empty result
	uint64_t checkSum2;    // total number of results
	uint64_t checkSum3;    // xor of the first, middle and last result of each query
};


#define	MAX_PATH_LEN	1024


typedef struct _regression_info {
	float fSlope;
	float fIntercept;
	uint32_t nRangeLeft;
	uint32_t nRangeRight;
} regression_info_t;


// shortest lists of each query in current batch    
uint32_t *d_shortest_lists;	

// offset of segoffset
uint32_t *h_offset;

// offset of every compressed segment
uint32_t *h_segoffset;
uint32_t *d_segoffset;	
uint32_t segoffset_size;

// base of every segment
int *h_base;
int *d_base;
uint32_t base_size;		


// regression info of every segment
regression_info_t *h_regression_info;
regression_info_t *d_regression_info;
uint32_t regression_info_size;	


uint32_t *h_hash_offset;
uint32_t hash_offset_size;

uint32_t *h_hash_info;
uint32_t *d_hash_info;
uint32_t hash_info_size;


// compressed ind2
uint32_t *h_lists;	
uint32_t *d_lists;	


// for cudppscan
uint32_t *d_isCommon;		// 0-1 array
uint32_t *d_scan_odata;		// scan result of d_isCommon 

// intersection result
uint32_t *h_result;		
uint32_t *d_result;				

// number of results per query
uint32_t *h_ucntResult;			
uint32_t *d_ucntResult;			

// queryID for each block
uint16_t *h_queryID_perBlock;		
uint16_t *d_queryID_perBlock;		


// constant space
uint32_t h_constant[16380];		
__constant__ uint32_t d_constant[16380]; 

uint32_t h_queries[16380];
uint32_t h_startBlockId[16380];
uint32_t h_queriesOffset[16380];
uint32_t h_baseOffset[16380];


__global__ void HS_SegLRC_Intersection(uint32_t *d_lists, uint32_t *d_isCommon, uint16_t *d_queryID_perBlock, uint32_t *d_shortest_lists, uint32_t *d_segoffset, regression_info_t *d_regression_info, int *d_base, uint32_t *d_hash_info) {
	//shared
	__shared__ uint32_t s_array[256];
	__shared__ uint32_t activeThreadNum;//thread number active in the block
	__shared__ uint32_t startIdx;		//start index of the shortest list in the query for this block
	__shared__ uint32_t s_tcount;		//tcount in shared
	__shared__ uint32_t constantOffset;	//segment offset from the head of constant memory
	//shared ends

	//registers
	uint32_t tid = threadIdx.x;
	uint32_t bid = blockIdx.x;
	uint32_t queryNum;
	__shared__ uint32_t startBlockId;	//the first block ID who deals with the query
	uint32_t queryID;					//my queryID
	uint32_t queriesOffset;
	uint32_t tcount;					//number of terms
	uint32_t *isCommon;
	uint32_t *shortest_list;
	uint32_t *ucnts, *uwlists;
	uint32_t *offset;
	uint32_t *hashoffset;
	regression_info_t *regression_info;
	//registers end

	

	if (tid == 0)
	{
		//thread0, fetch queryNum from the first segment of constant memory
		queryNum = ((batchInfo*)d_constant)->nTotalQueryNum;

		//search in startBlockId segment, confirm which query I will deal with
		constantOffset = batchInfoElementNum;

		//fetch the result of reduction; and set the queryID and startBlockId
		queryID = d_queryID_perBlock[bid]; //queryNum - s_array[0] - 1;
		startBlockId = (d_constant + constantOffset)[queryID];

		//calculate startIdx
		startIdx = (bid - startBlockId) * THREAD_NUM;

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
	if (tid < tcount * 4 + 1) 
	{
		s_array[tid] = (d_constant + constantOffset)[tid];
	}
	__syncthreads();
	
	
	//set my own ucnts and wulists
	ucnts = s_array + 1;
	uwlists = ucnts + tcount;
	offset = uwlists + tcount;
	hashoffset = offset + tcount;
	isCommon = d_isCommon + s_array[0];
	shortest_list = d_shortest_lists + s_array[0];

	//calculate activeThreadNum
	if (tid == 0)
	{
		activeThreadNum = (ucnts[0] - startIdx >= THREAD_NUM ? THREAD_NUM : ucnts[0] - startIdx);
	}
	__syncthreads();
	
	
	// LR Decompression start
	uint32_t eleIndex;	
	uint32_t shortest_segstart;	// start of current shortest list's segment
	uint32_t descriptor;		// descriptor of current segment
	uint32_t lb;				// lower lb bits of all the elements
	uint32_t idx;			    // index of elements to be decompressed
	uint32_t shift;
	uint32_t mask;				// bit-mask
	
	
	// locating the segment for hash
	uint32_t nhash_num;		 // hash segment number
	int middle, left, right; // variable for binary search			
	uint32_t nhash_segeleid; // element id within segment
	float fSlope;			 // slope of list
	float fIntercept;		 // intercept of list
	uint32_t nLeftRange;	 // LeftRange
	uint32_t nRightRange;	 // RightRange
	int nBase;				 // base of list
	uint32_t DocID;			 // DocID to be found
	
	// searching my DocID
	int listIdx;			 // varibale control which list is being searched
	int found;				 // found or not		
	uint32_t q;				 // DocID in longer lists
	int segment_start;		 // start of corresponding segment in current longer list
	int Pivot;				 // Pivot of LR and HASH


	// decompressing, then searching
	if (tid < activeThreadNum) {
		eleIndex = startIdx + tid;//element id within list
		nhash_num = d_hash_info[hashoffset[0] + 1];
		left = 0;						
		right = nhash_num;

		while (left < right) {
			middle = left + (right - left) / 2;
			if ((d_hash_info + hashoffset[0] + 2)[middle] <= eleIndex) {
				left = middle + 1;
			}
			else {
				right = middle;
			}
		}
		--left;

		/*
		while (left < right) {
			middle = (left + right) >> 1;

			if ((d_hash_info + hashoffset[0] + 2)[middle] < eleIndex)
				left = middle + 1;
			else
				right = middle;
		}
		
		if (eleIndex < (d_hash_info + hashoffset[0] + 2)[left]) {
			--left;
		}
		*/
		
		

		//start decompressing
		nhash_segeleid = eleIndex - (d_hash_info + hashoffset[0] + 2)[left];
		shortest_segstart = uwlists[0] + (d_segoffset + offset[0])[left];

		descriptor = d_lists[shortest_segstart];		
		++shortest_segstart;
		lb = (descriptor >> 26) % 32;		
		idx = shortest_segstart + ((nhash_segeleid * lb) >> 5);
		shift = (nhash_segeleid * lb) & 0x1f;
		mask = ((1U << lb) - 1);

		regression_info = d_regression_info + offset[0] + left;
		fSlope = regression_info->fSlope;
		fIntercept = regression_info->fIntercept;
		nBase = (d_base + offset[0])[left];

		uint64_t codeword = d_lists[idx] | (static_cast<uint64_t>(d_lists[idx + 1]) << 32);
		DocID = (codeword >> shift) & mask;
		DocID += (int)(nhash_segeleid * fSlope + fIntercept) + nBase;
		
		shortest_list[eleIndex] = DocID;
		

		
		// Start search
		isCommon[eleIndex] = 1;	// set to 1 at first

		for (listIdx = 1; listIdx < tcount; ++listIdx) {
			found = 0;

			Pivot = (DocID >> d_hash_info[hashoffset[listIdx]]);
			nhash_num = d_hash_info[hashoffset[listIdx] + 1];

			if (Pivot >= nhash_num) {
				isCommon[eleIndex] = 0;
				break;
			}


			segment_start = uwlists[listIdx] + (d_segoffset + offset[listIdx])[Pivot];
			descriptor = *(d_lists + segment_start);
			++segment_start;
			lb = (descriptor >> 26) % 32;
			mask = (1U << lb) - 1;

			regression_info = (d_regression_info + offset[listIdx] + Pivot);
			fSlope = regression_info->fSlope;
			fIntercept = regression_info->fIntercept;
			nLeftRange = regression_info->nRangeLeft;
			nRightRange = regression_info->nRangeRight;
			nBase = (d_base + offset[listIdx])[Pivot];


			// binary search within segment
			left = 0;
			right = (d_hash_info + hashoffset[listIdx] + 3)[Pivot] - 1 - (d_hash_info + hashoffset[listIdx] + 2)[Pivot];
			
			Pivot = (DocID - fIntercept) / fSlope;
//			Pivot = __fdividef(DocID - fIntercept, fSlope);
			if (Pivot - (int)nLeftRange > left)
				left = Pivot - nLeftRange;
			if (Pivot + (int)nRightRange < right)
				right = Pivot + nRightRange;

			while (left <= right) {
				middle = (left + right) >> 1;

				idx = segment_start + ((middle * lb) >> 5);
				shift = (middle * lb) & 0x1f;

				uint64_t codeword = d_lists[idx] | (static_cast<uint64_t>(d_lists[idx + 1]) << 32);
				q = (codeword >> shift) & mask;
				
				q += (int)(middle * fSlope + fIntercept) + nBase;
				

				//Compare
				if (DocID == q)
				{
					found = 1;
					break;
				}

				if (DocID > q)
					left = middle + 1;
				else
					right = middle - 1;
			}

			if (!found)	{
				isCommon[eleIndex] = 0;
				break;
			}
			
		}
	}
	__syncthreads();
}


__global__ void saCompact(uint32_t *d_shortest_lists, uint32_t* d_isCommon, uint32_t *d_scan_odata, uint32_t *d_result, uint16_t *d_queryID_perBlock)
{
	//shared
	__shared__ uint32_t ucnt;
	__shared__ uint32_t activeThreadNum;	//thread number needed in the block
	__shared__ uint32_t startIdx;	//start index of the shortest list in the query for this block
	__shared__ uint32_t constantOffset;	//offset to the head of constant memory, as pointer to constant memory is not allowed
	__shared__ uint32_t baseOffset;
	//shared ends

	//registers
	uint32_t tid = threadIdx.x;
	uint32_t bid = blockIdx.x;
	uint32_t queryNum;
	uint32_t startBlockId;	//the first block ID who deals with the query
	uint32_t queryID;		//my queryID
	uint32_t queriesOffset;
	uint32_t *isCommon, *scan_odata;
	//registers end


	if (tid == 0)
	{
		//thread0, fetch queryNum from the first segment of constant memory
		queryNum = ((batchInfo*)d_constant)->nTotalQueryNum;

		//search in startBlockId segment, confirm which query I will deal with
		constantOffset = batchInfoElementNum;

		//fetch the queryID from d_queryID_per_block, and set startBlockId
		queryID = d_queryID_perBlock[bid];
		startBlockId = (d_constant + constantOffset)[queryID];

		//calculate startIdx
		startIdx = (bid - startBlockId) * THREAD_NUM;

		//get the query offset in queries segment
		constantOffset += queryNum;
		queriesOffset = (d_constant + constantOffset)[queryID];
		constantOffset += queryNum + queriesOffset;

		baseOffset = *(d_constant + constantOffset + 1);
		constantOffset += 2;
		
		//retieve detailed lists info
		ucnt = ((d_constant + constantOffset))[0];
	}
	__syncthreads();


	//calculate activeThreadNum
	if (tid == 0)
	{
		activeThreadNum = (ucnt - startIdx >= THREAD_NUM ? THREAD_NUM : ucnt - startIdx);
	}
	__syncthreads();

	//compact
	uint32_t eleIndex = startIdx + tid;
	isCommon = d_isCommon + baseOffset;
	scan_odata = d_scan_odata + baseOffset;
	uint32_t *shortest_list = d_shortest_lists + baseOffset;


	if (tid < activeThreadNum)
	{
		if (isCommon[eleIndex])
		{
			d_result[scan_odata[eleIndex] - 1] = shortest_list[eleIndex];
		}
	}
	__syncthreads();
}


__global__ void ucntResult(uint32_t *d_scan_odata, uint32_t *d_ucntResult)
{
	//shared
	__shared__ uint32_t queryNum;
	//shared ends

	//registers
	uint32_t queryID = blockDim.x * blockIdx.x + threadIdx.x;	//each thread deals with one query
	uint32_t baseOffset;
	uint32_t *segPointer = d_constant;	//for baseOffset segment in constant memory
	//registers end

	if (threadIdx.x == 0)
	{
		queryNum = ((batchInfo*)d_constant)->nTotalQueryNum;
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


