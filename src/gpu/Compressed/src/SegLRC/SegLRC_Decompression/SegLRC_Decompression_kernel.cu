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
	int m_tcount;		
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
	uint32_t segNum[MAX_QUERY_LEN];
}; 


#define MAX_BLOCK_NUM 65535 
#define THREAD_NUM 256 
#define THREAD_NUM_BIT 8
const uint64_t baseSize = MAX_BLOCK_NUM * THREAD_NUM * sizeof(uint32_t);
uint64_t Threshold = 1024 * 1024;


// filled by CPUPreprocess
struct batchInfo {
	uint32_t blockNum;			// block number needed by kernel
	uint32_t constantUsedInByte;// capacity used in constant memory
	uint32_t ucntShortest_sum;	// length sum of shortest lists, which is the actual docID number in d_isCommon and d_scan_odata
	uint32_t nTotalQueryNum;	// query number in this batch, including those who have only keyword
	uint32_t nLinearInfoInByte; // bytes d_fLinearInfo needs to transfer
};

const uint32_t batchInfoElementNum = sizeof(struct batchInfo) / sizeof(uint32_t);	


#define	MAX_PATH_LEN	1024


typedef struct _regression_info {
	float fSlope;
	float fIntercept;
	uint32_t nRangeLeft;
	uint32_t nRangeRight;
} regression_info_t;


// shortest lists of each query in current batch    
uint32_t *d_shortest_lists;	

// offset of segOffset
uint32_t *h_segNum;

// offset of every compressed segment
uint32_t *h_segOffset;
uint32_t *d_segOffset;	
uint32_t segOffset_size;

// base of every segment
int *h_base;
int *d_base;
uint32_t base_size;	

regression_info_t *h_regression_info;
regression_info_t *d_regression_info;
uint32_t regression_info_size;	// size of at_regression_info


// compressed ind2
uint32_t *h_lists;	
uint32_t *d_lists;	

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


__global__ void SegLRC_Decompression(uint32_t *d_lists, uint16_t *d_queryID_perBlock, uint32_t *d_shortest_lists, uint32_t *d_segOffset, regression_info_t *d_regression_info, int *d_base)
{
	//shared
	__shared__ uint32_t s_array[192];
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
	uint32_t *shortest_list;
	uint32_t *ucnts, *uwlists;
	uint32_t *segNum;
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
	if (tid < tcount * 3 + 2) 
	{
		s_array[tid] = (d_constant + constantOffset)[tid];
	}
	__syncthreads();
	
	//set my own ucnts and wulists
	ucnts = s_array + 2;
	uwlists = ucnts + tcount;
	segNum = uwlists + tcount;
	shortest_list = d_shortest_lists + s_array[0];

	//calculate activeThreadNum
	if (tid == 0)
	{
		activeThreadNum = (ucnts[0] - startIdx >= THREAD_NUM ? THREAD_NUM : ucnts[0] - startIdx);
	}
	__syncthreads();

	
	
	
	uint32_t shortest_segstart = uwlists[0] + (d_segOffset + segNum[0])[bid - startBlockId];
	
	uint32_t descriptor = d_lists[shortest_segstart];
	++shortest_segstart;
	uint32_t lb = ((descriptor >> 26) % 32);
	uint32_t eb = ((descriptor >> 21) & 0x1f);
	uint32_t hb = ((descriptor >> 16) & 0x1f);
	uint32_t en = ((descriptor & 0xffff));
	uint32_t idx = shortest_segstart + ((tid * lb) >> 5);
	uint32_t shift = (tid * lb) & 0x1f;
	uint32_t mask = ((1U << lb) - 1);					
	

	regression_info = d_regression_info + segNum[0] + bid - startBlockId;
	float fSlope = regression_info->fSlope;				
	float fIntercept = regression_info->fIntercept;	
	int nBase = (d_base + segNum[0])[bid - startBlockId];

	__shared__ uint32_t shortest_segment[THREAD_NUM];
	uint64_t codeword = 0;
	if (tid < activeThreadNum) {
		codeword = d_lists[idx] | (static_cast<uint64_t>(d_lists[idx + 1]) << 32);
		shortest_segment[tid] = (codeword >> shift) & mask;
	}
	__syncthreads();
	
	
	uint32_t ep = 0;
	if (tid < en) {
		shortest_segstart += ((activeThreadNum * lb) >> 5) + (((activeThreadNum * lb) & 31) > 0 ? 1 : 0);
		idx = shortest_segstart + ((tid * eb) >> 5);
		shift = (tid * eb) & 0x1f;
		mask = ((1U << eb) - 1);

		codeword = d_lists[idx] | (static_cast<uint64_t>(d_lists[idx + 1]) << 32);
		ep = (codeword >> shift) & mask;


		shortest_segstart += ((en * eb) >> 5) + (((en * eb) & 31) > 0 ? 1 : 0);
		idx = shortest_segstart + ((tid * hb) >> 5);
		shift = (tid * hb) & 0x1f;
		mask = ((1U << hb) - 1);

		codeword = d_lists[idx] | (static_cast<uint64_t>(d_lists[idx + 1]) << 32);
		shortest_segment[ep] |= ((codeword >> shift) & mask) << lb;
	}
	__syncthreads();


	uint32_t eleIndex = startIdx + tid;
	if (tid < activeThreadNum) {
		shortest_segment[tid] += (int)(tid * fSlope + fIntercept) + nBase;
		shortest_list[eleIndex] = shortest_segment[tid];
	}
	__syncthreads();
}

#endif // #ifndef _TEMPLATE_KERNEL_H_


