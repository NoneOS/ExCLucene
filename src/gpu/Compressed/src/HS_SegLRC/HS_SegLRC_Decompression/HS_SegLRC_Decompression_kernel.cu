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



#define MAX_BLOCK_NUM 65535 
#define THREAD_NUM 256 
#define THREAD_NUM_BIT 8
const uint64_t baseSize = MAX_BLOCK_NUM * THREAD_NUM * sizeof(uint32_t);
uint64_t Threshold = 1024 * 1024;


// filled by CPUPreprocess
struct batchInfo {
	uint32_t blockNum;		 	 // block number needed by kernel
	uint32_t constantUsedInByte; // capacity used in constant memory
	uint32_t ucntShortest_sum;	 // length sum of shortest lists, which is the actual docID number in d_isCommon and d_scan_odata
	uint32_t nTotalQueryNum;	 // query number in this batch, including those who have only keyword
	uint32_t nLinearInfoInByte;	 // bytes d_fLinearInfo needs to transfer
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



__global__ void HS_SegLRC_Decompression(uint32_t *d_lists, uint16_t *d_queryID_perBlock, uint32_t *d_shortest_lists, uint32_t *d_segoffset, regression_info_t *d_regression_info, int *d_base, uint32_t *d_hash_info)
{
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
	shortest_list = d_shortest_lists + s_array[0];

	//calculate activeThreadNum
	if (tid == 0)
	{
		activeThreadNum = (ucnts[0] - startIdx >= THREAD_NUM ? THREAD_NUM : ucnts[0] - startIdx);
	}
	__syncthreads();
	

	uint32_t eleIndex;
	uint32_t shortest_segstart;
	uint32_t descriptor;				
	uint32_t lb;				
	uint32_t eb;
	uint32_t hb;
	uint32_t en;
	uint32_t idx;
	uint32_t shift;
	uint32_t mask;	
	
	
	// locating the segment for hash
	uint32_t nhash_num;		
	int middle, left, right;
	uint32_t nhash_segeleid;
	float fSlope;		
	float fIntercept;
	int nBase;		
	
	// decompressing, then searching
	uint64_t codeword = 0;
	if (tid < activeThreadNum) {
		// binary search locating hash segment
		eleIndex = startIdx + tid;//element id within list
		nhash_num = d_hash_info[hashoffset[0] + 1];
		left = 0;						
		right = nhash_num;

		while (left < right)
		{
			middle = (left + right) >> 1;

			if ((d_hash_info + hashoffset[0] + 2)[middle] < eleIndex)
				left = middle + 1;
			else
				right = middle;
		}
		if (eleIndex < (d_hash_info + hashoffset[0] + 2)[left]) {
			--left;
		}
		

		// start decompressing
		nhash_segeleid = eleIndex - (d_hash_info + hashoffset[0] + 2)[left];
		shortest_segstart = uwlists[0] + (d_segoffset + offset[0])[left];

		descriptor = d_lists[shortest_segstart];
		++shortest_segstart;
		lb = ((descriptor >> 26) % 32);	
		eb = ((descriptor >> 21) & 0x1f);
		hb = ((descriptor >> 16) & 0x1f);
		en = (descriptor & 0xffff);

		idx = shortest_segstart + ((nhash_segeleid * lb) >> 5);
		shift = (nhash_segeleid * lb) & 0x1f;
		mask = ((1U << lb) - 1);

		regression_info = d_regression_info + offset[0] + left;
		fSlope = regression_info->fSlope;			//slope of shortest list
		fIntercept = regression_info->fIntercept;	//intercept of shortest list
		nBase = (d_base + offset[0])[left];			//base of shortest list

		codeword = d_lists[idx] | (static_cast<uint64_t>(d_lists[idx + 1]) << 32);
		shortest_list[eleIndex] = (codeword >> shift) & mask;
	}
	__syncthreads();


	uint32_t ep;
	uint32_t nhashleft, nhashright;
	uint32_t nhashseglen;
	if (tid < activeThreadNum) {
		if (nhash_segeleid < en) {
			nhashleft = (d_hash_info + hashoffset[0] + 2)[left];
			nhashright = (d_hash_info + hashoffset[0] + 3)[left];
			nhashseglen = nhashright - nhashleft;


			shortest_segstart += ((nhashseglen * lb) >> 5) + (((nhashseglen * lb) & 31) > 0 ? 1 : 0);
			idx = shortest_segstart + ((nhash_segeleid * eb) >> 5);
			shift = (nhash_segeleid * eb) & 0x1f;
			mask = ((1U << eb) - 1);

			codeword = d_lists[idx] | (static_cast<uint64_t>(d_lists[idx + 1]) << 32);
			ep = (codeword >> shift) & mask;


			shortest_segstart += ((en * eb) >> 5) + (((en * eb) & 31) > 0 ? 1 : 0);
			idx = shortest_segstart + ((nhash_segeleid * hb) >> 5);
			shift = (nhash_segeleid * hb) & 0x1f; 
			mask = ((1U << hb) - 1);

			codeword = d_lists[idx] | (static_cast<uint64_t>(d_lists[idx + 1]) << 32);
			shortest_list[nhashleft + ep] |= ((codeword >> shift) & mask) << lb;
		}
	}
	__syncthreads();


	if (tid < activeThreadNum) {
		shortest_list[eleIndex] += (int)(nhash_segeleid * fSlope + fIntercept) + nBase;
	}
	__syncthreads();
}


#endif // #ifndef _TEMPLATE_KERNEL_H_


