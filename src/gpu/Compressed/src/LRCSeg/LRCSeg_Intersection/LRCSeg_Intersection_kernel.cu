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
	uint32_t segNum[MAX_QUERY_LEN];
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


#define	MAX_PATH_LEN 1024


typedef struct _regression_info {
	float fSlope;
	float fIntercept;
	uint32_t nRangeLeft;
	uint32_t nRangeRight;
} regression_info_t;


// shortest lists of each query in current batch    
uint32_t *d_shortest_lists;	

// offset of segOffset
uint32_t *h_offset;

// offset of every compressed segment
uint32_t *h_segOffset;
uint32_t *d_segOffset;	
uint32_t segOffset_size;

// head of every segment
uint32_t *h_segHead;
uint32_t *d_segHead;	
uint32_t segHead_size;	

// minimum of every segment
int *h_base;
int *d_base;		
uint32_t base_size;

// regression info
float *h_fLinearInfo;						
float *d_fLinearInfo;			

// regression info of every list
regression_info_t *h_regression_info;
uint32_t regression_info_size;	


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


__global__ void LRCSeg_Intersection(uint32_t *d_lists, uint32_t *d_isCommon, uint16_t *d_queryID_perBlock, uint32_t *d_shortest_lists, uint32_t *d_segOffset, uint32_t *d_segHead, int *d_base, float *d_fLinearInfo) {
	// shared
	__shared__ uint32_t s_array[192];
	__shared__ float s_linearInfo[192];
	__shared__ uint32_t activeThreadNum; // number of active threads in the block
	__shared__ uint32_t startIdx;	
	__shared__ uint32_t s_tcount;	
	__shared__ uint32_t constantOffset;	
	// shared ends

	// registers
	uint32_t tid = threadIdx.x;
	uint32_t bid = blockIdx.x;
	uint32_t queryNum;
	__shared__ uint32_t startBlockId;
	uint32_t queryID;				
	uint32_t queriesOffset;
	uint32_t tcount;			
	uint32_t *isCommon;
	uint32_t *shortest_list;
	uint32_t *ucnts, *uwlists;
	uint32_t *segNum;
	// registers end


	if (tid == 0) {
		// thread0, fetch queryNum from the first segment of constant memory
		queryNum = ((batchInfo*)d_constant)->nTotalQueryNum;

		// search in startBlockId segment, confirm which query I will deal with
		constantOffset = batchInfoElementNum;

		// fetch the result of reduction; and set the queryID and startBlockId
		queryID = d_queryID_perBlock[bid]; //queryNum - s_array[0] - 1;
		startBlockId = (d_constant + constantOffset)[queryID];

		// calculate startIdx
		startIdx = (bid - startBlockId) * THREAD_NUM;

		// get the query offset in queries segment
		constantOffset += queryNum;
		queriesOffset = (d_constant + constantOffset)[queryID];
		constantOffset += queryNum + queriesOffset;

		// get tcount
		s_tcount = (d_constant + constantOffset)[0];
		constantOffset += 1;

	}
	__syncthreads();


	// retrieve baseOffset, ucnts and uwlists
	// s_array[0] is baseOffset
	tcount = s_tcount;
	if (tid < tcount * 3 + 2) {
		s_array[tid] = (d_constant + constantOffset)[tid];
	}
	__syncthreads();
	
	// set my own ucnts and wulists
	ucnts = s_array + 2;
	uwlists = ucnts + tcount;
	segNum = uwlists + tcount;
	isCommon = d_isCommon + s_array[0];
	shortest_list = d_shortest_lists + s_array[0];

	// calculate activeThreadNum
	if (tid == 0) {
		activeThreadNum = (ucnts[0] - startIdx >= THREAD_NUM ? THREAD_NUM : ucnts[0] - startIdx);
	}
	__syncthreads();

	// get linearInfo
	if (tid < 4 * tcount) {
		s_linearInfo[tid] = d_fLinearInfo[s_array[1] + tid];
	}
	__syncthreads();

	
	// decompress the shortest list
	uint32_t eleIndex = startIdx + tid;
	uint32_t shortest_segstart = uwlists[0] + (d_segOffset + segNum[0])[bid - startBlockId];
	uint32_t descriptor = d_lists[shortest_segstart];
	++shortest_segstart;			
	uint32_t lb = (descriptor >> 26) % 32;		
	uint32_t idx = shortest_segstart + ((tid * lb) >> 5);
	uint32_t shift = (tid * lb) & 0x1f;
	uint32_t mask = (1U << lb) - 1;
	
	float fSlope = s_linearInfo[0];		
	float fIntercept = s_linearInfo[tcount];
	int base = (d_base + segNum[0])[bid - startBlockId];
	uint32_t DocID = 0;

	if (tid < activeThreadNum) {
		// misaglined address error
//		DocID = reinterpret_cast<uint64_t *>(d_lists + idx)[0] >> shift; 

		// right but slow
//		DocID = d_lists[idx] >> shift;
//		if (lb > 32 - shift) 
//			DocID |= d_lists[idx + 1] << (32 - shift);

		uint64_t codeword = d_lists[idx] | (static_cast<uint64_t>(d_lists[idx + 1]) << 32);
		DocID = codeword >> shift;


		DocID &= mask;
		DocID += (int)(eleIndex * fSlope + fIntercept) + base;

		shortest_list[eleIndex] = DocID;
	}
	__syncthreads();


	// calculate safe search range
	__shared__ uint32_t s_head, s_tail;
	if (tid == 0) {
		s_head = shortest_list[eleIndex];
		s_tail = shortest_list[eleIndex + activeThreadNum - 1];
	}
	__syncthreads();
	
	__shared__ uint32_t s_Border[80];
	if (tid < (tcount << 1)) {
		uint32_t i = tid >> 1;
		fSlope = s_linearInfo[i];
		fIntercept = s_linearInfo[tcount + i];
		
		if (tid & 1) { // right border
//			uint32_t pivot = (s_tail - fIntercept) / fSlope; // slow

			uint32_t pivot = __fdividef(s_tail - fIntercept, fSlope);
			uint32_t nRangeRight = s_linearInfo[tcount * 3 + i];
			uint32_t rightBorder = pivot + nRangeRight;

			if (rightBorder < ucnts[i])
				s_Border[tid] = rightBorder;
			else
				s_Border[tid] = ucnts[i] - 1;
		}
		else {  // left border
//			uint32_t pivot = (s_head - fIntercept) / fSlope;   // slow

			uint32_t pivot = __fdividef(s_head - fIntercept, fSlope);
			uint32_t nRangeLeft = s_linearInfo[(tcount << 1) + i];
			int leftBorder = int(pivot - nRangeLeft);

			if (leftBorder < 0)
				s_Border[tid] = 0;
			else
				s_Border[tid] = leftBorder;
		}

		s_Border[tid] >>= THREAD_NUM_BIT;
	}
	__syncthreads();


	bool found = false;	
	int middle, left, right;	// for binary search
	uint32_t q;				    // DocID in longer lists
	if (tid < activeThreadNum) {
		isCommon[eleIndex] = 1;	// set to 1 at first

		for (uint32_t listIdx = 1; listIdx < tcount; ++listIdx) {
			found = false;

			// locate segment 
			left = s_Border[listIdx << 1];
			right = s_Border[(listIdx << 1) + 1];

			/*
			   while (left <= right) {
			   middle = (left + right) >> 1;
			   q = (d_segHead + segNum[listIdx])[middle];

			   if (q == DocID) {
			   right = middle;
			   break;
			   }

			   if (q < DocID)
			   left = middle + 1;
			   else
			   right = middle - 1;
			   }
			   left = right; // FIXME: there might be a bug here
			 */


			while (left < right) {
				middle = (left + right) >> 1;
				q = (d_segHead + segNum[listIdx])[middle];

				if (DocID > q)
					left = middle + 1;
				else
					right = middle;
			}

			if (left != right) {   // important
				isCommon[eleIndex] = 0;
				break;
			}


			q = (d_segHead + segNum[listIdx])[left];
			if (DocID < q && left > 0) {
				--left;
			}


			// calculate number of segments int current longer list
			uint32_t start_eleindex = (left << THREAD_NUM_BIT);
			uint32_t segment_size = ucnts[listIdx] & (THREAD_NUM - 1);
			uint32_t segment_num = ucnts[listIdx] >> THREAD_NUM_BIT;
			if (segment_size == 0 || left < segment_num)
				segment_size = THREAD_NUM;
			

			uint32_t segment_start = uwlists[listIdx] + (d_segOffset + segNum[listIdx])[left];
			descriptor = *(d_lists + segment_start);
			++segment_start;		
			lb = (descriptor >> 26) % 32;
			mask = (1U << lb) - 1;


			// binary search among the segment
			fSlope = s_linearInfo[listIdx];
			fIntercept = s_linearInfo[tcount + listIdx];
			base = (d_base + segNum[listIdx])[left];

			left = 0;
			right = segment_size - 1;
			while (left <= right) {
				middle = (left + right) >> 1;

				idx = segment_start + ((middle * lb) >> 5);
				shift = (middle * lb) & 0x1f;

				// misaglined address error
//				q = reinterpret_cast<uint64_t *>(d_lists + idx)[0] >> shift;  

				// right but slow
//				q = d_lists[idx] >> shift;
//				if (lb > 32 - shift) 
//					q |= d_lists[idx + 1] << (32 - shift);

				uint64_t codeword = d_lists[idx] | (static_cast<uint64_t>(d_lists[idx + 1]) << 32);
				q = codeword >> shift;


				q &= mask;
				q += (int)((start_eleindex + middle) * fSlope + fIntercept) + base;


				if (DocID == q) {
					found = true;
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


__global__ void saCompact(uint32_t *d_shortest_lists, uint32_t *d_isCommon, uint32_t *d_scan_odata, uint32_t *d_result, uint16_t *d_queryID_perBlock) {
	__shared__ uint32_t ucnt;
	__shared__ uint32_t activeThreadNum;	
	__shared__ uint32_t startIdx;	
	__shared__ uint32_t constantOffset;	
	__shared__ uint32_t baseOffset;

	uint32_t tid = threadIdx.x;
	uint32_t bid = blockIdx.x;
	uint32_t queryNum;
	uint32_t startBlockId;	//the first block ID who deals with the query
	uint32_t queryID;		//my queryID
	uint32_t queriesOffset;
	uint32_t *isCommon, *scan_odata;


	if (tid == 0) {
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
		constantOffset += 3;
		
		//retieve detailed lists info
		ucnt = ((d_constant + constantOffset))[0];
	}
	__syncthreads();


	//calculate activeThreadNum
	if (tid == 0) {
		activeThreadNum = (ucnt - startIdx >= THREAD_NUM ? THREAD_NUM : ucnt - startIdx);
	}
	__syncthreads();

	//compact
	uint32_t eleIndex = startIdx + tid;
	isCommon = d_isCommon + baseOffset;
	scan_odata = d_scan_odata + baseOffset;
	uint32_t *shortest_list = d_shortest_lists + baseOffset;


	if (tid < activeThreadNum) {
		if (isCommon[eleIndex]) {
			d_result[scan_odata[eleIndex] - 1] = shortest_list[eleIndex];
		}
	}
	__syncthreads();
}


__global__ void ucntResult(uint32_t *d_scan_odata, uint32_t *d_ucntResult) {
	__shared__ uint32_t queryNum;

	uint32_t queryID = blockDim.x * blockIdx.x + threadIdx.x;	
	uint32_t baseOffset;
	uint32_t *segPointer = d_constant;	

	if (threadIdx.x == 0) {
		queryNum = ((batchInfo*)d_constant)->nTotalQueryNum;
	}
	__syncthreads();

	if (queryID < queryNum - 1) {
		segPointer += batchInfoElementNum + queryNum;
		segPointer += queryNum + segPointer[queryID + 1];
		baseOffset = segPointer[1];
		d_ucntResult[queryID] = d_scan_odata[baseOffset - 1];
	}
	else if (queryNum - 1 == queryID) {
		baseOffset = ((batchInfo*)d_constant)->ucntShortest_sum;
		d_ucntResult[queryID] = d_scan_odata[baseOffset - 1];
	}
	__syncthreads();
}

#endif // #ifndef _TEMPLATE_KERNEL_H_

