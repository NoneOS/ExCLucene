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
const uint64_t baseSize = MAX_BLOCK_NUM * THREAD_NUM * sizeof(uint32_t);
uint64_t Threshold = 0;


// filled by CPUPreprocess
struct batchInfo {
	uint32_t blockNum;		 	 // block number needed by kernel
	uint32_t constantUsedInByte; // capacity used in constant memory
	uint32_t ucntShortest_sum;	 // length sum of shortest lists, which is the actual docID number in d_isCommon and d_scan_odata
	uint32_t nTotalQueryNum;	 // query number in this batch, including those who have only keyword
};

const uint32_t batchInfoElementNum = sizeof(struct batchInfo) / sizeof(uint32_t);


struct checkSum {
	uint64_t checkSum1;	// number of queries with empty result
	uint64_t checkSum2;    // total number of results
	uint64_t checkSum3;    // xor of the first, middle and last result of each query
};


#define	MAX_PATH_LEN	1024


// shortest lists of each query in current batch    
uint32_t *d_shortest_lists;	

// offset of segOffset
uint32_t *h_segNum;

// offset of every compressed segment
uint32_t *h_segOffset;
uint32_t *d_segOffset;	
uint32_t segOffset_size;

// head of every segment
uint32_t *h_segHead;
uint32_t *d_segHead;	
uint32_t segHead_size;	

// median of every segment
uint32_t *h_segMedian;
uint32_t *d_segMedian;	
uint32_t segMedian_size;


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


__global__ void ParaPFD_Intersection(uint32_t *d_lists, uint32_t *d_isCommon, uint16_t *d_queryID_perBlock, uint32_t *d_shortest_lists, uint32_t *d_segOffset, uint32_t *d_segHead, uint32_t *d_segMedian) {
	// shared
	__shared__ uint32_t s_array[256];
	uint32_t activeThreadNum;			//thread number active in the block
	__shared__ uint32_t startIdx;		//start index of the shortest list in the query for this block
	__shared__ uint32_t s_tcount;		//tcount in shared
	__shared__ uint32_t constantOffset;	//segment offset from the head of constant memory
	// shared ends

	// registers
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

	tcount = s_tcount;

	// retrieve baseOffset, ucnts and uwlists
	// s_array[0] is baseOffset
	if (tcount * 3 + 1 <= THREAD_NUM) {
		if (tid < tcount * 3 + 1) {
			s_array[tid] = (d_constant + constantOffset)[tid];
		}
	}
	else {
		s_array[tid] = (d_constant + constantOffset)[tid];

		if (tid < tcount * 3 + 1 - THREAD_NUM) {
			s_array[tid + THREAD_NUM] = (d_constant + constantOffset)[tid + THREAD_NUM];
		}
	}
	__syncthreads();
	

	// set my own ucnts and wulists
	ucnts = s_array + 1;
	uwlists = ucnts + tcount;
	segNum = uwlists + tcount;
	isCommon = d_isCommon + s_array[0];
	shortest_list = d_shortest_lists + s_array[0];

	// calculate activeThreadNum
	activeThreadNum = ucnts[0] - startIdx >= THREAD_NUM ? THREAD_NUM : ucnts[0] - startIdx;

	int eleIndex = startIdx + tid;
	uint32_t shortest_segstart = uwlists[0] + (d_segOffset + segNum[0])[bid - startBlockId];
	
	uint32_t descriptor = d_lists[shortest_segstart];
	++shortest_segstart;			
	uint32_t lb = (descriptor >> 26) % 32;		
	uint32_t idx = shortest_segstart + ((tid * lb) >> 5);
	uint32_t shift = (tid * lb) & 0x1f;
	uint32_t mask = (1U << lb) - 1;

	__shared__ uint32_t shortest_segment[THREAD_NUM << 1];
	if (tid < activeThreadNum) {
		uint64_t codeword = d_lists[idx] | (static_cast<uint64_t>(d_lists[idx + 1]) << 32);
		shortest_segment[tid] = (codeword >> shift) & mask;
	}
	__syncthreads();



	// scan the gap lists just decompressed
	int pout = 0, pin = 1;		// varialbes controling swapping
	int off;					// varialbes controling the offset of scan

	if (tid == 0) {
		shortest_segment[0] = (d_segHead + segNum[0])[bid - startBlockId];//head of current segment
	}
	__syncthreads();

	for (off = 1; off < THREAD_NUM; off <<= 1) {
		pout = 1 - pout;
		pin = 1 - pout;
		
		__syncthreads();		// important

		shortest_segment[pout * THREAD_NUM + tid] = shortest_segment[pin * THREAD_NUM + tid];

		if (off <= tid) {
			shortest_segment[pout * THREAD_NUM + tid] += shortest_segment[pin * THREAD_NUM + tid - off];
		}
	}
	__syncthreads();


	//searching my DocID
	int listIdx;				//varibale control which list is being searched
	int found;							
	int middle, left, right;
	uint32_t p, q;
	int segment_num;			//number of segments in current longer list
	int segment_size;			//size of segment found in current longer list
	int segment_start;			//start of corresponding segment in current longer list
	uint32_t cur_DocID;
	uint32_t last_DocID;

	if (tid < activeThreadNum) {
		isCommon[eleIndex] = 1;	
		p = shortest_segment[pout * THREAD_NUM + tid]; 
		shortest_list[eleIndex] = p;


		// searching for the first longer list to the last longer list if necessary
		for (listIdx = 0; listIdx < tcount - 1; ++listIdx) {
			found = 0;

			// calculate number of segments int current longer list
			segment_size = ucnts[listIdx + 1] & (THREAD_NUM - 1);
			segment_num = ucnts[listIdx + 1] / THREAD_NUM;

			if (segment_size)
				++segment_num;
			else	
				segment_size = THREAD_NUM;


			// binary search locating segment in current longer list
			left = 0;
			right = segment_num - 1;
			while (left < right) {
				middle = (left + right) >> 1;
				q = (d_segHead + segNum[listIdx + 1])[middle];

				if (p > q)
					left = middle + 1;
				else
					right = middle;
			}

			q = (d_segHead + segNum[listIdx + 1])[left];
			if (p < q && left > 0) {
				--left;
			}
			

			// calculate start and size of the segment just found
			segment_start = uwlists[listIdx + 1] + (d_segOffset + segNum[listIdx + 1])[left];
			if (left < segment_num - 1) {
				segment_size = THREAD_NUM;
			}
			
			// calculate the lb,eb,hb and en descriptors of the segment just found
			descriptor = *(d_lists + segment_start);
			++segment_start;	
			lb = (descriptor >> 26) % 32;
			mask = (1U << lb) - 1;
			cur_DocID = (d_segMedian + segNum[listIdx + 1])[left];

			if (p < cur_DocID) {
				cur_DocID = (d_segHead + segNum[listIdx + 1])[left];
				left = lb;

				segment_size >>= 1;
			}
			else {
				left = (segment_size >> 1) * lb + lb;
				segment_size -= (segment_size >> 1);
			}
						

			for (middle = 0; middle < segment_size; middle++, left += lb) {
				if (cur_DocID < p) {
					last_DocID = cur_DocID;

					idx = segment_start + (left >> 5);
					shift = left & 0x1f;

					uint64_t codeword = d_lists[idx] | (static_cast<uint64_t>(d_lists[idx + 1]) << 32);
					cur_DocID = (codeword >> shift) & mask;

					cur_DocID += last_DocID;
				}
				else {			
					if (cur_DocID == p) {
						found = 1;
					}

					break;
				}
			}

			if (!found) {
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
	uint32_t activeThreadNum;	//thread number needed in the block
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
	activeThreadNum = (ucnt - startIdx >= THREAD_NUM ? THREAD_NUM : ucnt - startIdx);

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


