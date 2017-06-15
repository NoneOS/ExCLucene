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



/* more optimized handcoded edition */
#define S16_DECODE(_w, _p)	\
{ \
  uint32_t _k = (*_w)>>28; \
  switch(_k) \
  { \
    case 0: \
      *_p = (*_w) & 1;     _p++; \
      *_p = (*_w>>1) & 1;  _p++; \
      *_p = (*_w>>2) & 1;  _p++; \
      *_p = (*_w>>3) & 1;  _p++; \
      *_p = (*_w>>4) & 1;  _p++; \
      *_p = (*_w>>5) & 1;  _p++; \
      *_p = (*_w>>6) & 1;  _p++; \
      *_p = (*_w>>7) & 1;  _p++; \
      *_p = (*_w>>8) & 1;  _p++; \
      *_p = (*_w>>9) & 1;  _p++; \
      *_p = (*_w>>10) & 1;  _p++; \
      *_p = (*_w>>11) & 1;  _p++; \
      *_p = (*_w>>12) & 1;  _p++; \
      *_p = (*_w>>13) & 1;  _p++; \
      *_p = (*_w>>14) & 1;  _p++; \
      *_p = (*_w>>15) & 1;  _p++; \
      *_p = (*_w>>16) & 1;  _p++; \
      *_p = (*_w>>17) & 1;  _p++; \
      *_p = (*_w>>18) & 1;  _p++; \
      *_p = (*_w>>19) & 1;  _p++; \
      *_p = (*_w>>20) & 1;  _p++; \
      *_p = (*_w>>21) & 1;  _p++; \
      *_p = (*_w>>22) & 1;  _p++; \
      *_p = (*_w>>23) & 1;  _p++; \
      *_p = (*_w>>24) & 1;  _p++; \
      *_p = (*_w>>25) & 1;  _p++; \
      *_p = (*_w>>26) & 1;  _p++; \
      *_p = (*_w>>27) & 1;  _p++; \
      break; \
    case 1: \
      *_p = (*_w) & 3;     _p++; \
      *_p = (*_w>>2) & 3;  _p++; \
      *_p = (*_w>>4) & 3;  _p++; \
      *_p = (*_w>>6) & 3;  _p++; \
      *_p = (*_w>>8) & 3;  _p++; \
      *_p = (*_w>>10) & 3;  _p++; \
      *_p = (*_w>>12) & 3;  _p++; \
      *_p = (*_w>>14) & 1;  _p++; \
      *_p = (*_w>>15) & 1;  _p++; \
      *_p = (*_w>>16) & 1;  _p++; \
      *_p = (*_w>>17) & 1;  _p++; \
      *_p = (*_w>>18) & 1;  _p++; \
      *_p = (*_w>>19) & 1;  _p++; \
      *_p = (*_w>>20) & 1;  _p++; \
      *_p = (*_w>>21) & 1;  _p++; \
      *_p = (*_w>>22) & 1;  _p++; \
      *_p = (*_w>>23) & 1;  _p++; \
      *_p = (*_w>>24) & 1;  _p++; \
      *_p = (*_w>>25) & 1;  _p++; \
      *_p = (*_w>>26) & 1;  _p++; \
      *_p = (*_w>>27) & 1;  _p++; \
      break; \
    case 2: \
      *_p = (*_w) & 1;     _p++; \
      *_p = (*_w>>1) & 1;  _p++; \
      *_p = (*_w>>2) & 1;  _p++; \
      *_p = (*_w>>3) & 1;  _p++; \
      *_p = (*_w>>4) & 1;  _p++; \
      *_p = (*_w>>5) & 1;  _p++; \
      *_p = (*_w>>6) & 1;  _p++; \
      *_p = (*_w>>7) & 3;  _p++; \
      *_p = (*_w>>9) & 3;  _p++; \
      *_p = (*_w>>11) & 3;  _p++; \
      *_p = (*_w>>13) & 3;  _p++; \
      *_p = (*_w>>15) & 3;  _p++; \
      *_p = (*_w>>17) & 3;  _p++; \
      *_p = (*_w>>19) & 3;  _p++; \
      *_p = (*_w>>21) & 1;  _p++; \
      *_p = (*_w>>22) & 1;  _p++; \
      *_p = (*_w>>23) & 1;  _p++; \
      *_p = (*_w>>24) & 1;  _p++; \
      *_p = (*_w>>25) & 1;  _p++; \
      *_p = (*_w>>26) & 1;  _p++; \
      *_p = (*_w>>27) & 1;  _p++; \
      break; \
    case 3: \
      *_p = (*_w) & 1;     _p++; \
      *_p = (*_w>>1) & 1;  _p++; \
      *_p = (*_w>>2) & 1;  _p++; \
      *_p = (*_w>>3) & 1;  _p++; \
      *_p = (*_w>>4) & 1;  _p++; \
      *_p = (*_w>>5) & 1;  _p++; \
      *_p = (*_w>>6) & 1;  _p++; \
      *_p = (*_w>>7) & 1;  _p++; \
      *_p = (*_w>>8) & 1;  _p++; \
      *_p = (*_w>>9) & 1;  _p++; \
      *_p = (*_w>>10) & 1;  _p++; \
      *_p = (*_w>>11) & 1;  _p++; \
      *_p = (*_w>>12) & 1;  _p++; \
      *_p = (*_w>>13) & 1;  _p++; \
      *_p = (*_w>>14) & 3;  _p++; \
      *_p = (*_w>>16) & 3;  _p++; \
      *_p = (*_w>>18) & 3;  _p++; \
      *_p = (*_w>>20) & 3;  _p++; \
      *_p = (*_w>>22) & 3;  _p++; \
      *_p = (*_w>>24) & 3;  _p++; \
      *_p = (*_w>>26) & 3;  _p++; \
      break; \
    case 4: \
      *_p = (*_w) & 3;     _p++; \
      *_p = (*_w>>2) & 3;  _p++; \
      *_p = (*_w>>4) & 3;  _p++; \
      *_p = (*_w>>6) & 3;  _p++; \
      *_p = (*_w>>8) & 3;  _p++; \
      *_p = (*_w>>10) & 3;  _p++; \
      *_p = (*_w>>12) & 3;  _p++; \
      *_p = (*_w>>14) & 3;  _p++; \
      *_p = (*_w>>16) & 3;  _p++; \
      *_p = (*_w>>18) & 3;  _p++; \
      *_p = (*_w>>20) & 3;  _p++; \
      *_p = (*_w>>22) & 3;  _p++; \
      *_p = (*_w>>24) & 3;  _p++; \
      *_p = (*_w>>26) & 3;  _p++; \
      break; \
    case 5: \
      *_p = (*_w) & 15;     _p++; \
      *_p = (*_w>>4) & 7;  _p++; \
      *_p = (*_w>>7) & 7;  _p++; \
      *_p = (*_w>>10) & 7;  _p++; \
      *_p = (*_w>>13) & 7;  _p++; \
      *_p = (*_w>>16) & 7;  _p++; \
      *_p = (*_w>>19) & 7;  _p++; \
      *_p = (*_w>>22) & 7;  _p++; \
      *_p = (*_w>>25) & 7;  _p++; \
      break; \
    case 6: \
      *_p = (*_w) & 7;     _p++; \
      *_p = (*_w>>3) & 15;  _p++; \
      *_p = (*_w>>7) & 15;  _p++; \
      *_p = (*_w>>11) & 15;  _p++; \
      *_p = (*_w>>15) & 15;  _p++; \
      *_p = (*_w>>19) & 7;  _p++; \
      *_p = (*_w>>22) & 7;  _p++; \
      *_p = (*_w>>25) & 7;  _p++; \
      break; \
    case 7: \
      *_p = (*_w) & 15;     _p++; \
      *_p = (*_w>>4) & 15;  _p++; \
      *_p = (*_w>>8) & 15;  _p++; \
      *_p = (*_w>>12) & 15;  _p++; \
      *_p = (*_w>>16) & 15;  _p++; \
      *_p = (*_w>>20) & 15;  _p++; \
      *_p = (*_w>>24) & 15;  _p++; \
      break; \
    case 8: \
      *_p = (*_w) & 31;     _p++; \
      *_p = (*_w>>5) & 31;  _p++; \
      *_p = (*_w>>10) & 31;  _p++; \
      *_p = (*_w>>15) & 31;  _p++; \
      *_p = (*_w>>20) & 15;  _p++; \
      *_p = (*_w>>24) & 15;  _p++; \
      break; \
    case 9: \
      *_p = (*_w) & 15;     _p++; \
      *_p = (*_w>>4) & 15;  _p++; \
      *_p = (*_w>>8) & 31;  _p++; \
      *_p = (*_w>>13) & 31;  _p++; \
      *_p = (*_w>>18) & 31;  _p++; \
      *_p = (*_w>>23) & 31;  _p++; \
      break; \
    case 10: \
      *_p = (*_w) & 63;     _p++; \
      *_p = (*_w>>6) & 63;  _p++; \
      *_p = (*_w>>12) & 63;  _p++; \
      *_p = (*_w>>18) & 31;  _p++; \
      *_p = (*_w>>23) & 31;  _p++; \
      break; \
    case 11: \
      *_p = (*_w) & 31;     _p++; \
      *_p = (*_w>>5) & 31;  _p++; \
      *_p = (*_w>>10) & 63;  _p++; \
      *_p = (*_w>>16) & 63;  _p++; \
      *_p = (*_w>>22) & 63;  _p++; \
      break; \
    case 12: \
      *_p = (*_w) & 127;     _p++; \
      *_p = (*_w>>7) & 127;  _p++; \
      *_p = (*_w>>14) & 127;  _p++; \
      *_p = (*_w>>21) & 127;  _p++; \
      break; \
    case 13: \
      *_p = (*_w) & 1023;     _p++; \
      *_p = (*_w>>10) & 511;  _p++; \
      *_p = (*_w>>19) & 511;  _p++; \
      break; \
    case 14: \
      *_p = (*_w) & 16383;     _p++; \
      *_p = (*_w>>14) & 16383;  _p++; \
      break; \
    case 15: \
      *_p = (*_w) & ((1<<28)-1);     _p++; \
      break; \
  }\
  _w++; \
}


__global__ void NewPFD_Decompression(uint32_t *d_lists, uint16_t *d_queryID_perBlock, uint32_t *d_shortest_lists, uint32_t *d_segOffset, uint32_t *d_segHead)
{
	//shared
	__shared__ uint32_t s_array[256];
	uint32_t activeThreadNum;			//thread number active in the block
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
	if (tcount * 3 + 1 <= THREAD_NUM) 
	{
		if (tid < tcount * 3 + 1)
		{
			s_array[tid] = (d_constant + constantOffset)[tid];
		}
	}
	else
	{
		s_array[tid] = (d_constant + constantOffset)[tid];

		if (tid < tcount * 3 + 1 - THREAD_NUM)
		{
			s_array[THREAD_NUM + tid] = (d_constant + constantOffset)[THREAD_NUM + tid];
		}
	}
	__syncthreads();
	
	// set my own ucnts and wulists
	ucnts = s_array + 1;
	uwlists = ucnts + tcount;
	segNum = uwlists + tcount;
	shortest_list = d_shortest_lists + s_array[0];

	// calculate activeThreadNum
	activeThreadNum = ucnts[0] - startIdx >= THREAD_NUM ? THREAD_NUM : ucnts[0] - startIdx;


	uint32_t eleIndex = startIdx + tid;
	uint32_t shortest_segstart = uwlists[0] + (d_segOffset + segNum[0])[bid - startBlockId];
	uint32_t descriptor = d_lists[shortest_segstart];
	++shortest_segstart;
	uint32_t b = ((descriptor >> 26) % 32);
	uint32_t en = (descriptor & 0xffff);
	uint32_t idx = shortest_segstart + ((tid * b) >> 5);
	uint32_t shift = (tid * b) & 0x1f;
	uint32_t mask = ((1 << b) - 1);


	__shared__ uint32_t shortest_segment[THREAD_NUM << 1];//shortest lists' segment
	uint64_t codeword = 0;
	if (tid < activeThreadNum) {
		codeword = d_lists[idx] | (static_cast<uint64_t>(d_lists[idx + 1]) << 32);
		shortest_segment[tid] = (codeword >> shift) & mask;
	}
	__syncthreads();


	__shared__ uint32_t all[THREAD_NUM << 1];
	uint32_t *_pp, *_ww;
	uint32_t psum = 0;
	uint32_t i;
	if (en != 0)
	{
		if (tid == 0)
		{
			shortest_segstart += ((activeThreadNum * b) >> 5) + (((activeThreadNum * b) & 0x1f) > 0 ? 1 : 0);
		
			for (_pp = all, _ww = (uint32_t *)(d_lists + shortest_segstart); _pp < &(all[2 * en]); )
			{
				S16_DECODE(_ww, _pp);
			}

			psum = all[0];
			for (i = 0; i < en; i++)
			{
				shortest_segment[psum] += (all[en + i] << b);
				psum += all[i + 1] + 1;
			}
		}
	}
	__syncthreads();


	//scan the gap lists just decompressed
	int pout = 0, pin = 1;		//varialbes controling swapping
	int off;					//varialbes controling the offset of scan

	if (tid == 0)
	{
		shortest_segment[0] = (d_segHead + segNum[0])[bid - startBlockId];//head of current segment
	}
	__syncthreads();

	for (off = 1; off < THREAD_NUM; off <<= 1)
	{
		pout = 1 - pout;
		pin = 1 - pout;
		
		__syncthreads();		//important

		shortest_segment[pout * THREAD_NUM + tid] = shortest_segment[pin * THREAD_NUM + tid];

		if (off <= tid)
		{
			shortest_segment[pout * THREAD_NUM + tid] += shortest_segment[pin * THREAD_NUM + tid - off];
		}
	}
	__syncthreads();


	if (tid < activeThreadNum)
	{
		shortest_list[eleIndex] = shortest_segment[pout * THREAD_NUM + tid];//DocID of shortest list to be searched
	}
	__syncthreads();
}


#endif // #ifndef _TEMPLATE_KERNEL_H_


