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


#define	MAX_PATH_LEN   1024
#define THREAD_NUM     256
#define MAX_LIST_LEN   30000000


// ind1
typedef struct _at_term_ind1 {
	uint32_t m_urlcount;  // length 
	uint64_t m_off;       // offset (in bytes)
} at_term_ind1_t;

typedef struct _at_search_ind {
	FILE* fd_ind1;				
	FILE* fd_ind2;			
	uint64_t sz_ind1;	
	uint64_t sz_ind2;
	at_term_ind1_t *m_pind1;
	unsigned char *m_pind2;	
	uint32_t m_tcount;		
} at_search_ind_t;

typedef struct _regression_info {
	float fSlope;
	float fIntercept;
	uint32_t nRangeLeft;
	uint32_t nRangeRight;
} regression_info_t;

typedef struct _stat_info {
	double dRSquare;
	double dContractionRatio;
	uint32_t nBitsNeeded;
} stat_info_t;


uint32_t *d_list;
regression_info_t *d_regressioninfo;
int *d_distance;
uint32_t *d_hashinfo;


__device__ float getDistance(uint32_t eleIndex, uint32_t nDocID, float fSlope, float fIntercept) {
	return (nDocID - (eleIndex * fSlope + fIntercept));
}


__global__ void HS_SegLRC_Generator(uint32_t *d_list, uint32_t nListLen, regression_info_t *d_regressioninfo, uint32_t nRegressionInfoOffset, int *d_distance, uint32_t *d_hashinfo, uint32_t nHashOffset) {
	uint32_t tid = threadIdx.x;
	uint32_t bid = blockIdx.x;
	uint32_t eleIndex = bid * blockDim.x + tid;
	if (eleIndex < nListLen) {
		uint32_t *thisHashInfo = d_hashinfo + nHashOffset;
		uint32_t kBucketNum = thisHashInfo[1];
		thisHashInfo += 2;

		int left = 0, right = kBucketNum;
		while (left < right) {
			int middle = (left + right) >> 1;

			if (thisHashInfo[middle] < eleIndex)
				left = middle + 1;
			else
				right = middle;
		}
		if (eleIndex < thisHashInfo[left]) {
			--left;
		}

		regression_info_t *thisRegressionInfo = d_regressioninfo + nRegressionInfoOffset + left;
		float fSlope = thisRegressionInfo->fSlope;
		float fIntercept = thisRegressionInfo->fIntercept;

		uint32_t nDocID = d_list[eleIndex];
		uint32_t idx = eleIndex - thisHashInfo[left];
		float fOriGap = getDistance(idx, nDocID, fSlope, fIntercept);
		int nDistance = (int)fOriGap;
		float fYTheory = idx * 1 * fSlope + fIntercept;
		int nY = (int)fYTheory;

		int nDistanceMinus = nDistance, nDistancePlus = nDistance;
		while (1) {
			if ((uint32_t)(nY + nDistanceMinus) == nDocID) {
				d_distance[eleIndex] = nDistanceMinus;
				break;
			}

			if ((uint32_t)(nY + nDistancePlus) == nDocID) {
				d_distance[eleIndex] = nDistancePlus;
				break;
			}

			--nDistanceMinus;
			++nDistancePlus;
		}
	}
	__syncthreads();
}

#endif // #ifndef _TEMPLATE_KERNEL_H_

