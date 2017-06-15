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



__device__ float DcalcVDis(uint32_t eleIndex, uint32_t nDocID, float fSlope, float fIntercept) {
	return (nDocID - (eleIndex * fSlope + fIntercept));
}


__global__ void SegLRC_Generator(uint32_t *d_list, regression_info_t *d_regressioninfo, int *d_distance, uint32_t offset, uint32_t nListLen) {
	uint32_t tid = threadIdx.x;
	uint32_t bid = blockIdx.x;
	uint32_t eleIndex = bid * blockDim.x + tid;

	float fSlope;
	float fIntercept;
	uint32_t nDocID;
	float fOriGap;
	int nDistance;
	float fYTheory;
	int nY;

	if (eleIndex < nListLen) {
		fSlope = (d_regressioninfo + offset + bid)->fSlope;
		fIntercept = (d_regressioninfo + offset + bid)->fIntercept;
		nDocID = d_list[eleIndex];
		fOriGap = DcalcVDis(tid, nDocID, fSlope, fIntercept);
		nDistance = (int)fOriGap;
		
		fYTheory = tid * 1 * fSlope + fIntercept;
		nY = (int)fYTheory;

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


