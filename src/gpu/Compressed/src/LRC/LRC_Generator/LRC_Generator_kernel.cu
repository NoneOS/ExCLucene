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
	int iIntercept;
	float fSearchRange;
} regression_info_t;

typedef struct _stat_info {
	double dRSquare;
	double dContractionRatio;
	uint32_t nBitsNeeded;
} stat_info_t;


uint32_t *d_list;
regression_info_t *d_regressioninfo;
int *d_distance;


__global__ void LRC_Generator(uint32_t *d_list, regression_info_t *d_regressioninfo, int *d_distance, uint32_t nListLen, uint32_t nListIdx)
{
	uint32_t tid = threadIdx.x;
	uint32_t bid = blockIdx.x;
	int eleIndex = bid * blockDim.x + tid;

	float fSlope = (d_regressioninfo + nListIdx)->fSlope;
	int iIntercept = (d_regressioninfo + nListIdx)->iIntercept;
	int nDocID;
	int nDistance;
	int nY;


	if (eleIndex < nListLen) {
		nDocID = d_list[eleIndex];
		nY = int(eleIndex * fSlope) + iIntercept;
		nDistance = nDocID - nY;
		

		int nDistanceMinus = nDistance, nDistancePlus = nDistance;
		while (1) {
			if ((int)(nY + nDistanceMinus) == nDocID) {
				d_distance[eleIndex] = nDistanceMinus;
				break;
			}

			if ((int)(nY + nDistancePlus) == nDocID) {
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

