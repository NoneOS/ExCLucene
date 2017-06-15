// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include </usr/include/sys/stat.h>
#include <vector>
#include </usr/include/sys/time.h>
#include <errno.h>

#include <stdint.h>
#include <iostream>
#include <string>

#include "cutil_inc.h"
#include <cuda.h>

using namespace std;

// includes, kernels
#include "cuda_hashSegment_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C"
void computeGold( float* reference, float* idata, const unsigned int len);

//extern "C"
//CUDPPResult cudppPlan(CUDPPHandle *planHandle, CUDPPConfiguration config,  size_t n,  size_t rows,  size_t rowPitch);

//iteration times
#define ITERATION 1 

#define fullProcess
//#define displayCPU
//#define displayKernel
#define KERNEL_DISPLAY_SHORTEST_LEN 572

//you can modify it for expected hash segment length
#define UNI_LEN 256


unsigned char* ptr;
at_search_ind_t *patind;

#define BUFFER_SIZE 2048
unsigned char buffer[BUFFER_SIZE];
#define MAX_URL_NO		0x1FFFFFF
//transplant GetTickCount under windows
//???���0
unsigned int GetTickCount()
{
	struct timeval tv;
	if (!gettimeofday(&tv, NULL))  //gettimeofday, ?ɹ???0
	{
		return tv.tv_sec * 1000 + tv.tv_usec / 1000;
	}
	else
	{
		return 0;
	}
}


inline void readFile(unsigned char* ptr, FILE* fp){
	int start = GetTickCount();
	uint64_t count = 0, bytes;
	while ((bytes = fread(ptr+count, 1, BUFFER_SIZE, fp))>0) {
		count += bytes;
	}

	cout << "size: " << count / double(1024 * 1024) << "MB" << endl;
	cout << "time: " << GetTickCount() - start << "ms" << endl;
}

at_search_ind_t* as_load_atind(const char *dbi_dir, const char *ind_name)
{
	
	patind = (at_search_ind_t *)malloc(sizeof (*patind));
	struct stat buf;

	char file_name[MAX_PATH_LEN];
	patind->fd_ind1 = 0;
	patind->fd_ind2 = 0;
	patind->m_pind1 = 0;
	patind->m_pind2 = 0;

	// ind1
	sprintf (file_name, "%s%s.ind1", dbi_dir, ind_name);
	cout << "reading " << file_name << endl;
	stat(file_name, &buf);
	patind->sz_ind1 = buf.st_size;
	patind->m_tcount = buf.st_size / sizeof (at_term_ind1_t);
	patind->fd_ind1 = fopen(file_name, "rb");
	patind->m_pind1 = (at_term_ind1_t *)malloc(buf.st_size);
	readFile((unsigned char*)patind->m_pind1 , patind->fd_ind1);

	//ind2
	sprintf (file_name, "%s%s.ind2", dbi_dir, ind_name);
	cout << "reading " << file_name << endl;
	stat(file_name, &buf);
	patind->sz_ind2 = buf.st_size;
	patind->fd_ind2 = fopen(file_name,"rb");
	patind->m_pind2 = (unsigned char*) malloc(buf.st_size);
	readFile(patind->m_pind2 , patind->fd_ind2);

	return patind;
}





int last;
int* result_seq;
int gggcount  = 0;


struct single_keyword_struct
{
	unsigned int queryID;  //??0??ʼ
	unsigned int length;	//Ͱ??
	unsigned int offset;	//?ڵ????????е?ƫ??
};

char resultFileName[100];
FILE *fpTotalStat;


//Ԥ??????Դ???????Ĵ???
//??ȷ??0,???���1??????ģʽ??
int allocateResource(/*host??Դ*/unsigned int **h_constant, uint16_t **h_queryID_perBlock, unsigned int **h_result, unsigned int **h_ucntResult, unsigned char **h_lists, unsigned char **h_bloom, regresstion_info_t **h_pSLinearInfoDetail, float **h_fLinearInfo, unsigned int **h_hashSegInfo, unsigned int **h_hashSegOffset, /*device??Դ*/uint16_t **d_queryID_perBlock, unsigned int **d_lists, unsigned int **d_isCommon, unsigned int **d_scan_odata, unsigned int **d_result, unsigned int **d_ucntResult, unsigned int size_d_lists, unsigned char **d_bloom, unsigned int **d_batchInfo, float ** d_fLinearInfo, unsigned int **d_hashSegInfo, /*cudpp??Դ*/CUDPPConfiguration *config)
{
	// by ysharp: initialize cudpp library
	cudppCreate(&theCudpp);


	//host??Դ
	*h_result = (unsigned int*)malloc(baseSize);
	*h_ucntResult = (unsigned int*)malloc(50000 * sizeof(unsigned int));
	CUDA_SAFE_CALL(cudaMallocHost((void**)h_queryID_perBlock, 65535 * sizeof(uint16_t)));
	CUDA_SAFE_CALL(cudaMallocHost((void**)h_constant, 163800 * sizeof(unsigned int)));

	//device??Դ
	CUDA_SAFE_CALL(cudaMalloc((void**)d_isCommon, baseSize));
	CUDA_SAFE_CALL(cudaMalloc((void**)d_scan_odata, baseSize));
	CUDA_SAFE_CALL(cudaMalloc((void**)d_result, baseSize));
	CUDA_SAFE_CALL(cudaMalloc((void**)d_ucntResult, 50000 * sizeof(unsigned int)));	//no more than 5000 queries in a batch, else the response time will be unacceptable
	CUDA_SAFE_CALL(cudaMalloc((void**)d_lists, size_d_lists));
	CUDA_SAFE_CALL(cudaMalloc((void**)d_queryID_perBlock, 65535 * sizeof(uint16_t)));
	CUDA_SAFE_CALL(cudaMalloc((void**)d_batchInfo, 1024 * 1024));



	//at_HashSeg_info
	FILE *fpHashSegInfo = fopen("./data/at_HashSeg_info", "rb");
	struct stat buf;
	if (!fpHashSegInfo)
	{
		perror("can't open hashSegInfo");
		exit(1);
	}
	stat("./data/at_HashSeg_info", &buf);
	*h_hashSegInfo = (unsigned int*)malloc(buf.st_size);
	printf("reading and transferring hashSegInfo...\n");
	readFile((unsigned char*)*h_hashSegInfo, fpHashSegInfo);
	fclose(fpHashSegInfo);
	CUDA_SAFE_CALL(cudaMalloc((void**)d_hashSegInfo, buf.st_size));
	CUDA_SAFE_CALL(cudaMemcpy(*d_hashSegInfo, *h_hashSegInfo, buf.st_size, cudaMemcpyHostToDevice));
	//at_HashSeg_info ends

	//at_HashSeg_offset
	FILE *fpHashSegOffset = fopen("./data/at_HashSeg_offset", "rb");
	if (!fpHashSegOffset)
	{
		perror("can't open hashSegOffset");
		exit(1);
	}
	stat("./data/at_HashSeg_offset", &buf);
	*h_hashSegOffset = (unsigned int*)malloc(buf.st_size);
	printf("reading hashSegOffset\n");
	readFile((unsigned char*)*h_hashSegOffset, fpHashSegOffset);
	fclose(fpHashSegOffset);
	//at_HashSeg_offset ends


	if (!(*h_result && *d_lists && *d_isCommon && *d_scan_odata && *d_result))
	{
		return 1;
	}

	//transfer lists
	CUDA_SAFE_CALL(cudaMemcpy(*d_lists, *h_lists, size_d_lists, cudaMemcpyHostToDevice));

	//cudpp alloc
	(*config).op = CUDPP_ADD;
	(*config).datatype = CUDPP_INT;
	(*config).algorithm = CUDPP_SCAN;
	(*config).options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
	CUDPPResult result = cudppPlan(theCudpp, &scanplan, *config, 30000000, 1, 0);

	if (CUDPP_SUCCESS != result)
	{
		return 1;
	}

	return 0;
}
unsigned int nBFQueryNum = 0;
FILE *fpCPUPre = NULL;
batchInfo CPUPreprocess(/*host resource*/unsigned int *h_constant, unsigned int *h_queries, unsigned int *h_startBlockId, unsigned int *h_queriesOffset, uint16_t *h_queryID_perBlock, regresstion_info_t *h_pSLinearInfoDetail, float *h_fLinearInfo, /*query resource*/at_search_ind_t *patind, char **ptr_in, char **ptr_end, vector<singleKeyword_t> *pvSSingleKeyword)
{
	//CPU preprocess temp memory
	query_input_t input;  //save current query
	testCell tc;			//save detailed info of the current query
	at_term_ind1_t *pind1_temp;		//temp ind1 pointer for getting detailed query info
	register unsigned int i, j, k;	//loop control variables
	register unsigned int tcount = 0;	//number of terms; used frequently
	batchInfo bi;	//infor of this batch
	unsigned int current_blockNum;	//block number for current query
	pvSSingleKeyword->clear();
	//CPU preprocess temp memory ends

		
	//local temp variables
	unsigned int ucntShortest_sum = 0; 
	unsigned int blockNum= 0;
	unsigned int query_num = 0;
	unsigned int nTotalQueryNum= 0;  //queries  whose lists number is more than 1
	unsigned int queries_offset = 0;	//offset for h_queries
	unsigned int linearInfo_offset = 0;  //offset for h_fLinearInfo
	singleKeyword_t SSingleKey;  //struct of singleKeyword
	unsigned int nRoute = ROUTE_BINARY;
	char *ptr_in_begin = *ptr_in;
	//local temp variables end

	while (ucntShortest_sum <= DOCID_LIMIT && *ptr_in < *ptr_end)
	{
		//get current query
		input.tnum = **ptr_in;
		*ptr_in += sizeof(unsigned int);
		memcpy(input.tno, *ptr_in, sizeof(unsigned int) * input.tnum);
		*ptr_in += sizeof(unsigned int) * input.tnum;

		//get detailed query info: ucnt and uwlist
		tc.tcount = input.tnum;
		tcount = tc.tcount;
		for (i = 0; i < tcount; ++i)
		{
			pind1_temp = patind->m_pind1 + input.tno[i];
			tc.ucnt[i] = pind1_temp->m_urlcount;  
			tc.uwlist[i] = (pind1_temp->m_off) / sizeof(unsigned int);	
			tc.tno[i] = input.tno[i];
			tc.hashOffset[i] = h_hashSegOffset[tc.tno[i]];
#ifdef debug
			printf("ucnt:%u\t", tc.ucnt[i]);
			printf("offset:%u\n", tc.uwlist[i]);
#endif
		}

		//single keyword process
		if (1 == tcount)
		{
			SSingleKey.nLen = tc.ucnt[0];
			SSingleKey.nOffset = tc.uwlist[0];
			SSingleKey.nQueryIDInBatch = nTotalQueryNum;
			pvSSingleKeyword->push_back(SSingleKey);
			nTotalQueryNum++;
			continue;
		}

		//insertion sort
		for (i = 1; i < tcount; ++i)
		{
			k = i;
			unsigned int uwlist_tmp = tc.uwlist[i];
			unsigned int ucnt_tmp = tc.ucnt[i];
			unsigned int tno_tmp = tc.tno[i];
			unsigned int hashOffset_tmp = tc.hashOffset[i];

			while (k && ucnt_tmp < tc.ucnt[k - 1])
			{
				tc.uwlist[k] = tc.uwlist[k - 1];
				tc.ucnt[k] = tc.ucnt[k - 1];
				tc.tno[k] = tc.tno[k - 1];
				tc.hashOffset[k] = tc.hashOffset[k - 1];
				k--;
			};

			if (k != i)
			{
				tc.uwlist[k] = uwlist_tmp;
				tc.ucnt[k] = ucnt_tmp;
				tc.tno[k] = tno_tmp;
				tc.hashOffset[k] = hashOffset_tmp;
			}
		}


#ifdef displayCPU
		printf("tcount: %u\n", tcount);
		for (int m = 0; m < tcount; ++m)
		{
			printf("ucnt:%u\thashOffset:%u\ttermID:%u\n", tc.ucnt[m], tc.hashOffset[m], tc.tno[m]);
		}
#endif

		//algorithm route
		//double dRoute = ((double)(tc.ucnt[0])) / ((double)10000) * ((double)0.007156283) + 0.0032;  //0.57R2
//		double dRoute = tc.ucnt[0] * tc.ucnt[0] / ((double)100000000) * (0 - 0.000018996) + tc.ucnt[0] / ((double)10000) * 0.021 + 0.07939;
//		printf("%lf\n", dRoute);
		/*
		if (tc.ucnt[0] > 10000)
		{
			nBFQueryNum++;
			nRoute = ROUTE_BLOOM;  //nRoute was set ROUTE_BINARY as default
		}
		*/

		//calculate block number needed by current query
		current_blockNum = tc.ucnt[0] % THREAD_NUM ? tc.ucnt[0] / THREAD_NUM + 1 : tc.ucnt[0] / THREAD_NUM;

		//set host memory for constant memory
		h_startBlockId[query_num] = blockNum;
		h_baseOffset[query_num] = ucntShortest_sum;
		h_queriesOffset[query_num] = queries_offset;
		/*
		//fill the linearInfo
		for (int listIndex = 0; listIndex < tcount; ++listIndex)
		{
			h_fLinearInfo[listIndex + linearInfo_offset] = (h_pSLinearInfoDetail + tc.tno[listIndex])->fSlope;
			h_fLinearInfo[tcount + listIndex + linearInfo_offset] = (h_pSLinearInfoDetail + tc.tno[listIndex])->fIntercept;
			h_fLinearInfo[tcount * 2 + listIndex + linearInfo_offset] = (h_pSLinearInfoDetail + tc.tno[listIndex])->nRangeLeft;
			h_fLinearInfo[tcount * 3 + listIndex + linearInfo_offset] = (h_pSLinearInfoDetail + tc.tno[listIndex])->nRangeRight;

		}
		*/
		//process ends
		//copy the query from tc to h_queries
		*(h_queries + queries_offset) = tc.tcount;
		*(h_queries + queries_offset + 1) = ucntShortest_sum;
		*(h_queries + queries_offset + 2) = linearInfo_offset;
		*(h_queries + queries_offset + 3) = nRoute;
		memcpy(h_queries + queries_offset + 4, tc.ucnt, sizeof(unsigned int) * tc.tcount);
		memcpy(h_queries + queries_offset + 4 + tc.tcount, tc.uwlist, sizeof(unsigned int) * tc.tcount);
		memcpy(h_queries + queries_offset + 4 + tc.tcount * 2, tc.hashOffset, sizeof(unsigned int) * tc.tcount);
		//copy ends

		//set queryID for each block
		for (k = 0; k < current_blockNum; ++k)
		{
			h_queryID_perBlock[blockNum + k] = query_num;	
		}

		//set several local variables for next loop
		blockNum += current_blockNum;
		ucntShortest_sum += tc.ucnt[0];
		queries_offset += tc.tcount * 3 + 4;
		linearInfo_offset += 4 * tcount;
		query_num++;
		nTotalQueryNum++;

/*
		//wakensky
		if (3000 < blockNum)
		{
			printf("blockNum: %u\tqueries: %u\tptr_offset: %u\n", blockNum, nTotalQueryNum, *ptr_in - ptr_in_begin);
			exit(0);
		}
		*/
	};

	//whether the base arrays is not big enough
	if (baseSize / sizeof(unsigned int) < ucntShortest_sum)
	{
		printf("!!!!!!!!!!!!!!!!!!!!!!!!\nresource allocated as d_isCommon is insuffient\n!!!!!!!!!!!!!!!!!!!!!!!!\n");
		exit(1);
	}


	//whether the blockNum is over 65535
	if (65535 < blockNum)
	{
		printf("!!!!!!!!!!!!!!!!!!!!!!!!\nblockNum is over 65535\n!!!!!!!!!!!!!!!!!!!!!!!!\n");
		exit(1);
	}

	//prepare for return value
	bi.blockNum = blockNum;
	bi.queryNum = query_num;  //wakensky ??д?ɱ?֤GPU????ֱ??ʹ?õ???ʽ
	bi.constantUsedInByte = (query_num * 3 + queries_offset + batchInfoElementNum) * sizeof(unsigned int);  
	bi.ucntShortest_sum = ucntShortest_sum;
	bi.nTotalQueryNum = nTotalQueryNum;
	bi.nLinearInfoInByte = linearInfo_offset * sizeof(float);

	//printf("constant used: %u\n", bi.constantUsedInByte);

	//integrate five arrays into h_constant
	memcpy(h_constant, &bi, sizeof(struct batchInfo));
	memcpy(h_constant + batchInfoElementNum, h_startBlockId, sizeof(unsigned int) * query_num);
	//memcpy(h_constant + batchInfoElementNum + query_num, h_baseOffset, sizeof(unsigned int) * query_num);
	memcpy(h_constant + batchInfoElementNum + query_num, h_queriesOffset, sizeof(unsigned int) * query_num);
	memcpy(h_constant + batchInfoElementNum + query_num * 2, h_queries, queries_offset * sizeof(unsigned int));


	return bi;
}

void htodTransfer(unsigned int *h_constant, uint16_t *h_queryID_perBlock, float *h_fLinearInfo, unsigned int *d_constantLocal, uint16_t *d_queryID_perBlock, batchInfo bi, unsigned int *d_batchInfo, float *d_fLinearInfo)
{
//	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_constant, h_constant, bi.constantUsedInByte));
	CUDA_SAFE_CALL(cudaMemcpy(d_batchInfo, h_constant, bi.constantUsedInByte, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_queryID_perBlock, h_queryID_perBlock, bi.blockNum * sizeof(uint16_t), cudaMemcpyHostToDevice));

}

void kernelInvoke(/*host resource*/batchInfo bi,  /*device resource*/unsigned int *d_lists, unsigned int *d_isCommon, unsigned int *d_scan_odata, unsigned int *d_result, unsigned int *d_ucntResult, uint16_t *d_queryID_perBlock, unsigned int *d_batchInfo, float *d_fLinearInfo, unsigned int *d_hashSegInfo)
{
	
#ifdef debug	
	printf("blockNum:%u\n",bi.blockNum);
#endif

	mqSearch<<<bi.blockNum, THREAD_NUM >>>(d_batchInfo, d_lists, d_isCommon, d_queryID_perBlock, d_hashSegInfo);
	CUDA_SAFE_CALL(cudaThreadSynchronize());

#ifdef displayKernel
	if (bi.ucntShortest_sum == KERNEL_DISPLAY_SHORTEST_LEN)
	{
	//debug; transback h_isCommon and h_scan_odata
		FILE *fileTemp = fopen("new_temp_file", "w+");
		char tempString[256];
	unsigned int *h_isCommon = (unsigned int *)malloc(baseSize);
	CUDA_SAFE_CALL(cudaMemcpy(h_isCommon, d_isCommon, baseSize, cudaMemcpyDeviceToHost));
	/*
	for (int i = 0; i < bi.ucntShortest_sum; ++i)
	{
		printf("%d ", h_isCommon[i]);
		sprintf(tempString, "%u:%u\n", i, h_isCommon[i]);
		fputs(tempString, fileTemp);
		fflush(fileTemp);
	}
	*/
	printf("377: %d\n", h_isCommon[377]);
	printf("\n");
	free(h_isCommon);
		fclose(fileTemp);
	//debug ends
	}
#endif

	cudppScan(scanplan, d_scan_odata, d_isCommon, bi.ucntShortest_sum);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifdef displayKernel
	//debug; transback h_isCommon and h_scan_odata
	unsigned int *h_scan_odata = (unsigned int *)malloc(baseSize);
	CUDA_SAFE_CALL(cudaMemcpy(h_scan_odata, d_scan_odata, baseSize, cudaMemcpyDeviceToHost));
	//for (int i = 0; i < 600; ++i)
	if (bi.ucntShortest_sum == KERNEL_DISPLAY_SHORTEST_LEN)
	{
		printf("resNumFromScan: %u \n\n", h_scan_odata[KERNEL_DISPLAY_SHORTEST_LEN - 1]);
	}
	free(h_scan_odata);
	//debug ends
#endif

#ifdef fullProcess
	
	saCompact<<<bi.blockNum, THREAD_NUM>>>(d_batchInfo, d_lists, d_isCommon, d_scan_odata, d_result, d_queryID_perBlock);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	

#ifdef debug
	//debug; transback h_isCommon and h_scan_odata
	unsigned int *h_result = (unsigned int *)malloc(baseSize);
	CUDA_SAFE_CALL(cudaMemcpy(h_result, d_result, baseSize, cudaMemcpyDeviceToHost));
	for (int i = 0; i < 600; ++i)
	{
		printf("%u ", h_result[i]);
	}
	printf("\n\n");
	free(h_isCommon);
	//debug ends
#endif

	ucntResult<<<bi.queryNum / 64 + 1, 64>>>(d_batchInfo, d_scan_odata, d_ucntResult);
	CUDA_SAFE_CALL(cudaThreadSynchronize());

#endif

}

void dtohTransfer(/*host resource*/unsigned int *h_ucntResult, unsigned int *h_result, batchInfo bi, /*device resource*/unsigned int *d_ucntResult, unsigned int *d_result)
{
	CUDA_SAFE_CALL(cudaMemcpy(h_ucntResult, d_ucntResult, sizeof(unsigned int) * bi.queryNum, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(h_result, d_result, sizeof(unsigned int) * h_ucntResult[bi.queryNum - 1], cudaMemcpyDeviceToHost));
}

void releaseResource(/*host resource*/unsigned int **h_constant, unsigned int **h_result, unsigned int **h_ucntResult, uint16_t **h_queryID_perBlock, unsigned char **h_bloom, regresstion_info_t **h_pSLinearInfoDetail, float **h_fLinearInfo, unsigned int **h_hashSegInfo, unsigned int **h_hashSegOffset, /*device resource*/unsigned int **d_lists, unsigned int **d_isCommon, unsigned int **d_scan_odata, unsigned int **d_result, unsigned int **d_ucntResult, uint16_t **d_queryID_perBlock, unsigned char **d_bloom, unsigned int **d_batchInfo, float **d_fLinearInfo, unsigned int **d_hashSegInfo)
{
	free(*h_result);
	*h_result = NULL;
	free(*h_ucntResult);
	*h_ucntResult = NULL;
	CUDA_SAFE_CALL(cudaFreeHost(*h_queryID_perBlock));
	*h_queryID_perBlock = NULL;
	CUDA_SAFE_CALL(cudaFreeHost(*h_constant));
	*h_constant = NULL;
	free(*h_hashSegInfo);
	*h_hashSegInfo = NULL;
	free(*h_hashSegOffset);
	*h_hashSegOffset = NULL;


	CUDA_SAFE_CALL(cudaFree(*d_lists));
	*d_lists = NULL;
	CUDA_SAFE_CALL(cudaFree(*d_isCommon));
	*d_isCommon = NULL;
	CUDA_SAFE_CALL(cudaFree(*d_scan_odata));
	*d_scan_odata = NULL;
	CUDA_SAFE_CALL(cudaFree(*d_result));
	*d_result = NULL;
	CUDA_SAFE_CALL(cudaFree(*d_ucntResult));
	*d_ucntResult = NULL;
	CUDA_SAFE_CALL(cudaFree(*d_queryID_perBlock));
	*d_queryID_perBlock = NULL;
	CUDA_SAFE_CALL(cudaFree(*d_batchInfo));
	*d_batchInfo = NULL;
	CUDA_SAFE_CALL(cudaFree(*d_hashSegInfo));
	*d_hashSegInfo = NULL;


	CUDPPResult res = cudppDestroyPlan(scanplan);
	if (CUDPP_SUCCESS != res)
	{
		printf("Error destroying CUDPPPlan\n");
		exit(1);
	}

	// by ysharp: release cudpplibrary	
	cudppDestroy(theCudpp);
}

unsigned int queryID = 0;
FILE *fpResNum = NULL;
void verify(batchInfo bi, unsigned int *h_ucntResult, unsigned int *h_result, checkSum *cs, vector<singleKeyword_t> *pvSSingleKey, unsigned int *h_lists)
{
	//local variables
	unsigned int i, j;	//loop controller
	unsigned int  nValidQueryIndex = 0;
	unsigned int resultNum;	//no more than at_trunc_count
	vector<singleKeyword_t>::iterator irSingleKey = pvSSingleKey->begin();
	irSingleKey = pvSSingleKey->begin();
	//local variables end

	char buf[256];

	for (i = 0; i < bi.nTotalQueryNum; ++i)
	{
		if (nValidQueryIndex)	//not the first valid query in the batch
		{
			if (irSingleKey != pvSSingleKey->end() && i == irSingleKey->nQueryIDInBatch)  //single keyword query
			{
				resultNum = irSingleKey->nLen < at_trunc_count ? irSingleKey->nLen : at_trunc_count;
				cs->check_sum2 += resultNum;
				cs->check_sum3 ^= ((h_lists + irSingleKey->nOffset)[0]);
				cs->check_sum3 ^= ((h_lists + irSingleKey->nOffset + resultNum - 1)[0]);
				cs->check_sum3 ^= ((h_lists + irSingleKey->nOffset + resultNum / 2)[0]);

				irSingleKey++;
			}
			else if(h_ucntResult[nValidQueryIndex] - h_ucntResult[nValidQueryIndex - 1])
			{
				resultNum = h_ucntResult[nValidQueryIndex] - h_ucntResult[nValidQueryIndex - 1] < at_trunc_count ? h_ucntResult[nValidQueryIndex] - h_ucntResult[nValidQueryIndex - 1] : at_trunc_count;
				cs->check_sum2 += resultNum;
				cs->check_sum3 ^= (h_result[h_ucntResult[nValidQueryIndex - 1]]);
				cs->check_sum3 ^= (h_result[h_ucntResult[nValidQueryIndex - 1] + resultNum - 1]);
				cs->check_sum3 ^= (h_result[h_ucntResult[nValidQueryIndex - 1] + resultNum / 2]);
				nValidQueryIndex++;

				//printf("%u:%u\r\n", queryID++, resultNum);
				sprintf(buf, "%u,%u\n", queryID++, resultNum);
				fputs(buf, fpResNum);
			}
			else
			{
				cs->check_sum1++;
				nValidQueryIndex++;
				queryID++;
			}
		}
		else
		{
			if (irSingleKey != pvSSingleKey->end() && i == irSingleKey->nQueryIDInBatch)  //single keyword query
			{
				resultNum = irSingleKey->nLen < at_trunc_count ? irSingleKey->nLen : at_trunc_count;
				cs->check_sum2 += resultNum;
				cs->check_sum3 ^= ((h_lists + irSingleKey->nOffset)[0]);
				cs->check_sum3 ^= ((h_lists + irSingleKey->nOffset + resultNum - 1)[0]);
				cs->check_sum3 ^= ((h_lists + irSingleKey->nOffset + resultNum / 2)[0]);

				irSingleKey++;
			}
			else if (h_ucntResult[nValidQueryIndex])
			{
				resultNum = h_ucntResult[nValidQueryIndex] < at_trunc_count ? h_ucntResult[nValidQueryIndex] : at_trunc_count;
				cs->check_sum2 += resultNum;
				cs->check_sum3 ^= (h_result[0]);
				cs->check_sum3 ^= (h_result[resultNum - 1]);
				cs->check_sum3 ^= (h_result[resultNum / 2]);
				nValidQueryIndex++;
				//printf("%u:%u\r\n", queryID++, resultNum);
				/*
				for (int c = 0; c < resultNum; ++c)
				{
					printf("%u ", (h_result[c]));
				}
				printf("\n\n");
				*/
				sprintf(buf, "%u,%u\n", queryID++, resultNum);
				fputs(buf, fpResNum);
			}
			else
			{
				cs->check_sum1++;
				nValidQueryIndex++;
				queryID++;
			}
		}
	}
}

//function
FILE *fpRes = NULL;
void Run(at_search_ind_t *patind, void *userdata, char *ptr_in, char *ptr_end)
{
	//local variables
	batchInfo bi;
	checkSum cs;
	vector<singleKeyword_t> vSSingleKey;
	char *ptr_in_old = ptr_in;
	char *ptr_end_old = ptr_end;
	//local ends


	//time recorders
	//CPU
	unsigned int timer_CPU = 0;
	double time_CPU = 0;
	double cur_time_CPU = 0;
	//htod transfer
	unsigned int timer_htod = 0;
	double time_htod = 0;
	double cur_time_htod = 0;
	//dtoh transfer
	unsigned int timer_dtoh = 0;
	double time_dtoh = 0;
	double cur_time_dtoh = 0;
	//kernel
	unsigned int timer_kernel = 0;
	double time_kernel = 0;
	double cur_time_kernel = 0;
	//time recorders end

	allocateResource(&h_constant, &h_queryID_perBlock, &h_result, &h_ucntResult, &(patind->m_pind2), &h_bloom, &h_pSLinearInfoDetail, &h_fLinearInfo, &h_hashSegInfo, &h_hashSegOffset, &d_queryID_perBlock, &d_lists, &d_isCommon, &d_scan_odata, &d_result, &d_ucntResult, patind->sz_ind2, &d_bloom, &d_batchInfo, &d_fLinearInfo, &d_hashSegInfo, &config);

	//	DOCID_LIMIT = 30 * 0 * THREAD_NUM; 
	for (DOCID_LIMIT = 32 * 128 * THREAD_NUM; DOCID_LIMIT < 32 *  256 * THREAD_NUM; DOCID_LIMIT *= 2)
	{
		for (int iteration = 0; iteration < ITERATION; ++iteration)
		{
			//time recorders
			//CPU
			timer_CPU = 0;
			time_CPU = 0;
			//htod transfer
			timer_htod = 0;
			time_htod = 0;
			//dtoh transfer
			timer_dtoh = 0;
			time_dtoh = 0;
			//kernel
			timer_kernel = 0;
			time_kernel = 0;
			//time recorders end
			memset(&cs, 0, sizeof(struct checkSum));

			//create timers
			cutilCheckError(cutCreateTimer(&timer_CPU));
			cutilCheckError(cutCreateTimer(&timer_htod));
			cutilCheckError(cutCreateTimer(&timer_kernel));
			cutilCheckError(cutCreateTimer(&timer_dtoh));
			//create timers end

			unsigned int nShortestSum = 0;


			printf("begin calculating...\n");
			unsigned int nThreshold = DOCID_LIMIT / 1024;
			printf("==========%uk ========== THREADNUM:%u\n", DOCID_LIMIT / 1024, THREAD_NUM);

			unsigned int batchID = 0;
			unsigned int nValidQueryNum = 0;
			ptr_in = ptr_in_old;
			ptr_end = ptr_end_old;
			while (ptr_in < ptr_end /*&&  batchID < 8784*/)
			{
				cutilCheckError(cutResetTimer(timer_CPU));
				cutilCheckError(cutStartTimer(timer_CPU));
				bi = CPUPreprocess(h_constant, h_queries, h_startBlockId, h_queriesOffset, h_queryID_perBlock, h_pSLinearInfoDetail, h_fLinearInfo, patind, &ptr_in, &ptr_end, &vSSingleKey);
				cutilCheckError(cutStopTimer(timer_CPU));
				time_CPU += (cur_time_CPU = cutGetTimerValue(timer_CPU));

				cutilCheckError(cutResetTimer(timer_htod));
				cutilCheckError(cutStartTimer(timer_htod));
				htodTransfer(h_constant, h_queryID_perBlock, h_fLinearInfo, d_batchInfo, d_queryID_perBlock, bi, d_batchInfo, d_fLinearInfo);
				cutilCheckError(cutStopTimer(timer_htod));
				time_htod += (cur_time_htod = cutGetTimerValue(timer_htod));

				cutilCheckError(cutResetTimer(timer_kernel));
				cutilCheckError(cutStartTimer(timer_kernel));
				kernelInvoke(bi, d_lists, d_isCommon, d_scan_odata, d_result, d_ucntResult, d_queryID_perBlock, d_batchInfo, d_fLinearInfo, d_hashSegInfo);
				cutilCheckError(cutStopTimer(timer_kernel));
				time_kernel += (cur_time_kernel = cutGetTimerValue(timer_kernel));

#ifdef fullProcess
				cutilCheckError(cutResetTimer(timer_dtoh));
				cutilCheckError(cutStartTimer(timer_dtoh));
				dtohTransfer(h_ucntResult, h_result, bi, d_ucntResult, d_result);
				cutilCheckError(cutStopTimer(timer_dtoh));
				time_dtoh += (cur_time_dtoh = cutGetTimerValue(timer_dtoh));

				verify(bi, h_ucntResult, h_result, &cs, &vSSingleKey, (unsigned int *)(patind->m_pind2));
#endif


#ifdef debug
				//debug; print batch info
				printf("---------batch Info ----------\n"); 
				printf("batchID: %u\n", batchID);
				printf("queryNum:%u\tblockNum:%u\tconstantUsed:%u\tshortestSum:%u\n", bi.queryNum, bi.blockNum, bi.constantUsedInByte, bi.ucntShortest_sum);
				printf("first resultNum: ======%u======\n", h_ucntResult[0]);
				printf("---------batch Info ends-----\n");
				//debug ends
#endif

				batchID++;
				nValidQueryNum += bi.queryNum;
				nShortestSum += bi.ucntShortest_sum;

				/*
				//response and throughput
				char buf[256];
				double dResponse = (cur_time_CPU + cur_time_htod + cur_time_kernel + cur_time_dtoh);
				sprintf(buf, "%lf\n", dResponse);
				fputs(buf, fpRes);
				fflush(fpRes);
				//ends
				*/

			}

			printf("total batch Num:%u\tshortestSum: %u\tvalidQueryNum:%u\tBFQueryNum:%u\n", batchID, nShortestSum, nValidQueryNum, nBFQueryNum);
			//print time
			printf("CPU time:%lf ms\nhtod transfer time:%lf ms\n===kernel time===:%lf ms\ndtoh transfer time:%lf ms\ntotal time: %lf\n", time_CPU, time_htod, time_kernel, time_dtoh, time_CPU + time_htod + time_kernel + time_dtoh);
			printf("valid proportion: %f\n", (float)time_kernel / (time_CPU + time_htod + time_kernel + time_dtoh));


			//print check_sum; for baidu:16392f 22203,2743515,4174671
			printf("************************\nchecksum:%x\ncheck_sum1:%u\tcheck_sum2: %u\tcheck_sum3:%u\n************************\n", cs.check_sum1 ^ cs.check_sum2 ^ cs.check_sum3, cs.check_sum1, cs.check_sum2, cs.check_sum3);

			//response and throughput
			char buf[256];
			double dResponse = (time_CPU + time_htod + time_kernel + time_dtoh) / ((double)batchID);
			double dThroughput = ((double)nValidQueryNum) / (time_CPU + time_htod + time_kernel + time_dtoh) * 1000;
			sprintf(buf, "%uk\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf", nThreshold, dResponse, dThroughput, dThroughput / SPEEDUP, time_CPU, time_kernel, time_htod + time_dtoh);
			//sprintf(buf, "%lf\n", dResponse);
			fputs(buf, fpRes);
			fflush(fpRes);
			//ends

			printf("res: %lf\tthroughput: %lf\n", dResponse, dThroughput);
			printf("########### resNum: %u\tredundancy rate: %lf\tactualFalsePositive: %lf #############\n", cs.check_sum2, (cs.check_sum2 - GOV_RESNUM) / ((double)GOV_RESNUM), (cs.check_sum2 - GOV_RESNUM) / ((double)nShortestSum));
			//delete timers
			cutilCheckError(cutDeleteTimer(timer_CPU));
			cutilCheckError(cutDeleteTimer(timer_htod));
			cutilCheckError(cutDeleteTimer(timer_kernel));
			cutilCheckError(cutDeleteTimer(timer_dtoh));
			//delete timers end
		}

		fputs("\n", fpRes);
		fflush(fpRes);
	}

	releaseResource(&h_constant, &h_result, &h_ucntResult, &h_queryID_perBlock, &h_bloom, &h_pSLinearInfoDetail, &h_fLinearInfo, &h_hashSegInfo, &h_hashSegOffset, &d_lists, &d_isCommon, &d_scan_odata, &d_result, &d_ucntResult, &d_queryID_perBlock, &d_bloom, &d_batchInfo, &d_fLinearInfo, &d_hashSegInfo);

}

// Do some free operations
void terminator(){
	if(ptr!=NULL) free(ptr);
	if(patind->m_pind1!=NULL) free(patind->m_pind1);
	if(patind->m_pind2!=NULL) free(patind->m_pind2);
	if(patind!=NULL) free(patind);
}

//hash segment
const unsigned int MIN_SEG_LEN = UNI_LEN;  //if ucnt is shorter than it, the list will not be partitioned 
const unsigned int AVER_SEG_LEN = UNI_LEN;  //the segNum is ucnt / AVER_SEG_LEN + (ucnt % AVER_SEG_LEN > 0)
unsigned int DOCID_BITNUM = 24;	  //bit number the max docid uses

void RunHashSegInfoGenerator(at_search_ind_t *patind, char *ptr_in, char *ptr_end)
{
	//local variables
	unsigned int nListNum = patind->sz_ind1 / sizeof(at_term_ind1_t);
	at_term_ind1_t *pTmp_ind1 = 0;
	unsigned int nListLen = 0;
	FILE *fpSegInfo = NULL, *fpSegOffset = NULL;
	unsigned int *pnSegInfo = (unsigned int*)malloc(20000000 * sizeof(unsigned int));  //2,000,000 is the virtual max docID, 64 is the virtual min aver_seg_len
	if (!pnSegInfo)
	{
		perror("pnSegInfo allocate failed");
		exit(1);
	}
	unsigned int *pnSegOffset = (unsigned int*)malloc(nListNum * sizeof(unsigned int));
	unsigned int nSegOffset = 0;
	//local variables end

	//open files
	fpSegInfo = fopen("./data/at_HashSeg_info", "wb+");
	if (!fpSegInfo)
	{
		perror("can't open at_hashseg_info");
		exit(1);
	}
	fpSegOffset = fopen("./data/at_HashSeg_offset", "wb+");
	if (!fpSegInfo)
	{
		perror("can't open at_hashseg_offset");
		exit(1);
	}
	//open ends

	for (unsigned int i = 0; i < nListNum; ++i)
	{
		//fill the pnSegOffset
		pnSegOffset[i] = nSegOffset;
		//filling ends

		unsigned int nSegNum = 0;
		pTmp_ind1 = patind->m_pind1 + i;
		nListLen = pTmp_ind1->m_urlcount;
		unsigned int *pnList = (unsigned int *)(patind->m_pind2 + pTmp_ind1->m_off);

		
		double dBITNUM = log((double)((pnList[nListLen - 1]))) / log((double)2);
		uint32_t HS_SHIFT = 0;
		if (dBITNUM - (unsigned int)dBITNUM > 0)
		{
			DOCID_BITNUM = ((unsigned int)dBITNUM) + 1;
		}
		else
		{
			DOCID_BITNUM = (unsigned int)dBITNUM;
		}
	

		if (nListLen <= MIN_SEG_LEN)
		{
			pnSegInfo[0] = DOCID_BITNUM;
			pnSegInfo[1] = 0;
			pnSegInfo[2] = nListLen;

			nSegOffset += 3;
			nSegNum = 1;
		}
		else
		{
			double dLog = log(((double)nListLen) / AVER_SEG_LEN) / log((double)2);
			unsigned int nK = 0;
			if (dLog - (unsigned int)dLog > 0)
			{
				nK = ((unsigned int)dLog) + 1;
			}
			else
			{
				nK = (unsigned int)dLog;
			}
			HS_SHIFT = DOCID_BITNUM - nK;
			pnSegInfo[0] = HS_SHIFT;
			
			//travel the list
			unsigned int nCurSegID = 0, nSegID = 0;
			pnSegInfo[1 + nCurSegID] = 0;
			for (unsigned int docIndex = 0; docIndex < nListLen; ++docIndex)
			{
				unsigned int nDocID = (pnList[docIndex]);
				nSegID = nDocID >> HS_SHIFT;
				if (nSegID > nCurSegID) //a new seg head is found
				{
					while (nCurSegID < nSegID)
					{
						++nCurSegID;
						pnSegInfo[1 + nCurSegID] = docIndex;
					}
				}
			}
			//travelling ends

			//fill the tail with the bucket size
			nCurSegID++;
			pnSegInfo[1 + nCurSegID] = nListLen;
			//filling ends

			nSegNum = nCurSegID;
			nSegOffset += (nSegNum + 2);
		}
		//write segInfo to the disk
		fwrite(pnSegInfo, (nSegNum + 2) * sizeof(unsigned int), 1, fpSegInfo);
		//writing ends

		unsigned int nNonEmpSegNum = 0;
		for (unsigned int nSegID = 0; nSegID < nSegNum; nSegID++)
		{
			if (pnSegInfo[nSegID + 2] - pnSegInfo[nSegID + 1] > 0)
			{
				nNonEmpSegNum++;
			}
		}
//		fprintf(stderr, "%d\t%lf\n", DOCID_BITNUM, (double)nListLen / (double)nNonEmpSegNum);

	}

	//write segOffset to the disk
	fwrite(pnSegOffset, nListNum * sizeof(unsigned int), 1, fpSegOffset);
	//writing ends

	fclose(fpSegInfo);
	fclose(fpSegOffset);
	free(pnSegInfo);
	free(pnSegOffset);

	printf("hashSeg generating finished\n");
}

//read seg info with termID
void ReadSegInfo(at_search_ind_t *patind, char *ptr_in, char *ptr_end, int iTermID)
{
	//local variables
	unsigned int nListNum = patind->sz_ind1 / sizeof(at_term_ind1_t);
	at_term_ind1_t *pTmp_ind1 = 0;
	unsigned int nListLen = 0;
	FILE *fpSegInfo = NULL, *fpSegOffset = NULL;
	unsigned int nSegOffset = 0;
	unsigned int pnSegLenDistribution[2000];
	bzero(pnSegLenDistribution, sizeof(int) * 1000);
	//local variables end

	//read open file
	fpSegInfo = fopen("data/at_HashSeg_info", "rb");
	if (!fpSegInfo)
	{
		perror("can't open rb info file");
		exit(1);
	}
	fpSegOffset = fopen("data/at_HashSeg_offset", "rb");
	if (!fpSegOffset)
	{
		perror("can't open rb offset file");
		exit(1);
	}
	struct stat buf;
	stat("data/at_HashSeg_info", &buf);
	unsigned int *pnSegInfo = (unsigned int*)malloc(buf.st_size); 
	if (!pnSegInfo)
	{
		perror("malloc pnSegInfo failed");
		exit(1);
	}
	printf("reading segInfo ...\n");
	readFile((unsigned char*)pnSegInfo, fpSegInfo);
	stat("data/at_HashSeg_offset", &buf);
	unsigned int *pnSegOffset = (unsigned int*)malloc(buf.st_size);
	if (!pnSegOffset)
	{
		perror("malloc pnSegOffset failed");
		exit(1);
	}
	printf("reading segOffset ...\n");
	readFile((unsigned char*)pnSegOffset, fpSegOffset);

	fclose(fpSegInfo);
	fclose(fpSegOffset);
	//openning ends

	//write open file
	FILE *fpSegLenDetail = fopen("segLenDetail", "w+");
	unsigned int nZeroCount = 0;
	//openning ends

	for (unsigned int i = 0; i < nListNum; ++i)
	{
		pTmp_ind1 = patind->m_pind1 + i;
		nSegOffset = pnSegOffset[i];
		unsigned int nListLen = pTmp_ind1->m_urlcount;
		unsigned int nSegNum = (unsigned int)pow(2.0, 0 - (((int)pnSegInfo[nSegOffset]) - (int)DOCID_BITNUM));

		//not normalized
		/*if (nSegNum == 32)  //fix the most significant k bits*/
		/*
		{
			for (unsigned int seg = 1; seg < nSegNum + 1; ++seg)
			{
				unsigned int nSegLen = pnSegInfo[nSegOffset + seg + 1] - pnSegInfo[nSegOffset + seg];

				if (nSegLen / 50 < 1000)
				{
					pnSegLenDistribution[nSegLen / 50]++;
				}
				else
				{
					pnSegLenDistribution[999]++;
				}

				if (0 == nSegLen)
				{
					nZeroCount++;
				}
			}
		}
		*/

		//normalized
		for (unsigned int seg = 1; seg < nSegNum + 1; ++seg)
		{
			unsigned int nSegLen = pnSegInfo[nSegOffset + seg + 1] - pnSegInfo[nSegOffset + seg];
			if (0 == nSegLen)
			{
				nZeroCount++;
			}
			else
			{
				unsigned int nPos = (unsigned int)((((double)nSegLen) / nListLen) * 1000);
				pnSegLenDistribution[nPos]++;
			}
		}
	}

	//print distribution
	printf("zeroCount: %u\n", nZeroCount);
	for (int i = 0; i < 1000; ++i)
	{
		/*printf("%u,%u\n", i, pnSegLenDistribution[i]);*/
		printf("%u\n", pnSegLenDistribution[i]);
	}
	//print ends

	for (int i = 0; i < 2200; i += 50)
	{
		printf("[%u,%u)\n", i, i + 50);
	}

	/*
	//single list stat
	pTmp_ind1 = patind->m_pind1 + iTermID;
	nListLen = pTmp_ind1->m_urlcount;
	nSegOffset = pnSegOffset[iTermID];
	char bufTmp[256];
	unsigned int nSegNum = 0;


	nSegNum = (unsigned int)pow(2.0, 0 - (((int)pnSegInfo[nSegOffset]) - (int)DOCID_BITNUM));
	//debug
	fprintf(stderr, "pow: %d\n", (((int)pnSegInfo[nSegOffset]) - (int)DOCID_BITNUM));
	//debug ends

	sprintf(bufTmp, "----segNum:%u len:%u----\n", nSegNum, nListLen);
	fputs(bufTmp, fpSegLenDetail);
	for (unsigned int i = 1; i < nSegNum + 1; ++i)
	{
		//seg len
		sprintf(bufTmp, "%u\n", pnSegInfo[nSegOffset + i + 1] - pnSegInfo[nSegOffset + i]);
		//seg head
		//sprintf(bufTmp, "%u\n", pnSegInfo[nSegOffset + i]);
		fputs(bufTmp, fpSegLenDetail);
	}
	*/


	fclose(fpSegLenDetail);
	free(pnSegInfo);
	free(pnSegOffset);
}
//hash ends

void runTest( int argc, char** argv) {
	printf("program start\n");
	if (argc != 2) {
		std::cout << "wrong number of arguments" << std::endl;
		exit(1);
	}

	string dataset = argv[1];
	string index_dir = "/media/gov2disk/naiyong/dataset/" + dataset + "/";

	string input_file = index_dir + dataset + ".query";
	at_search_ind_t *at_ind = NULL;

	FILE* fp = fopen(input_file.c_str(),"rb");
	if (!fp)
	{
		printf("at_test_data open failed\terr code:%u\n", errno);
		exit(1);
	}


	cout << "reading " << input_file << endl;
	struct stat buf;
	stat(input_file.c_str(), &buf);  
	int size = buf.st_size;  //????test?Ĵ?С
	ptr = (unsigned char*) malloc(size);;  //????test??ָ??
	readFile(ptr,fp);
	at_ind = as_load_atind(index_dir.c_str(), dataset.c_str());
	fclose(fp);


	//CUT_DEVICE_INIT(argc, argv);
	if (argc > 2)
	{
		printf("%s\n", argv[2]);
		cutilSafeCall(cudaSetDevice(strtol(argv[2], NULL, 10)));
	}
	else //default set to Tesla
	{
		cutilSafeCall(cudaSetDevice(0));
	}



	fpRes = fopen("PPOPP_RES_HASH.txt", "w+");
	fpResNum = fopen("resNumFromVerify.txt", "w+");

	//run
	cout << "1" << endl;
	RunHashSegInfoGenerator(at_ind, (char *)ptr, (char *)ptr + buf.st_size);
	cout << "2" << endl;
	ReadSegInfo(at_ind, (char*)ptr, (char*)ptr + buf.st_size, 0);
	cout << "3" << endl;
	Run(at_ind, 0, (char *)ptr, (char *)ptr + buf.st_size);


	fclose(fpRes);
	fclose(fpResNum);

	terminator();
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
    runTest( argc, argv);
	cudaThreadExit();
}
