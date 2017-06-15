// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include </usr/include/sys/stat.h>
#include <vector>
#include </usr/include/sys/time.h>
#include <errno.h>

using namespace std;

//#include "cudpp/cudpp.h"

// includes, project
#include <cutil_inline.h>
#include <cuda.h>

// includes, kernels
#include <ppopp_binary_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C"
void computeGold( float* reference, float* idata, const unsigned int len);

//extern "C"
//CUDPPResult cudppPlan(CUDPPHandle *planHandle, CUDPPConfiguration config,  size_t n,  size_t rows,  size_t rowPitch);

#define ITERATION 1 
#define fullProcess
//#define displayCPU
//#define displayKernel



unsigned char* ptr;
at_search_ind_t *patind;

#define BUFFER_SIZE 2048
unsigned char buffer[BUFFER_SIZE];
#define GET_URL_NO(x)   ((x)&0x1FFFFFF)
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
	int count = 0, bytes;
	while((bytes = fread(ptr+count,1,BUFFER_SIZE,fp))>0) {
		//memcpy(ptr+count,buffer,bytes);	
		//CUDA_SAFE_CALL(cudaMemcpy((void*)(ptr+count),(void*)(buffer),bytes,cudaMemcpyHostToHost));
		
		count+= bytes;
		//printf("%d\n",count);
	}
	printf("%d ms\n", GetTickCount()-start);
}



at_search_ind_t* as_load_atind(char *dbi_dir, char *ind_name)
{
	
	patind = (at_search_ind_t *) malloc (sizeof (*patind));
	struct stat buf;

	char file_name[MAX_PATH_LEN];
	patind->fd_ind1 = 0;
	patind->fd_ind2 = 0;
	patind->m_pind1 = 0;
	patind->m_pind2 = 0;

	sprintf (file_name, "%s%s.ind1", dbi_dir, ind_name);

	printf("%s\n",file_name);

	stat(file_name, &buf);
	patind->sz_ind1 = buf.st_size;
	patind->m_tcount = buf.st_size / sizeof (at_term_ind1_t);
	patind->fd_ind1 = fopen(file_name,"rb");
	patind->m_pind1 = (at_term_ind1_t *) malloc(buf.st_size);
	readFile((unsigned char*)patind->m_pind1 , patind->fd_ind1);

	
	//patind->m_pind1
	//patind->m_pind1 = (at_term_ind1_t *)mmap(NULL, buf.st_size, PROT_READ, MAP_SHARED, patind->fd_ind1, 0);

	//ind2
	sprintf (file_name, "%s%s.ind2", dbi_dir, ind_name);


	stat(file_name, &buf);

	
	printf("zzzzzzzzzzzzzzzzzzzzzzzzzzzz%u\n",(unsigned int)buf.st_size);
	patind->sz_ind2 = buf.st_size;
	patind->fd_ind2 = fopen(file_name,"rb");
	patind->m_pind2 = (unsigned char*) malloc(buf.st_size);
	printf("%u\n",(unsigned int)buf.st_size);
	printf("ok");
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
int allocateResource(/*host??Դ*/unsigned int **h_queries, unsigned int **h_startBlockId, unsigned int **h_queriesOffset, unsigned int **h_baseOffset, unsigned int **h_constant, uint16_t **h_queryID_perBlock, unsigned int **h_result, unsigned int **h_ucntResult, unsigned char **h_lists, unsigned char **h_bloom, regresstion_info_t **h_pSLinearInfoDetail, float **h_fLinearInfo, /*device??Դ*/uint16_t **d_queryID_perBlock, unsigned int **d_lists, unsigned int **d_isCommon, unsigned int **d_scan_odata, unsigned int **d_result, unsigned int **d_ucntResult, unsigned int size_d_lists, unsigned char **d_bloom, unsigned int **d_batchInfo, float ** d_fLinearInfo, /*cudpp??Դ*/CUDPPConfiguration *config, CUDPPHandle *scanplan)
{
	//host??Դ
	CUDA_SAFE_CALL(cudaMallocHost((void**)h_queries, STREAM_NUM * 16380 * sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMallocHost((void**)h_startBlockId, STREAM_NUM * 16380 * sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMallocHost((void**)h_queriesOffset, STREAM_NUM * 16380 * sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMallocHost((void**)h_baseOffset, STREAM_NUM * 16380 * sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMallocHost((void**)h_result, STREAM_NUM * baseSize));
	CUDA_SAFE_CALL(cudaMallocHost((void**)h_ucntResult, STREAM_NUM * 5000 * sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMallocHost((void**)h_queryID_perBlock, STREAM_NUM * 65535 * sizeof(uint16_t)));
	CUDA_SAFE_CALL(cudaMallocHost((void**)h_constant, STREAM_NUM * 163800 * sizeof(unsigned int)));
	//CUDA_SAFE_CALL(cudaMallocHost((void**)h_fLinearInfo, 16380 * sizeof(float)));

	//device??Դ
	CUDA_SAFE_CALL(cudaMalloc((void**)d_isCommon, STREAM_NUM * baseSize));
	CUDA_SAFE_CALL(cudaMalloc((void**)d_scan_odata, STREAM_NUM * baseSize));
	CUDA_SAFE_CALL(cudaMalloc((void**)d_result, STREAM_NUM * baseSize));
	CUDA_SAFE_CALL(cudaMalloc((void**)d_ucntResult, STREAM_NUM * 50000 * sizeof(unsigned int)));	//no more than 5000 queries in a batch, else the response time will be unacceptable
	CUDA_SAFE_CALL(cudaMalloc((void**)d_queryID_perBlock, STREAM_NUM * 65535 * sizeof(uint16_t)));
	CUDA_SAFE_CALL(cudaMalloc((void**)d_batchInfo, STREAM_NUM * 1024 * 1024));
//	CUDA_SAFE_CALL(cudaMalloc((void**)d_fLinearInfo, 163800 * sizeof(float)));

	CUDA_SAFE_CALL(cudaMalloc((void**)d_lists, size_d_lists));


	/*
	//bloomfilter
	FILE *fpBloom = fopen("data/gov/bloom5.ind", "rb");
	if (!fpBloom)
	{
		perror("can't open bloom.ind");
		exit(1);
	}
	unsigned int nSize = 0;
	struct stat buf;
	stat("data/gov/bloom5.ind", &buf); 
	nSize = buf.st_size;
	printf("size of bloom.ind:%u", nSize);
	*h_bloom = (unsigned char *)malloc(nSize);
	if (!(*h_bloom))
	{
		perror("h_bloom allocated failed");
	}
	printf("reading bloom.ind...\n");
	readFile(*h_bloom, fpBloom);
	fclose(fpBloom);
	*/

	/*
	//at_regression_info
	FILE *fpLinear = fopen("data/gov/at_regression_info", "rb");
	if (!fpLinear)
	{
		perror("can't open at_regression_info");
		exit(1);
	}
	unsigned int nSize = 0;
	struct stat buf;
	stat("data/gov/at_regression_info", &buf);
	nSize = buf.st_size;
	*h_pSLinearInfoDetail = (regresstion_info_t *)malloc(nSize);
	if (!(*h_pSLinearInfoDetail))
	{
		perror("h_pSLinearInfoDetail allocated failed");
		exit(1);
	}
	printf("reading at_regression_info\n");
	readFile((unsigned char *)(*h_pSLinearInfoDetail), fpLinear);
	fclose(fpLinear);
	fpLinear = NULL;
	// at_regression_info ends
	*/

	//CUDA_SAFE_CALL(cudaMalloc((void**)d_bloom, nSize));
	//CUDA_SAFE_CALL(cudaMemcpy(*d_bloom, *h_bloom, nSize, cudaMemcpyHostToDevice));

	if (!(*h_result && *d_lists && *d_isCommon && *d_scan_odata && *d_result))
	{
		return 1;
	}

	//transfer lists
	printf("transferring ind2...\n");
	CUDA_SAFE_CALL(cudaMemcpy(*d_lists, *h_lists, size_d_lists, cudaMemcpyHostToDevice));

	//cudpp alloc
	(*config).op = CUDPP_ADD;
	(*config).datatype = CUDPP_INT;
	(*config).algorithm = CUDPP_SCAN;
	(*config).options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
	CUDPPResult result = cudppPlan(scanplan, *config, 5000000, 1, 0);

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
	singleKeyword_t SSingleKey;  //struct of singleKeyword
	unsigned int nRoute = ROUTE_BINARY;
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
			
#ifdef debug
			printf("ucnt:%u\t", tc.ucnt[i]);
			printf("offset:%u\n", tc.uwlist[i]);
#endif
		}

		//single keyword process
		if (1 == tcount)
		/*if (2 != tcount)*/
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

			while (k && ucnt_tmp < tc.ucnt[k - 1])
			{
				tc.uwlist[k] = tc.uwlist[k - 1];
				tc.ucnt[k] = tc.ucnt[k - 1];
				tc.tno[k] = tc.tno[k - 1];
				k--;
			};

			if (k != i)
			{
				tc.uwlist[k] = uwlist_tmp;
				tc.ucnt[k] = ucnt_tmp;
				tc.tno[k] = tno_tmp;
			}
		}

#ifdef displayCPU
		printf("tcount: %u\n", tcount);
		for (int m = 0; m < tcount; ++m)
		{
			printf("ucnt:%u\n", tc.ucnt[m]);
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
		//process ends
		//copy the query from tc to h_queries
		*(h_queries + queries_offset) = tc.tcount;
		*(h_queries + queries_offset + 1) = ucntShortest_sum;
		*(h_queries + queries_offset + 2) = 0;//linearInfo_offset;
		*(h_queries + queries_offset + 3) = nRoute;
		memcpy(h_queries + queries_offset + 4, tc.ucnt, sizeof(unsigned int) * tc.tcount);
		memcpy(h_queries + queries_offset + 4 + tc.tcount, tc.uwlist, sizeof(unsigned int) * tc.tcount);
		//copy ends

		//set queryID for each block
		for (k = 0; k < current_blockNum; ++k)
		{
			h_queryID_perBlock[blockNum + k] = query_num;	
		}

		//set several local variables for next loop
		blockNum += current_blockNum;
		ucntShortest_sum += tc.ucnt[0];
		queries_offset += tc.tcount * 2 + 4;
		query_num++;
		nTotalQueryNum++;
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


	//printf("constant used: %u\n", bi.constantUsedInByte);

	//integrate five arrays into h_constant
	memcpy(h_constant, &bi, sizeof(struct batchInfo));
	memcpy(h_constant + batchInfoElementNum, h_startBlockId, sizeof(unsigned int) * query_num);
	//memcpy(h_constant + batchInfoElementNum + query_num, h_baseOffset, sizeof(unsigned int) * query_num);
	memcpy(h_constant + batchInfoElementNum + query_num, h_queriesOffset, sizeof(unsigned int) * query_num);
	memcpy(h_constant + batchInfoElementNum + query_num * 2, h_queries, queries_offset * sizeof(unsigned int));


	return bi;
}

void htodTransfer(unsigned int *h_constant, uint16_t *h_queryID_perBlock, float *h_fLinearInfo, unsigned int *d_constantLocal, uint16_t *d_queryID_perBlock, batchInfo bi, unsigned int *d_batchInfo, float *d_fLinearInfo, int sid)
{
//	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_constant, h_constant, bi.constantUsedInByte));
	CUDA_SAFE_CALL(cudaMemcpyAsync(d_batchInfo, h_constant, bi.constantUsedInByte, cudaMemcpyHostToDevice, stream[sid]));
	CUDA_SAFE_CALL(cudaMemcpyAsync(d_queryID_perBlock, h_queryID_perBlock, bi.blockNum * sizeof(uint16_t), cudaMemcpyHostToDevice, stream[sid]));

}

void kernelInvoke(/*host resource*/batchInfo bi,  /*device resource*/unsigned int *d_lists, unsigned int *d_isCommon, unsigned int *d_scan_odata, unsigned int *d_result, unsigned int *d_ucntResult, uint16_t *d_queryID_perBlock, unsigned int *d_batchInfo, float *d_fLinearInfo, int sid)
{
	
#ifdef debug	
	printf("blockNum:%u\n",bi.blockNum);
#endif

	mqSearch<<<bi.blockNum, THREAD_NUM, stream[sid]>>>(d_batchInfo, d_lists, d_isCommon, d_queryID_perBlock);
	CUDA_SAFE_CALL(cudaThreadSynchronize());

#ifdef displayKernel
	if (bi.ucntShortest_sum == 884)
	{
	//debug; transback h_isCommon and h_scan_odata
		FILE *fileTemp = fopen("new_temp_file", "w+");
		char tempString[256];
	unsigned int *h_isCommon = (unsigned int *)malloc(baseSize);
	CUDA_SAFE_CALL(cudaMemcpy(h_isCommon, d_isCommon, baseSize, cudaMemcpyDeviceToHost));
	for (int i = 0; i < bi.ucntShortest_sum; ++i)
	{
		printf("%d ", h_isCommon[i]);
		sprintf(tempString, "%u:%u\n", i, h_isCommon[i]);
		fputs(tempString, fileTemp);
		fflush(fileTemp);
	}
	printf("\n\n");
	free(h_isCommon);
		fclose(fileTemp);
	//debug ends
	}
#endif

/*	cudppScan(scanplan, d_scan_odata, d_isCommon, bi.ucntShortest_sum);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifdef displayKernel
	//debug; transback h_isCommon and h_scan_odata
	unsigned int *h_scan_odata = (unsigned int *)malloc(baseSize);
	CUDA_SAFE_CALL(cudaMemcpy(h_scan_odata, d_scan_odata, baseSize, cudaMemcpyDeviceToHost));
	//for (int i = 0; i < 600; ++i)
	if (bi.ucntShortest_sum == 884)
	{
		printf("resNumFromScan: %u ", h_scan_odata[883]);
	}
	printf("\n\n");
	free(h_scan_odata);
	//debug ends
#endif

#ifdef fullProcess
	
	saCompact<<<bi.blockNum, THREAD_NUM, stream[sid]>>>(d_batchInfo, d_lists, d_isCommon, d_scan_odata, d_result, d_queryID_perBlock);
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

	ucntResult<<<bi.queryNum / 64+ 1, 64, stream[sid]>>>(d_batchInfo, d_scan_odata, d_ucntResult);
	CUDA_SAFE_CALL(cudaThreadSynchronize());

#endif
*/
}

void dtohTransfer(/*host resource*/unsigned int *h_ucntResult, unsigned int *h_result, batchInfo bi, /*device resource*/unsigned int *d_ucntResult, unsigned int *d_result, int sid)
{
	CUDA_SAFE_CALL(cudaMemcpyAsync(h_ucntResult, d_ucntResult, sizeof(unsigned int) * bi.queryNum, cudaMemcpyDeviceToHost, stream[sid]));
	CUDA_SAFE_CALL(cudaMemcpyAsync(h_result, d_result, sizeof(unsigned int) * h_ucntResult[bi.queryNum - 1], cudaMemcpyDeviceToHost, stream[sid]));
}

void releaseResource(/*host resource*/unsigned int **h_queries, unsigned int **h_startBlockId, unsigned int **h_queriesOffset, unsigned int **h_baseOffset, unsigned int **h_constant, unsigned int **h_result, unsigned int **h_ucntResult, uint16_t **h_queryID_perBlock, unsigned char **h_bloom, regresstion_info_t **h_pSLinearInfoDetail, float **h_fLinearInfo, /*device resource*/unsigned int **d_lists, unsigned int **d_isCommon, unsigned int **d_scan_odata, unsigned int **d_result, unsigned int **d_ucntResult, uint16_t **d_queryID_perBlock, unsigned char **d_bloom, unsigned int **d_batchInfo, float **d_fLinearInfo)
{
	CUDA_SAFE_CALL(cudaFreeHost(*h_queries));
	*h_queries = NULL;
	CUDA_SAFE_CALL(cudaFreeHost(*h_startBlockId));
	*h_startBlockId = NULL;
	CUDA_SAFE_CALL(cudaFreeHost(*h_queriesOffset));
	*h_queriesOffset = NULL;
	CUDA_SAFE_CALL(cudaFreeHost(*h_baseOffset));
	*h_baseOffset = NULL;
	CUDA_SAFE_CALL(cudaFreeHost(*h_result));
	*h_result = NULL;
	CUDA_SAFE_CALL(cudaFreeHost(*h_ucntResult));
	*h_ucntResult = NULL;
	CUDA_SAFE_CALL(cudaFreeHost(*h_queryID_perBlock));
	*h_queryID_perBlock = NULL;
	CUDA_SAFE_CALL(cudaFreeHost(*h_constant));
	*h_constant = NULL;
	CUDA_SAFE_CALL(cudaFreeHost(*h_fLinearInfo));
	*h_fLinearInfo = NULL;
	free(*h_bloom);
	*h_bloom = NULL;
	/*
	free(*h_pSLinearInfoDetail);
	*h_pSLinearInfoDetail = NULL;
	*/


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
	CUDA_SAFE_CALL(cudaFree(*d_bloom));
	*d_bloom = NULL;
	CUDA_SAFE_CALL(cudaFree(*d_batchInfo));
	*d_batchInfo = NULL;
	/*
	CUDA_SAFE_CALL(cudaFree(*d_fLinearInfo));
	*d_fLinearInfo = NULL;
	*/
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
				cs->check_sum3 ^= GET_URL_NO((h_lists + irSingleKey->nOffset)[0]);
				cs->check_sum3 ^= GET_URL_NO((h_lists + irSingleKey->nOffset + resultNum - 1)[0]);
				cs->check_sum3 ^= GET_URL_NO((h_lists + irSingleKey->nOffset + resultNum / 2)[0]);

				irSingleKey++;
			}
			else if(h_ucntResult[nValidQueryIndex] - h_ucntResult[nValidQueryIndex - 1])
			{
				resultNum = h_ucntResult[nValidQueryIndex] - h_ucntResult[nValidQueryIndex - 1] < at_trunc_count ? h_ucntResult[nValidQueryIndex] - h_ucntResult[nValidQueryIndex - 1] : at_trunc_count;
				cs->check_sum2 += resultNum;
				cs->check_sum3 ^= GET_URL_NO(h_result[h_ucntResult[nValidQueryIndex - 1]]);
				cs->check_sum3 ^= GET_URL_NO(h_result[h_ucntResult[nValidQueryIndex - 1] + resultNum - 1]);
				cs->check_sum3 ^= GET_URL_NO(h_result[h_ucntResult[nValidQueryIndex - 1] + resultNum / 2]);
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
				cs->check_sum3 ^= GET_URL_NO((h_lists + irSingleKey->nOffset)[0]);
				cs->check_sum3 ^= GET_URL_NO((h_lists + irSingleKey->nOffset + resultNum - 1)[0]);
				cs->check_sum3 ^= GET_URL_NO((h_lists + irSingleKey->nOffset + resultNum / 2)[0]);

				irSingleKey++;
			}
			else if (h_ucntResult[nValidQueryIndex])
			{
				resultNum = h_ucntResult[nValidQueryIndex] < at_trunc_count ? h_ucntResult[nValidQueryIndex] : at_trunc_count;
				cs->check_sum2 += resultNum;
				cs->check_sum3 ^= GET_URL_NO(h_result[0]);
				cs->check_sum3 ^= GET_URL_NO(h_result[resultNum - 1]);
				cs->check_sum3 ^= GET_URL_NO(h_result[resultNum / 2]);
				nValidQueryIndex++;
				//printf("%u:%u\r\n", queryID++, resultNum);
				/*
				for (int c = 0; c < resultNum; ++c)
				{
					printf("%u ", GET_URL_NO(h_result[c]));
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
	batchInfo bi[STREAM_NUM];
	checkSum cs;
	vector<singleKeyword_t> vSSingleKey[STREAM_NUM];
	char *ptr_in_old = ptr_in;
	char *ptr_end_old = ptr_end;
	//local ends


	//time recorders
	//CPU
	unsigned int timer_CPU;
	double time_CPU;
	double cur_time_CPU;
	//htod transfer
	unsigned int timer_htod;
	double time_htod;
	double cur_time_htod;
	//dtoh transfer
	unsigned int timer_dtoh;
	double time_dtoh;
	double cur_time_dtoh;
	//kernel
	unsigned int timer_kernel;
	double time_kernel;
	double cur_time_kernel;
	//time recorders end

	allocateResource(&h_queries, &h_startBlockId, &h_queriesOffset, &h_baseOffset, &h_constant, &h_queryID_perBlock, &h_result, &h_ucntResult, &(patind->m_pind2), &h_bloom, &h_pSLinearInfoDetail, &h_fLinearInfo, &d_queryID_perBlock, &d_lists, &d_isCommon, &d_scan_odata, &d_result, &d_ucntResult, patind->sz_ind2, &d_bloom, &d_batchInfo, &d_fLinearInfo, &config, &scanplan);

	//	DOCID_LIMIT = 30 * 0 * THREAD_NUM; 
	for (DOCID_LIMIT = 32 * 1 * THREAD_NUM; DOCID_LIMIT <= 32 * 512 * THREAD_NUM; DOCID_LIMIT *= 2)
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
			unsigned int nThreshold = DOCID_LIMIT / 32 / THREAD_NUM;
			printf("===========B %u   THREAD_NUM %u==========\n", DOCID_LIMIT / 32 / THREAD_NUM, THREAD_NUM);

			unsigned int batchID = 0;
			unsigned int nValidQueryNum = 0;
			unsigned int nTotalQueryNum = 0;
			ptr_in = ptr_in_old;
			ptr_end = ptr_end_old;
			while (ptr_in < ptr_end /*&& batchID < 1024*/)
			{
				for (int sid = 0; sid < STREAM_NUM; sid++)
				{
					cutilCheckError(cutResetTimer(timer_CPU));
					cutilCheckError(cutStartTimer(timer_CPU));
					bi[sid] = CPUPreprocess(h_constant + sid * 163800, h_queries + sid * 16380, h_startBlockId + sid * 16380, h_queriesOffset + sid * 16380, h_queryID_perBlock + sid * 65535, h_pSLinearInfoDetail, h_fLinearInfo, patind, &ptr_in, &ptr_end, vSSingleKey + sid);
					cutilCheckError(cutStopTimer(timer_CPU));
					time_CPU += (cur_time_CPU = cutGetTimerValue(timer_CPU));

					cutilCheckError(cutResetTimer(timer_htod));
					cutilCheckError(cutStartTimer(timer_htod));
					htodTransfer(h_constant + sid * 163800, h_queryID_perBlock + sid * 65535, h_fLinearInfo, d_batchInfo + sid * 1024 * 1024 / sizeof(unsigned int), d_queryID_perBlock + sid * 65535, bi[sid], d_batchInfo + sid * 1024 * 1024 / sizeof(unsigned int), d_fLinearInfo, sid);
					cutilCheckError(cutStopTimer(timer_htod));
					time_htod += (cur_time_htod = cutGetTimerValue(timer_htod));

					cutilCheckError(cutResetTimer(timer_kernel));
					cutilCheckError(cutStartTimer(timer_kernel));
					kernelInvoke(bi[sid], d_lists, d_isCommon + sid * baseSize / sizeof(unsigned int), d_scan_odata + sid * baseSize / sizeof(unsigned int), d_result + sid * baseSize / sizeof(unsigned int), d_ucntResult + sid * 5000, d_queryID_perBlock + sid * 65535, d_batchInfo + sid * 1024 * 1024 / sizeof(unsigned int), d_fLinearInfo, sid);
					cutilCheckError(cutStopTimer(timer_kernel));
					time_kernel += (cur_time_kernel = cutGetTimerValue(timer_kernel));

#ifdef fullProcess
					cutilCheckError(cutResetTimer(timer_dtoh));
					cutilCheckError(cutStartTimer(timer_dtoh));
					dtohTransfer(h_ucntResult + sid * 5000, h_result + sid * baseSize / sizeof(unsigned int), bi[sid], d_ucntResult + sid * 5000, d_result + sid * baseSize / sizeof(unsigned int), sid);
					cutilCheckError(cutStopTimer(timer_dtoh));
					time_dtoh += (cur_time_dtoh = cutGetTimerValue(timer_dtoh));

					verify(bi[sid], h_ucntResult + sid * 5000, h_result + sid * baseSize / sizeof(unsigned int), &cs, vSSingleKey + sid, (unsigned int *)(patind->m_pind2));
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
					nValidQueryNum += bi[sid].queryNum;
					nTotalQueryNum += bi[sid].nTotalQueryNum;
					nShortestSum += bi[sid].ucntShortest_sum;

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

			}

			printf("total batch Num:%u\tshortestSum: %u\tvalidQueryNum:%u\tBFQueryNum:%u\n", batchID, nShortestSum, nValidQueryNum, nBFQueryNum);
			//print time
			printf("CPU time:%lf ms\nhtod transfer time:%lf ms\n===kernel time===:%lf ms\ndtoh transfer time:%lf ms\ntotal time: %lf\n", time_CPU, time_htod, time_kernel, time_dtoh, time_CPU + time_htod + time_kernel + time_dtoh);


			//print check_sum; for baidu:16392f 22203,2743515,4174671
			printf("************************\nchecksum:%x\ncheck_sum1:%u\tcheck_sum2: %u\tcheck_sum3:%u\n************************\n", cs.check_sum1 ^ cs.check_sum2 ^ cs.check_sum3, cs.check_sum1, cs.check_sum2, cs.check_sum3);

			//response and throughput
			char buf[256];
			printf("totalQueryNum: %u\n", nTotalQueryNum);
			double dResponse = (time_CPU + time_htod + time_kernel + time_dtoh) / ((double)batchID);
			double dThroughput = ((double)nTotalQueryNum) / (time_CPU + time_htod + time_kernel + time_dtoh) * 1000;
			//sprintf(buf, "%lf\t%lf\n", dResponse, dThroughput);
			/*sprintf(buf, "%lf\n", dResponse);*/
			sprintf(buf, "%u\t%lf\t%lf\t%lf\t%lf\t%lf\n", nThreshold, dResponse, dThroughput, time_CPU, time_kernel, time_htod + time_dtoh);
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

		printf("\n");
	}

	releaseResource(&h_queries, &h_startBlockId, &h_queriesOffset, &h_baseOffset, &h_constant, &h_result, &h_ucntResult, &h_queryID_perBlock, &h_bloom, &h_pSLinearInfoDetail, &h_fLinearInfo, &d_lists, &d_isCommon, &d_scan_odata, &d_result, &d_ucntResult, &d_queryID_perBlock, &d_bloom, &d_batchInfo, &d_fLinearInfo);

}

// Do some free operations
void terminator(){
	if(ptr!=NULL) free(ptr);
	if(patind->m_pind1!=NULL) free(patind->m_pind1);
	if(patind->m_pind2!=NULL) free(patind->m_pind2);
	if(patind!=NULL) free(patind);
}


void
runTest( int argc, char** argv) 
{
	printf("program start\n");
	char index_dir[] = "data/gov/";
	char input_file[] = "data/gov/at_test_data";
	at_search_ind_t *at_ind = NULL;

	FILE* fp = fopen(input_file,"rb");
	if (!fp)
	{
		printf("at_test_data open failed\terr code:%u\n", errno);
		exit(1);
	}

	struct stat buf;
	stat(input_file, &buf);  
	int size = buf.st_size;  //????test?Ĵ?С
	ptr = (unsigned char*) malloc(size);;  //????test??ָ??
	readFile(ptr,fp);
	at_ind = as_load_atind(index_dir, "at");
	fclose(fp);

	//CUT_DEVICE_INIT(argc, argv);
	if (argc > 1)
	{
		printf("%s\n", argv[1]);
		cutilSafeCall(cudaSetDevice(strtol(argv[1], NULL, 10)));
	}
	else //default set to Tesla
	{
		cutilSafeCall(cudaSetDevice(0));
	}

	fpRes = fopen("PPOPP_RES_BINARY.txt", "w+");
	fpResNum = fopen("resNumFromVerify.txt", "w+");
                                

	/*
	for (unsigned int nHashNum = 3; nHashNum <= 9; ++nHashNum)
	{
		printf("==========hashNum %u==========\n", nHashNum);
		GenerateBloom(at_ind, nHashNum, 16);
	}
	*/

	fpCPUPre = fopen("fileCPUPre", "w+");


	
	/*
	 * create stream
	 */
	for (int sid = 0; sid < STREAM_NUM; sid++)
	{
		cudaStreamCreate(&stream[sid]);
	}



	//run
	Run(at_ind, 0, (char *)ptr, (char *)ptr + buf.st_size);



	/*
	 * destroy stream
	 */
	for (int sid = 0; sid < STREAM_NUM; sid++)
	{
		cudaStreamDestroy(stream[sid]);
	}



	fclose(fpCPUPre);




	/*
	for (unsigned int nM= 8; nM <= 64; nM *= 2)
	{
		printf("==========M %u==========\n", nM);
		GenerateBloom(at_ind, 5, nM);
	}
	*/

	fclose(fpRes);
	fclose(fpResNum);

	terminator();
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);
	cudaThreadExit();	
//    cutilExit(argc, argv);
}
