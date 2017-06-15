#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include </usr/include/sys/stat.h>
#include <errno.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

#include <cudpp.h>
#include <cuda_runtime.h>

#include "ParaPFD_Intersection_kernel.cu"

using namespace std;



string dataset_dir = "/media/indexDisk/naiyong/dataset/";
string data_dir = "/media/indexDisk/naiyong/data/ParaPFD/Compression/3/";
string result_dir = "/media/indexDisk/naiyong/result/ParaPFD/Intersection/";



#define ITERATION 2
uint32_t batchID = 0;

unsigned char *ptr = NULL;       // pointer of queryset
at_search_ind_t *patind = NULL;  // pointer of struct for ind1 and ind2


#define BUFFER_SIZE 2048
unsigned char buffer[BUFFER_SIZE];
inline void readFile(unsigned char *ptr, FILE *fp) {
	uint64_t count = 0, bytes = 0;
	while ((bytes = fread(ptr+count, 1, BUFFER_SIZE, fp)) > 0) {
		count += bytes;
	}
}


void as_load_atind(const char *dbi_dir, const char *ind_name) {
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
}


void allocateResource() {
	// host resources
	h_result = (uint32_t *)malloc(baseSize);
	h_ucntResult = (uint32_t *)malloc(5000 * sizeof(uint32_t));
	CUDA_SAFE_CALL(cudaMallocHost((void**)&h_queryID_perBlock, MAX_BLOCK_NUM * sizeof(uint16_t)));
	
	// device resources
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_shortest_lists, baseSize));					
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_segOffset, segOffset_size));				
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_segHead, segHead_size));				
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_segMedian, segMedian_size));


	CUDA_SAFE_CALL(cudaMalloc((void**)&d_isCommon, baseSize));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_scan_odata, baseSize));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_result, baseSize));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_ucntResult, 5000 * sizeof(uint32_t)));	// no more than 5000 queries in a batch, otherwise the response time will be unacceptable
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_lists, patind->sz_ind2));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_queryID_perBlock, MAX_BLOCK_NUM * sizeof(uint16_t)));

	if (!(h_result && d_lists && d_isCommon && d_scan_odata && d_result)) {
		cout << "allocation failed" << endl;
		exit(1);
	}
	

	// transfer lists
	h_lists = (uint32_t *)patind->m_pind2;
	CUDA_SAFE_CALL(cudaMemcpy(d_lists, h_lists, patind->sz_ind2, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_segOffset, h_segOffset, segOffset_size, cudaMemcpyHostToDevice));	
	CUDA_SAFE_CALL(cudaMemcpy(d_segHead, h_segHead, segHead_size, cudaMemcpyHostToDevice));	
	CUDA_SAFE_CALL(cudaMemcpy(d_segMedian, h_segMedian, segMedian_size, cudaMemcpyHostToDevice)); 


	// cudpp alloc
	cudppCreate(&theCudpp);

	config.op = CUDPP_ADD;
	config.datatype = CUDPP_INT;
	config.algorithm = CUDPP_SCAN;
	config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
	CUDPPResult result = cudppPlan(theCudpp, &scanplan, config, 40000000, 1, 0);

	if (CUDPP_SUCCESS != result) {
		cout << "cudpp allocationg falied" << endl;
		exit(1);
	}
}


batchInfo CPUPreprocess(unsigned char *&ptr_in, const unsigned char *ptr_end) {
	batchInfo bi;			
	query_input_t input;
	testCell tc;	
	register uint32_t tcount = 0;

	uint32_t current_blockNum;		
	uint32_t ucntShortest_sum = 0; 
	uint32_t blockNum = 0;
	uint32_t nTotalQueryNum = 0;
	uint32_t queries_offset = 0;	// offset for h_queries

	while (ucntShortest_sum <= Threshold && ptr_in < ptr_end) {
		input.tnum = *ptr_in;
		ptr_in += sizeof(uint32_t);
		memcpy(input.tno, ptr_in, sizeof(uint32_t) * input.tnum);
		ptr_in += sizeof(uint32_t) * input.tnum;


		tc.tcount = input.tnum;
		tcount = tc.tcount;
		for (uint32_t i = 0; i < tcount; ++i) {
			at_term_ind1_t *pind1 = patind->m_pind1 + input.tno[i];
			tc.ucnt[i] = pind1->m_urlcount;  
			tc.uwlist[i] = (pind1->m_off) / sizeof(uint32_t);	
			tc.segNum[i] = *(h_segNum + input.tno[i]);
		}

		// insertion sort
		for (uint32_t i = 1; i < tcount; ++i) {
			uint32_t k = i;
			uint32_t uwlist_tmp = tc.uwlist[i];
			uint32_t ucnt_tmp = tc.ucnt[i];
			uint32_t segNum_tmp = tc.segNum[i];

			while (k && ucnt_tmp < tc.ucnt[k - 1]) {
				tc.uwlist[k] = tc.uwlist[k - 1];
				tc.ucnt[k] = tc.ucnt[k - 1];
				tc.segNum[k] = tc.segNum[k - 1];
				--k;
			};

			if (k != i) {
				tc.uwlist[k] = uwlist_tmp;
				tc.ucnt[k] = ucnt_tmp;
				tc.segNum[k] = segNum_tmp;
			}
		}

		// calculate block number needed by current query
		current_blockNum = tc.ucnt[0] % THREAD_NUM ? tc.ucnt[0] / THREAD_NUM + 1 : tc.ucnt[0] / THREAD_NUM;

		// set host memory for constant memory
		h_startBlockId[nTotalQueryNum] = blockNum;
		h_baseOffset[nTotalQueryNum] = ucntShortest_sum;
		h_queriesOffset[nTotalQueryNum] = queries_offset;

		// copy the query from tc to h_queries
		*(h_queries + queries_offset) = tc.tcount;
		*(h_queries + queries_offset + 1) = ucntShortest_sum;
		memcpy(h_queries + queries_offset + 2, tc.ucnt, sizeof(uint32_t) * tc.tcount);
		memcpy(h_queries + queries_offset + 2 + tc.tcount, tc.uwlist, sizeof(uint32_t) * tc.tcount);
		memcpy(h_queries + queries_offset + 2 + 2 * tc.tcount, tc.segNum, sizeof(uint32_t) * tc.tcount);
		// copy ends

		// set queryID for each block
		for (uint32_t k = 0; k < current_blockNum; ++k) {
			h_queryID_perBlock[blockNum + k] = nTotalQueryNum;	
		}

		// set several local variables for next loop
		blockNum += current_blockNum;
		ucntShortest_sum += tc.ucnt[0];
		queries_offset += tc.tcount * 3 + 2;
		++nTotalQueryNum;
	};


	if (baseSize / sizeof(uint32_t) < ucntShortest_sum) {
		cout << "ucntShortest_sum: " << ucntShortest_sum << "exceeds baseSize: " << baseSize << endl;
		exit(1);
	}

	if (MAX_BLOCK_NUM < blockNum) {
		cout << "blockNum is over " << MAX_BLOCK_NUM  << endl;;
		exit(1);
	}

	// prepare for return value
	bi.blockNum = blockNum;
	bi.constantUsedInByte = (nTotalQueryNum * 3 + queries_offset + batchInfoElementNum) * sizeof(uint32_t);  
	bi.ucntShortest_sum = ucntShortest_sum;
	bi.nTotalQueryNum = nTotalQueryNum;

	// integrate five arrays into h_constant
	memcpy(h_constant, &bi, sizeof(struct batchInfo));
	memcpy(h_constant + batchInfoElementNum, h_startBlockId, sizeof(uint32_t) * nTotalQueryNum);
	memcpy(h_constant + batchInfoElementNum + nTotalQueryNum, h_queriesOffset, sizeof(uint32_t) * nTotalQueryNum);
	memcpy(h_constant + batchInfoElementNum + nTotalQueryNum * 2, h_queries, queries_offset * sizeof(uint32_t));

	return bi;
}


void htodTransfer(const batchInfo &bi) {
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_constant, h_constant, bi.constantUsedInByte));
	CUDA_SAFE_CALL(cudaMemcpy(d_queryID_perBlock, h_queryID_perBlock, bi.blockNum * sizeof(uint16_t), cudaMemcpyHostToDevice));
}


void kernelInvoke(const batchInfo &bi) {
#ifdef debug	
	printf("blockNum:%u\n",bi.blockNum);
#endif

	ParaPFD_Intersection<<<bi.blockNum, THREAD_NUM>>>(d_lists, d_isCommon, d_queryID_perBlock, d_shortest_lists, d_segOffset, d_segHead, d_segMedian);
	CUDA_SAFE_CALL(cudaThreadSynchronize());


#ifdef debug
	// debug; transback h_isCommon and h_scan_odata
	FILE *fisCommon = fopen("isCommon.txt", "a+");
	char isCommon[256];
	uint32_t *h_isCommon = (uint32_t *)malloc(baseSize);
	CUDA_SAFE_CALL(cudaMemcpy(h_isCommon, d_isCommon, baseSize, cudaMemcpyDeviceToHost));
	for (int i = 0; i < bi.ucntShortest_sum; ++i)
	{
		sprintf(isCommon, "i:%d\t%d\n", i, h_isCommon[i]);
		fputs(isCommon, fisCommon);
		fflush(fisCommon);
	}
	free(h_isCommon);
	fclose(fisCommon);
	// debug ends
#endif

	cudppScan(scanplan, d_scan_odata, d_isCommon, bi.ucntShortest_sum);
	CUDA_SAFE_CALL(cudaThreadSynchronize());


#ifdef debug
	//debug; transback h_isCommon and h_scan_odata
	uint32_t *h_scan_odata = (uint32_t *)malloc(baseSize);
	CUDA_SAFE_CALL(cudaMemcpy(h_scan_odata, d_scan_odata, baseSize, cudaMemcpyDeviceToHost));
	for (int i = 0; i < 600; ++i)
	{
		printf("%u ", h_scan_odata[i]);
	}
	printf("\n\n");
	free(h_scan_odata);
	//debug ends
#endif

	saCompact<<<bi.blockNum, THREAD_NUM>>>(d_shortest_lists, d_isCommon, d_scan_odata, d_result, d_queryID_perBlock);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	

#ifdef debug
	if (batchID < 10) {
		cout << "batchID = " << batchID << ", nTotalQueryNum = " << bi.nTotalQueryNum << endl;

		//debug; transback h_isCommon and h_scan_odata
		uint32_t *h_result = (uint32_t *)malloc(baseSize);
		CUDA_SAFE_CALL(cudaMemcpy(h_result, d_result, baseSize, cudaMemcpyDeviceToHost));
		for (int i = 0; i < 600; ++i) {
			printf("%u ", h_result[i]);
			if (i % 100 == 0) 
				cout << endl;
		}
		printf("\n\n");
		free(h_result);
		//debug ends
	}
#endif

	ucntResult<<<bi.nTotalQueryNum/ 64 + 1, 64>>>(d_scan_odata, d_ucntResult);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
}


void dtohTransfer(const batchInfo &bi) {
	CUDA_SAFE_CALL(cudaMemcpy(h_ucntResult, d_ucntResult, sizeof(uint32_t) * bi.nTotalQueryNum, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(h_result, d_result, sizeof(uint32_t) * h_ucntResult[bi.nTotalQueryNum - 1], cudaMemcpyDeviceToHost));
}


void releaseResource() {
	free(h_result);
	h_result = NULL;

	free(h_ucntResult);
	h_ucntResult = NULL;

	free(h_segNum);
	h_segNum = NULL;

	free(h_segOffset);
	h_segOffset = NULL;

	free(h_segHead);
	h_segHead = NULL;

	free(h_segMedian);
	h_segMedian = NULL;


	CUDA_SAFE_CALL(cudaFreeHost(h_queryID_perBlock));
	h_queryID_perBlock = NULL;


	CUDA_SAFE_CALL(cudaFree(d_shortest_lists));
	d_shortest_lists = NULL;

	CUDA_SAFE_CALL(cudaFree(d_segOffset));
	d_segOffset = NULL;

	CUDA_SAFE_CALL(cudaFree(d_segHead));
	d_segHead = NULL;

	CUDA_SAFE_CALL(cudaFree(d_segMedian));
	d_segMedian = NULL;


	CUDA_SAFE_CALL(cudaFree(d_lists));
	d_lists = NULL;

	CUDA_SAFE_CALL(cudaFree(d_isCommon));
	d_isCommon = NULL;

	CUDA_SAFE_CALL(cudaFree(d_scan_odata));
	d_scan_odata = NULL;

	CUDA_SAFE_CALL(cudaFree(d_result));
	d_result = NULL;

	CUDA_SAFE_CALL(cudaFree(d_ucntResult));
	d_ucntResult = NULL;

	CUDA_SAFE_CALL(cudaFree(d_queryID_perBlock));
	d_queryID_perBlock = NULL;


	CUDPPResult res = cudppDestroyPlan(scanplan);
	if (CUDPP_SUCCESS != res) {
		printf("Error destroying CUDPPPlan\n");
		exit(1);
	}

	cudppDestroy(theCudpp);
}


void verify(batchInfo bi, checkSum &cs) {
	for (uint32_t i = 0; i < bi.nTotalQueryNum; ++i) {
		if (i) { // all queries except the first one
			if (h_ucntResult[i] - h_ucntResult[i - 1]) {
				uint32_t resultNum = h_ucntResult[i] - h_ucntResult[i - 1];
				cs.checkSum2 += resultNum;
				cs.checkSum3 ^= h_result[h_ucntResult[i - 1]];
				cs.checkSum3 ^= h_result[h_ucntResult[i - 1] + resultNum - 1];
				cs.checkSum3 ^= h_result[h_ucntResult[i - 1] + resultNum / 2];
			}
			else 
				++cs.checkSum1;
		}
		else { // the first query
			if (h_ucntResult[i]) { 
				uint32_t resultNum = h_ucntResult[i];
				cs.checkSum2 += resultNum;
				cs.checkSum3 ^= h_result[0];
				cs.checkSum3 ^= h_result[resultNum - 1];
				cs.checkSum3 ^= h_result[resultNum / 2];
			}
			else 
				++cs.checkSum1;
		}
	}
}


template <uint64_t beginThreshold, uint64_t endThreshold>
void Run(const string &dataset, unsigned char *ptr_in, unsigned char *ptr_end) {
	string result = result_dir + dataset + "_ParaPFD_Intersection_FRAC.txt";
	ofstream ofsresult(result.c_str());
	ofsresult << "response: ms/batch" << endl
		      << "throughput: queries/s" << endl
			  << "time: ms" << endl << endl;
	ofsresult << "threshold\tresponse\tthroughput\tcpu\tkernel\ttransfer" << endl;


	batchInfo bi;
	checkSum cs;
	unsigned char *ptr_in_old = ptr_in, *ptr_end_old = ptr_end;

	cudaEvent_t start, stop;
	float time_CPU = 0, time_htod = 0, time_kernel = 0, time_dtoh = 0;
	float elapsedTime = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	allocateResource();
	
	for (Threshold = beginThreshold; Threshold <= endThreshold; Threshold *= 2) {
		cout << "begin calculating..." << endl;
		if (Threshold < 1024 * 1024) 
			cout << "Threshold: " << Threshold / 1024 << "K";
		else 
			cout << "Threshold: " << Threshold / (1024 * 1024) << "M";

		cout << ", THREAD_NUM: " << THREAD_NUM << endl;

		time_CPU = 0;
		time_htod = 0;
		time_dtoh = 0;
		time_kernel = 0;

		uint32_t nTotalQueryNum = 0;
		for (int iteration = 0; iteration < ITERATION; ++iteration) {
			memset(&cs, 0, sizeof(struct checkSum));

			ptr_in = ptr_in_old;
			ptr_end = ptr_end_old;

			batchID = 0;
			nTotalQueryNum = 0;
			while (ptr_in < ptr_end) {
				cudaEventRecord(start, 0);

				bi = CPUPreprocess(ptr_in, ptr_end);

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsedTime, start, stop);
				time_CPU += elapsedTime;


				cudaEventRecord(start, 0);

				htodTransfer(bi);

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsedTime, start, stop);
				time_htod += elapsedTime;


				cudaEventRecord(start, 0);

				kernelInvoke(bi);

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsedTime, start, stop);
				time_kernel += elapsedTime;


				cudaEventRecord(start, 0);

				dtohTransfer(bi);

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsedTime, start, stop);
				time_dtoh += elapsedTime; 


				verify(bi, cs);


				++batchID;
				nTotalQueryNum += bi.nTotalQueryNum;
			}

		}

		time_CPU /= ITERATION;
		time_htod /= ITERATION;
		time_kernel /= ITERATION;
		time_dtoh /= ITERATION;

		float time_total = time_CPU + time_htod + time_kernel + time_dtoh;

		cout << "number of queries: " << nTotalQueryNum << endl;
		cout << "number of batches: " << batchID << endl;
		cout << "CPU time: " <<  time_CPU << "ms" << endl
			 << "htod time: " <<  time_htod << "ms" << endl
			 << "kernel time: " <<  time_kernel << "ms" << endl
			 << "dtoh time: " <<  time_dtoh << "ms" << endl
			 << "total time: " <<  time_total << "ms" << endl;

		cout << "*********************************************" << endl
			 << "checkSum: " << hex << (cs.checkSum1 ^ cs.checkSum2 ^ cs.checkSum3) << endl
			 << "checkSum1: " << dec << cs.checkSum1 << "\t"
			 << "checkSum2: " << cs.checkSum2 << "\t"
			 << "checkSum3: " << cs.checkSum3 << endl
			 << "*********************************************" << endl << endl;


		float fResponse = time_total / batchID;
		float fThroughput = nTotalQueryNum / time_total * 1000;
		if (Threshold < 1024 * 1024) 
			ofsresult << Threshold / 1024 << "K\t";
		else 
			ofsresult << Threshold / (1024 * 1024) << "M\t";

		ofsresult << fResponse << "\t"
			      << fThroughput << "\t"
			      << time_CPU << "\t"
			      << time_kernel << "\t"
			      << time_htod + time_dtoh << endl;

		sleep(5);
	}

	releaseResource();


	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	ofsresult.close();
}


void terminator() {
	if (ptr != NULL) 
		free(ptr);
	if (patind->m_pind1 != NULL) 
		free(patind->m_pind1);
	if (patind->m_pind2 != NULL) 
		free(patind->m_pind2);
	if (patind != NULL) 
		free(patind);

	ptr = NULL;
	patind->m_pind1 = NULL;
	patind->m_pind2 = NULL;
	patind = NULL;
}




void runTest(int argc, char** argv) {
	if (argc < 2) {
		cout << "wrong number of arguments" << endl;
		exit(1);
	}
	string dataset = argv[1];
	cout << "ParaPFD_Intersection_FRAC" << endl;
	cout << "dataset = " << dataset << endl;

	if (argc > 2) {
		CUDA_SAFE_CALL(cudaSetDevice(strtol(argv[2], NULL, 10)));
	}
	else {
		CUDA_SAFE_CALL(cudaSetDevice(0));
	}

	string queryset = dataset_dir + dataset + "/" + dataset + ".query";

	string segNum_file = data_dir + dataset + ".segNum"; 
	string segOffset_file = data_dir + dataset + ".segOffset";
	string segHead_file = data_dir + dataset + ".segHead"; 
	string segMedian_file = data_dir + dataset + ".segMedian";

	struct stat buf;

	FILE *fsegNum = fopen(segNum_file.c_str(), "rb");
	if (!fsegNum) {
		cout << segNum_file << " open failed\t" << "err code:" << errno << endl;
		exit(1);
	}
	stat(segNum_file.c_str(), &buf);
	h_segNum = (uint32_t*)malloc(buf.st_size);		
	readFile((unsigned char*)h_segNum, fsegNum);
	fclose(fsegNum);

	FILE *fsegOffset = fopen(segOffset_file.c_str(), "rb");
	if (!fsegOffset) {
		cout << segOffset_file << " open failed\t" << "err code:" << errno << endl;
		exit(1);
	}
	stat(segOffset_file.c_str(), &buf);
	segOffset_size = buf.st_size;				//size of segOffset
	h_segOffset = (uint32_t*)malloc(segOffset_size);	//pointer of segOffset
	readFile((unsigned char*)h_segOffset, fsegOffset);
	fclose(fsegOffset);

	FILE *fsegHead = fopen(segHead_file.c_str(), "rb");
	if (!fsegHead) {
		cout << segHead_file << " open failed\t" << "err code:" << errno << endl;
		exit(1);
	}
	stat(segHead_file.c_str(), &buf);
	segHead_size = buf.st_size;								//size of segHead	
	h_segHead = (uint32_t*)malloc(segHead_size);		//pointer of segHead
	readFile((unsigned char*)h_segHead, fsegHead);
	fclose(fsegHead);

	FILE *fsegMedian = fopen(segMedian_file.c_str(), "rb");
	if (!fsegMedian) {
		cout << segMedian_file << " open failed\t" << "err code:" << errno << endl;
		exit(1);
	}
	stat(segMedian_file.c_str(), &buf);
	segMedian_size = buf.st_size;
	h_segMedian = (uint32_t *)malloc(segMedian_size);
	readFile((unsigned char*)h_segMedian, fsegMedian);
	fclose(fsegMedian);

	FILE *fquery = fopen(queryset.c_str(),"rb");
	if (!fquery) {
		cout << queryset << " open failed\t" << "err code:" << errno << endl;
		exit(1);
	}
	stat(queryset.c_str(), &buf);  
	uint32_t querysize = buf.st_size;
	ptr = (unsigned char *)malloc(querysize);
	readFile(ptr, fquery);
	fclose(fquery);


	as_load_atind(data_dir.c_str(), dataset.c_str());


	// <1K, 2M>
	Run<1024, 2 * 1024 * 1024>(dataset, ptr, ptr + querysize);

	terminator();
}


int main(int argc, char** argv) {
    runTest(argc, argv);
}
