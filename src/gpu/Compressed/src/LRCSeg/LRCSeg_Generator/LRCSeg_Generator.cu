#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include </usr/include/sys/stat.h>
#include <errno.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "LRCSeg_Generator_kernel.cu"

using namespace std;


string dataset_dir = "/media/indexDisk/naiyong/dataset/";
string regressioninfo_dir = "/media/indexDisk/naiyong/data/LRCSeg/Generator/";
string result_dir = "/media/indexDisk/naiyong/result/LRCSeg/Generator/";

ofstream ofsresult;
const uint32_t interval = 100 * 1000;
uint32_t intervalNum = MAX_LIST_LEN / interval;


#define MAX_TOTAL_BLOCK_NUM 500000000 

at_search_ind_t *patind = NULL;  // pointer of struct for ind1 and ind2


inline uint64_t div_roundup(uint64_t v, uint32_t divisor) {
	    return (v + (divisor - 1)) / divisor;
}

inline uint32_t gccbits(const uint32_t v) {
	return v == 0 ? 0 : 32 - __builtin_clz(v);
}

inline uint32_t maxbits(const uint32_t *in, uint32_t nvalue) {
	uint32_t accumulator = 0;
	for (uint32_t i = 0; i < nvalue; ++i) {
		accumulator |= in[i];
	}
	return gccbits(accumulator);
}


#define BUFFER_SIZE 2048
unsigned char buffer[BUFFER_SIZE];
inline void readFile(unsigned char* ptr, FILE* fp){
	uint64_t count = 0, bytes;
	while ((bytes = fread(ptr+count, 1, BUFFER_SIZE, fp))>0) {
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

	// ind2
	sprintf (file_name, "%s%s.ind2", dbi_dir, ind_name);
	cout << "reading " << file_name << endl;
	stat(file_name, &buf);
	patind->sz_ind2 = buf.st_size;
	patind->fd_ind2 = fopen(file_name,"rb");
	patind->m_pind2 = (unsigned char*) malloc(buf.st_size);
	readFile(patind->m_pind2 , patind->fd_ind2);
}




void generateInd2Distance(const string &dataset) {
	struct stat buf;

	// read at_regression_info file
	string strRegressionInfo = regressioninfo_dir + dataset + ".regression_info";
	FILE *fRegressionInfo = fopen(strRegressionInfo.c_str(), "rb");
	if (!fRegressionInfo) {
		cout << "read binary file " << strRegressionInfo << " failed;error:" << endl;
		exit(1);
	}

	stat(strRegressionInfo.c_str(), &buf);
	uint32_t szRegressionInfo = buf.st_size;
	regression_info_t *h_regressioninfo = (regression_info_t *)malloc(szRegressionInfo);
	if (h_regressioninfo == 0) {
		perror("alloc error\n");
		exit(1);
	}
	readFile((unsigned char *)h_regressioninfo, fRegressionInfo);
	// read ends


	// create ind2_distance
	string strDistance = regressioninfo_dir + dataset + ".ind2_distance";
	FILE *fDistance = fopen(strDistance.c_str(), "wb+");
	if (!fDistance) {
		cout << strDistance << " create failed" << endl;
		exit(1);
	}
	// create ends

	uint64_t ind2Size = patind->sz_ind2;
	uint32_t *VDs = (uint32_t *)malloc(ind2Size);
	if (VDs == 0) {
		perror("VDs error\n");
		exit(1);
	}


	// ind2_distance_base
	string strBase = regressioninfo_dir + dataset + ".ind2_distance_base";
	FILE *fBase = fopen(strBase.c_str(), "wb+");
	if (!fBase) {
		cout << strBase << " create failed" << endl;
		exit(1);
	}

    int *pBase = (int *)malloc(MAX_TOTAL_BLOCK_NUM * sizeof(int));
	if (pBase == 0) {
		perror("ind2_distance_base error\n");
		exit(1);
	}


	CUDA_SAFE_CALL(cudaMalloc((void**)&d_list, MAX_LIST_LEN * sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_distance, MAX_LIST_LEN * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_regressioninfo, szRegressionInfo));

	CUDA_SAFE_CALL(cudaMemcpy(d_regressioninfo, h_regressioninfo, szRegressionInfo, cudaMemcpyHostToDevice));


	vector<uint64_t> nBitsNeeded(intervalNum, 0);
	vector<uint64_t> length(intervalNum, 0);

	uint64_t totalBlockNum = 0;

	uint32_t nListNum = patind->sz_ind1 / sizeof(at_term_ind1_t);
	for (uint32_t i = 0; i < nListNum; ++i) {
		// progress
		if (!(i % (nListNum / 10))) {
			printf("%f%%...\n", ((float)i) / nListNum * 100);
		}

		at_term_ind1_t *pind1 = patind->m_pind1 + i;
		uint32_t nListLen = pind1->m_urlcount;
		uint32_t *h_list = (uint32_t *)(patind->m_pind2 + pind1->m_off);
		int *h_distance = (int *)(VDs + pind1->m_off / sizeof(uint32_t));


		CUDA_SAFE_CALL(cudaMemcpy(d_list, h_list, nListLen * sizeof(uint32_t), cudaMemcpyHostToDevice));

		LRCSeg_Generator<<<nListLen / THREAD_NUM + 1, THREAD_NUM>>>(d_list, d_regressioninfo, d_distance, nListLen, i);
		CUDA_SAFE_CALL(cudaThreadSynchronize());

		CUDA_SAFE_CALL(cudaMemcpy(h_distance, d_distance, nListLen * sizeof(int), cudaMemcpyDeviceToHost));


		uint32_t idx = nListLen / interval;
		length[idx] += nListLen;

		uint32_t kBlockSize = THREAD_NUM;
		uint32_t kBlockNum = div_roundup(nListLen, kBlockSize);

		// all blocks except the last one
		for (uint32_t j = 0; j < kBlockNum - 1; ++j) {
			uint32_t offset = j * kBlockSize;

			// get minimum
			int nBase = h_distance[offset];
			for (uint32_t k = 1; k < kBlockSize; ++k) {
				if (nBase > h_distance[offset + k])
					nBase = h_distance[offset + k];
			}
			pBase[totalBlockNum + j] = nBase;

	        // reduction	
			for (uint32_t k = 0; k < kBlockSize; ++k) 
				h_distance[offset + k] -= nBase;

			// stat
			uint32_t mb = maxbits(reinterpret_cast<const uint32_t *>(h_distance + offset), kBlockSize);
			uint32_t nwords = div_roundup(kBlockSize * mb, 32);
			nBitsNeeded[idx] += nwords * 32;
		}

		// tail block
		uint32_t offset = (kBlockNum - 1) * kBlockSize;
		int nBase = h_distance[offset];
		for (uint32_t j = offset + 1; j < nListLen; ++j) {
			if (nBase > h_distance[j])
				nBase = h_distance[j];
		}
		pBase[totalBlockNum + kBlockNum - 1] = nBase;

		for (uint32_t j = offset; j < nListLen; ++j)
			h_distance[j] -= nBase;

		// stat
		uint32_t mb = maxbits(reinterpret_cast<const uint32_t *>(h_distance + offset), nListLen - offset);
		uint32_t nwords = div_roundup((nListLen - offset) * mb, 32);
		nBitsNeeded[idx] += nwords * 32;
		

		totalBlockNum += kBlockNum; // update totalBlockNum
	}

	fwrite(pBase, sizeof(int), totalBlockNum, fBase);
	fflush(fBase);

	fwrite(VDs, ind2Size, 1, fDistance);
	fflush(fDistance);

	ofsresult << "length\t" << "Compression (bits/interger)" << endl;
	for (uint32_t i = 0; i < intervalNum; ++i) {
		if (length[i] > 0) {
			nBitsNeeded[i] /= length[i];
			ofsresult << "[" << (i * interval) / 1000 << "K, "
				      << ((i+1) * interval) / 1000 << "K)\t"
					  << nBitsNeeded[i] << endl;
		}
	}

	CUDA_SAFE_CALL(cudaFree(d_list));
	CUDA_SAFE_CALL(cudaFree(d_distance));
	CUDA_SAFE_CALL(cudaFree(d_regressioninfo));

	free(h_regressioninfo);
	free(VDs);
	free(pBase);

	fclose(fRegressionInfo);
	fclose(fDistance);
	fclose(fBase);
}


// average edition
void getRegressionInfo(const uint32_t *pList, uint32_t nLen, regression_info_t &regressionInfo, stat_info_t &statInfo) {
	if (nLen == 0) {
		regressionInfo.fSlope = 1;
		regressionInfo.fIntercept = 0;
		regressionInfo.nRangeLeft = 0;
		regressionInfo.nRangeRight = 0;

		statInfo.dRSquare = 1;
		statInfo.dContractionRatio = 0;

		return;
	}

	if (nLen == 1) {
		regressionInfo.fSlope = 1;
		regressionInfo.fIntercept = pList[0];
		regressionInfo.nRangeLeft = 0;
		regressionInfo.nRangeRight = 0;

		statInfo.dRSquare = 1;
		statInfo.dContractionRatio = 0;

		return;
	}


	// average
	double dXA = 0, dYA = 0;
	for (uint32_t i = 0; i < nLen; ++i) {
		dXA += i;
		dYA += pList[i];
	}
	dXA /= nLen;
	dYA /= nLen;

	// diff sum
	double dDiffSumX = 0, dDiffSumXY = 0, dDiffSumY = 0;
	for (uint32_t i = 0; i < nLen; ++i) {
		double dValueY = (double)(pList[i]);
		dDiffSumX += (double)((i - dXA) * (i - dXA));
		dDiffSumXY += (double)((i - dXA) * (dValueY - dYA));
		dDiffSumY += (double)((dValueY - dYA) * (dValueY - dYA));
	}

	// slope and intercept
	regressionInfo.fSlope = (float)(dDiffSumXY / dDiffSumX);
	regressionInfo.fIntercept = (float)(dYA - regressionInfo.fSlope * dXA);

	// furthest points
	float fRangeLeft = 0, fRangeRight = 0;
	for (uint32_t i = 0; i < nLen; ++i) {
		float fPrivateX = (pList[i] - regressionInfo.fIntercept) / regressionInfo.fSlope;
		if (fPrivateX - i > fRangeLeft) {
			fRangeLeft = fPrivateX - i;
		}
		else if (fPrivateX - i < fRangeRight) {
			fRangeRight = fPrivateX - i;
		}
	}
	// safe search range
	regressionInfo.nRangeLeft = (uint32_t)(fRangeLeft) + 1;
	regressionInfo.nRangeRight = (uint32_t)(0 - fRangeRight) + 1;


	// stat info
	statInfo.dRSquare = (dDiffSumXY * dDiffSumXY) / (dDiffSumX * dDiffSumY);
	statInfo.dContractionRatio = double(regressionInfo.nRangeLeft + regressionInfo.nRangeRight) / nLen;
}


// generate detailed regression data
void RunGetRegressionInfo(const string &dataset) {
	// open file
	string strRegressionInfo = regressioninfo_dir + dataset + ".regression_info";
	FILE *fRegressionInfo = fopen(strRegressionInfo.c_str(), "wb+");
	if (!fRegressionInfo) {
		cout << "open binary file " << strRegressionInfo << " failed; error:" << endl;
		exit(1);
	}


	vector<stat_info_t> statInfo(intervalNum);
	memset(&statInfo[0], 0, intervalNum * sizeof(stat_info_t));
	vector<uint32_t> count(intervalNum, 0);

	uint32_t nListNum = patind->sz_ind1 / sizeof(at_term_ind1_t);
	vector<regression_info_t> regressionInfo(nListNum);

	for (uint32_t i = 0; i < nListNum; ++i) {
		at_term_ind1_t *pind1 = patind->m_pind1 + i;
		uint32_t nListLen = pind1->m_urlcount;
		uint32_t *pList = (uint32_t *)(patind->m_pind2 + pind1->m_off);

		stat_info_t thisStatInfo;
		memset(&thisStatInfo, 0, sizeof(stat_info_t));
		getRegressionInfo(pList, nListLen, regressionInfo[i], thisStatInfo);

		if (nListLen > 1) {
			uint32_t idx = nListLen / interval;
			++count[idx];
			statInfo[idx].dRSquare += thisStatInfo.dRSquare;
			statInfo[idx].dContractionRatio += thisStatInfo.dContractionRatio;
		}

	}
	// write result to binary file
	fwrite(&regressionInfo[0], sizeof(regression_info_t), nListNum, fRegressionInfo);
	fflush(fRegressionInfo);


	ofsresult << "length\t" << "count\t"
			  << "R^2\t" << "Ratio" << endl;
	for (uint32_t i = 0; i < intervalNum; ++i) {
		if (count[i] > 0) {
			statInfo[i].dRSquare /= count[i];
			statInfo[i].dContractionRatio /= count[i];
			ofsresult << "[" << (i * interval) / 1000 << "K, "
				      << ((i+1) * interval) / 1000 << "K)\t"
					  << count[i] << "\t" 
					  << statInfo[i].dRSquare << "\t"
					  << statInfo[i].dContractionRatio << endl;
		}
	}
	ofsresult << endl;


	fclose(fRegressionInfo);
}


// Do some free operations
void terminator(){
	if (patind->m_pind1 != NULL) 
		free(patind->m_pind1);
	if (patind->m_pind2 != NULL) 
		free(patind->m_pind2);
	if (patind != NULL) 
		free(patind);

	patind->m_pind1 = NULL;
	patind->m_pind2 = NULL;
	patind = NULL;
}


void runTest(int argc, char** argv) {
	if (argc < 2) {
		std::cout << "wrong number of arguments" << std::endl;
		exit(1);
	}
	string dataset = argv[1];
	cout << "LRCSeg_Generator" << endl;
	cout << "dataset = " << dataset << endl;

	if (argc > 2) {
		CUDA_SAFE_CALL(cudaSetDevice(strtol(argv[2], NULL, 10)));
	}
	else // default set to Tesla
	{
		CUDA_SAFE_CALL(cudaSetDevice(0));
	}

	string index_dir = dataset_dir + dataset + "/";
	as_load_atind(index_dir.c_str(), dataset.c_str());


	string result = result_dir + dataset + "_LRCSeg_Generator.txt";
	ofsresult.open(result.c_str());

	cout << "Getting regression info..." << endl;
	RunGetRegressionInfo(dataset);

	cout << "Generating ind2 distance..." << endl;
	generateInd2Distance(dataset);

	ofsresult.close();


	terminator();
}


int main(int argc, char **argv) {
    runTest(argc, argv);

	return 0;
}
