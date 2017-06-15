#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include </usr/include/sys/stat.h>
#include <errno.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>


#include "HS_SegLRC_Generator_kernel.cu"

using namespace std;


string dataset_dir = "/media/indexDisk/naiyong/dataset/";
string regressioninfo_dir = "/media/indexDisk/naiyong/data/HS_SegLRC/Generator/";
string result_dir = "/media/indexDisk/naiyong/result/HS_SegLRC/Generator/";


#define MAX_TOTAL_BLOCK_NUM 500000000 

ofstream ofsresult;
const uint32_t interval = 100 * 1000;
uint32_t intervalNum = MAX_LIST_LEN / interval;


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
 
	// read regression_infoe
	string strRegressionInfo = regressioninfo_dir + dataset + ".regression_info";
	FILE *fRegressionInfo = fopen(strRegressionInfo.c_str(), "rb");
	if (!fRegressionInfo) {
		cout << "read binary file " << strRegressionInfo << " failed;error:" << endl;
		exit(1);
	}

	stat(strRegressionInfo.c_str(), &buf);
	uint32_t szRegresstionInfo = buf.st_size;
	regression_info_t *h_regressioninfo = (regression_info_t*)malloc(szRegresstionInfo);
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

	uint64_t ind2Size = patind->sz_ind2;
	uint32_t *VDs = (uint32_t *)malloc(ind2Size);
	if (VDs == 0) {
		perror("VDs error\n");
		exit(1);
	}
	

	// create base
	string strBase = regressioninfo_dir + dataset + ".ind2_distance_base";
	FILE *fBase = fopen(strBase.c_str(), "wb+");
	if (!fBase) {
		cout << strBase << " create failed" << endl;
		exit(1);
	}
	int *pBase = (int *)malloc(MAX_TOTAL_BLOCK_NUM * sizeof(int));
	if (pBase == 0) {
		perror("pBase error\n");
		exit(1);
	}
	// create ends


	// read HashSeg_offset
	string strHashOffset = regressioninfo_dir + dataset + ".HashSeg_offset";
	FILE *fHashOffset = fopen(strHashOffset.c_str(), "rb");
	if (!fHashOffset) {
		cout << "open binary file " << strHashOffset << " failed;error:" << endl;
		exit(1);
	}

	stat(strHashOffset.c_str(), &buf);
	uint32_t *hashOffset = (uint32_t *)malloc(buf.st_size);
	if (hashOffset == 0) {
		perror("hashOffset error\n");
		exit(1);
	}
	readFile((unsigned char*)hashOffset, fHashOffset);
	// read ends


	// read HashSeg_info
	string strHashInfo = regressioninfo_dir + dataset + ".HashSeg_info";
	FILE *fHashInfo = fopen(strHashInfo.c_str(), "rb");
	if (!fHashInfo) {
		cout << "open binary file " << strHashInfo << " failed;error: " << endl;
		exit(1);
	}

	stat(strHashInfo.c_str(), &buf);
	uint32_t szHashInfo = buf.st_size;
	uint32_t *h_hashinfo = (uint32_t*)malloc(szHashInfo);
	if (h_hashinfo == 0) {
		perror("hashInfo error\n");
		exit(1);
	}
	readFile((unsigned char*)h_hashinfo, fHashInfo);
	// read ends


	CUDA_SAFE_CALL(cudaMalloc((void**)&d_list, MAX_LIST_LEN * sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_distance, MAX_LIST_LEN * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_hashinfo, szHashInfo));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_regressioninfo, szRegresstionInfo));

	CUDA_SAFE_CALL(cudaMemcpy(d_hashinfo, h_hashinfo, szHashInfo, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_regressioninfo, h_regressioninfo, szRegresstionInfo, cudaMemcpyHostToDevice));


	// stats
	vector<uint64_t> nBitsNeeded(intervalNum, 0);
	vector<uint64_t> length(intervalNum, 0);


	uint64_t totalBucketNum = 0;
	uint32_t nListNum = patind->sz_ind1 / sizeof(at_term_ind1_t);
	for (uint32_t i = 0; i < nListNum; ++i) {
		// progress
		if (!(i % (nListNum / 10))) {
			printf("%f%%...\n", ((float)i) / nListNum * 100);
		}

		at_term_ind1_t *pind1 = patind->m_pind1 + i;
		uint32_t nListLen = pind1->m_urlcount;
		uint32_t *pList = (uint32_t *)(patind->m_pind2 + pind1->m_off);
		int *h_distance = (int *)(VDs + pind1->m_off / sizeof(uint32_t));

		
		CUDA_SAFE_CALL(cudaMemcpy(d_list, pList, nListLen * sizeof(uint32_t), cudaMemcpyHostToDevice));

		HS_SegLRC_Generator<<<nListLen / THREAD_NUM + 1, THREAD_NUM>>>(d_list, nListLen, d_regressioninfo, totalBucketNum, d_distance, d_hashinfo, hashOffset[i]);
		CUDA_SAFE_CALL(cudaThreadSynchronize());

		CUDA_SAFE_CALL(cudaMemcpy(h_distance, d_distance, nListLen * sizeof(int), cudaMemcpyDeviceToHost));


		uint32_t idx = nListLen / interval;
		length[idx] += nListLen;


		uint32_t *thisHashInfo = h_hashinfo + hashOffset[i];
		uint32_t kBucketNum = thisHashInfo[1];
		thisHashInfo += 2;

		for (uint32_t j = 0; j < kBucketNum; ++j) {
			uint32_t offset = thisHashInfo[j];
			uint32_t kBucketSize = thisHashInfo[j + 1] - thisHashInfo[j];

			int nBase = h_distance[offset];
			for (uint32_t k = 1; k < kBucketSize; ++k) {
				if (nBase > h_distance[offset + k])
					nBase = h_distance[offset + k];
			}
			pBase[totalBucketNum + j] = nBase;

			for (uint32_t k = 0; k < kBucketSize; ++k) {
				h_distance[offset + k] -= nBase;
			}

			// stat
			uint32_t mb = maxbits(reinterpret_cast<const uint32_t *>(h_distance + offset), kBucketSize);
			uint32_t nwords = div_roundup(kBucketSize * mb, 32);
			nBitsNeeded[idx] += nwords * 32;
		}


		totalBucketNum += kBucketNum;
	}
	fwrite(pBase, sizeof(int), totalBucketNum, fBase);
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
	CUDA_SAFE_CALL(cudaFree(d_regressioninfo));
	CUDA_SAFE_CALL(cudaFree(d_distance));
	CUDA_SAFE_CALL(cudaFree(d_hashinfo));


	free(h_regressioninfo);
	free(VDs);
	free(h_hashinfo);
	free(pBase);
	free(hashOffset);

	fclose(fRegressionInfo);
	fclose(fDistance);
	fclose(fBase);
	fclose(fHashOffset);
	fclose(fHashInfo);
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
		cout << "open binary file " << strRegressionInfo << " failed;error:" << endl;
		exit(1);
	}
	regression_info_t *regressionInfo = (regression_info_t *)malloc(MAX_TOTAL_BLOCK_NUM * sizeof(regression_info_t));
	

	struct stat buf;

	string strHashOffset = regressioninfo_dir + dataset + ".HashSeg_offset";
	FILE *fHashOffset = fopen(strHashOffset.c_str(), "rb");
	if (!fHashOffset) {
		cout << "open binary file " << strHashOffset << " failed;error;" << endl;
		exit(1);
	}
	stat(strHashOffset.c_str(), &buf);
	uint32_t *hashOffset = (uint32_t*)malloc(buf.st_size);
	readFile((unsigned char*)hashOffset, fHashOffset);


	string strHashInfo = regressioninfo_dir + dataset + ".HashSeg_info";
	FILE *fHashInfo = fopen(strHashInfo.c_str(), "rb");
	if (!fHashInfo) {
		cout << "open binary file " << strHashInfo << " failed;error" << endl;
		exit(1);
	}
	stat(strHashInfo.c_str(), &buf);
	uint32_t *hashInfo = (uint32_t*)malloc(buf.st_size);
	readFile((unsigned char*)hashInfo, fHashInfo);


	// stats
	vector<stat_info_t> statInfo(intervalNum);
	memset(&statInfo[0], 0, intervalNum * sizeof(stat_info_t));
	vector<uint32_t> count(intervalNum, 0);


	const uint32_t kMaxBucketSize = 1000, kIntervalSize = 50;
	const uint32_t kIntervalNum = kMaxBucketSize / kIntervalSize;
	uint32_t kEmptyBucketNum = 0;
	vector<uint64_t> countBucketSize(kIntervalNum + 1);

	
	uint64_t totalBucketNum = 0;

	uint32_t nListNum = patind->sz_ind1 / sizeof(at_term_ind1_t);
	for (uint32_t i = 0; i < nListNum; ++i) {
		at_term_ind1_t *pind1 = patind->m_pind1 + i;
		uint32_t nListLen = pind1->m_urlcount;
		uint32_t *pList = (uint32_t *)(patind->m_pind2 + pind1->m_off);
		

		uint32_t *thisHashInfo = hashInfo + hashOffset[i];
		uint32_t kBucketNum = thisHashInfo[1];
		thisHashInfo += 2;

		vector<stat_info_t> thisStatInfo(kBucketNum);
		for (uint32_t j = 0; j < kBucketNum; ++j) {
			uint32_t offset = thisHashInfo[j];
			uint32_t kBucketSize = thisHashInfo[j + 1] - thisHashInfo[j];
			getRegressionInfo(pList + offset, kBucketSize, regressionInfo[totalBucketNum + j], thisStatInfo[j]);
		}


		// stats
		if (nListLen > 1) {
			uint32_t idx = nListLen / interval;
			++count[idx];

			uint32_t kNonEmptyBucketNum = 0;
			double dSumRSquare = 0, dSumContractionRatio = 0;
			for (uint32_t j = 0; j < kBucketNum; ++j) {
				uint32_t kBucketSize = thisHashInfo[j + 1] - thisHashInfo[j];
				if (kBucketSize > 0) {
					++kNonEmptyBucketNum;

					dSumRSquare += thisStatInfo[j].dRSquare;
					dSumContractionRatio += thisStatInfo[j].dContractionRatio;

					if (kBucketSize < kMaxBucketSize) 
						++countBucketSize[kBucketSize / kIntervalSize];
					else
						++countBucketSize[kIntervalNum];
				}
				else 
					++kEmptyBucketNum;
			}

			statInfo[idx].dRSquare += dSumRSquare / kNonEmptyBucketNum;
			statInfo[idx].dContractionRatio += dSumContractionRatio / kNonEmptyBucketNum;
		}


		totalBucketNum += kBucketNum;
	}
	fwrite(regressionInfo, sizeof(regression_info_t), totalBucketNum, fRegressionInfo);
	fflush(fRegressionInfo);

	ofsresult << "kEmptyBucketNum = " << kEmptyBucketNum << endl;
	ofsresult << "kBucketSize\tNumber of Buckets" << endl;
	for (uint32_t i = 0; i <= kIntervalNum; ++i) {
		ofsresult << (i == 0 ? '(' : '[') << i * kIntervalSize << ", "
			<< (i == kIntervalNum ? MAX_LIST_LEN : (i+1) * kIntervalSize) << ")\t"
			<< countBucketSize[i] << endl;
	}
	ofsresult << endl;


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


	free(regressionInfo);
	free(hashOffset);
	free(hashInfo);

	fclose(fRegressionInfo);
	fclose(fHashOffset);
	fclose(fHashInfo);
}


uint32_t kExceptedBucketSize = 256;

void RunGetHashInfo(const string &dataset) {
	uint32_t nListNum = patind->sz_ind1 / sizeof(at_term_ind1_t);

	string strHashInfo = regressioninfo_dir + dataset + ".HashSeg_info";
	FILE *fHashInfo = fopen(strHashInfo.c_str(), "wb+");
	if (!fHashInfo) {
		cout << "can't open " << strHashInfo << endl;
		exit(1);
	}

	uint32_t *hashInfo = (uint32_t*)malloc(MAX_TOTAL_BLOCK_NUM * sizeof(uint32_t));
	if (!hashInfo) {
		perror("hashInfo allocate failed");
		exit(1);
	}


	string strHashOffset = regressioninfo_dir + dataset + ".HashSeg_offset";
	FILE *fHashOffset = fopen(strHashOffset.c_str(), "wb+");
	if (!fHashOffset) {
		cout << "can't open " << strHashOffset << endl;
		exit(1);
	}

	uint32_t *hashOffset = (uint32_t*)malloc(nListNum * sizeof(uint32_t));
	if (!hashOffset) {
		perror("hashOffset allocate failed");
		exit(1);
	}


	uint32_t nSegOffset = 0;
	for (uint32_t i = 0; i < nListNum; ++i) {
		at_term_ind1_t *pind1 = patind->m_pind1 + i;
		uint32_t nListLen = pind1->m_urlcount;
		uint32_t *pList = (uint32_t *)(patind->m_pind2 + pind1->m_off);

		hashOffset[i] = nSegOffset;
		uint32_t *thisHashInfo = hashInfo + nSegOffset;

		uint32_t nSegNum = 0;

		const uint32_t kMaxBits = gccbits(pList[nListLen - 1]);
		if (nListLen <= kExceptedBucketSize) {
			thisHashInfo[0] = kMaxBits;  // hash function
			thisHashInfo[1] = 1;         // number of buckets
			thisHashInfo[2] = 0;         // offset
			thisHashInfo[3] = nListLen;  // list length 

			nSegOffset += 4;
			nSegNum = 1;
		}
		else {
			uint32_t kExpectedBucketNum = div_roundup(nListLen, kExceptedBucketSize);
			uint32_t shift = kMaxBits - gccbits(kExpectedBucketNum);

			// hash function
			thisHashInfo[0] = shift;
			
			// travel the list
			uint32_t nCurSegID = 0, nSegID = 0;
			thisHashInfo[2 + nCurSegID] = 0;
			for (uint32_t docIndex = 0; docIndex < nListLen; ++docIndex) {
				uint32_t nDocID = pList[docIndex];
				nSegID = nDocID >> shift;
				if (nSegID > nCurSegID) { // a new seg head is found
					while (nCurSegID < nSegID) {
						++nCurSegID;
						thisHashInfo[2 + nCurSegID] = docIndex;
					}
				}
			}
			// fill the tail with the bucket size
			++nCurSegID;
			thisHashInfo[2 + nCurSegID] = nListLen;

			nSegNum = nCurSegID;
			thisHashInfo[1] = nSegNum;
			nSegOffset += (nSegNum + 3);
		}

	}
	fwrite(hashOffset,  sizeof(uint32_t), nListNum, fHashOffset);
	fwrite(hashInfo, sizeof(uint32_t), nSegOffset, fHashInfo);


	free(hashInfo);
	free(hashOffset);

	fclose(fHashInfo);
	fclose(fHashOffset);
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

	if (argc > 2) {
		kExceptedBucketSize = atoi(argv[2]);
	}
	cout << "HS" << kExceptedBucketSize << "_SegLRC_Generator" << endl;
	cout << "dataset = " << dataset << endl;

	if (argc > 3) {
		CUDA_SAFE_CALL(cudaSetDevice(strtol(argv[3], NULL, 10)));
	}
	else // default set to Tesla
	{
		CUDA_SAFE_CALL(cudaSetDevice(0));
	}


	string index_dir = dataset_dir + dataset + "/";
	as_load_atind(index_dir.c_str(), dataset.c_str());



	ostringstream result;
	result << result_dir << dataset << "_"
		   << "HS" << kExceptedBucketSize << "_SegLRC_Generator.txt";
	ofsresult.open((result.str()).c_str());
	
	cout << "Getting hash info..." << endl;
	RunGetHashInfo(dataset);

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
