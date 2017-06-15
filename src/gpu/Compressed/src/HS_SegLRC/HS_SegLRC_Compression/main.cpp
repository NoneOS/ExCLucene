/*************************************************************************
	> File: main.cpp
	> Author: Naiyong Ao
	> Email: aonaiyong@gmail.com 
	> Time: Sat 31 Jan 2015 08:38:57 PM CST
 ************************************************************************/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include</usr/include/sys/stat.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "ParaPFor.h"
#include "util.h"

using namespace std;


string dataset_dir = "/media/indexDisk/naiyong/dataset/";
string data_dir = "/media/indexDisk/naiyong/data/HS_SegLRC/Compression/";
string regressioninfo_dir = "/media/indexDisk/naiyong/data/HS_SegLRC/Generator/";
string result_dir = "/media/indexDisk/naiyong/result/HS_SegLRC/Compression/";

uint32_t kExceptedBucketSize = 256;

typedef struct _at_term_ind1 {
	uint32_t length;
	uint64_t offset;
} at_term_ind1_t;


#define MAX_PATH_LEN 1024
#define BUFFER_SIZE 2048
#define MAX_LIST_LEN 30000000
#define MAX_TOTAL_BLOCK_NUM 500000000 


at_term_ind1_t *ind1;
unsigned char buffer[BUFFER_SIZE];

// for compression ratio
ofstream ofsresult;
float FRAC_begin, FRAC_step, FRAC_end;

int step = 0;
extern float FRAC;


inline void readFile(unsigned char *ptr, FILE *fp) {
    size_t count = 0;
	size_t	bytes = 0;

    while ((bytes = fread(buffer, 1, BUFFER_SIZE, fp)) > 0) {
        memcpy(ptr + count, buffer, bytes);    
        count += bytes;
    }
}


void Compression(const string &dataset) {
	string index_dir = dataset_dir + dataset + "/";

	char file_name[MAX_PATH_LEN];

	// for compression ratio
	uint64_t initSize = 0, compSize = 0;


	struct stat buf;
	cout << "reading at_HashSeg_offset..." << endl;
	sprintf(file_name, "%s%s%s", regressioninfo_dir.c_str(), dataset.c_str(), ".HashSeg_offset");
	stat(file_name, &buf);
	FILE *fHashOffset = fopen(file_name, "rb");
	uint32_t *hashOffset = (uint32_t*)malloc(buf.st_size);
	readFile((unsigned char*)hashOffset, fHashOffset);
	uint32_t *hashOffset_ptr = hashOffset;
	compSize += buf.st_size;
	

	cout << "reading at_HashSeg_info..." << endl;
	sprintf(file_name, "%s%s%s", regressioninfo_dir.c_str(), dataset.c_str(), ".HashSeg_info");
	stat(file_name, &buf);
	FILE *fHashInfo = fopen(file_name, "rb");
	uint32_t *hashInfo = (uint32_t*)malloc(buf.st_size);
	readFile((unsigned char*)hashInfo, fHashInfo);
	uint32_t *hashInfo_ptr = hashInfo;
	compSize += buf.st_size;


	sprintf(file_name, "%s%s%s", regressioninfo_dir.c_str(), dataset.c_str(), ".regression_info");
	stat(file_name, &buf);
	compSize += buf.st_size;

	sprintf(file_name, "%s%s%s", regressioninfo_dir.c_str(), dataset.c_str(), ".ind2_distance_base");
	stat(file_name, &buf);
	compSize += buf.st_size;


	cout << "reading ind1..." << endl;
	sprintf(file_name, "%s%s.ind1", index_dir.c_str(), dataset.c_str());
	stat(file_name, &buf);
	uint64_t ind1Size = buf.st_size;
	FILE *find1 = fopen(file_name, "rb");
	ind1 = (at_term_ind1_t *)malloc(ind1Size);
	readFile((unsigned char *)ind1, find1);
  

	sprintf(file_name, "%s%s%s", regressioninfo_dir.c_str(), dataset.c_str(), ".ind2_distance");
	FILE *fVDs = fopen(file_name, "rb");
	stat(file_name, &buf);
	uint64_t ind2Size = buf.st_size;
	initSize += ind2Size;


	sprintf(file_name, "%s%d/%s.ind1", data_dir.c_str(), step, dataset.c_str());
	FILE *fNewInd1 = fopen(file_name, "wb");

	sprintf(file_name, "%s%d/%s.ind2", data_dir.c_str(), step, dataset.c_str());
	FILE *fcompInd2 = fopen(file_name, "wb");

	sprintf(file_name, "%s%d/%s%s", data_dir.c_str(), step, dataset.c_str(), ".segOffset");
	FILE *fsegOffset = fopen(file_name, "wb");

	sprintf(file_name, "%s%d/%s%s", data_dir.c_str(), step, dataset.c_str(), ".segNum");
	FILE *fbucketNum = fopen(file_name, "wb");


	uint32_t *VDs = (uint32_t *)malloc(MAX_LIST_LEN * sizeof(uint32_t));
	uint32_t *compInd2 = (uint32_t *)malloc(ind2Size);
	uint32_t *segOffset = (uint32_t *)malloc(MAX_TOTAL_BLOCK_NUM * sizeof(uint32_t));


	uint64_t nListNum = ind1Size / sizeof(at_term_ind1_t);
	vector<at_term_ind1_t> newInd1(ind1, ind1 + nListNum);
	vector<uint32_t> bucketNum(nListNum, 0);

	uint64_t totalWords = 0;
	uint64_t totalBucketNum = 0;


	cout << "compressing..." << endl;
	for (uint64_t i = 0; i < nListNum; ++i) {
		newInd1[i].offset = totalWords * sizeof(uint32_t);
		bucketNum[i] = totalBucketNum;


		uint64_t nListLen = (ind1 + i)->length;
		uint64_t nListOffset = (ind1 + i)->offset;
		fseek(fVDs, nListOffset, SEEK_SET);
		fread(VDs, sizeof(uint32_t), nListLen, fVDs);


		// for hash
		uint32_t *thisHashInfo = hashInfo + hashOffset[i];
		uint64_t kBucketNum = thisHashInfo[1];
		thisHashInfo += 2;
		
		uint64_t nwords = 0; // offset of compressed block 
		for (uint64_t j = 0; j < kBucketNum; ++j)	{
			segOffset[totalBucketNum + j] = nwords;

			uint64_t nBucketStart = thisHashInfo[j];
			uint64_t BucketSize = thisHashInfo[j + 1] - nBucketStart;
			if (BucketSize > 0)
				nwords += encodeArray(VDs + nBucketStart, compInd2 + totalWords + nwords, BucketSize);
		}


		totalBucketNum += kBucketNum;
		totalWords += nwords;
	}

	fwrite(&newInd1[0], sizeof(at_term_ind1_t), nListNum, fNewInd1);

	// bucketNum
	fwrite(&bucketNum[0], sizeof(uint32_t), nListNum, fbucketNum);
	compSize += nListNum * sizeof(uint32_t); 

	// segOffset
	fwrite(segOffset, sizeof(uint32_t), totalBucketNum, fsegOffset);	
	compSize += totalBucketNum * sizeof(uint32_t);

	// compInd2
	fwrite(compInd2, sizeof(uint32_t), totalWords, fcompInd2);		
	compSize += totalWords * sizeof(uint32_t);


	double ratio = (double)initSize / compSize;
	cout << "compressiong raito = " << ratio << endl << endl;
	ofsresult << FRAC << "\t" << 32 / ratio << endl;


	free(VDs);
	free(compInd2);
	free(segOffset);
	free(hashOffset);
	free(hashInfo);

	fclose(find1);
	fclose(fVDs);
	fclose(fNewInd1);
	fclose(fcompInd2);
	fclose(fsegOffset);
	fclose(fbucketNum);
	fclose(fHashOffset);
	fclose(fHashInfo);
}



void terminator() {
	if (ind1 != NULL) free(ind1);
}


void runTest(const string &dataset) {
	Compression(dataset);
	
	terminator();
}


int main(int argc, char** argv) {
	if (argc < 2) {
		cout << "wrong number of arguments" << endl;
		exit(1);
	}

	if (argc > 2) {
		kExceptedBucketSize = atoi(argv[2]);
	}
	string dataset = argv[1];
	cout << "HS" << kExceptedBucketSize << "_SegLRC_Compression" << endl;
	cout << "dataset = " << dataset << endl;

	ostringstream result;
	result << result_dir << dataset << "_HS" << kExceptedBucketSize << "_SegLRC_Compression.txt";
	ofsresult.open((result.str()).c_str());
	ofsresult << "Compression: bits/integer" << endl << endl;
	ofsresult << "FRAC\tCompression" << endl;

	FRAC_begin = 0.0;
	FRAC_step = 0.04;
	FRAC_end = 0.0;

	for (FRAC = FRAC_begin, step = 0; FRAC <= FRAC_end;  FRAC += FRAC_step, step++) {
		cout << "step = " << step << ", FRAC = " << FRAC << endl;
		runTest(dataset);
	}

	ofsresult.close();

	return 0;
}

