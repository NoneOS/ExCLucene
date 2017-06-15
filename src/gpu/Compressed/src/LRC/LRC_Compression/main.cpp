/*************************************************************************
	> File: main.cpp
	> Author: Naiyong Ao
	> Email: aonaiyong@gmail.com 
	> Time: Fri 30 Jan 2015 03:52:59 PM CST
 ************************************************************************/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include</usr/include/sys/stat.h>
#include</usr/include/sys/time.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "ParaPFor.h"
#include "util.h"

using namespace std;




string dataset_dir = "/media/indexDisk/naiyong/dataset/";
string data_dir = "/media/indexDisk/naiyong/data/LRC/Compression/";
string regressioninfo_dir = "/media/indexDisk/naiyong/data/LRC/Generator/";
string result_dir = "/media/indexDisk/naiyong/result/LRC/Compression/";


typedef struct _at_term_ind1 {
	uint32_t length;
	uint64_t offset;
} at_term_ind1_t;


#define SegmentSize 256
#define MAX_LIST_LEN 30000000 // upper bound of length
#define MAX_TOTAL_BLOCK_NUM 500000000 

#define MAX_PATH_LEN 1024	
#define BUFFER_SIZE 2048

at_term_ind1_t *ind1;	
unsigned char buffer[BUFFER_SIZE];


// for compression ratio
ofstream ofsresult;
float FRAC_begin, FRAC_step, FRAC_end;

int step = 0;
extern float FRAC;


double GetTickCount() {
	struct timeval tv;

	if (!gettimeofday(&tv, NULL)) {
		return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
	}
	else {
		return 0;
	}
}


inline void readFile(unsigned char *ptr, FILE *fp) {
    double start;
    size_t count;
	size_t	bytes;

	start = GetTickCount();				
	count = 0;							

    while ((bytes = fread(buffer, 1, BUFFER_SIZE, fp)) > 0) {
        memcpy(ptr + count, buffer, bytes);    
        count += bytes;
    }
}


void Compression(const string &dataset) {
	string index_dir = dataset_dir + dataset + "/";

	char file_name[MAX_PATH_LEN];

	uint64_t initSize = 0; // size of uncompressed ind2
	uint64_t compSize = 0; // size of compressed ind2 plus auxiliary info

	struct stat buf;
	// ind1
	cout << "reading ind1..." << endl;
	sprintf(file_name, "%s%s.ind1", index_dir.c_str(), dataset.c_str());
	stat(file_name, &buf);
	uint64_t ind1Size = buf.st_size;
	FILE *find1 = fopen(file_name, "rb");
	ind1 = (at_term_ind1_t *)malloc(ind1Size);
	readFile((unsigned char *)ind1, find1);
  
	// new ind1
	sprintf(file_name, "%s%d/%s.ind1", data_dir.c_str(), step, dataset.c_str());
	FILE *fNewInd1 = fopen(file_name, "wb");


	// ind2
	sprintf(file_name, "%s%s.ind2", index_dir.c_str(), dataset.c_str());
	FILE *find2 = fopen(file_name, "rb");
	stat(file_name, &buf);
	uint64_t ind2Size = buf.st_size;
	initSize += ind2Size; 


	// vertical deviations
	sprintf(file_name, "%s%s%s", regressioninfo_dir.c_str(), dataset.c_str(), ".ind2_distance");
	FILE *fVDs = fopen(file_name, "rb");


	// compressed ind2
	sprintf(file_name, "%s%d/%s.ind2", data_dir.c_str(), step, dataset.c_str());
	FILE *fcompInd2 = fopen(file_name, "wb");
	

	// auxiliary info
	
	// regression info
	sprintf(file_name, "%s%s%s", regressioninfo_dir.c_str(), dataset.c_str(), ".regression_info");
	stat(file_name, &buf);
	compSize += buf.st_size;

	// block info
	sprintf(file_name, "%s%d/%s%s", data_dir.c_str(), step, dataset.c_str(), ".segHead");
	FILE *fsegHead = fopen(file_name, "wb");

	sprintf(file_name, "%s%d/%s%s", data_dir.c_str(), step, dataset.c_str(), ".segOffset");
	FILE *fsegOffset = fopen(file_name, "wb");

	sprintf(file_name, "%s%d/%s%s", data_dir.c_str(), step, dataset.c_str(), ".segNum");
	FILE *fsegNum = fopen(file_name, "wb");



	uint32_t *docIDs = (uint32_t *)malloc(MAX_LIST_LEN * sizeof(uint32_t));
	uint32_t *VDs = (uint32_t *)malloc(MAX_LIST_LEN * sizeof(uint32_t));

	uint32_t *compInd2 = (uint32_t *)malloc(ind2Size);
	uint32_t *segHead = (uint32_t *)malloc(MAX_TOTAL_BLOCK_NUM * sizeof(uint32_t));
	uint32_t *segOffset = (uint32_t *)malloc(MAX_TOTAL_BLOCK_NUM * sizeof(uint32_t));


	uint64_t nListNum = ind1Size / sizeof(at_term_ind1_t);
	vector<at_term_ind1_t> newInd1(ind1, ind1 + nListNum);
	vector<uint32_t> segNum(nListNum, 0);

	uint64_t totalWords = 0;
	uint64_t totalSegNum = 0;


	cout << "compressing..." << endl;
	for (uint64_t i = 0; i < nListNum; ++i) {
		newInd1[i].offset = totalWords * sizeof(uint32_t);
		segNum[i] = totalSegNum;


		uint64_t nListLen = (ind1 + i)->length;				
		uint64_t nListOffset = (ind1 + i)->offset;

		// load VDs
		fseek(fVDs, nListOffset, SEEK_SET);
		fread(VDs, sizeof(uint32_t), nListLen, fVDs);

		// load docIDs
		fseek(find2, nListOffset, SEEK_SET);	
		fread(docIDs, sizeof(uint32_t), nListLen, find2);


		uint64_t nwords = 0; // offset of compressed block 
		uint64_t kSegNum = div_roundup(nListLen, SegmentSize);
		uint64_t nSegStart = 0;
		for (uint64_t j = 0; j < kSegNum - 1; ++j) {
			nSegStart = j * SegmentSize;
			segHead[totalSegNum + j] = docIDs[nSegStart];
			segOffset[totalSegNum + j] = nwords;

			nwords += encodeArray(VDs + nSegStart, compInd2 + totalWords + nwords, SegmentSize);
		}

		nSegStart = (kSegNum - 1) * SegmentSize;
		segHead[totalSegNum + kSegNum - 1] = docIDs[nSegStart];
		segOffset[totalSegNum + kSegNum - 1] = nwords;

		nwords += encodeArray(VDs + nSegStart, compInd2 + totalWords + nwords, nListLen - nSegStart);

		totalSegNum += kSegNum; // update total number of blocks (offset of segHead and segOffset)
		totalWords += nwords; // update offset of compressed ind2
	}
	
	fwrite(&newInd1[0], sizeof(at_term_ind1_t), nListNum, fNewInd1);

	// segNum
	fwrite(&segNum[0], sizeof(uint32_t), nListNum, fsegNum);
	compSize += nListNum * sizeof(uint32_t); 

	// segHead
	fwrite(segHead, sizeof(uint32_t), totalSegNum, fsegHead);		
	compSize += totalSegNum * sizeof(uint32_t); 

	// segOffset
	fwrite(segOffset, sizeof(uint32_t), totalSegNum, fsegOffset);	
	compSize += totalSegNum * sizeof(uint32_t);

	// compInd2
	fwrite(compInd2, sizeof(uint32_t), totalWords, fcompInd2);		
	compSize += totalWords * sizeof(uint32_t);

	
	double ratio = (double)initSize / compSize;
	cout << "compressiong raito = " << ratio << endl << endl;
	ofsresult << FRAC << "\t" << 32 / ratio << endl;

	free(docIDs);
	free(VDs);
	free(compInd2);
	free(segHead);
	free(segOffset);

	fclose(find1);
	fclose(fNewInd1);
	fclose(find2);
	fclose(fVDs);
	fclose(fcompInd2);
	fclose(fsegHead);
	fclose(fsegOffset);
	fclose(fsegNum);
}


void terminator() {
	if (ind1 != NULL) free(ind1);
}


void runTest(const string &dataset) {
	Compression(dataset);
	
	terminator();
}


int main(int argc, char** argv) {
	if (argc != 2) {
		cout << "wrong number of arguments" << endl;
		exit(1);
	}
	string dataset = argv[1];
	cout << "LRC_Compression" << endl;
	cout << "dataset = " << dataset << endl;

	string result_filename = result_dir + dataset + "_LRC_Compression.txt";
	ofsresult.open(result_filename.c_str());
	ofsresult << "Compression: bits/integer" << endl << endl;
	ofsresult << "FRAC\tCompression" << endl;

	FRAC_begin = 0.0;
	FRAC_step = 0.04;
	FRAC_end = 0.64;

	for (FRAC = FRAC_begin, step = 0; FRAC <= FRAC_end;  FRAC += FRAC_step, step++) {
		cout << "step = " << step << ", FRAC = " << FRAC << endl;
		runTest(dataset);
	}

	ofsresult.close();

	sleep(5);

	return 0;
}

