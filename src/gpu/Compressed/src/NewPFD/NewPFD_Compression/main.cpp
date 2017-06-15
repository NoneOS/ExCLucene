/*************************************************************************
	> File: main.cpp
	> Author: Naiyong Ao
	> Email: aonaiyong@gmail.com 
	> Time: Sun 01 Feb 2015 07:10:16 PM CST
 ************************************************************************/
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include </usr/include/sys/stat.h>
#include </usr/include/sys/time.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "NewPFor.h"
#include "util.h"


using namespace std;


string dataset_dir = "/media/indexDisk/naiyong/dataset/";
string data_dir = "/media/indexDisk/naiyong/data/NewPFD/Compression/"; 
string result_dir = "/media/indexDisk/naiyong/result/NewPFD/Compression/";


typedef struct _at_term_ind1 {
	uint32_t length;
	uint64_t offset;
} at_term_ind1_t;


#define SegmentSize 256
#define MAX_LIST_LEN 30000000
#define MAX_TOTAL_BLOCK_NUM 500000000 

#define MAX_PATH_LEN 1024				//filename length restriction
#define BUFFER_SIZE 2048				//size of buffer


at_term_ind1_t *ind1;					//ind1 file pointer
unsigned char buffer[BUFFER_SIZE];		//file reading buffer


//for compression ratio
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
    size_t count = 0;
	size_t	bytes = 0;

    while((bytes = fread(buffer, 1, BUFFER_SIZE, fp)) > 0) {
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


	// compressed ind2
	sprintf(file_name, "%s%d/%s.ind2", data_dir.c_str(), step, dataset.c_str());
	FILE *fcompInd2 = fopen(file_name, "wb");


	// auxiliary info
	sprintf(file_name, "%s%d/%s%s", data_dir.c_str(), step, dataset.c_str(), ".segHead");
	FILE *fsegHead = fopen(file_name, "wb");

	sprintf(file_name, "%s%d/%s%s", data_dir.c_str(), step, dataset.c_str(), ".segMedian");
	FILE *fsegMedian = fopen(file_name, "wb");

	sprintf(file_name, "%s%d/%s%s", data_dir.c_str(), step, dataset.c_str(), ".segOffset");
	FILE *fsegOffset = fopen(file_name, "wb");

	sprintf(file_name, "%s%d/%s%s", data_dir.c_str(), step, dataset.c_str(), ".segNum");
	FILE *fsegNum = fopen(file_name, "wb");


	uint32_t *docIDs = (uint32_t *)malloc(MAX_LIST_LEN * sizeof(uint32_t));	
	uint32_t *dgaps = (uint32_t *)malloc(MAX_LIST_LEN * sizeof(uint32_t));

	uint32_t *compInd2 = (uint32_t *)malloc(ind2Size);
	uint32_t *segHead = (uint32_t *)malloc(MAX_TOTAL_BLOCK_NUM * sizeof(uint32_t));
	uint32_t *segMedian = (uint32_t *)malloc(MAX_TOTAL_BLOCK_NUM * sizeof(uint32_t));
	uint32_t *segOffset = (uint32_t *)malloc(MAX_TOTAL_BLOCK_NUM * sizeof(uint32_t));


	uint64_t nListNum = ind1Size / sizeof(at_term_ind1_t);
	vector<at_term_ind1_t> newInd1(ind1, ind1 + nListNum);
	vector<uint32_t> segNum(nListNum, 0);

	uint64_t totalWords = 0;
	uint64_t totalSegNum = 0;


	cout << "compressing..." << endl;
	for (uint32_t i = 0; i < nListNum; i++) {
		newInd1[i].offset = totalWords * sizeof(uint32_t);
		segNum[i] = totalSegNum;


		uint64_t nListLen = (ind1 + i)->length;				
		uint64_t nListOffset = (ind1 + i)->offset;

		// load docIDs
		fseek(find2, nListOffset, SEEK_SET);	
		fread(docIDs, sizeof(uint32_t), nListLen, find2);

		dgaps[0] = docIDs[0];
		for (uint32_t j = 1; j < nListLen; j++) {
			dgaps[j] = docIDs[j] - docIDs[j-1];
		}

		uint64_t nwords = 0; // offset of compressed block 
		uint64_t kSegNum = div_roundup(nListLen, SegmentSize);
		uint64_t nSegStart = 0;
		for (uint64_t j = 0; j < kSegNum - 1; ++j) {
			nSegStart = j * SegmentSize;
			segHead[totalSegNum + j] = docIDs[nSegStart];
			segMedian[totalSegNum + j] = docIDs[nSegStart + SegmentSize / 2];
			segOffset[totalSegNum + j] = nwords;

			nwords += encodeArray(dgaps + nSegStart, compInd2 + totalWords + nwords, SegmentSize);
		}

		nSegStart = (kSegNum - 1) * SegmentSize;
		segHead[totalSegNum + kSegNum - 1] = docIDs[nSegStart];
		segMedian[totalSegNum + kSegNum - 1] = docIDs[nSegStart + (nListLen - nSegStart) /2];
		segOffset[totalSegNum + kSegNum - 1] = nwords;

		nwords += encodeArray(dgaps + nSegStart, compInd2 + totalWords + nwords, nListLen - nSegStart);

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

	// segMedian
	fwrite(segMedian, sizeof(uint32_t), totalSegNum, fsegMedian);		
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
	free(dgaps);
	free(compInd2);
	free(segHead);
	free(segMedian);
	free(segOffset);

	fclose(find1);
	fclose(find2);
	fclose(fNewInd1);
	fclose(fcompInd2);
	fclose(fsegHead);
	fclose(fsegMedian);
	fclose(fsegOffset);
	fclose(fsegNum);
}


void Decompression(const string &dataset) {
	string index_dir = dataset_dir + dataset + "/";

	char file_name[MAX_PATH_LEN];

	struct stat buf;
	// compressed ind2
	cout << "reading compressed ind2..." << endl;
	sprintf(file_name, "%s%d/%s.ind2", data_dir.c_str(), step, dataset.c_str());
	stat(file_name, &buf);
	FILE *fcompInd2 = fopen(file_name, "rb");
	uint32_t *compInd2 = (unsigned int *)malloc(buf.st_size);
	readFile((unsigned char *)compInd2, fcompInd2);
	uint32_t *compInd2_ptr = compInd2;
	
	// old ind1
	cout << "reading old ind1..." << endl;
	sprintf(file_name, "%s%s.ind1", index_dir.c_str(), dataset.c_str());
	stat(file_name, &buf);
	FILE *find1 = fopen(file_name, "rb");
	ind1 = (at_term_ind1_t *)malloc(buf.st_size);
	readFile((unsigned char *)ind1, find1);
	
	// old ind2
	sprintf(file_name, "%s%s.ind2", index_dir.c_str(), dataset.c_str());
	FILE *find2 = fopen(file_name, "rb");
	

	uint32_t *dgaps = (uint32_t *)malloc(MAX_LIST_LEN * sizeof(uint32_t));
	uint32_t *docIDs = (uint32_t *)malloc(MAX_LIST_LEN * sizeof(uint32_t));


	cout << "decompressing..." << endl;
	uint32_t size = 0;
	int flag = 1;
	double start = GetTickCount();
	for (uint32_t i = 0; i < buf.st_size / sizeof(at_term_ind1_t); i++) {
		uint32_t nListLen = (ind1 + i)->length;

		uint32_t *gap_ptr = dgaps;
		for(uint32_t j = 0; j < nListLen / BS; j++)	//segment of size BS
		{
			size = decodeArray(compInd2_ptr, gap_ptr, BS);

			compInd2_ptr += size;
			gap_ptr += BS;
		}

		size = (nListLen & (BS - 1));
		if (size != 0)						//segment of size < BS
		{
			size = decodeArray(compInd2_ptr, gap_ptr, size);
			compInd2_ptr += size;
		}
		

		 /*****************verification*****************/
		 for(uint32_t j = 0; j < nListLen - 1; j++)
			 dgaps[j + 1] += dgaps[j];
				 
		 fseek(find2, (ind1 + i)->offset, SEEK_SET);
		 fread(docIDs, sizeof(uint32_t), nListLen, find2);
		 for (uint32_t j = 0; j < nListLen; j++) {
			 if ((docIDs[j]) != dgaps[j]) {
				printf("docIDs:%u\tgap:%u\n", (docIDs[j]), dgaps[j]);
				printf("wrong position:%d \n", j);
				flag = 0;
				break;
			 }
		}

	}

	if (flag == 1)
		cout << "the result is right!" << endl;
	else
		cout << "the result is wrong!" << endl;

	cout << "time: " << GetTickCount() - start << "ms" << endl << endl << endl;


	free(compInd2);
	free(docIDs);
	free(dgaps);

	fclose(fcompInd2);
	fclose(find1);
	fclose(find2);
}


void terminator() {
	if (ind1 != NULL) free(ind1);
}

void runTest(const string &dataset) {
	Compression(dataset);

	Decompression(dataset);
	
	terminator();
}


int main(int argc, char** argv) {
	if (argc < 2) {
		cout << "wrong number of arguments" << endl;
		exit(1);
	}
	string dataset = argv[1];
	cout << "NewPFD_Compression" << endl;
	cout << "dataset = " << dataset << endl;

	string result_filename = result_dir + dataset + "_NewPFD_Compression.txt";
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
	cout << endl << endl;

	ofsresult.close();

	sleep(5);

	return 0;
}

