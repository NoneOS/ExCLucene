#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include </usr/include/sys/stat.h>
#include <vector>
#include <algorithm>
#include </usr/include/sys/time.h>
#include <errno.h>


using namespace std;


//#define MAX_DOCID 1053110	//for gov
//#define MAX_DOCID 15749656	//for baidu
#define MAX_DOCID 5038710 //for gov2 
#define MAXLEN 100000000
#define GET_URL_NO(x) ((x)&0x1ffffff)



//GOV IND1
typedef struct _at_term_ind1
{
	unsigned int m_urlcount;
	unsigned int m_off;
} at_term_ind1_t;



typedef struct _term_sign
{
	unsigned int sign1, sign2;
} term_sign_t;


/*
//BAIDU IND1
typedef struct _at_term_ind1
{
	term_sign_t m_term_sign;
	unsigned m_idf:3;
	unsigned m_urlcount:25;
	unsigned m_tag:4;
	unsigned int m_off;
} at_term_ind1_t;
*/


#define printErr()\
{\
	char buf[256];\
	sprintf(buf, "line:%u\tfunc:%s\tfile:%s", __LINE__, __FUNCTION__, __FILE__);\
	perror(buf);\
	exit(1);\
}


inline int readFileWithNum(unsigned char *ptr, FILE *fp, const uint32_t nNum)
{
	fread(ptr, sizeof(uint32_t), nNum, fp);
	if (ferror(fp))
	{
		printErr();
		return -1;
	}
	else
	{
		return 0;
	}
}


/*
 * get List from file
 */
int32_t getList(uint32_t *pnList, FILE *fpFile, uint32_t nOffset, uint32_t nNum)
{
	if (fseek(fpFile, nOffset, SEEK_SET) == -1)
	{
		printErr();
		return -1;
	}


	if (readFileWithNum((unsigned char*)pnList, fpFile, nNum) == -1)
	{
		printErr();
		return -1;
	}

	
	return 0;
}


void shuffleMap(at_term_ind1_t *pSInd1, FILE *fpInd2, const uint32_t nListNum)
{
    //local
    FILE *fpInd2R = fopen("/home/para/NVIDIA_GPU_Computing_SDK/C/bin/linux/release/data/gov2r/at.ind2", "wb+");
    if (!fpInd2)
    {
        printErr();
        exit(1);
    }
    FILE *fpMap = fopen("./data/map.txt", "r");
    if (!fpMap)
    {
        printErr();
        exit(1);
    }
    uint32_t *pnList = (uint32_t*)malloc(sizeof(uint32_t) * MAXLEN);
    if (!pnList)
    {
        printErr();
        exit(1);
    }
    uint32_t *pnTmpList = (uint32_t*)malloc(sizeof(uint32_t) * MAXLEN);
    if (!pnTmpList)
    {
        printErr();
        exit(1);
    }


	vector<uint32_t> vRandom(MAX_DOCID + 1);
	uint32_t nNew = 0;

    at_term_ind1_t *pTmp_ind1;
    uint32_t nListLen = 0;

	uint32_t nListIdx, nDocIdx;
    //local ends
    
    //establish oldNew Map
    printf("establishing map...\n");
	nDocIdx = 0;
    while (fscanf(fpMap, "%u", &nNew) != EOF)
    {
		vRandom[nDocIdx] = nNew;
		nDocIdx++;
    }

    //shuffle
    printf("shuffling...\n");
    for (nListIdx = 0; nListIdx < nListNum; ++nListIdx)
    {
        //progress
        if (nListIdx % 5000 == 0)
        {
            printf("%f%%...\n", nListIdx / (float)nListNum * 100);
        }

        pTmp_ind1  = pSInd1 + nListIdx;
        nListLen = pTmp_ind1->m_urlcount;
        if (getList(pnList, fpInd2, pTmp_ind1->m_off, nListLen) == -1)
        {
            printErr();
            exit(1);
        }

		
		/*
        //debug
        for (nDocIdx = 0; nDocIdx < nListLen; ++nDocIdx)
        {
            printf("%u ", GET_URL_NO(pnList[nDocIdx]));
        }
        printf("\n");
        //debug ends
		*/

        for (nDocIdx = 0; nDocIdx < nListLen; ++nDocIdx)
        {
//            printf("old %u, ", GET_URL_NO(pnList[nDocIdx]));
			if (GET_URL_NO(pnList[nDocIdx]) <= MAX_DOCID) 
            {
                pnList[nDocIdx] = vRandom[GET_URL_NO(pnList[nDocIdx])];
//                printf("new %u\n", GET_URL_NO(pnList[nDocIdx]));
			}
			else
            {
                printf("miss occurs!!! list:%u\tdocIdx:%u\tdocId:%u\n", nListIdx, nDocIdx, GET_URL_NO(pnList[nDocIdx]));
                exit(1);
            }
        }

       
		/*
        //debug
        for (nDocIdx = 0; nDocIdx < nListLen; ++nDocIdx)
        {
            printf("%u ", pnList[nDocIdx]);
        }
        printf("\n");
        //debug ends
        */
        

		sort(pnList, pnList + nListLen);
        
		
		/*	
        //debug
        for (nDocIdx = 0; nDocIdx < nListLen; ++nDocIdx)
        {
            printf("%u ", pnList[nDocIdx]);
        }
        printf("\n");
        //debug ends
        */

        //write to disk
        fwrite(pnList, sizeof(uint32_t), nListLen, fpInd2R);
    }
    
    //release
    fclose(fpInd2R);
    fpInd2R = NULL;
    fclose(fpMap);
    fpMap = NULL;
    free(pnList);
    pnList = NULL;
    free(pnTmpList);
    pnTmpList = NULL;
}


int main ()
{
	FILE *find1, *find2;
	char in_dir[256] = "/home/para/data/gov2/";
	char ind1_fn[256];
	char ind2_fn[256];
	uint32_t ind1_size;
	at_term_ind1_t *pind1;
	struct stat buf;


	//open and load FILE at.ind1
	sprintf(ind1_fn, "%s%s", in_dir, "at.ind1");
	find1 = fopen(ind1_fn, "rb");
	if (!find1)
	{
		printErr();
		exit(1);
	}
	stat(ind1_fn, &buf);
	ind1_size = buf.st_size;
	pind1 = (at_term_ind1_t *)malloc(ind1_size);
	if (!pind1)
	{
		printErr();
		exit(1);
	}
	fread(pind1, 1, ind1_size, find1);
	
	//open FILE at.ind2
	sprintf(ind2_fn, "%s%s", in_dir, "at.ind2");
	find2 = fopen(ind2_fn, "rb");
	if (!find2)
	{
		printErr();
		exit(1);
	}
	

	shuffleMap(pind1, find2, ind1_size / sizeof(at_term_ind1_t));


	fclose(find1);
	fclose(find2);
	free(pind1);
	pind1 = NULL;


	return 0;
}
