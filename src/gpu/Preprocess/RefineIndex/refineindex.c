#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include </usr/include/sys/stat.h>
#include </usr/include/sys/time.h>
#include <errno.h>


#define MAX_QUERY_LEN 80
#define MAX_LIST_NUM 4000000
#define MAX_IND2_SIZE 400 * 1024 * 1024


typedef struct _term_sign
{
	unsigned int sign1, sign2;
} term_sign_t;


/*
//baidu ind1
typedef struct _at_term_ind1
{
	term_sign_t m_term_sign;
	unsigned m_idf:3;
	unsigned m_urlcount:25;
	unsigned m_tag:4;
	unsigned int m_off;
} at_term_ind1_t;
*/


//gov ind1
typedef struct _at_term_ind1
{
	unsigned int m_urlcount;
	unsigned int m_off;
} at_term_ind1_t;


typedef struct _query_input
{
	int tnum;
	int tno[MAX_QUERY_LEN];
} query_input_t;


#define printErr()\
{\
	char buf[256];\
	sprintf(buf, "line:%u\tfunc:%s\tfile:%s", __LINE__, __FUNCTION__, __FILE__);\
	perror(buf);\
	exit(1);\
}


int main()
{
	char in_dir[] = "/home/para/data/gov/";
	char out_dir[] = "./data/";
	FILE *ftestdata, *find1, *find2;
	char testdata_fn[256], ind1_fn[256], ind2_fn[256];
	unsigned int testdata_size;
	unsigned int ind1_size;
	unsigned char *ptestdata;
	at_term_ind1_t *pind1;
	struct stat buf;
	FILE *fNtestdata, *fNind1, *fNind2;
	char Ntestdata_fn[256], Nind1_fn[256], Nind2_fn[256];
	unsigned char *ptr_in, *ptr_end;
	query_input_t input;
	int qid, tid;
	int *pMapOldNew;
	int nOld, nNew;
	unsigned int *url;
	at_term_ind1_t *pind1_tmp;
	unsigned int urlcount;
	int ind1_count_sum;
	unsigned int ind2_offset;
	at_term_ind1_t ind1;


	//open and load old at_test_data
	sprintf(testdata_fn, "%s%s", in_dir, "at_test_data");
	ftestdata = fopen(testdata_fn, "rb");
	if (!ftestdata)
	{
		printErr();
		exit(1);
	}
	stat(testdata_fn, &buf);
	testdata_size = buf.st_size;
	ptestdata = (unsigned char *)malloc(testdata_size);
	if (!ptestdata)
	{
		printErr();
		exit(1);
	}
	printf("read file %s\n", testdata_fn);
	fread(ptestdata, 1, testdata_size, ftestdata);

	//open and load old at.ind1
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
	printf("read file:%s\n", ind1_fn);
	fread(pind1, 1, ind1_size, find1);

	//open old at.ind2
	sprintf(ind2_fn, "%s%s", in_dir, "at.ind2");
	find2 = fopen(ind2_fn, "rb");
	if (!find2)
	{
		printErr();
		exit(1);
	}


	//open new at_test_data
	sprintf(Ntestdata_fn, "%s%s", out_dir, "at_test_data");
	fNtestdata = fopen(Ntestdata_fn, "wb+");
	if (!fNtestdata)
	{
		printErr();
		exit(1);
	}

	//open new at.ind1
	sprintf(Nind1_fn, "%s%s", out_dir, "at.ind1");
	fNind1 = fopen(Nind1_fn, "wb+");
	if (!fNind1)
	{
		printErr();
		exit(1);
	}

	//open new at.ind2
	sprintf(Nind2_fn, "%s%s", out_dir, "at.ind2");
	fNind2 = fopen(Nind2_fn, "wb+");
	if (!fNind2)
	{
		printErr();
		exit(1);
	}


	//allocate pmapOldNew
	pMapOldNew = (int *)malloc(MAX_LIST_NUM * sizeof(int));
	if (!pMapOldNew)
	{
		printErr();
		exit(1);
	}
	memset(pMapOldNew, -1, MAX_LIST_NUM * sizeof(int));


	ptr_in = ptestdata;
	ptr_end = ptestdata + testdata_size;
	qid = 0;
	ind1_count_sum = 0;
	ind2_offset = 0;
	while (ptr_in < ptr_end)
	{
		if (ind2_offset >= MAX_IND2_SIZE)
		{
			break;
		}

		if (qid % 10000 == 0)
		{
			printf("%d queries finished!\n", qid);
		}
		qid++;

		input.tnum = *(const int *)ptr_in; 
		ptr_in += sizeof(int);
		memcpy(input.tno, ptr_in, sizeof(int) * input.tnum);
		ptr_in += sizeof(int) * input.tnum;


		for (tid = 0; tid < input.tnum; tid++)
		{
			nOld = input.tno[tid];
			if (pMapOldNew[nOld] != -1)
			{
				input.tno[tid] = pMapOldNew[nOld];
			}
			else
			{
				//update map
				nNew = ind1_count_sum;
				input.tno[tid] = nNew;
				pMapOldNew[nOld] = nNew;

				
				//write at.ind1
				pind1_tmp = pind1 + nOld;
				ind1 = *pind1_tmp;
				ind1.m_off = ind2_offset;
				fwrite(&ind1, 1, sizeof(at_term_ind1_t), fNind1);
				fflush(fNind1);
				ind1_count_sum++;
			
			
				//write at.ind2
				fseek(find2, pind1_tmp->m_off, SEEK_SET);
				urlcount = pind1_tmp->m_urlcount;
				url = (unsigned int *)malloc(urlcount * sizeof(unsigned int));
				if (!url)
				{
					printErr();
					exit(1);
				}
				fread(url, sizeof(unsigned int), urlcount, find2);
				fwrite(url, sizeof(unsigned int), urlcount, fNind2);
				fflush(fNind2);
				ind2_offset += urlcount * sizeof(unsigned int);
				free(url);
				url = NULL;
			}
		}


		fwrite(&(input.tnum), sizeof(int), 1, fNtestdata);
		fwrite(input.tno, sizeof(int), input.tnum, fNtestdata);
		fflush(fNtestdata);
	}


	fclose(ftestdata);
	fclose(find1);
	fclose(find2);
	fclose(fNtestdata);
	fclose(fNind1);
	fclose(fNind2);
	free(ptestdata);
	ptestdata = NULL;
	free(pind1);
	pind1 = NULL;
	free(pMapOldNew);
	pMapOldNew = NULL;
}
