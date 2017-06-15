#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include </usr/include/sys/stat.h>
#include </usr/include/sys/time.h>
#include <errno.h>


#define MAX_QUERY_LEN 80
#define LIST_LEN_BASE 256
#define LIST_LEN_MAX 4096


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


//GOV IND1
typedef struct _at_term_ind1
{
	unsigned int m_urlcount;
	unsigned int m_off;
} at_term_ind1_t;


//at_testdata
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
	FILE *ftestdata, *find1, *fNtestdata;
	char in_dir[] = "/home/para/data/gov/";
	char out_dir[] = "./data/";
	char testdata_fn[256], ind1_fn[256], Ntestdata_fn[256];
	unsigned char *ptestdata;
	at_term_ind1_t *pind1;
	unsigned int testdata_size;
	unsigned int ind1_size;
	struct stat buf;
	unsigned char *ptr_in, *ptr_end;
	query_input_t input;
	unsigned int queryid;
	unsigned int ucnt, ucntShortest;
	int tid;
	unsigned int ListLen;


	//load at_test_data
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
	fread(ptestdata, 1, testdata_size, ftestdata);


	//load at.ind1
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


//	for (ListLen = LIST_LEN_BASE; ListLen <= LIST_LEN_MAX; ListLen *= 2)
	{
		//open new at_test_data
//		sprintf(Ntestdata_fn, "%s%s_%u", out_dir, "at_test_data", ListLen);
		sprintf(Ntestdata_fn, "%s%s_new", out_dir, "at_test_data");
		fNtestdata = fopen(Ntestdata_fn, "wb+");
		if (!fNtestdata)
		{
			printErr();
			exit(1);
		}


		//begin selecting
//		printf("\nselect queries <= %u\n", ListLen);
		ptr_in = ptestdata;
		ptr_end = ptestdata + testdata_size;
		queryid = 0;
		while (ptr_in < ptr_end)
		{
			if (queryid % 10000 == 0)
			{
				printf("%u queries finished!\n", queryid);
			}

			//load tnum and tno
			input.tnum = *(const int *)ptr_in;
			ptr_in += sizeof(int);
			memcpy(input.tno, ptr_in, sizeof(int) * input.tnum);
			ptr_in += sizeof(int) * input.tnum;

/*
			//obtain shortest list len
			ucntShortest = 80000000;
			for (tid = 0; tid < input.tnum; tid++)
			{
				ucnt = (pind1 + input.tno[tid])->m_urlcount;

				if (ucnt < ucntShortest)
				{
					ucntShortest = ucnt;
				}
			}


			//select 
			if (ucntShortest >= ListLen)
			{
				fwrite(&input, sizeof(int), 1 + input.tnum, fNtestdata);
				fflush(fNtestdata);
			}
			*/

			if (queryid < 31)
			{
				fwrite(&input, sizeof(int), 1 + input.tnum, fNtestdata);
				fflush(fNtestdata);
			}
			queryid++;
		}

		fclose(fNtestdata);
	}


	fclose(ftestdata);
	fclose(find1);
	free(ptestdata);
	ptestdata = NULL;
	free(pind1);
	pind1 = NULL;


	return 0;
}
