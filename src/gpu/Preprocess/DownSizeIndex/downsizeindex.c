#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include </usr/include/sys/stat.h>
#include <errno.h>


#define MAX_QUERY_LEN 80
#define MAX_IND2_SIZE 500 * 1024 * 1024


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
	struct stat buf;
	FILE *ftestdata, *find1, *find2;
	char testdata_fn[256], ind1_fn[256], ind2_fn[256];
	unsigned int ntestdata_size, nind1_size, nind2_size;
	unsigned char *ptestdata;
	at_term_ind1_t *pind1;
	unsigned int *pind2;
	FILE *fNtestdata, *fNind1, *fNind2;
	char Ntestdata_fn[256], Nind1_fn[256], Nind2_fn[256];
	unsigned int tno, size;
	unsigned char *ptr_in, *ptr_end;
	query_input_t input;
	int flag, tid;
	
	

	//load at_test_data
	sprintf(testdata_fn, "%s%s", in_dir, "at_test_data");
	ftestdata = fopen(testdata_fn, "rb");
	if (!ftestdata)
	{
		printErr();
	}
	stat(testdata_fn, &buf);
	ntestdata_size = buf.st_size;
	ptestdata = (unsigned char*)malloc(ntestdata_size);
	if (!ptestdata)
	{
		printErr();
	}
	printf("reading %s\n", testdata_fn);
	fread(ptestdata, 1, ntestdata_size, ftestdata);

	
	//load at.ind1
	sprintf(ind1_fn, "%s%s", in_dir, "at.ind1");
	find1 = fopen(ind1_fn, "rb");
	if (!find1)
	{
		printErr();
	}
	stat(ind1_fn, &buf);
	nind1_size = buf.st_size;
	pind1 = (at_term_ind1_t*)malloc(nind1_size);
	if (!pind1)
	{
		printErr();
	}
	printf("reading %s\n", ind1_fn);
	fread(pind1, 1, nind1_size, find1);

	
	//load at.ind2
	sprintf(ind2_fn, "%s%s", in_dir, "at.ind2");
	find2 = fopen(ind2_fn, "rb");
	if (!find2)
	{
		printErr();
	}
	stat(ind2_fn, &buf);
	nind2_size = buf.st_size;
	pind2 = (unsigned int*)malloc(nind2_size);
	if (!pind2)
	{
		printErr();
	}
	printf("reading %s\n", ind2_fn);
	fread(pind2, 1, nind2_size, find2);
	

	//open new at_test_data
	sprintf(Ntestdata_fn, "%s%s", out_dir, "at_test_data");
	fNtestdata = fopen(Ntestdata_fn, "wb");
	if (!fNtestdata)
	{
		printErr();
	}


	//open new at.ind1
	sprintf(Nind1_fn, "%s%s", out_dir, "at.ind1");
	fNind1 = fopen(Nind1_fn, "wb");
	if (!fNind1)
	{
		printErr();
	}


	//open new at.ind2
	sprintf(Nind2_fn, "%s%s", out_dir, "at.ind2");
	fNind2 = fopen(Nind2_fn, "wb");
	if (!fNind2)
	{
		printErr();
	}



	//select at.ind1 so that the corresponding size of at.ind2 is less than MAX_IND2_SIZE
	tno = -1;
	size = 0;
	while (size < MAX_IND2_SIZE && size < nind2_size)
	{
		tno++;
		size += ((pind1 + tno)->m_urlcount) * sizeof(unsigned int);

		if (tno % 100 == 0)
		{
			printf("tno:%dsize:%d\n", tno, size);
		}
	}
	printf("tno:%d\n", tno);


	//select and write new at_test_data
	ptr_in = ptestdata;
	ptr_end = ptestdata + ntestdata_size;
	while (ptr_in < ptr_end)
	{
		input.tnum = *(const int *)ptr_in;
		ptr_in += sizeof(unsigned int);
		memcpy(input.tno, ptr_in, input.tnum * sizeof(unsigned int));
		ptr_in += input.tnum * sizeof(unsigned int);

		flag = 1;
		for (tid = 0; tid < input.tnum; tid++)
		{
			if (input.tno[tid] > tno)
			{
				flag = 0;
				break;
			}
		}

		if (flag)
		{
			fwrite(&(input.tnum), sizeof(unsigned int), 1, fNtestdata);
			fwrite(input.tno, sizeof(unsigned int), input.tnum, fNtestdata);
			fflush(fNtestdata);
		}
	}


	//write new at.ind1 and at.ind2
	fwrite(pind1, sizeof(at_term_ind1_t), tno + 1, fNind1);
	fflush(fNind1);
	fwrite(pind2, 1, (pind1 + tno)->m_off + (pind1 + tno)->m_urlcount * sizeof(unsigned int), fNind2);
	fflush(fNind2);
	
	

	fclose(ftestdata);
	free(ptestdata);
	ptestdata = NULL;
	fclose(find1);
	free(pind1);
	pind1 = NULL;
	fclose(find2);
	fclose(fNtestdata);
	fclose(fNind1);
	fclose(fNind2);


	return 0;
}

