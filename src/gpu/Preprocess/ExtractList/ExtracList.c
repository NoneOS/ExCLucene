#include <stdio.h>
#include <stdlib.h>
#include </usr/include/sys/stat.h>
#include <errno.h>


typedef struct _term_sign
{
	unsigned sign1, sign2;
} term_sign_t;


/*
//baidu ind1
typedef struct _at_term_ind1
{
	term_sign_t m_term_sign;
	unsigned m_idf:3;
	unsigned m_urlcount:25;
	unsigned m_tag:4;
	unsigned m_off;
}
*/


//gov ind1
typedef struct _at_term_ind1
{
	unsigned m_urlcount;
	unsigned m_off;
} at_term_ind1_t;


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
	FILE *find1, *find2, *flist;
	char ind1_fn[256], ind2_fn[256], list_fn[256];
	unsigned ind1_size, ind2_size;
	at_term_ind1_t *pind1;
	unsigned *pind2;
	unsigned tid;
	unsigned urlcount;
	unsigned off;


	//read at.ind1
	sprintf(ind1_fn, "%s%s", in_dir, "at.ind1");
	find1 = fopen(ind1_fn, "rb");
	if (!find1)
	{
		printErr();
	}
	stat(ind1_fn, &buf);
	ind1_size = buf.st_size;
	pind1 = (at_term_ind1_t*)malloc(ind1_size);
	if (!pind1)
	{
		printErr();
	}
	fread(pind1, 1, ind1_size, find1);


	//reade at.ind2
	sprintf(ind2_fn, "%s%s", in_dir, "at.ind2");
	find2 = fopen(ind2_fn, "rb");
	if (!find2)
	{
		printErr();
	}
	stat(ind2_fn, &buf);
	ind2_size = buf.st_size;
	pind2 = (unsigned*)malloc(ind2_size);
	if (!pind2)
	{
		printErr();
	}
	fread(pind2, 1, ind2_size, find2);


	//creat list
	sprintf(list_fn, "%s%s", out_dir, "InvertedList");
	flist = fopen(list_fn, "wb");
	if (!flist)
	{
		printErr();
	}


	for (tid = 0; tid < ind1_size / sizeof(at_term_ind1_t); tid++)
	{
		urlcount = (pind1 + tid)->m_urlcount;

		if (100000 <= urlcount && urlcount <= 200000)
		{
			printf("urlcount: %u\n", urlcount);

			off = (pind1 + tid)->m_off;
			fwrite(pind2 + off, sizeof(unsigned), urlcount, flist);
			fflush(flist);
			
			break;
		}
	}


	//release resources
	fclose(find1);
	free(pind1);
	pind1 = NULL;
	fclose(find2);
	free(pind2);
	pind2 = NULL;
	fclose(flist);


	return 0;
}


