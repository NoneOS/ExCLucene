#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include </usr/include/sys/stat.h>
#include <errno.h>


#define MAX_QUERY_LEN 80


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



//struct of ind1 for texture memory
typedef struct _at_term_tm_ind1
{
	at_term_ind1_t m_ind1;
	unsigned int m_flag;
} at_term_tm_ind1_t;
	


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
	FILE *find1, *find2;
	char ind1_fn[256], ind2_fn[256];
	unsigned int nind1_size, nind2_size;
	at_term_ind1_t *pind1;
	FILE *fNind1;
	char Nind1_fn[256];
	at_term_tm_ind1_t tm_ind1;
	unsigned int tno, size; 
	unsigned int halvetno;
	
	
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


	//obtain the size of at.ind2
	sprintf(ind2_fn, "%s%s", in_dir, "at.ind2");
	find2 = fopen(ind2_fn, "rb");
	if (!find2)
	{
		printErr();
	}
	stat(ind2_fn, &buf);
	nind2_size = buf.st_size;


	//open new at.ind1
	sprintf(Nind1_fn, "%s%s", out_dir, "at.ind1");
	fNind1 = fopen(Nind1_fn, "wb");
	if (!fNind1)
	{
		printErr();
	}



	//halve at.ind1
	tno = -1;
	halvetno = -1;
	size = 0;
	while (size < nind2_size)
	{
		tno++;
		size += ((pind1 + tno)->m_urlcount) * sizeof(unsigned int);

	
		tm_ind1.m_ind1 = *(pind1 + tno);
		if (size < nind2_size / 2)
		{
			tm_ind1.m_flag = 0;
		}
		else 
		{
			if (halvetno == -1)
			{
				halvetno = tno;
			}

			(tm_ind1.m_ind1).m_off = (tm_ind1.m_ind1).m_off - (pind1 + halvetno)->m_off;
			tm_ind1.m_flag = 1;
		}
		fwrite(&tm_ind1, 1, sizeof(at_term_tm_ind1_t), fNind1);
		fflush(fNind1);

//		if (tno % 100 == 0)
//		{
//			printf("tno:%dsize:%d\n", tno, size);
//		}
	}
	printf("halvetno:%d\ttno:%d\n", halvetno, tno);
	printf("halvesize:%u\n", (pind1 + halvetno)->m_off);



	fclose(find1);
	free(pind1);
	pind1 = NULL;
	fclose(find2);
	fclose(fNind1);


	return 0;
}


