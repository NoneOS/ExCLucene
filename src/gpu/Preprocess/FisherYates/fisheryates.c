#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

using namespace std;

#define GOV_MAX_DOCID 1053110
//#define BD_MAX_DOCID 15749656 
#define BD_MAX_DOCID 5038710  //gov2 indeed 
uint32_t RAND_MAX_BITS = 0;

inline void swap(vector<uint32_t> *pvRandom, const uint32_t first, const uint32_t second)
{
	uint32_t tmp = 0;
	tmp = (*pvRandom)[first];
	(*pvRandom)[first] = (*pvRandom)[second];
	(*pvRandom)[second] = tmp;
}

inline uint32_t randomizer(const uint32_t i, const uint32_t nN)
{
//	uint32_t nSecond = ((rand() << RAND_MAX_BITS) + rand()) % nN;
	uint32_t nSecond = ((rand() << RAND_MAX_BITS) + rand()) % i;
	return nSecond;
}


/*
 * generate random map file
 */
int main()
{
	//local
	FILE *fpMap = fopen("../ShuffleIndex/data/map.txt", "w+");
//	uint32_t nN = pow(2, 25);
//	uint32_t nN = 10846563;
	uint32_t nN = GOV_MAX_DOCID + 1;	//for gov
//	uint32_t nN = BD_MAX_DOCID + 1;		//for baidu 
	printf("universe: %u\n", nN);
	vector<uint32_t> vRandom(nN);
	//local ends

	//RAND_MAX_BITS
	uint32_t nRandMax = RAND_MAX;
	printf("RAND_MAX: %u\n", RAND_MAX);
	for (uint32_t i = 0; nRandMax > 0; ++i)
	{
		nRandMax >>= 1;
		RAND_MAX_BITS++;
	}
	printf("RAND_MAX_BITS: %u\n", RAND_MAX_BITS);

	//init
	printf("initing beginning...\n");
	for (uint32_t i = 0; i < nN; ++i)
	{
		vRandom[i] = i;
	}
	printf("initing ending...\n");

	//random
	printf("randoming beginning...\n");
	for (uint32_t i = 0; i < nN; ++i)
	{
		if (i % 5000 == 0)
		{
			srand(time(0));
			printf("%f\%\n", i / (float)nN * 100);
		}
		swap(&vRandom, i, randomizer(i + 1, nN));
	}
	printf("randoming ending...\n");

	//output
	printf("outputing beginning...\n");
	char buf[256];
	for (uint32_t i = 0; i < nN; ++i)
	{
		sprintf(buf, "%u\n", vRandom[i]);
		fputs(buf, fpMap);
	}
	printf("outputing ending...\n");

	//release
	fclose(fpMap);
	fpMap = NULL;
	vRandom.clear();

	return 0;
}
