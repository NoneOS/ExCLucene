/*
 * type.h
 *
 */

#ifndef SRC_NEW_FUNCTION_HEADER_TYPE_H_
#define SRC_NEW_FUNCTION_HEADER_TYPE_H_

#include "common.h"


using namespace std;
using namespace lucene::analysis;
using namespace lucene::index;
using namespace lucene::util;
using namespace lucene::queryParser;
using namespace lucene::document;
using namespace lucene::search;

#define check_FILE(fstream,file) if(fstream == NULL){ cerr << file << "is NULL" << endl; exit(1);}

const unsigned topK = 20;
const unsigned BS = 64;

const int32_t MAXDID = 25205178;
const int32_t DOCNUM = MAXDID + 1;

class ListPtr;
typedef vector<ListPtr*> queryEntry;

typedef vector<ListPtr*>::iterator qE_iterator;

class ListPtr_B;
typedef vector<ListPtr_B*> queryEntry_B;

typedef vector<ListPtr_B*>::iterator qE_iterator_B;

template<typename It, typename T>
inline void sortByDid(It start, It end){ //assumes that derefernecing an iterator with *it gives a T* that has a did member
       std::sort(start, end, [](const T* a, const T* b){ return a->curDoc < b->curDoc; });
}

typedef struct result
{
	friend bool operator<(const result &r1, const result &r2)
	{
		return r1.score > r2.score;
	}
	unsigned int did;
	float score;
} result;

typedef struct doc
{
    unsigned docid;
    float score;
    float upper_score;
    vector<unsigned> vecTerms;
}doc;

typedef struct vecGlobalInfo
{
	vector<queryEntry> vecQuerySet;
	vecGlobalInfo():vecQuerySet(0){};
}VecGlbInfo;

typedef struct vecGlobalInfo_B
{
	vector<queryEntry_B> vecQuerySet;
	vecGlobalInfo_B():vecQuerySet(0){};
}VecGlbInfo_B;


static double getTime(timeval &begin_time, timeval &end_time)
{
    return 1000.0 * (end_time.tv_sec - begin_time.tv_sec) + (end_time.tv_usec - begin_time.tv_usec) / 1000.0;
}


static char Index[250] = "/dataset/gov2-essentialDocs";
static IndexReader* reader =IndexReader::open(Index);
static IndexSearcher* searcher = new IndexSearcher(reader);
static lucene::analysis::WhitespaceAnalyzer analyzer;

static int query_no = 0;

#endif /* SRC_NEW_FUNCTION_HEADER_TYPE_H_ */
