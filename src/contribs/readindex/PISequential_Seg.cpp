#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unordered_map>
#include "CLucene/_ApiHeader.h"
#include "CLucene/StdHeader.h"
#include "CLucene/_clucene-config.h"
#include "CLucene/search/Similarity.h"
#include "CLucene/search/BooleanQuery.h"
#include "CLucene.h"
#include "CLucene/config/repl_tchar.h"
#include "CLucene/config/repl_wchar.h"
#include "CLucene/util/Misc.h"
#include "CLucene/util/Equators.h"
#include "CLucene/search/Scorer.h"
#include "CLucene/search/Similarity.h"
#include "CLucene/search/Query.h"
#include "CLucene/search/Hits.h"
#include "ListPtr.hpp"
#include "CLucene/search/TermQuery.h"
#include "CLucene/index/Terms.h"


using namespace std;
using namespace lucene::analysis;
using namespace lucene::index;
using namespace lucene::util;
using namespace lucene::queryParser;
using namespace lucene::document;
using namespace lucene::search;

#include "CLucene/_ApiHeader.h"
#include "../core/CLucene/index/IndexReader.h"
#include "../core/CLucene/index/IndexWriter.h"

#include "Grammar.hpp"
#include "IOUtils.hpp"

static uint32_t outArray[Grammar_Consts::list_upperb];


int main(int argc, char* argv[])
{
	ios_base::sync_with_stdio(false);
	locale loc("en_US.utf8");
	locale::global(loc);

	FILE* outind1 = fopen("/media/indexDisk/gmData/tmp/g.ind1","wb");
        FILE* outind2 = fopen("/media/indexDisk/gmData/tmp/g.ind2","wb");

	assert(outind1 != nullptr);

	Term_index indexTerm;
	indexTerm.m_urlcount = 0;
	indexTerm.m_off = 0;

	unsigned freq;
	unsigned docId;
	int32_t count;
	int32_t length;
	IndexReader* reader = IndexReader::open("CLucene_index_path");
	if(reader == NULL)
	{
		perror("Open file recfile !");
		exit(1);
	}
	TermEnum* termEnum = reader->terms();
	while(termEnum->next()){
		count = 0;
		Term* term=termEnum->term();
		TermDocs* termDocs = reader->termDocs(term);
		while(termDocs->next())
		{
			docId = unsigned(termDocs->doc());
			freq = unsigned(termDocs->freq());
			outArray[count] = docId;
			count++;
		}

		indexTerm.m_urlcount = count;
		indexTerm.m_off += count * sizeof(uint32_t);
		fwrite(&indexTerm, sizeof(Term_index), 1, outind1);
		fwrite(outArray, sizeof(uint32_t), count, outind2);

		termDocs->close();
	}

        reader->close();
        termEnum->close();

	fclose(outind1);
	fclose(outind2);
	return 0;
}
