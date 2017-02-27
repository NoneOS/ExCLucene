#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <cwctype>
#include <locale>
#include <cwchar>
#include <unordered_map>

#include "CLucene/StdHeader.h"
#include "CLucene/_clucene-config.h"
#include "CLucene/search/_TermScorer.h"
#include "CLucene/search/Scorer.h"
#include "CLucene.h"
#include "CLucene/config/repl_tchar.h"
#include "CLucene/config/repl_wchar.h"
#include "CLucene/util/Misc.h"
#include "CLucene/util/Equators.h"
#include "CLucene/search/Similarity.h"

#include "CLucene/search/Query.h"
#include "CLucene/search/SearchHeader.h"

using namespace lucene::analysis;
using namespace lucene::index;
using namespace lucene::queryParser;
using namespace lucene::document;
using namespace lucene::search;
using namespace std;

void StoreMaxScorer(const char* index,const char* res)
{
	lucene::analysis::WhitespaceAnalyzer analyzer;
	std::wofstream wout(res);
	int32_t count = 0;
	IndexReader* reader = IndexReader::open(index);
	IndexSearcher searcher(reader);
	TermEnum* termEnum = reader->terms();
	while(termEnum->next())
	{

		float_t maxScorer = 0;
		count++;
		Term* term=termEnum->term();
		Query* q = QueryParser::parse(term->text(),_T("content"),&analyzer);
		IndexSearcher* searcherPoint = &searcher;
		Weight* weight = q->weight(searcherPoint);
		Similarity* similarity = q->getSimilarity(searcherPoint);
		Scorer* scorer = weight->scorer(reader);
		TermDocs* termDocs = reader->termDocs(term);

		while(scorer->next())
		{
			int32_t doc = scorer->doc();
			float_t score = scorer->score();
			if(score > maxScorer)
			{
				maxScorer = score;
			}

		}
		wout<<term->text() << " " <<maxScorer <<endl;

	}
	wout.close();

}

int main( int32_t argc, char** argv ){

	const char index[250] = "/gov2-essentialDocs";
	const char* res = "/maxscore.test";

	StoreMaxScorer(index,res);
	return 0;
}


