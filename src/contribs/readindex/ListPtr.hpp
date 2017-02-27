
#ifndef SRC_HJY_LISTPTR_H_
#define SRC_HJY_LISTPTR_H_

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
#include "CLucene/search/TermQuery.h"
#include "CLucene/index/Terms.h"

using namespace std;
using namespace lucene::analysis;
using namespace lucene::index;
using namespace lucene::util;
using namespace lucene::queryParser;
using namespace lucene::document;
using namespace lucene::search;

CL_CLASS_DEF(index,IndexReader)
CL_NS_DEF(search)

//char index[250] = "/home/hjy/dataset/gov2-essentialDocs";
//IndexReader* readerIndex =IndexReader::open(index);
//IndexSearcher* searcher = new IndexSearcher(readerIndex);
//lucene::analysis::WhitespaceAnalyzer analyzer;


class ListPtr{
public:
		float_t maxScore;
		const TCHAR* term;
		Scorer* termDocs;
		int32_t curDoc;



		ListPtr();
		ListPtr (Term* term,unordered_map<wstring,float_t> &my_map);

		ListPtr& operator=(const ListPtr &);
		void Skip(const int32_t &d);
		friend bool operator<(const ListPtr &la, const ListPtr &lb)
			{
			    return la.curDoc < lb.curDoc;
			}
		~ListPtr();

};

CL_NS_END

#endif /* SRC_HJY_LISTPTR_H_ */
