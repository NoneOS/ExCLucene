/*
 * SearchTest.cpp
 *
 */

#include "SearchTest.h"

CL_NS_USE2(search, cache)

const TCHAR *stopwords1[1] =
{
   _T("sjnasf")
};

SearchTest::SearchTest() {

}

SearchTest::SearchTest(InputParameters &ip) : hl_analyzer(stopwords1) {
    // TODO Auto-generated constructor stub
    initializeCacheInfo(ip);
}

SearchTest::~SearchTest() {
    // TODO Auto-generated destructor stub
    hl_searcher->close();
    hl_reader->close();
    _CLDELETE(hl_searcher);
    _CLDELETE(hl_reader);
    _CLDELETE(resultCache);
    _CLDELETE(snippetCache);
}

