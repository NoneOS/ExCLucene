/*************************************************************************
	> File Name: common.h
************************************************************************/

#ifndef HEADERS_COMMON_H_
#define HEADERS_COMMON_H_

#include <fstream>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>// for out_of_range
#include <cstring>
#include <sstream>
#include <fstream>
#include <cassert>
#include <set>
#include <map>
#include <unordered_map>
#include <list>
#include <iomanip>
#include <queue>
#include <cstddef> // for size_t
#include <cstdint>
#include <numeric>      // std::adjacent_difference
#include <ctime> // clock(), clock_t, CLOCKS_PER_SEC
#include <sys/time.h> //gettimeofday
#include <sys/stat.h>
#include <memory>

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
#include "CLucene/util/Misc.h"
#include "CLucene/index/Term.h"
#include "CLucene/search/_TermScorer.h"
#include "CLucene/search/Scorer.h"
#include "CLucene.h"
#include "CLucene/search/Similarity.h"
#include "CLucene/search/Query.h"
#include <errno.h>
#include <fcntl.h>
#include <immintrin.h>
#include <iso646.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>

#include "ListPtr.h"

using namespace std;
using namespace lucene::analysis;
using namespace lucene::index;
using namespace lucene::util;
using namespace lucene::queryParser;
using namespace lucene::document;
using namespace lucene::search;







// SSE&AVX headers
//#include <xmmintrin.h> // SSE
//#include <emmintrin.h> // SSE2
//#include <pmmintrin.h> // SSE3
//#include <tmmintrin.h> // SSSE3
//#include <smmintrin.h> // SSE4.1
//#include <nmmintrin.h> // SSE4.2
//#include <immintrin.h> // AVX&AVX2


#endif /* HEADERS_COMMON_H_ */

