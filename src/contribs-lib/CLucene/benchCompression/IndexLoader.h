/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef INDEXLOADER_H_
#define INDEXLOADER_H_

#include "common.h"
#include "Array.h"
#include "IndexInfo.h"

class IndexLoader {
public:
	std::string dictPath;
	std::string postingsPath;
	uint64_t kListNum = 0;
	std::vector<dictionary_t> dictionary;
	std::vector<Array> postings;

	IndexLoader(const std::string &indexDir, const std::string &dataset) :
		dictPath(indexDir + dataset + ".ind1"),
		postingsPath(indexDir + dataset + ".ind2") {
			struct stat buf;
			stat(dictPath.c_str(), &buf);
			kListNum = buf.st_size / sizeof(dictionary_t);
			assert(kListNum < MAXNUM);
		}

	void loadDictionary();
	void loadPostings();

	void loadIndex() {
		std::cout << "loading dictionary & postings lists" << std::endl;
		loadDictionary();
		loadPostings();
		std::cout << "loading indexes finished" << std::endl;
	}
};

#endif /* INDEXLOADER_H_ */
