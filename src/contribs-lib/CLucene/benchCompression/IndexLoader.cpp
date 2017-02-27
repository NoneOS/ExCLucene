/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#include "IndexLoader.h"

// load dictionary
void IndexLoader::loadDictionary() {
    FILE *fdict = fopen(dictPath.c_str(), "rb");
    if(!fdict) {
		std::cerr << "dictionary doesn't exist!" << std::endl;
		exit(1);
    }

	dictionary.resize(kListNum);
	fread(&dictionary[0], sizeof(dictionary_t), kListNum, fdict);

    fclose(fdict);
}

// load postings
void IndexLoader::loadPostings() {
	if (dictionary.empty()) {
		std::cerr << "please load dictionary first" << std::endl;
		exit(1);
	}

    FILE *fpostings = fopen(postingsPath.c_str(), "rb");
    if(!fpostings) {
		std::cerr << "postings doesn't exist!" << std::endl;
		exit(1);
    }

	postings.resize(kListNum);
	for (uint64_t uListIdx = 0; uListIdx < kListNum; ++uListIdx) {
		uint64_t length = dictionary[uListIdx].length;
		uint64_t offset = dictionary[uListIdx].offset;

		postings[uListIdx].reserve(length);
		fseek(fpostings, offset, SEEK_SET);
		fread(postings[uListIdx].data(), sizeof(uint32_t), length, fpostings);
	}

    fclose(fpostings);
}

