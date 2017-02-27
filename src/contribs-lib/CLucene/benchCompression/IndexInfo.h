/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef INDEXINFO_H_
#define INDEXINFO_H_

#include "common.h"

struct dictionary_t {
	uint32_t length; // length of postings list (how many ints)
	uint64_t offset; // offset of postings list (how many bytes)
};

enum {
	MAXNUM = 19000000,   // upper limit on number of postings lists
	MAXNUMLR = 3000000,  // upper limit on number of postings lists of length greater than MINLENLR

	MINLENDELTA = 3,   // we skip lists of length less than MINLENDELTA during Delta compression
	MINLENLR = 5,      // we skip lists of length less than MINLENLR during Linear Regression and Linear Regression compression
	MAXLEN = 12000000  // upper limit on length of postings lists
};

#endif /* INDEXINFO_H_ */
