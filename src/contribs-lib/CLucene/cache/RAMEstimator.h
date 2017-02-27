/*
 * RAMEstimator::h
 *
 */

#ifndef SRC_CONTRIBS_LIB_CLUCENE_CACHE_RAMESTIMATOR_H_
#define SRC_CONTRIBS_LIB_CLUCENE_CACHE_RAMESTIMATOR_H_

#include "CLucene.h"
#include "Configure.h"
#include "Snippet.h"
#include "util.h"

CL_NS_USE(index)
CL_NS_USE(analysis)
CL_NS_USE(util)
CL_NS_USE(store)
CL_NS_USE(document)

CL_NS_DEF2(search, cache)

class CLUCENE_EXPORT RAMEstimator : LUCENE_BASE {
public:

    // Coarse estimates used to measure RAM usage
    // FIXME:Maybe use sizeof(long))
    const static int LONG_NUM_BYTE = sizeof (long);
    const static int INT_NUM_BYTE = sizeof (int);
    const static int CHAR_NUM_BYTE = sizeof (char);
    const static int POINTER_NUM_BYTE = sizeof (void*);
    const static int OBJECT_HEADER_BYTES = 8; // e.g., Integer header
    // TODO: maybe useless for C++

    const static long ONE_KB = 1024;
    const static long ONE_MB = ONE_KB * ONE_KB;
    const static long ONE_GB = ONE_KB * ONE_MB;

    const static int INT_BLOCK_SHIFT = 13;
    const static int INT_BLOCK_SIZE = 1 << INT_BLOCK_SHIFT;
    const static int INT_BLOCK_MASK = INT_BLOCK_SIZE - 1;

    const static int PER_DOC_BLOCK_SIZE = 1024;
    const static int UN_LIMIT_MEMORY = 8192; // 8192MB

    static double bytesToMB(unsigned long long numBytes) {
        return numBytes / (double) ONE_MB;
    }

    template<typename strType>
    static int getNumBytesOfString(const strType& str) {
        return str.length() * sizeof (str[0]);
    }

    static int getNumBytesOfInteger() {
        return INT_NUM_BYTE;
    }

    // get the num of bytes taken of a Lucene's Document class

    static int getNumBytesOfDocument(Document* document) {
        int numBytes = 0;
        
        // There's no TITLE field in this index
    //    tstring title = document->get(Configure::ITEM_TITLE);
        tstring url = document->get(Configure::ITEM_URL);
        tstring content = document->get(Configure::ITEM_CONTENT);

    //    numBytes += RAMEstimator::getNumBytesOfString(title);
        numBytes += RAMEstimator::getNumBytesOfString(url);
        numBytes += RAMEstimator::getNumBytesOfString(content);
        

        return numBytes;
    }

    // get the num of bytes taken of a Snippet class

    static int getNumBytesOfSnippet(Snippet& snippet) {
        int numBytes = 0;
        numBytes += RAMEstimator::getNumBytesOfString(snippet.getTitle());
        numBytes += RAMEstimator::getNumBytesOfString(snippet.getUrl());
        numBytes += RAMEstimator::getNumBytesOfString(snippet.getSummarization());
        return numBytes;
    }

    // PostingEntry

    static int getNumBytesOfPostingEntry() {
        int numBytes = 0;
        numBytes = RAMEstimator::INT_NUM_BYTE + RAMEstimator::INT_NUM_BYTE;
        return numBytes;
    }


    // TODO: getNumBytesOfTermDocs needs to be test

    static int getNumBytesOfTermDocs(TermDocs &td) {
        int numBytes = 0;
        int length = 0; // num of entries in the TermDocs
        while (true) {
            length++;
            bool isEnd = false;
            try {
                isEnd = td.next();
            } catch (exception& e) {
                // TODO Auto-generated catch block
                e.what();
                isEnd = true;
            }
            if (isEnd) break;
        }
        // here, 2 means doc id and doc frequency
        // each takes 4 bytes
        // while num of entries is: length
        numBytes = RAMEstimator::INT_NUM_BYTE * 2 * length;
        return numBytes;
    }


    // refer to DocumentsWriter for estimating memory usage

    /* Rough logic: HashMap has an array[Entry] w/ varying
    load factor (say 2 * POINTER).  Entry is object w/ Term
    key, BufferedDeletes.Num val, int hash, Entry next
    (OBJ_HEADER + 3*POINTER + INT).  Term is object w/
    string field and string text (OBJ_HEADER + 2*POINTER).
    We don't count Term's field since it's interned.
    Term's text is string (OBJ_HEADER + 4*INT + POINTER +
    OBJ_HEADER + string.length*CHAR).  BufferedDeletes.num is
    OBJ_HEADER + INT. */

    const static int BYTES_PER_DEL_TERM = 8 * POINTER_NUM_BYTE + 5 * OBJECT_HEADER_BYTES + 6 * INT_NUM_BYTE;

    /* Rough #include "CLucene/cache/Cache.h"
             #include "CLucene/cache/StaticCache.h"
            #include "InputParameters.h"
            #include "SortedBySomething.h"logic: del docIDs are List<Integer>.  Say list
            allocates ~2X size (2*POINTER).  Integer is OBJ_HEADER
   + int */
    const static int BYTES_PER_DEL_DOCID = 2 * POINTER_NUM_BYTE + OBJECT_HEADER_BYTES + INT_NUM_BYTE;

    /* Rough logic: HashMap has an array[Entry] w/ varying
   load factor (say 2 * POINTER).  Entry is object w/
   Query key, Integer val, int hash, Entry next
   (OBJ_HEADER + 3*POINTER + INT).  Query we often
   undercount (say 24 bytes).  Integer is OBJ_HEADER + INT. */
    const static int BYTES_PER_DEL_QUERY = 5 * POINTER_NUM_BYTE + 2 * OBJECT_HEADER_BYTES + 2 * INT_NUM_BYTE + 24;

    /* Initial chunks size of the shared byte[] blocks used to
   store postings data */
    const static int BYTE_BLOCK_SHIFT = 15;
    const static int BYTE_BLOCK_SIZE = 1 << BYTE_BLOCK_SHIFT;
    const static int BYTE_BLOCK_MASK = BYTE_BLOCK_SIZE - 1;
    const static int BYTE_BLOCK_NOT_MASK = ~BYTE_BLOCK_MASK;

    /**
     * Return good default units based on byte size.
     */
    static std::string humanReadableUnits(long bytes) {
        std::string newSizeAndUnits;

        if (bytes / ONE_GB > 0) {
            newSizeAndUnits = doubleToString((float) bytes / ONE_GB)
                + " GB";
        } else if (bytes / ONE_MB > 0) {
            newSizeAndUnits = doubleToString((float) bytes / ONE_MB)
                + " MB";
        } else if (bytes / ONE_KB > 0) {
            newSizeAndUnits = doubleToString((float) bytes / ONE_KB)
                + " KB";
        } else {
            newSizeAndUnits = ToString(bytes) + " bytes";
        }
        return newSizeAndUnits;
    }


public:
    RAMEstimator();
    virtual ~RAMEstimator();
};

CL_NS_END2

#endif /* SRC_CONTRIBS_LIB_CLUCENE_CACHE_RAMESTIMATOR_H_ */
