/*
 * Configure.h
 *
 */

#ifndef SRC_CONTRIBS_LIB_CLUCENE_CACHE_CONFIGURE_H_
#define SRC_CONTRIBS_LIB_CLUCENE_CACHE_CONFIGURE_H_

#include "CLucene/StdHeader.h"

CL_NS_DEF2(search, cache)

class CLUCENE_INLINE_EXPORT Configure {
public:
    const static int TOPK_DISPLAY = 10; // top k results to display, here we choose 30
    static std::string QRC_RESULT_CACHE;
    static std::string PLC_POSTINGLIST_CACHE;
    static std::string DC_DOCUMENT_CACHE;
    static std::string SC_SNIPPET_CACHE;

    const static TCHAR* ITEM_ID; // doc id, used for referring DB, primary key
    const static TCHAR* ITEM_URL;
    const static TCHAR* ITEM_TITLE;
    const static TCHAR* ITEM_CONTENT;
    const static TCHAR* ITEM_PATH; // file path of the original item

public:
    Configure();
    virtual ~Configure();
};

CL_NS_END2

#endif /* SRC_CONTRIBS_LIB_CLUCENE_CACHE_CONFIGURE_H_ */
