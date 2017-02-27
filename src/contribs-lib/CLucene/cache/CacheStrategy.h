/*
 * CacheStrategy.h
 *
 */

#ifndef SRC_CONTRIBS_LIB_CLUCENE_CACHE_CACHESTRATEGY_H_
#define SRC_CONTRIBS_LIB_CLUCENE_CACHE_CACHESTRATEGY_H_

#include "CLucene/StdHeader.h"

CL_NS_DEF2(search, cache)

class CLUCENE_INLINE_EXPORT CacheStrategy {
public:
    // dynamic cache strategies, starts with "Dyn"
    const static std::string LRU;
    const static std::string LFU;
    const static std::string DynQTFDF;
    const static std::string DynCA;
    const static std::string DynFB;
    const static std::string DynGDSFK;

    // static cache strategies, starts with "Sta"
    const static std::string StaQTF; // query frequency based
    const static std::string StaQTFDF; // sorted by frequency/size
    const static std::string StaCA; // cost aware

    // check whether static or not

    static bool isStaticCache(std::string cacheStrategy) {
        // may be better to set a PREFIX for static cache
        if (cacheStrategy.substr(0, 3) == "Sta")
            return true;
        else
            return false;
    }
    CacheStrategy();
    virtual ~CacheStrategy();
};

const std::string CacheStrategy::LRU = "DynLRU";
const std::string CacheStrategy::LFU = "DynLFU";
const std::string CacheStrategy::DynQTFDF = "DynQTFDF";
const std::string CacheStrategy::DynCA = "DynCA";
const std::string CacheStrategy::DynFB = "DynFB";
const std::string CacheStrategy::DynGDSFK = "DynGDSFK";

const std::string CacheStrategy::StaQTF = "StaQTF"; // query frequency based
const std::string CacheStrategy::StaQTFDF = "StaQTFDF"; // sorted by frequency/size
const std::string CacheStrategy::StaCA = "StaCA"; // cost aware

CL_NS_END2

#endif /* SRC_CONTRIBS_LIB_CLUCENE_CACHE_CACHESTRATEGY_H_ */
