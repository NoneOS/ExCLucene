/*
 * Cache.h
 *
 */

#ifndef SRC_CONTRIBS_LIB_CLUCENE_CACHE_CACHE_H_
#define SRC_CONTRIBS_LIB_CLUCENE_CACHE_CACHE_H_

#include "CacheNode.h"
#include "RAMEstimator.h"

#include <unordered_map>
#define HashMap std::unordered_map

CL_NS_DEF2(search, cache)

template<typename K, typename V>
class CLUCENE_EXPORT Cache : LUCENE_BASE {
public:

    typedef typename HashMap<K, CacheNode<K, V> >::iterator HashMap_It;
    typedef typename HashMap<K, CacheNode<K, V> >::size_type HashMap_St;
    typedef typename HashMap<K, CacheNode<K, V> >::value_type HashMap_Vt;

protected:

    HashMap<K, CacheNode<K, V> > cache;

    uint64_t maxMemory_Byte; // memory limit allocated for caching in terms of Bytes
    uint64_t usedMemory_Byte; // memory are already used by the cache

    uint64_t numOfLookups;
    uint64_t numOfHits;

    uint64_t bytesOfLookUps;
    uint64_t bytesOfHits;

public:

    Cache(double maxMemory_MByte = 1) : cache(),
    maxMemory_Byte(maxMemory_MByte * RAMEstimator::ONE_MB), usedMemory_Byte(0),
    numOfLookups(0), numOfHits(0), bytesOfLookUps(0), bytesOfHits(0) {
    }

//    Cache(const Cache<K, V> & c) :
//    cache(c.cache), maxMemory_Byte(c.maxMemory_Byte), usedMemory_Byte(c.usedMemory_Byte),
//    numOfLookups(c.numOfLookups), numOfHits(c.numOfHits),
//    bytesOfLookUps(c.bytesOfLookUps), bytesOfHits(c.bytesOfHits)
//     {
//    }

    bool isCacheFullPlusOneNode(const CacheNode<K, V>& node) {
        if (this->usedMemory_Byte + node.numOfBytes > this->maxMemory_Byte)
            return true;
        return false;
    }

    void addByetsOfLookUps(const uint32_t CacheNodebytes) {
        this->bytesOfLookUps += CacheNodebytes;
    }

//    V* getVptr(const K &key) {
//        HashMap_It node_it = getcacheItemIterator(key);
//        if (node_it == cache.end()) return NULL; // have to add it
//        return &(node_it->second.value);
//    }
//
//    V getVvalue(const K &key) {
//        HashMap_It node_it = getcacheItemIterator(key);
//        if (node_it == cache.end()) return V(); // have to add it
//        return node_it->second.value;
//    }
//
//    bool isIncache(const K &key) {
//        if (cache.find(key) != cache.end())
//            return true;
//        else
//            return false;
//    }

    string getAbsoluateHitRatio() {
        if (numOfLookups == 0)
            return "0/x = 0.000000";
        if (numOfLookups == numOfHits) return ToString(numOfLookups)
            + "/" + ToString(numOfHits) + " = 1.000000";
        double hitRatio = (double) numOfHits / (double) numOfLookups;
        string strRatio = ToString(numOfHits) + "/" + ToString(numOfLookups)
            + " = " + doubleToString(hitRatio);
        return strRatio;
    }

    string getByteHitRatio() {

        if (this->bytesOfLookUps == 0) return "0/x = 0.000000";
        if (bytesOfLookUps == bytesOfHits) return doubleToString(bytesOfLookUps) + "/" + doubleToString(bytesOfHits) + " = 1.000000";
        double hitRatio = (double) bytesOfHits / (double) bytesOfLookUps;
        string strRatio = doubleToString(bytesOfHits) + "/" + doubleToString(bytesOfLookUps) + " = " + doubleToString(hitRatio);
        return strRatio;
    }

    string getHitRatio() {
        string strRatio = this->getAbsoluateHitRatio() + ", " + this->getByteHitRatio();
        return strRatio;
    }

    void setMemoryLimitMB(double memoryLimitMB) {
        this->maxMemory_Byte = (long) (memoryLimitMB * RAMEstimator::ONE_MB);
    }

    double getMemoryLimitMB() {
        return RAMEstimator::bytesToMB(this->maxMemory_Byte);
    }

    double getUsedMemoryInCacheMB() {
        return RAMEstimator::bytesToMB(this->usedMemory_Byte);
    }

    // num of entries (CacheNodes) in the cache
    // may not be reach up to the memory hit

    HashMap_St cacheSize() {
        return cache.size();
    }

    void reset() {
        this->numOfHits = 0;
        this->numOfLookups = 0;
        this->bytesOfHits = 0;
        this->bytesOfLookUps = 0;
    }

    void destroy() {
        this->reset();
        cache.clear();
        this->usedMemory_Byte = 0;
    }

    //TODO: deprecate this 

    HashMap<K, CacheNode<K, V> >& getCacheContent() {
        return this->cache;
    }

    bool isCacheEmpty() {
        if (this->cache.size() == 0) return true;
        else return false;
    }

    HashMap_It cache_end() {
        return cache.end();
    }

    virtual void setenable(const bool n) {
    }

    // PLEASE NOTE: check getCacheEntry != cache_end() before put()
    virtual HashMap_It getCacheEntry(const K &key) = 0;

    virtual bool push(CacheNode<K, V>& node) = 0;
    //By Rui,used for delete document
    virtual bool push(CacheNode<K, V>& node, vector<V>& evictedCacheNodes) = 0;

    virtual bool pop(const K &Itemkey) = 0;

    virtual ~Cache() {
        this->destroy();
    }
};

CL_NS_END2

#endif /* SRC_CONTRIBS_LIB_CLUCENE_CACHE_CACHE_H_ */
