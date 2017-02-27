/*
 * StaticCache.h
 *
 */

#ifndef SRC_CONTRIBS_LIB_CLUCENE_CACHE_STATICCACHE_H_
#define SRC_CONTRIBS_LIB_CLUCENE_CACHE_STATICCACHE_H_

#include "Cache.h"

CL_NS_DEF2(search, cache)

template<typename K, typename V>
class StaticCache : public Cache<K, V> {
    typedef Cache<K, V> _Base;
    typedef typename _Base::HashMap_It HashMap_It;
    typedef typename _Base::HashMap_St HashMap_St;
    typedef typename _Base::HashMap_Vt HashMap_Vt;
    typedef typename std::pair<HashMap_It, bool> HashMap_Ret;

protected:
    bool enableCounting;

public:

    StaticCache(double memoryLimitMB, bool en = true) : Cache<K, V>(memoryLimitMB), enableCounting(en) {
    }

    HashMap_It getCacheEntry(const K &key) {
        HashMap_It ret = this->cache.find(key);
        this->numOfLookups++;
        if (ret != this->cache.end()) {
            if (this->enableCounting)
                ret->second.increaseFrequency();
            this->numOfHits++;
            this->bytesOfHits += ret->second.numOfBytes;
            this->bytesOfLookUps += ret->second.numOfBytes;
        }
        return ret;
    }

    bool push(CacheNode<K, V>& cnode) {
        // for the static cache, reject any new items if it is already full
        if (_Base::isCacheFullPlusOneNode(cnode))
            return false;
        
        if (this->enableCounting)// at the training phrase
            cnode.increaseFrequency();
        HashMap_Ret ret = this->cache.insert(HashMap_Vt(cnode.key, cnode));
        if (ret.second) {
            this->usedMemory_Byte += cnode.numOfBytes;
            return true;
        } else {
            _CLTHROWA_DEL(CL_ERR_Runtime, "push Static CacheNode failed.");
            return false;
        }
    }

    bool pop(const K &Itemkey) {
        HashMap_It ptr = this->cache.find(Itemkey);
        if (ptr == this->cache.end())
            return false;
        this->usedMemory_Byte -= ptr->second.numOfBytes;
        this->cache.erase(ptr);        
        return true;
    }

    void setenable(const bool status) {
        this->enableCounting = status;
    }

    bool getCountStatus() const {
        return this->enableCounting;
    }
    
    //By Rui,used for delete document
    virtual bool push(CacheNode<K, V> &node, vector<V>& evictedCacheNodes){
    }

    StaticCache() : Cache<K, V>(0) {
        enableCounting = true;
    }

    virtual ~StaticCache() {
    }
};

CL_NS_END2

#endif /* SRC_CONTRIBS_LIB_CLUCENE_CACHE_STATICCACHE_H_ */
