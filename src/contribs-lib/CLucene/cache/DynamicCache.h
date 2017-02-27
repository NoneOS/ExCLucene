/*
 * DynamicCache.h
 *
 */

#ifndef SRC_CONTRIBS_LIB_CLUCENE_CACHE_DYNAMICCACHE_H_
#define SRC_CONTRIBS_LIB_CLUCENE_CACHE_DYNAMICCACHE_H_

#include "Cache.h"

CL_NS_DEF2(search, cache)

template<class K, class V>
class DynamicCache : public Cache<K, V> {
protected:
    typedef Cache<K, V> _Base;
    typedef typename _Base::HashMap_It HashMap_It;
    typedef typename _Base::HashMap_St HashMap_St;
    typedef typename _Base::HashMap_Vt HashMap_Vt;
    typedef typename std::pair<HashMap_It, bool> HashMap_Ret;

    // to be over-written later by its sub-class
    // whenever, inserting or getting a node, invoke this function
    virtual void moveToHead(CacheNode<K, V> &node) = 0;

    virtual bool insert(CacheNode<K, V>& cnode) = 0;
    
    // get a node to evict
    virtual bool getNodeAndEvict() = 0;
    //By Rui,used for delete document
    virtual bool getNodeAndEvict(vector<V>& evictedCacheNodes) = 0;

public:

    DynamicCache(double maxMemory_MByte) : Cache<K, V>(maxMemory_MByte) {
    }

    HashMap_It getCacheEntry(const K &key) {
        HashMap_It ret = this->cache.find(key);
        this->numOfLookups++;
        if (ret != this->cache.end()) {
            this->numOfHits++;
            this->bytesOfHits += ret->second.numOfBytes;
            this->bytesOfLookUps += ret->second.numOfBytes;
            moveToHead(ret->second); // move to head
        }
        return ret;
    }

    // Pleas NOTE: use getCacheEntry() instead of push() for existing node
    virtual bool push(CacheNode<K, V> &node) {
        if (node.numOfBytes > _Base::maxMemory_Byte)
            return false;

        // HashMap_It it = this->cache.find(node.key);
        // if (it != this->cache.end()) {
        //    moveToHead(it->second);
        //    return true;
        // }

        while (_Base::isCacheFullPlusOneNode(node)) {
            if (getNodeAndEvict() == false)
                break;
        }
        return this->insert(node);
    }
    //By Rui,used for delete document
    virtual bool push(CacheNode<K, V> &node, vector<V>& evictedCacheNodes) {
        if (node.numOfBytes > _Base::maxMemory_Byte)
            return false;

        // HashMap_It it = this->cache.find(node.key);
        // if (it != this->cache.end()) {
        //    moveToHead(it->second);
        //    return true;
        // }

        while (_Base::isCacheFullPlusOneNode(node)) {
            if (getNodeAndEvict(evictedCacheNodes) == false)
                break;
        }
        return this->insert(node);
    }

    virtual bool pop(const K &Itemkey) = 0;

    virtual ~DynamicCache() {
    }

};

CL_NS_END2



#endif /* SRC_CONTRIBS_LIB_CLUCENE_CACHE_DYNAMICCACHE_H_ */
