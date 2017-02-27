/*
 * DynLRU.h
 *
 */

#ifndef SRC_CONTRIBS_LIB_CLUCENE_CACHE_DYNLRU_H_
#define SRC_CONTRIBS_LIB_CLUCENE_CACHE_DYNLRU_H_

#include "DynamicCache.h"

CL_NS_DEF2(search, cache)

template<class K, class V>
class DynLRU : public DynamicCache<K, V> {
    CacheNode<K, V>* first; // head of the double-linked list
    CacheNode<K, V>* last; // tail of the double-linked list

protected:

    typedef DynamicCache<K, V> _Base;
    typedef typename _Base::HashMap_It HashMap_It;
    typedef typename _Base::HashMap_St HashMap_St;
    typedef typename _Base::HashMap_Vt HashMap_Vt;
    typedef typename std::pair<HashMap_It, bool> HashMap_Ret;

    void moveToHead(CacheNode<K, V> &node) {
        if (&node == this->first)
            return;
        if (node.prev != NULL)
            node.prev->next = node.next;
        if (node.next != NULL)
            node.next->prev = node.prev;
        if (&node == this->last)
            this->last = node.prev;
        if (this->first != NULL) { // move the current node to the front
            node.next = this->first;
            this->first->prev = &node;
        }
        node.prev = NULL;
        this->first = &node;
        if (last == NULL) // initialization
            last = first;
    }

    bool insert(CacheNode<K, V> &node) {
        HashMap_Ret ret = _Base::cache.insert(HashMap_Vt(node.key, node));
        if (ret.second) {
            this->usedMemory_Byte += node.numOfBytes;
            moveToHead(ret.first->second);
            return true;
        } else {
            _CLTHROWA_DEL(CL_ERR_Runtime, "push LRU CacheNode failed.");
            return false;
        }
    }

    // remove the least recent use CacheNode, which is at the last one of List
    bool getNodeAndEvict() {
        if (this->last == NULL)
            return false;
        K& key = last->key;         // reference
        HashMap_It ptr = this->cache.find(key);
        if (ptr == this->cache.end())
        {
            _CLTHROWA_DEL(CL_ERR_Runtime, "getEvictNode LRU failed.");
            return false;
        }
        if (last->prev != NULL) {
            last->prev->next = NULL;
        }
        this->usedMemory_Byte -= last->numOfBytes;
        last = last->prev;
        this->cache.erase(ptr);    
        
        return true;
        //return removeLast();
    }
    //By Rui,used for delete document
    bool getNodeAndEvict(vector<V>& evictedCacheNodes) {
        if (this->last == NULL)
            return false;
        K& key = last->key;         // reference
        evictedCacheNodes.push_back(last->value);
        HashMap_It ptr = this->cache.find(key);
        if (ptr == this->cache.end())
        {
            _CLTHROWA_DEL(CL_ERR_Runtime, "getEvictNode LRU failed.");
            return false;
        }
        if (last->prev != NULL) {
            last->prev->next = NULL;
        }
        this->usedMemory_Byte -= last->numOfBytes;
        last = last->prev;
        this->cache.erase(ptr);    
        
        return true;
        //return removeLast();
    }
    

    //    bool removeLast() {
    //        if (this->last == NULL)
    //            return false;
    //        K key = last->key;
    //        HashMap_It hm_it = this->cache.find(key);
    //        if (hm_it == this->cache.end())
    //            assert(false);
    //
    //        if (last->prev != NULL) {
    //            last->prev->next = NULL;
    //        }
    //
    //        this->usedMemory_Byte -= last->numOfBytes;
    //        last = last->prev;
    //        this->cache.erase(hm_it);
    //        return true;
    //    }

public:

    DynLRU(double maxMemory_MByte) : DynamicCache<K, V>(maxMemory_MByte),
    first(NULL), last(NULL) {

    }

    bool pop(const K &Itemkey) {
        HashMap_It hm_it = this->cache.find(Itemkey);
        if (hm_it == this->cache.end())
            return false;

        CacheNode<K, V> &node = hm_it->second;
        // step 1: set its pointers
        if (node.prev != NULL) {
            node.prev->next = node.next;
        }
        if (node.next != NULL) {
            node.next->prev = node.prev;
        }
        if (last == &node) // if there is only one element to delete, after that, last = first = null
            last = node.prev;
        if (first == &node)
            first = node.next;

        // release space consumed
        this->usedMemory_Byte -= node.numOfBytes;

        // step 2: remove it from cache
        this->cache.erase(hm_it);
        
        /*Document*dd;
        if(typeid(node.value).name()==typeid(dd).name()){
            std::cout << "Delete document" << std::endl;
            _CLDELETE(node.value);
        }
        _CLDELETE(dd);*/
        
        return true;
    }

};

CL_NS_END2

#endif /* SRC_CONTRIBS_LIB_CLUCENE_CACHE_DYNLRU_H_ */
