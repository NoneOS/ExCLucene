/*
 * DynQTFDF.h
 *
 */

#ifndef SRC_CONTRIBS_LIB_CLUCENE_CACHE_DYNQTFDF_H_
#define SRC_CONTRIBS_LIB_CLUCENE_CACHE_DYNQTFDF_H_

#include "DynamicCache.h"

CL_NS_DEF2(search, cache)

template<class K, class V, class Compare>
class DynQTFDF: public DynamicCache<K, V>
{
protected:
    typedef DynamicCache<K, V> _Base;
    typedef typename _Base::HashMap_It HashMap_It;
    typedef typename _Base::HashMap_St HashMap_St;
    typedef typename _Base::HashMap_Vt HashMap_Vt;
    typedef typename std::pair<HashMap_It, bool> HashMap_Ret;
    typedef typename vector<CacheNode<K,V>* >::iterator CacheNodePtrVec_It;
    
    Compare comp;
	vector<CacheNode<K,V>* > CacheNodePtr_vec;

    //@Override
    // update the statistical info
    void moveToHead(CacheNode<K, V> &node) {
        node.increaseFrequency();
        if (CacheNodePtr_vec.front() == &node) {
            make_heap(CacheNodePtr_vec.begin(), CacheNodePtr_vec.end(), comp);
//            CacheNodePtrVec_It nodePtr_it = min_element(CacheNodePtr_vec.begin(), CacheNodePtr_vec.end(), Compare(false));
//            cout << "in moveToHead: " << endl;
//            cout << "min_frequency: " << (*nodePtr_it)->frequency << endl;
//            cout << "front_frequency: " << CacheNodePtr_vec.front()->frequency << endl;
        }
    }
    
    //node isn't in cache, we insert it.Before inserting it, CacheNodePtr_vec is a heap.
    bool insert(CacheNode<K, V> &node) {
        HashMap_Ret ret = _Base::cache.insert(HashMap_Vt(node.key, node));
        if (ret.second) {
            this->usedMemory_Byte += node.numOfBytes;
            CacheNodePtr_vec.push_back(&(ret.first->second));
            if (CacheNodePtr_vec.front()->frequency > 1)//node.frequency = 1
                swap(CacheNodePtr_vec.front(), CacheNodePtr_vec.back());
            //push_heap(CacheNodePtr_vec.begin(), CacheNodePtr_vec.end(), comp);
            return true;
        } else {
            _CLTHROWA_DEL(CL_ERR_Runtime, "push QTFDF CacheNode failed.");
            return false;
        }
    }

    bool getNodeAndEvict()
    {
    	if(CacheNodePtr_vec.empty())
    		return false;

    //	unsigned prev_front_frequency = CacheNodePtr_vec.front()->frequency;
    //	unsigned prev_back_frequency = CacheNodePtr_vec.back()->frequency;
    //	assert(prev_front_frequency <=  prev_back_frequency);

    	pop_heap(CacheNodePtr_vec.begin(), CacheNodePtr_vec.end(), comp);
    	K key = CacheNodePtr_vec.back()->key;
	this->usedMemory_Byte -= CacheNodePtr_vec.back()->numOfBytes;

		/*if(CacheNodePtr_vec.back()->frequency > CacheNodePtr_vec.front()->frequency)
		{
			cout << "error: in DynQTFDF's getNodeAndRemove()" << endl;
			cout << "before pop_heap: " << endl;
			cout << "prev_front_frequency: " << prev_front_frequency << endl;
			cout << "prev_back_frequency: " << prev_back_frequency << endl;
			cout << "after pop_heap: " << endl;

			cout << "CacheNodePtr_vec.front()->frequency: " << CacheNodePtr_vec.front()->frequency << endl;
			cout << "CacheNodePtr_vec.back()->frequency: " << CacheNodePtr_vec.back()->frequency << endl;
			exit(1);
		}*/
    	CacheNodePtr_vec.pop_back();
    	_Base::cache.erase(key);
        
        /*Document*dd;
        if(typeid(CacheNodePtr_vec.back()->value).name()==typeid(dd).name()){
            std::cout << "Delete document" << std::endl;
            _CLDELETE(CacheNodePtr_vec.back()->value);
        }
        _CLDELETE(dd);*/

    	return true;
    }
    //By Rui,used for delete document
    bool getNodeAndEvict(vector<V>& evictedCacheNodes){
    	if(CacheNodePtr_vec.empty())
    		return false;

    //	unsigned prev_front_frequency = CacheNodePtr_vec.front()->frequency;
    //	unsigned prev_back_frequency = CacheNodePtr_vec.back()->frequency;
    //	assert(prev_front_frequency <=  prev_back_frequency);

    	pop_heap(CacheNodePtr_vec.begin(), CacheNodePtr_vec.end(), comp);
    	K key = CacheNodePtr_vec.back()->key;
        evictedCacheNodes.push_back(CacheNodePtr_vec.back()->value);
	this->usedMemory_Byte -= CacheNodePtr_vec.back()->numOfBytes;

		/*if(CacheNodePtr_vec.back()->frequency > CacheNodePtr_vec.front()->frequency)
		{
			cout << "error: in DynQTFDF's getNodeAndRemove()" << endl;
			cout << "before pop_heap: " << endl;
			cout << "prev_front_frequency: " << prev_front_frequency << endl;
			cout << "prev_back_frequency: " << prev_back_frequency << endl;
			cout << "after pop_heap: " << endl;

			cout << "CacheNodePtr_vec.front()->frequency: " << CacheNodePtr_vec.front()->frequency << endl;
			cout << "CacheNodePtr_vec.back()->frequency: " << CacheNodePtr_vec.back()->frequency << endl;
			exit(1);
		}*/
    	CacheNodePtr_vec.pop_back();
    	_Base::cache.erase(key);
        
        /*Document*dd;
        if(typeid(CacheNodePtr_vec.back()->value).name()==typeid(dd).name()){
            std::cout << "Delete document" << std::endl;
            _CLDELETE(CacheNodePtr_vec.back()->value);
        }
        _CLDELETE(dd);*/

    	return true;
    }
    
public:

    DynQTFDF(double memoryLimit, const Compare& comp = Compare(true)) :
    DynamicCache<K, V>(memoryLimit), comp(comp), CacheNodePtr_vec() {
        // TODO Auto-generated constructor stub
    }

    bool pop(const K &Itemkey) {
        HashMap_It hm_it = this->cache.find(Itemkey);
        if (hm_it == this->cache.end())
            return false;

        CacheNode<K, V>* nodePtr = &(hm_it->second);
        if (nodePtr == CacheNodePtr_vec.front()) {
            pop_heap(CacheNodePtr_vec.begin(), CacheNodePtr_vec.end(), comp);
            CacheNodePtr_vec.pop_back();
//			swap(CacheNodePtr_vec.front(), CacheNodePtr_vec.back());
//			CacheNodePtr_vec.pop_back();
//			make_heap(CacheNodePtr_vec.begin(), CacheNodePtr_vec.end(), comp);
        } else {
            swap(nodePtr, CacheNodePtr_vec.back());
            CacheNodePtr_vec.pop_back();
        }
        this->usedMemory_Byte -= nodePtr->numOfBytes;
        _Base::cache.erase(hm_it);
        
        /*Document*dd;
        if(typeid(nodePtr->value).name()==typeid(dd).name()){
            std::cout << "Delete document" << std::endl;
            _CLDELETE(nodePtr->value);
        }
        _CLDELETE(dd);*/
        
        return true;
    }

	~DynQTFDF()
	{}
};

CL_NS_END2

#endif /* SRC_CONTRIBS_LIB_CLUCENE_CACHE_DYNQTFDF_H_ */
