/*
 * SortedBySomething.h
 *
 */

#ifndef SRC_CONTRIBS_SEARCHING_SORTEDBYSOMETHING_H_
#define SRC_CONTRIBS_SEARCHING_SORTEDBYSOMETHING_H_

#include "CLucene/cache/CacheNode.h"
#include "CLucene/cache/Snippet.h"

CL_NS_USE2(search, cache)

// Maybe we should use them as friend functions of Cache/CacheNode

template<typename It, typename T>
inline void SortByFrequencyWithPtr(It start, It end){ //assumes that derefernecing an iterator with *it gives a T* that has a frequency member
        std::sort(start, end, [](const T* a, const T* b){ return a->frequency < b->frequency; });
}

//template<typename It, typename T>
//inline void SortByFrequencyWithRef(It start, It end){ //assumes that derefernecing an iterator with *it gives a T* that has a frequency member
//        std::sort(start, end, [](const T& a, const T& b){ return a.frequency < b.frequency; });
//}
//
//template<typename It, typename T>
//inline void SortByCost(It start, It end){ //assumes that derefernecing an iterator with *it gives a T* that has a cost member
//        std::sort(start, end, [](const T* a, const T* b){ return a->cost < b->cost; });
//}
//
//template<typename It, typename T>
//inline void SortByCostSize(It start, It end){ //assumes that derefernecing an iterator with *it gives a T* that has a getCostSize() member
//        std::sort(start, end, [](const T* a, const T* b){ return a->getCostSize() < b->getCostSize(); });
//}
//
//template<typename It, typename T>
//inline void SortByFrequencySize(It start, It end){ //assumes that derefernecing an iterator with *it gives a T* that has a getFrequencySize() member
//        std::sort(start, end, [](const T* a, const T* b){ return a->getFrequencySize() < b->getFrequencySize(); });
//}
//
//template<typename It, typename T>
//inline void SortByFrequencyCostSize(It start, It end){ //assumes that derefernecing an iterator with *it gives a T* that has a did member
//        std::sort(start, end, [](const T* a, const T* b){ return a->getFrequencyCostSize() < b->getFrequencyCostSize(); });
//}
//
//template<typename It, typename T>
//inline void SortByFrequencyTimesCost(It start, It end){ //assumes that derefernecing an iterator with *it gives a T* that has a getFrequencyTimesCost() member
//        std::sort(start, end, [](const T* a, const T* b){ return a->getFrequencyTimesCost() < b->getFrequencyTimesCost(); });
//}

template<typename key, typename value>
bool SortCacheNodeByFrequency(const CacheNode<key, value> *a, const CacheNode<key, value> *b) {
    return a->frequency > b->frequency;
}

template<typename key, typename value>
bool SortCacheNodeByFrequencyAndSize(const CacheNode<key, value> *a, const CacheNode<key, value> *b) {
    return a->getFrequencySize() > b->getFrequencySize();
}

template<typename key, typename value>
class SortFrequencyClass
{
	  bool reverse;
public:
	 SortFrequencyClass(const bool& revparam = false)
	{
		reverse = revparam;
	}
	bool operator() (const CacheNode<key, value>* lhs, const CacheNode<key, value>* rhs) const
	{
		if (reverse) return (lhs->frequency > rhs->frequency); // min heap, heap top is the smallest.
		else return (lhs->frequency < rhs->frequency); // max heap, heap top is the biggest.
	}
};

template<typename key, typename value>
class SortFrequencySizeClass
{
	  bool reverse;
public:
	 SortFrequencySizeClass(const bool& revparam = false)
	{
		reverse = revparam;
	}
	bool operator() (const CacheNode<key, value>* lhs, const CacheNode<key, value>* rhs) const
	{
		if (reverse) return (lhs->getFrequencySize() > rhs->getFrequencySize()); // min heap, heap top is the smallest.
		else return (lhs->getFrequencySize() < rhs->getFrequencySize()); // max heap, heap top is the biggest.
	}
};

template<typename key, typename value>
class SortFrequencyTimesCostClass
{
	  bool reverse;
public:
	  SortFrequencyTimesCostClass(const bool& revparam = false)
	{
		reverse = revparam;
	}
	bool operator() (const CacheNode<key, value>* lhs, const CacheNode<key, value>* rhs) const
	{
		if (reverse) return (lhs->getFrequencyTimesCost() > rhs->getFrequencyTimesCost()); // min heap, heap top is the smallest.
		else return (lhs->getFrequencyTimesCost() < rhs->getFrequencyTimesCost()); // max heap, heap top is the biggest.
	}
};


template<typename key, typename value>
class SortFrequencyCostSizeClass
{
	  bool reverse;
public:
	  SortFrequencyCostSizeClass(const bool& revparam = false)
	{
		reverse = revparam;
	}
	bool operator() (const CacheNode<key, value>* lhs, const CacheNode<key, value>* rhs) const
	{
		if (reverse) return (lhs->getFrequencyCostSize() > rhs->getFrequencyCostSize()); // min heap, heap top is the smallest.
		else return (lhs->getFrequencyCostSize() < rhs->getFrequencyCostSize()); // max heap, heap top is the biggest.
	}
};

template<typename key, typename value>
class SortFrequencyCostSizeKClass
{
	  bool reverse;
public:
	  SortFrequencyCostSizeKClass(const bool& revparam = false)
	{
		reverse = revparam;
	}
	bool operator() (const CacheNode<key, value>* lhs, const CacheNode<key, value>* rhs) const
	{
		if (reverse) return (lhs->getFrequencyCostSize_K() > rhs->getFrequencyCostSize_K()); // min heap, heap top is the smallest.
		else return (lhs->getFrequencyCostSize_K() < rhs->getFrequencyCostSize_K()); // max heap, heap top is the biggest.
	}
};



template<typename key, typename value>
class SortCostClass
{
	  bool reverse;
public:
	  SortCostClass(const bool& revparam = false)
	{
		reverse = revparam;
	}
	bool operator() (const CacheNode<key, value>* lhs, const CacheNode<key, value>* rhs) const
	{
		if (reverse) return (lhs->cost > rhs->cost); // min heap, heap top is the smallest.
		else return (lhs->cost < rhs->cost); // max heap, heap top is the biggest.
	}
};

template<typename key, typename value>
class SortCostSizeClass
{
	  bool reverse;
public:
	  SortCostSizeClass(const bool& revparam = false)
	{
		reverse = revparam;
	}
	bool operator() (const CacheNode<key, value>* lhs, const CacheNode<key, value>* rhs) const
	{
		if (reverse) return (lhs->getCostSize() > rhs->getCostSize()); // min heap, heap top is the smallest.
		else return (lhs->getCostSize() < rhs->getCostSize()); // max heap, heap top is the biggest.
	}
};

template<typename key, typename value>
class SortGDSClass
{
	  bool reverse;
public:
	  SortGDSClass(const bool& revparam = false)
	{
		reverse = revparam;
	}
	bool operator() (const CacheNode<key, value>* lhs, const CacheNode<key, value>* rhs) const
	{
		if (reverse) return (lhs->getHValue() > rhs->getHValue()); // min heap, heap top is the smallest.
		else return (lhs->getHValue() < rhs->getHValue()); // max heap, heap top is the biggest.
	}
};

template<typename key, typename value>
class SortGDSFKClass
{
	  bool reverse;
public:
	  SortGDSFKClass(const bool& revparam = false)
	{
		reverse = revparam;
	}
	bool operator() (const CacheNode<key, value>* lhs, const CacheNode<key, value>* rhs) const
	{
		if (reverse) return (lhs->getHValue_K() > rhs->getHValue_K()); // min heap, heap top is the smallest.
		else return (lhs->getHValue_K() < rhs->getHValue_K()); // max heap, heap top is the biggest.
	}
};




//



#endif /* SRC_CONTRIBS_SEARCHING_SORTEDBYSOMETHING_H_ */
