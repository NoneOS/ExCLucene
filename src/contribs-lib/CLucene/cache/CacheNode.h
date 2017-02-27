/*
 * CacheNode.h
 *
 */

#ifndef SRC_CONTRIBS_LIB_CLUCENE_CACHE_CACHENODE_H_
#define SRC_CONTRIBS_LIB_CLUCENE_CACHE_CACHENODE_H_

#include "util.h"

CL_NS_DEF2(search, cache)

template<class K, class V>
class CLUCENE_EXPORT CacheNode:LUCENE_BASE {
public:
    K key; // key   in my clucene K is pointer
    V value; // value in my clucene V is pointer
    int numOfBytes; // memory usage, in terms of bytes, e.g., one integer takes 4 bytes

    unsigned frequency;
    double cost; // time to get it
    double H_Value; // only used for the GDS replacement, and its variations
    // refer to "Cost-Aware WWW Proxy Caching Algorithms, 1997"
    double H_Value_K; // only used for the GDSFK replacment, refer to "Cost-aware
    // strategies for query result caching in Web Search Engines
    //  , TWEB 2011"

    // only used for the feature-based method, refer to "Improved Techniques for
    //  Result Caching in Web Search Engines"
    long FB_timeStamp_Curr; // time stamp for the current reference
    long FB_timeStamp_Prev; // time stamp for the previous reference
    long FB_timeStamp_Prev2; // time stamp for the previous 2nd reference

    CacheNode<K, V>* prev; // previous node
    CacheNode<K, V>* next; // next node

public:

    CacheNode() : key(), value(),numOfBytes(0)
    {
        frequency = 0;
        cost = 0;
        H_Value = 0;
        H_Value_K = 0;
        FB_timeStamp_Curr = -1;
        FB_timeStamp_Prev = -1;
        FB_timeStamp_Prev2 = -1;
        prev = NULL;
        next = NULL;
    }
    
    CacheNode(K key, V value, int numBytes) :
    key(key), value(value), numOfBytes(numBytes) {
        frequency = 1;
        cost = 0;
        H_Value = 0;
        H_Value_K = 0;
        FB_timeStamp_Curr = -1;
        FB_timeStamp_Prev = -1;
        FB_timeStamp_Prev2 = -1;
        prev = NULL;
        next = NULL;
    }

    CacheNode(const CacheNode<K, V> & cn) :
    key(cn.key), value(cn.value),
    H_Value(cn.H_Value), H_Value_K(cn.H_Value_K),
    numOfBytes(cn.numOfBytes), frequency(cn.frequency), cost(cn.cost),
    FB_timeStamp_Curr(cn.FB_timeStamp_Curr), FB_timeStamp_Prev(cn.FB_timeStamp_Prev), 
    FB_timeStamp_Prev2(cn.FB_timeStamp_Prev2),
    prev(cn.prev), next(cn.next) {
    }

    CacheNode<K, V> & operator=(const CacheNode<K, V> & cn) {
        key = cn.key;
        value = cn.value;
        H_Value = cn.H_Value;
        H_Value_K = cn.H_Value_K;
        numOfBytes = cn.numOfBytes;
        frequency = cn.frequency;
        cost = cn.cost;
        FB_timeStamp_Curr = cn.FB_timeStamp_Curr;
        FB_timeStamp_Prev = cn.FB_timeStamp_Prev;
        FB_timeStamp_Prev2 = cn.FB_timeStamp_Prev2;
        prev = cn.prev;
        next = cn.next;

        return *this;
    }

    // set up the time stamp
    // used for the feature-based method

    void setTimeStamp_FB(long currTimeStamp) {
        this->FB_timeStamp_Prev2 = this->FB_timeStamp_Prev;
        this->FB_timeStamp_Prev = this->FB_timeStamp_Curr;
        this->FB_timeStamp_Curr = currTimeStamp;
    }

    std::string getFB_Key() {
        std::string FBKey = "";
        std::string F1 = "null", F2 = "null", F3 = "null";

        int mod = 8; // each bucket has 8 bins
        if (FB_timeStamp_Curr != -1 & FB_timeStamp_Prev != -1)
            F1 = ToString((FB_timeStamp_Curr - FB_timeStamp_Prev) % mod);
        if (FB_timeStamp_Prev != -1 & FB_timeStamp_Prev2 != -1)
            F2 = ToString((FB_timeStamp_Prev - FB_timeStamp_Prev2) % mod);
        F3 = ToString(frequency % mod);

        FBKey = F1 + "_" + F2 + "_" + F3;
        return FBKey;
    }

    bool equals(CacheNode<K, V> &other) {
        if (this->key == other.key && this->value == other.value)
            return true;
        else return false;
    }

    //FIXME: key & value .tostring()
    std::string toString() {
        std::string str = "";
        str = "[" + key + " = " + value
                + ", numBytes " + this->numOfBytes
                + ", frequency " + this->frequency
                + ", cost " + this->cost + "]";
        return str;
    }

    void increaseFrequency() {
        this->frequency++;
    }

    // HValue depends on an extra LValue
    void setHValue(double LValue) {
        this->H_Value = this->cost / this->numOfBytes + LValue;
    }

    // HValue_K depends on an extra LValue
    void setHValue_K(double LValue) {
        // double freq_k = Math.pow(this->frequency, 2.5);
        double freq_k = pow(this->frequency, 1);
        // freq^k, where k is set to 2.5, refer to "Cost-aware strategies for 
        //  query result caching in Web Search Engines, TWEB 2011"
        this->H_Value_K = (this->cost * freq_k) / this->numOfBytes + LValue;
    }

    // return the h values
    double getHValue() const{
        return this->H_Value;
    }

    // return the h_k values
    double getHValue_K() const{
        return this->H_Value_K;
    }

    double getFrequencyTimesCost() const{
        double value = this->frequency * this->cost;
        return value;
    }

    // return the frequency/szie
    double getFrequencySize() const {
        double value = (this->frequency * 1.0) / this->numOfBytes;
        return value;
    }

    // return the cost/size
    double getCostSize() const{
        double value = this->cost / this->numOfBytes;
        return value;
    }

    // return frequency*cost/size
    double getFrequencyCostSize() const{
        double value = this->getFrequencySize() * this->cost;
        return value;
    }

    // return frequency^k*cost/size
    double getFrequencyCostSize_K() const{
        /**
         *  the equation is: F^k, here k is set to 2.5, which
         *  is recommended by "A cost-aware strategy for query result caching 
         *  in Web Search Engines, ECIR 2009".
         */
        double freq_k = pow(this->getFrequencySize(), 2.5);
        double value = freq_k * this->cost;
        return value;
    }

    virtual ~CacheNode() {
    }
};

CL_NS_END2

#endif /* SRC_CONTRIBS_LIB_CLUCENE_CACHE_CACHENODE_H_ */
