/*
 * SegBlock.hpp
 *
 */

#ifndef SEGBLOCK_HPP_
#define SEGBLOCK_HPP_

#include "Array.hpp"
#include <algorithm>
#include <iomanip>

typedef struct Partition {
    uint32_t part_id;
    uint32_t part_beg;
    uint32_t part_len;
} Partition;

class SegBlockBase {
public:
    std::vector<Partition> PartTable;

    SegBlockBase() {
    }

    virtual ~SegBlockBase() {
    }

    void reset();
};

void SegBlockBase::reset() {
    PartTable.clear();
}

/**
 * HashSegment refers to domain segmentation
 */

class HashSegment : public SegBlockBase {

    class Partition_Finder {
        uint32_t val;
    public:

        Partition_Finder(const uint32_t in) :
        val(in) {
        }

        bool operator()(const std::vector<Partition>::value_type &value) {
            return value.part_id == this->val;
        }
    };

    uint32_t mask;
    uint32_t shift;

public:

    HashSegment() :
    mask(0), shift(0) {
    }

    HashSegment(uint32_t highBits) {
        assert(highBits <= 25);
        std::cout << "Segment hbits: " << highBits << std::endl;
        shift = 25 - highBits;
        mask = ((1 << highBits) - 1) << shift;
    }

    ~HashSegment() {
    }

    void buildSegment(Array *alist);
    void buildSegment(const uint32_t *list, const uint32_t llen);

    uint32_t hashfunc(const uint32_t docID);
    uint32_t getSeg(const uint32_t segID, uint32_t& beg);

    uint32_t getSegSize();

    void debuginfo();
    void clear();
};

void HashSegment::buildSegment(Array *alist) //TODO
{
}

void HashSegment::buildSegment(const uint32_t *list, const uint32_t llen) {
    uint32_t pos = 0;
    std::vector<Partition>::iterator it;
    Partition elem;

    while (pos < llen) {
        it = std::find_if(PartTable.begin(), PartTable.end(),
                Partition_Finder(hashfunc(list[pos])));
        if (it == PartTable.end()) {
            elem.part_id = hashfunc(list[pos]);
            elem.part_beg = pos;
            elem.part_len = 1;
            PartTable.push_back(elem);
        } else {
            it->part_len++;
        }
        pos++;
    }
}

uint32_t HashSegment::hashfunc(const uint32_t docID) {
    return ((docID & mask) >> shift);
}

uint32_t HashSegment::getSeg(const uint32_t segID, uint32_t& beg) {
    std::vector<Partition>::iterator it;
    it = std::find_if(PartTable.begin(), PartTable.end(),
            Partition_Finder(segID));
    if (it == PartTable.end())
        return 0;
    else {
        beg = it->part_beg;
        return it->part_len;
    }
}

uint32_t HashSegment::getSegSize() {
    return this->PartTable.size();
}

void HashSegment::debuginfo() {
    std::vector<Partition>::iterator it = PartTable.begin();
    std::cout << "============== debug segment ==============" << std::endl;
    std::cout << " Total Segment: " << PartTable.size() << ", mask_code: 0x"
            << std::hex << this->mask << std::endl;
    while (it != PartTable.end()) {
        std::cout << " * " << "segID:" << std::setw(3) << std::dec << it->part_id
                << ", " << "segBegin:" << std::setw(6) << it->part_beg << ", "
                << "segLength:" << std::setw(5) << it->part_len << std::endl;
        it++;
    }
}

void HashSegment::clear() {
    this->reset();
}

#endif /* SEGBLOCK_HPP_ */
