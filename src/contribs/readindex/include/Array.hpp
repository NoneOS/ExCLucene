/*
 * Array.hpp
 *
 */

#ifndef ARRAY_HPP_
#define ARRAY_HPP_

#include <iostream>
#include <cassert>
#include <cstring>
#include <vector>

struct Array {
    uint32_t *pt;
    uint32_t *index;
    uint32_t div;
    uint32_t length;
    Array();
    Array(uint32_t len);
    void reset();
    void assign(uint32_t len);
    void final();
    uint32_t &operator[](uint32_t idx);
    void push_back(uint32_t val);
    size_t size();
    void buildIndex();
    void releaseIndex();
};

Array::Array() : pt(nullptr), index(nullptr), div(0), length(0) {
}

Array::Array(uint32_t len) : index(0), div(0) {
    try {
        pt = new uint32_t[len];
    }    catch (const std::bad_alloc &e) {
        std::cout << "Array(n) alloc failed: " << e.what() << std::endl;
    }
    length = 0;
}

void Array::reset() {
    length = 0;
}

void Array::assign(uint32_t len) {
    try {
        pt = new uint32_t[len];
    }    catch (const std::bad_alloc &e) {
        std::cout << "Alloc Array failed: " << e.what() << std::endl;
    }
}

void Array::final() {
    if (pt != NULL)
        delete []pt;
}

uint32_t &Array::operator[](uint32_t idx) {
    return pt[idx];
}

void Array::push_back(uint32_t val) {
    pt[this->length++] = val;
}

size_t Array::size() {
    return this->length;
}

void Array::releaseIndex() {
    delete []index;
}

void Array::buildIndex() {
    uint32_t div = (pt[length - 1] + 31) / 32 + 1;
    this->index = new uint32_t[div];
    memset(this->index, 0, sizeof (uint32_t) * div);
    for (uint32_t i = 0; i < length; i++) {
        uint32_t idx = pt[i] >> 5;
        uint32_t shift = pt[i] & 0x1f;
        assert(idx < div);
        index[idx] |= 1 << shift;
    }
    //extern space_cost += 4 * div;     //FIXME:CHECK extern
}

bool equal(Array &A, Array &B) {
    if (A.size() != B.size()) {
        std::cout << A.size() << "\t" << B.size() << std::endl;
        return false;
    }
    for (uint32_t i = 0; i < A.size(); i++) {
        if (A[i] != B[i]) {
            return false;
        }
    }
    return true;
}

uint32_t check_sum(Array &a) {
    uint32_t check = 0;
    for (uint32_t i = 0; i < a.size(); i++) {
        check += a[i];
    }
    return check;
}

void print(Array &a) {
    std::cout << "> ";
    for (uint32_t i = 0; i < a.size(); i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}

bool check_wrong(Array &a, Array &b) {
    if (a.size() != b.size()) {
        std::cout << "size is not equal\n";
        std::cout << "1 size is " << a.size() << ", 2 size is " << b.size() << "\n";
        return false;
    } else
        for (uint32_t j = 0; j < a.size(); j++) {
            if (a.pt[j] != b.pt[j]) {
                std::cout << "elem is not equal\n";
                // print(a);
                // print(b);
                return false;
                break;
            }
        }
    return true;
}

void print(std::vector<uint32_t> &vec) {
    std::cout << "> ";
    for (uint32_t i = 0; i < vec.size(); i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

void printInterval(std::vector<uint32_t> &vec) {
    assert(vec.size() % 2 == 0);
    for (uint32_t i = 0; i < vec.size(); i += 2) {
        std::cout << "[" << vec[i] << ", " << vec[i + 1] << "]";
    }
    std::cout << std::endl;
}

void final(Array *arr) {
    arr->final();
    delete arr;
}

#endif /* ARRAY_HPP_ */
