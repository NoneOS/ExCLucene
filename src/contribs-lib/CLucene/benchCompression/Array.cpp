/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#include "Array.h"

Array::Array(uint64_t len): length(len), 
	                        pt(new uint32_t[len]) { }

Array::Array(const Array &arr) {
	reserve(arr.size());
	for (uint64_t i = 0; i < length; ++i) {
		pt[i] = arr[i];
	}
}

void Array::reserve(uint64_t len) {
	if (len <= length)
		return;

	uint32_t *arr = new uint32_t[len];
	if (length > 0) {
		for (uint64_t i = 0; i < length; ++i) {
			arr[i] = pt[i];
		}
		delete[] pt;
		length = 0;
	}

	length = len;
    pt = arr;
}

Array::~Array() {
	if (pt == nullptr) 
		return;

    delete[] pt;
	length = 0;
}

