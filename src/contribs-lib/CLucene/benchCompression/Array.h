/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef ARRAY_H_
#define ARRAY_H_

#include "common.h"

class Array {
public:
    Array() = default;
    Array(uint64_t len);
    Array(const Array &arr);
    ~Array();


    void reserve(uint64_t len);


	uint32_t *begin() { return pt; }
	const uint32_t *begin() const { return pt; }

	uint32_t *end() { return pt + length; }
	const uint32_t *end() const { return pt + length; }


    uint32_t &operator[](uint64_t idx) { return pt[idx]; }
    const uint32_t &operator[](uint64_t idx) const { return pt[idx]; }

    uint32_t *data() { return pt; }
	const uint32_t *data() const { return pt; }

    uint64_t size() const { return length; }

private:
    uint32_t *pt = nullptr;
    uint64_t length = 0;
};

#endif /* ARRAY_H_ */
