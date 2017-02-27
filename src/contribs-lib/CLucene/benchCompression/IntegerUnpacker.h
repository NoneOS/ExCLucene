/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef INTEGERUNPACKER_H_
#define INTEGERUNPACKER_H_

#include "common.h"

class IntegerUnpacker {
public:
	virtual void unpack(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit) = 0;

	/* assumes that integers fit in the prescribed number of bits */
	virtual void packwithoutmask(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit) = 0;

	virtual void pack(const uint32_t *  __restrict__  in, uint32_t *  __restrict__  out,
			const uint32_t bit) = 0;

	virtual std::string name() const = 0;

	virtual ~IntegerUnpacker() = default;
};

#endif /* INTEGERUNPACKER_H_ */
