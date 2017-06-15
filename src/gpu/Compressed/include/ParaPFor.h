/*************************************************************************
	> File: ParaPFor.h
	> Author: Naiyong Ao
	> Email: aonaiyong@gmail.com 
	> Time: Wed 28 Jan 2015 11:33:14 PM CST
 ************************************************************************/

#ifndef PARAPFOR_H_
#define PARAPFOR_H_

#include <stdint.h>

#define BS 256

uint32_t encodeArray(const uint32_t *in, uint32_t *out, uint32_t blockSize);
uint32_t decodeArray(const uint32_t *in, uint32_t *out, uint32_t blockSize);

#endif /* PARAPFOR_H_ */

