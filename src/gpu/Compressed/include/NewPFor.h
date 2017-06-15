/*************************************************************************
	> File: NewPFor.h
	> Author: Naiyong Ao
	> Email: aonaiyong@gmail.com 
	> Time: Sun 01 Feb 2015 07:14:06 PM CST
 ************************************************************************/

#ifndef NEWPFOR_H_
#define NEWPFOR_H_

#include <stdint.h>

#define BS 256 // block size

uint32_t encodeArray(const uint32_t *in, uint32_t *out, uint32_t blockSize);
uint32_t decodeArray(const uint32_t *in, uint32_t *out, uint32_t blockSize);

#endif /* NEWPFOR_H_ */
