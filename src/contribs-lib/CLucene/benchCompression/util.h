/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef UTIL_H_
#define UTIL_H_

#include "common.h"

#ifdef __linux__
#define USE_O_DIRECT
#endif

//#define STATS
// taken from stackoverflow
#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif


#define memset16(dest)       \
	__asm__ __volatile__(    \
			"pxor   %%xmm0, %%xmm0\n\t"    \
			"movdqu %%xmm0, %0\n\t"        \
			"movdqu %%xmm0, %1\n\t"        \
			"movdqu %%xmm0, %2\n\t"        \
			"movdqu %%xmm0, %3\n\t"        \
			:"=m" (dest[0]), "=m" (dest[4]), "=m" (dest[8]), "=m" (dest[12])      \
			 ::"memory", "%xmm0")

#define memset32(dest)       \
	__asm__ __volatile__(    \
			"pxor   %%xmm0, %%xmm0\n\t"    \
			"movdqu %%xmm0, %0\n\t"        \
			"movdqu %%xmm0, %1\n\t"        \
			"movdqu %%xmm0, %2\n\t"        \
			"movdqu %%xmm0, %3\n\t"        \
			"movdqu %%xmm0, %4\n\t"        \
			"movdqu %%xmm0, %5\n\t"        \
			"movdqu %%xmm0, %6\n\t"        \
			"movdqu %%xmm0, %7\n\t"        \
			:"=m" (dest[0]), "=m" (dest[4]), "=m" (dest[8]), "=m" (dest[12]) ,     \
			 "=m" (dest[16]), "=m" (dest[20]), "=m" (dest[24]), "=m" (dest[28])    \
			 ::"memory", "%xmm0")

#define memset64(dest)       \
	__asm__ __volatile__(    \
			"pxor   %%xmm0, %%xmm0\n\t"    \
			"movdqu %%xmm0, %0\n\t"        \
			"movdqu %%xmm0, %1\n\t"        \
			"movdqu %%xmm0, %2\n\t"        \
			"movdqu %%xmm0, %3\n\t"        \
			"movdqu %%xmm0, %4\n\t"        \
			"movdqu %%xmm0, %5\n\t"        \
			"movdqu %%xmm0, %6\n\t"        \
			"movdqu %%xmm0, %7\n\t"        \
			"movdqu %%xmm0, %8\n\t"        \
			"movdqu %%xmm0, %9\n\t"        \
			"movdqu %%xmm0, %10\n\t"        \
			"movdqu %%xmm0, %11\n\t"        \
			"movdqu %%xmm0, %12\n\t"        \
			"movdqu %%xmm0, %13\n\t"        \
			"movdqu %%xmm0, %14\n\t"        \
			"movdqu %%xmm0, %15\n\t"        \
			:"=m" (dest[0]), "=m" (dest[4]), "=m" (dest[8]), "=m" (dest[12]) ,     \
			 "=m" (dest[16]), "=m" (dest[20]), "=m" (dest[24]), "=m" (dest[28]),   \
			 "=m" (dest[32]), "=m" (dest[36]), "=m" (dest[40]), "=m" (dest[44]),   \
			 "=m" (dest[48]), "=m" (dest[52]), "=m" (dest[56]), "=m" (dest[60])    \
			 ::"memory", "%xmm0")

#define memcpy16(src, dest)    \
	__asm__ __volatile__(      \
			"movdqu %4, %%xmm0\n\t"    \
			"movdqu %5, %%xmm1\n\t"    \
			"movdqu %6, %%xmm2\n\t"    \
			"movdqu %7, %%xmm3\n\t"    \
			"movdqu %%xmm0, %0\n\t"    \
			"movdqu %%xmm1, %1\n\t"    \
			"movdqu %%xmm2, %2\n\t"    \
			"movdqu %%xmm3, %3\n\t"    \
			:"=m" (dest[0]), "=m" (dest[4]), "=m" (dest[8]), "=m" (dest[12])    \
			 :"m" (src[0]), "m" (src[4]), "m" (src[8]), "m" (src[12])           \
			  :"memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3")

#define memcpy32(src, dest)    \
	__asm__ __volatile__(      \
			"movdqu %8,  %%xmm0\n\t"    \
			"movdqu %9,  %%xmm1\n\t"    \
			"movdqu %10, %%xmm2\n\t"    \
			"movdqu %11, %%xmm3\n\t"    \
			"movdqu %12, %%xmm4\n\t"    \
			"movdqu %13, %%xmm5\n\t"    \
			"movdqu %14, %%xmm6\n\t"    \
			"movdqu %15, %%xmm7\n\t"    \
			"movdqu %%xmm0, %0\n\t"     \
			"movdqu %%xmm1, %1\n\t"     \
			"movdqu %%xmm2, %2\n\t"     \
			"movdqu %%xmm3, %3\n\t"     \
			"movdqu %%xmm4, %4\n\t"     \
			"movdqu %%xmm5, %5\n\t"     \
			"movdqu %%xmm6, %6\n\t"     \
			"movdqu %%xmm7, %7\n\t"     \
			:"=m" (dest[0]), "=m" (dest[4]), "=m" (dest[8]), "=m" (dest[12]),      \
			 "=m" (dest[16]), "=m" (dest[20]), "=m" (dest[24]), "=m" (dest[28])    \
			 :"m" (src[0]), "m" (src[4]), "m" (src[8]), "m" (src[12]),             \
              "m" (src[16]), "m" (src[20]), "m" (src[24]), "m" (src[28])           \
			  :"memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7")

/**
 * Computes the greatest common divisor
 */
constexpr __attribute__ ((const))
uint32_t gcd(uint32_t x, uint32_t y) {
    return (x % y) == 0 ? y :  gcd(y,x % y);
}

template <class T>
__attribute__ ((const))
T * padTo32bits(T * inbyte) {
    return reinterpret_cast< T *> ((reinterpret_cast<uintptr_t> (inbyte)
            + 3) & ~3);
}

template <class T>
__attribute__ ((const))
const T * padTo32bits(const T * inbyte) {
    return reinterpret_cast<const T *> ((reinterpret_cast<uintptr_t> (inbyte)
            + 3) & ~3);
}

template <class T>
__attribute__ ((const))
T * padTo64bits(T * inbyte) {
    return reinterpret_cast<T *> ((reinterpret_cast<uintptr_t> (inbyte)
            + 7) & ~7);
}

template <class T>
__attribute__ ((const))
const T * padTo64bits(const  T * inbyte) {
    return reinterpret_cast<const T *> ((reinterpret_cast<uintptr_t> (inbyte)
            + 7) & ~7);
}

template <class T>
__attribute__ ((const))
T * padTo128bits(T * inbyte) {
    return reinterpret_cast<T *> ((reinterpret_cast<uintptr_t> (inbyte)
            + 15) & ~15);
}

template <class T>
__attribute__ ((const))
const T * padTo128bits(const T * inbyte) {
    return reinterpret_cast<const T *> ((reinterpret_cast<uintptr_t> (inbyte)
            + 15) & ~15);
}

template <class T>
__attribute__ ((const))
T * padTo64bytes(T * inbyte) {
    return reinterpret_cast<T *> ((reinterpret_cast<uintptr_t> (inbyte)
            + 63) & ~63);
}

template <class T>
__attribute__ ((const))
const T * padTo64bytes(const T * inbyte) {
    return reinterpret_cast<T *> ((reinterpret_cast<uintptr_t> (inbyte)
            + 63) & ~63);
}

template <class T>
__attribute__ ((const))
bool needPaddingTo32Bits(const T * inbyte) {
    return (reinterpret_cast<uintptr_t> (inbyte) & 3) != 0;
}

template <class T>
__attribute__ ((const))
bool needPaddingTo64Bits(const T * inbyte) {
    return (reinterpret_cast<uintptr_t> (inbyte) & 7) != 0;
}

template <class T>
__attribute__ ((const))
bool needPaddingTo128Bits(const T * inbyte) {
    return (reinterpret_cast<uintptr_t> (inbyte) & 15) != 0;
}

template <class T>
bool  needPaddingTo64bytes(const T * inbyte) {
    return (reinterpret_cast<uintptr_t> (inbyte) & 63) != 0;
}

__attribute__ ((const))
inline uint32_t gccbits(const uint32_t v) {
#ifdef _MSC_VER
    if (v == 0) {
        return 0;
    }
    unsigned long answer;
    _BitScanReverse(&answer, v);
    return answer + 1;
#else
    return v == 0 ? 0 : 32 - __builtin_clz(v);
#endif
}

__attribute__ ((const))
inline bool divisibleby(size_t a, uint32_t x) {
    return (a % x == 0);
}

/**
 * compute the deltas, you do not want to use this
 * function if speed matters. This is only for convenience.
 */
template <class container>
container diffs(const container & in, const bool aredistinct) {
    container out;
    if (in.empty())
        return out;
    out.resize(in.size()-1);
    for (size_t k = 0; k < in.size() - 1; ++k)
        if (aredistinct)
            out.push_back(in[k + 1] - in[k] - 1);
        else
            out.push_back(in[k + 1] - in[k]);
    return out;
}

inline void checkifdivisibleby(size_t a, uint32_t x) {
    if (!divisibleby(a, x)) {
        std::ostringstream convert;
        convert << a << " not divisible by " << x;
        throw std::logic_error(convert.str());
    }
}

template<class iter>
void printme(iter i, iter b) {
    for (iter j = i; j != b; ++j)
        std::cout << *j << " ";
    std::cout << std::endl;
}

__attribute__ ((const))
inline uint32_t asmbits(const uint32_t v) {
#ifdef _MSC_VER
    return gccbits(v);
#else
    if (v == 0)
        return 0;
    uint32_t answer;
    __asm__("bsr %1, %0;" :"=r"(answer) :"r"(v));
    return answer + 1;
#endif
}

__attribute__ ((const))
inline uint32_t slowbits(uint32_t v) {
    uint32_t r = 0;
    while (v) {
        r++;
        v = v >> 1;
    }
    return r;
}

__attribute__ ((const))
inline uint32_t bits(uint32_t v) {
    uint32_t r(0);
    if (v >= (1U << 15)) {
        v >>= 16;
        r += 16;
    }
    if (v >= (1U << 7)) {
        v >>= 8;
        r += 8;
    }
    if (v >= (1U << 3)) {
        v >>= 4;
        r += 4;
    }
    if (v >= (1U << 1)) {
        v >>= 2;
        r += 2;
    }
    if (v >= (1U << 0)) {
        v >>= 1;
        r += 1;
    }
    return r;
}

#ifndef _MSC_VER
__attribute__ ((const))
constexpr uint32_t constexprbits(uint32_t v) {
    return v >= (1U << 15) ? 16 + constexprbits(v>>16) :
            (v >= (1U << 7)) ? 8 + constexprbits(v>>8) :
                    (v >= (1U << 3)) ? 4 + constexprbits(v>>4) :
                            (v >= (1U << 1)) ? 2 + constexprbits(v>>2) :
                                    (v >= (1U << 0)) ? 1 + constexprbits(v>>1) :
                                            0;
}
#else

template <int N>
struct exprbits
{
    enum { value = 1 + exprbits<(N >> 1)>::value };
};

template <>
struct exprbits<0>
{
    enum { value = 0 };
};

#define constexprbits(n) exprbits<n>::value

#endif

constexpr uint64_t div_roundup(uint64_t v, uint32_t divisor) {
    return (v + (divisor - 1)) / divisor;
}

template<class iterator>
__attribute__ ((pure))
uint32_t maxbits(const iterator & begin, const iterator & end) {
    uint32_t accumulator = 0;
    for (iterator k = begin; k != end; ++k) {
        accumulator |= *k;
    }
    return gccbits(accumulator);
}

template<class iterator>
uint32_t slowmaxbits(const iterator & begin, const iterator & end) {
    uint32_t accumulator = 0;
    for (iterator k = begin; k != end; ++k) {
        const uint32_t tb = gccbits(*k);
        if (tb > accumulator)
            accumulator = tb;
    }
    return accumulator;
}

__attribute__ ((const))
inline uint32_t getNumberOfBytes(const uint32_t value) {
    if (value > 0x000000FF) {
        if (value > 0x0000FFFF) {
            if (value > 0x00FFFFFF) {
                return 4;
            } else {
                return 3;
            }
        } else {
            return 2;
        }
    } else {
        return 1;
    }
}

//basically, we can sometimes memoize the maxbits computation
// Since the first scan looks at b input words, the second looks
// at b/2, the third looks at b/3... (total related to harmonic numbers)
// it is probably only worthwhile to memoize the first maybe 20% prefix
// (rest can be "naively"  re-scanned if needed)
// also, a useful heuristic should be to start with however many
// bits are required for the first number in the sequence.  Or OR
// the first two or three values together (danger, what if you OR
// more than you'd actually use?)
// alternative heuristic is to start with however many bits you used for the
// last encoding.  See if it works.  Yes: start sequential scan downward.  No:
// start sequential scan upward.
// To be tried...

//template<class t>
//struct bitwise_or : public binary_function<t, t, t> {
  //t operator()(t x, t y) { return x|y; }
//};

//
template<int b, class t, class iterator>
int greedy_bit_size_lookahead( const iterator &begin,
       const iterator &end) {
  //  assert(end- begin <= b);
  std::vector<t> prefixOrBuffer(end-begin);  // consider a  preallocated buffer...

  partial_sum(begin, end, prefixOrBuffer.begin(),
	      [](t x, t y) { return x | y; }  // change dl's + to |
          //bitwise_or<t>()
          );
  // do the bitwise or-ing once only.
  if (end-begin == b) { // expected case, to help out compiler.  Should be unrolled
    for (int i=1; i < 31; ++i)
      if (prefixOrBuffer[ b/i -1] < (static_cast<t>(1)<<i)) return i;
    //assert(false); // cannot get here unless 32+ bits required
    return -1;
  }
  else { // general case, maybe less data than we could pack with 1-bit fields
    for (int i=1; i < 31; ++i) {
      uint64_t indexToCheck = b/i - 1;
      if (indexToCheck >= prefixOrBuffer.size())
        indexToCheck = prefixOrBuffer.size()-1;

      if (prefixOrBuffer[indexToCheck] < (static_cast<t>(1)<<i)) return i;
    }
    // assert(false);
    return -1;
  }
}


// assume the previous bit size is close to the required bit size
template<int b, class t, class iterator>
int greedy_bit_size_lookahead( const iterator &begin,
			       const iterator &end, uint32_t previous_size) {

  uint32_t  span_length = end-begin;
  if (span_length == b) {  // work on the specialization later...
    // try previous size
    if (maxbits(begin, begin+(b/previous_size)) > previous_size) {
      // previous_size is too small; go until you find something bigger that works
      for (uint32_t i=previous_size+1; i < previous_size+32 /* was nothing */ ; ++i)  // upper bound is only to encourage compiler to unroll
	if (maxbits(begin, begin+(b/i)) <= i) return i;
      return -1; // impossible
    }
    else { // previous_size works, but perhaps we can find something smaller that also works
      uint32_t i;
      for (i=previous_size-1; i /* > 0 */ != previous_size-32; --i) {
        if (i == 0) break;  // This funkiness is to encourage unrolling.
	if (maxbits(begin, begin+(b/i)) > i) break;
      }
      return i+1;  //either i=0 and we return 1....or i is the first too-small size
    }
  }
  else {
    // same thing with careful checks to avoid reading past end of buffer
    uint32_t endIdx = b/previous_size;
    if (endIdx >= span_length) endIdx=span_length;

    if (maxbits(begin, begin+endIdx) > previous_size) {
      for (uint32_t i=previous_size+1; ; ++i) {
	endIdx = b/i;
	if (endIdx >= span_length) endIdx=span_length;
	if (maxbits(begin, begin+endIdx) <= i) return i;
      }
      return -1; // impossible
    }
    else {
      uint32_t i;
      for (i=previous_size-1; i > 0; --i) {
	endIdx = b/i;
	if (endIdx >= span_length) endIdx=span_length;
	if (maxbits(begin, begin+endIdx) > i) break;
      }
      return i+1;
    }
  }
}

class BitWidthHistoGram {
public:
    std::vector<double> histo;
    BitWidthHistoGram():histo(33, 0) {}

	std::ostream &display(std::ostream &os = std::cout, const std::string &prefix = "") {
        double sum = 0;
        for (size_t k = 0; k < histo.size(); ++k)
            sum += histo[k];
        if(sum == 0) return os;

		os << prefix << std::endl;
		os << "bitWidth\t%" << std::endl;
        for (size_t k = 0; k < histo.size(); ++k) {
			os << k << "\t" << histo[k] / sum << std::endl;
        }
		return os;
    }

    void eatIntegers(const uint32_t *data, size_t size) {
        for (uint32_t i = 0; i < size; ++i) {
			++histo[asmbits(data[i])];
        }
    }

    void eatDGaps(const uint32_t *data, size_t size) {
        if (size <= 1)
            return;

        for (uint32_t i = 0; i < size - 1; ++i) {
            assert(data[i + 1] > data[i]);
            uint32_t gap = data[i + 1] - data[i] - 1;
            assert(gap < data[i + 1]);
            ++histo[asmbits(gap)];
        }
    }
};

#endif /* UTIL_H_ */
