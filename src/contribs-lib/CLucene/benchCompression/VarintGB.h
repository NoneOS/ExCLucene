/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef VARINTGB_H_
#define VARINTGB_H_

#include "common.h"
#include "IntegerCodec.h"

#ifdef __SSSE3__
#include <x86intrin.h>
namespace VarintTables {
extern const __m128i VarintGBSSEMasks[256];
}  // namespace VarintTables
#endif

namespace VarintTables {
extern const uint8_t VarintGBInputOffset[256];
extern const uint8_t VarintGBKeys[256][4];
}  // namespace VarintTables


struct G4B {
    enum {
        kHeaderSize = 1, // descriptor size (in bytes)
        kGroupSize = 4   // number of encoded integers in a group
    };
};

struct G8B {
    enum {
        kHeaderSize = 2, // descriptor size (in bytes)
        kGroupSize = 8   // number of encoded integers in a groups
    };
};


template<class T>
class VarintGB;

using VarintG4B = VarintGB<G4B>;
using VarintG8B = VarintGB<G8B>;


template<class T>
class VarintGBBase : public IntegerCodec {

//#undef __AVX2__
//#undef __SSSE3__

public:
	/**
	 * Number of bytes for the descriptor.
	 */
	enum { kHeaderSize = T::kHeaderSize };

    /**
     * Number of integers encoded / decoded in one pass.
     */
    enum { kGroupSize = T::kGroupSize };

    /**
     * Maximum encoded size.
     */
    enum { kMaxSize = kHeaderSize + sizeof(uint32_t) * kGroupSize };

    /**
     * Maximum size for n values.
     */
    static size_t maxSize(size_t n) {
    	// Full groups
    	size_t total = (n / kGroupSize) * kFullGroupSize;

        n %= kGroupSize;
        // Incomplete last group, if any
        if (n) {
            total += kHeaderSize + n * sizeof(uint32_t);
        }
        return total;
    }

    /**
     * Size of n values starting at p.
     */
    static size_t totalSize(const uint32_t *p, size_t n) {
        size_t size = 0;
        for ( ; n >= kGroupSize; n -= kGroupSize, p += kGroupSize)
            size += Derived::size(p);
        if (n)
            size += Derived::partialSize(p, n);
        return size;
    }


    virtual void encodeArray(const uint32_t *in, size_t nvalue,
                uint32_t *out, size_t &csize) {
        uint8_t *dst = reinterpret_cast<uint8_t *>(out);
        const uint8_t *const initdst = dst;

        // Full groups
        for ( ; nvalue >= kGroupSize; nvalue -= kGroupSize, in += kGroupSize)
        	dst = Derived::encode(dst, in);
        // Incomplete last group, if any
        if (nvalue)
        	dst = Derived::encode(dst, in, nvalue);

        csize = ((dst - initdst) + 3) / 4;  // number of words consumed
    }

    virtual const uint32_t * decodeArray(const uint32_t *in, size_t csize,
    		uint32_t *out, size_t nvalue) {
        const uint32_t *const endout = out + nvalue;
        const uint8_t *src = reinterpret_cast<const uint8_t *>(in);
        const uint8_t *const initsrc = src;
        while (endout > out) {
            Derived::decode(src, out);
        }

        csize = ((src - initsrc) + 3) / 4;
        return in + csize;
    }

    virtual std::string name() const {
    	std::ostringstream codecname;
    	std::string platform;
#ifdef __AVX2__
    	platform = "AVX";
#else  /* !__AVX2__ */
#ifdef __SSSE3__
    	platform = "SSE";
#else  /* !__SSSE3__ */
    	platform = "Scalar";
#endif  /* __SSSE3__ */
#endif  /* __AVX2__ */
    	codecname << "VarintG" << kGroupSize << "B_" << platform;
    	return codecname.str();
    }

protected:
    static uint8_t key(uint32_t x) {
    	// __builtin_clz is undefined for the x==0 case
    	return 3 - (__builtin_clz(x|1) / 8);
    }

    static uint8_t b0key(uint8_t x) { return x & 3; }
    static uint8_t b1key(uint8_t x) { return (x >> 2) & 3; }
    static uint8_t b2key(uint8_t x) { return (x >> 4) & 3; }
    static uint8_t b3key(uint8_t x) { return (x >> 6) & 3; }

    static const uint32_t kMask[];

private:
    typedef VarintGB<T> Derived;
    enum { kFullGroupSize = kHeaderSize + kGroupSize * sizeof(uint32_t) };
};

template <typename T>
const uint32_t VarintGBBase<T>::kMask[] = {
    0xff, 0xffff, 0xffffff, 0xffffffff
};


/**
 * VarintGB encoding for 32-bit values.
 *
 * Encodes 4 32-bit integers at once, each using 1-4 bytes depending on size.
 * There is one byte of overhead.  (The first byte contains the lengths of
 * the four integers encoded as two bits each; 00=1 byte .. 11=4 bytes)
 *
 * This implementation assumes little-endian and does unaligned 32-bit
 * accesses, so it's basically not portable outside of the x86[_64] world.
 */
template <>
class VarintGB<G4B> : public VarintGBBase<G4B> {
public:
	/**
	 * Return the number of bytes used to encode these four values.
	 */
	static size_t size(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
		return kHeaderSize + kGroupSize + key(a) + key(b) + key(c) + key(d);
	}

	/**
	 * Return the number of bytes used to encode four uint32_t values stored
	 * at consecutive positions in an array.
	 */
	static size_t size(const uint32_t *p) {
		return size(p[0], p[1], p[2], p[3]);
	}

	/**
	 * Return the number of bytes used to encode count (<= 4) values.
	 * If you clip a buffer after these many bytes, you can still decode
	 * the first "count" values correctly (if the remaining size() -
	 * partialSize() bytes are filled with garbage).
	 */
    static size_t partialSize(const uint32_t *p, size_t count) {
        assert(count <= kGroupSize);
        size_t s = kHeaderSize + count;
        for ( ; count; --count, ++p) {
        	s += key(*p);
        }
        return s;
    }

    /**
     * Return the number of values from *p that are valid from an encoded
     * buffer of size bytes.
     */
    static size_t partialCount(const uint8_t *p, size_t size) {
        uint8_t v = *p;
        size_t s = kHeaderSize;
        s += 1 + b0key(v);
        if (s > size) return 0;
        s += 1 + b1key(v);
        if (s > size) return 1;
        s += 1 + b2key(v);
        if (s > size) return 2;
        s += 1 + b3key(v);
        if (s > size) return 3;
        return 4;
    }

    /**
     * Given a pointer to the beginning of an VarintGB-encoded block,
     * return the number of bytes used by the encoding.
     */
    static size_t encodedSize(const uint8_t *p) {
    	return (kHeaderSize + kGroupSize +
    			b0key(*p) + b1key(*p) + b2key(*p) + b3key(*p));
    }

    /**
     * Encode four uint32_t values into the buffer pointed-to by p, and return
     * the next position in the buffer (that is, one character past the last
     * encoded byte).  p needs to have at least size()+4 bytes available.
     */
    static uint8_t * encode(uint8_t *dst, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    	uint8_t k0 = key(a);
    	uint8_t k1 = key(b);
    	uint8_t k2 = key(c);
    	uint8_t k3 = key(d);
    	*dst++ = (k3 << 6) | (k2 << 4) | (k1 << 2) | k0;
    	*reinterpret_cast<uint32_t *>(dst) = a;
    	dst += k0+1;
    	*reinterpret_cast<uint32_t *>(dst) = b;
    	dst += k1+1;
    	*reinterpret_cast<uint32_t *>(dst) = c;
    	dst += k2+1;
    	*reinterpret_cast<uint32_t *>(dst) = d;
    	dst += k3+1;
    	return dst;
    }

    /**
     * Encode four uint32_t values from the array pointed-to by src into the
     * buffer pointed-to by dst, similar to encode(dst,a,b,c,d) above.
     */
    static uint8_t * encode(uint8_t *dst, const uint32_t *src) {
    	return encode(dst, src[0], src[1], src[2], src[3]);
    }

    /**
     * Encode into the buffer pointed-to by dst with count (< 4) uint32_t values
     * from the array pointed-to by src and (4 - count) 0s.
     */
    static uint8_t * encode(uint8_t *dst, const uint32_t *src, size_t count) {
    	assert(count < kGroupSize);

        switch (count) {
        case 3: return dst = encode(dst, src[0], src[1], src[2], 0);

        case 2: return dst = encode(dst, src[0], src[1], 0, 0);

        case 1: return dst = encode(dst, src[0], 0, 0, 0);

        default: return dst;
        }
    }

    /**
     * Decode four uint32_t values from a buffer, and return the next position
     * in the buffer (that is, one character past the last encoded byte).
     * The buffer needs to have at least 3 extra bytes available (they
     * may be read but ignored).
     */
//    static const uint8_t * decode_simple(const uint8_t *src, uint32_t *a, uint32_t *b,
//                                                             uint32_t *c, uint32_t *d) {
//    	size_t k = *reinterpret_cast<const uint8_t*>(src);
//    	const uint8_t *end = src + VarintTables::VarintGBInputOffset[k];
//    	++src;
//    	size_t k0 = b0key(k);
//    	*a = *reinterpret_cast<const uint32_t *>(src) & kMask[k0];
//    	src += k0 + 1;
//    	size_t k1 = b1key(k);
//    	*b = *reinterpret_cast<const uint32_t *>(src) & kMask[k1];
//    	src += k1 + 1;
//    	size_t k2 = b2key(k);
//    	*c = *reinterpret_cast<const uint32_t *>(src) & kMask[k2];
//    	src += k2 + 1;
//    	size_t k3 = b3key(k);
//    	*d = *reinterpret_cast<const uint32_t *>(src) & kMask[k3];
//    	src += k3 + 1;
//    	return end;
//    }

    // table lookup is slower
    static const uint8_t * decode_simple(const uint8_t *src, uint32_t *a, uint32_t *b,
                                                             uint32_t *c, uint32_t *d) {
    	size_t k = *reinterpret_cast<const uint8_t*>(src);
    	const uint8_t *end = src + VarintTables::VarintGBInputOffset[k];
    	++src;
    	size_t k0 = VarintTables::VarintGBKeys[k][0];
    	*a = *reinterpret_cast<const uint32_t *>(src) & kMask[k0];
    	src += k0 + 1;
    	size_t k1 = VarintTables::VarintGBKeys[k][1];
    	*b = *reinterpret_cast<const uint32_t *>(src) & kMask[k1];
    	src += k1 + 1;
    	size_t k2 = VarintTables::VarintGBKeys[k][2];
    	*c = *reinterpret_cast<const uint32_t *>(src) & kMask[k2];
    	src += k2 + 1;
    	size_t k3 = VarintTables::VarintGBKeys[k][3];
    	*d = *reinterpret_cast<const uint32_t *>(src) & kMask[k3];
    	src += k3 + 1;
    	return end;
    }

    /**
     * Decode four uint32_t values from a buffer and store them in the array
     * pointed-to by dst, similar to decode(src,a,b,c,d) above.
     */
    static const uint8_t *decode_simple(const uint8_t *src, uint32_t *dst) {
    	return decode_simple(src, dst, dst + 1, dst + 2, dst + 3);
    }

#ifdef __AVX2__
    static const void decode(const uint8_t * &src, uint32_t * &dst) {
    	uint8_t k0 = src[0];
    	__m128i val0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src + 1));
    	uint8_t k1 = src[VarintTables::VarintGBInputOffset[k0]];
    	__m128i val1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src + 1 + VarintTables::VarintGBInputOffset[k0]));
    	__m256i val = _mm256_inserti128_si256(_mm256_castsi128_si256(val0), val1, 1);

    	__m128i mask0 = VarintTables::VarintGBSSEMasks[k0];
    	__m128i mask1 = VarintTables::VarintGBSSEMasks[k1];
    	__m256i mask = _mm256_inserti128_si256(_mm256_castsi128_si256(mask0), mask1, 1);

        __m256i result = _mm256_shuffle_epi8(val, mask);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), result);

        src += VarintTables::VarintGBInputOffset[k0] + VarintTables::VarintGBInputOffset[k1];
        dst += 2 * kGroupSize;
    }
#else /* !__AVX2__ */
#ifdef __SSSE3__
    static void decode(const uint8_t * &src, uint32_t * &dst) {
        uint8_t k = src[0];
        __m128i val = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src + 1));
        __m128i mask = VarintTables::VarintGBSSEMasks[k];
        __m128i result = _mm_shuffle_epi8(val, mask);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), result);

        src += VarintTables::VarintGBInputOffset[k];
        dst += kGroupSize;
    }
#else  /* !__AVX2__ && !__SSSE3__ */
    static void decode(const uint8_t * &src, uint32_t * &dst) {
    	src = decode_simple(src, dst);
    	dst += kGroupSize;
    }
#endif  /* __SSSE3__ */
#endif  /* __AVX2__ */


#ifdef __SSSE3__
    static const uint8_t * decode(const uint8_t *src, uint32_t *a, uint32_t *b,
    		                                          uint32_t *c, uint32_t *d) {
        uint8_t k = src[0];
        __m128i val = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src + 1));
        __m128i mask = VarintTables::VarintGBSSEMasks[k];
        __m128i result = _mm_shuffle_epi8(val, mask);

        // Extracting 32 bits at a time out of an XMM register is a SSE4 feature
#ifdef __SSE4__
        *a = _mm_extract_epi32(result, 0);
        *b = _mm_extract_epi32(result, 1);
        *c = _mm_extract_epi32(result, 2);
        *d = _mm_extract_epi32(result, 3);
#else  /* !__SSE4__ */
        *a = _mm_extract_epi16(result, 0) + (_mm_extract_epi16(result, 1) << 16);
        *b = _mm_extract_epi16(result, 2) + (_mm_extract_epi16(result, 3) << 16);
        *c = _mm_extract_epi16(result, 4) + (_mm_extract_epi16(result, 5) << 16);
        *d = _mm_extract_epi16(result, 6) + (_mm_extract_epi16(result, 7) << 16);
#endif  /* __SSE4__ */

        return src + VarintTables::VarintGBInputOffset[k];
    }
#else  /* !__SSSE3 */
    static const uint8_t * decode(const uint8_t *src, uint32_t *a, uint32_t *b,
    		                                          uint32_t *c, uint32_t *d) {
    	return decode_simple(src, a, b, c, d);
    }
#endif /* __SSSE3__ */
};


template <>
class VarintGB<G8B> : public VarintGBBase<G8B> {
public:
	/**
	 * Return the number of bytes used to encode these eight values.
	 */
    static size_t size(uint32_t a, uint32_t b, uint32_t c, uint32_t d,
                       uint32_t e, uint32_t f, uint32_t g, uint32_t h) {
        return kHeaderSize + kGroupSize + key(a) + key(b) + key(c) + key(d)
                                        + key(e) + key(f) + key(g) + key(h);
    }

    /**
     * Return the number of bytes used to encode four uint32_t values stored
     * at consecutive positions in an array.
     */
    static size_t size(const uint32_t *p) {
        return size(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    }

    /**
     * Return the number of bytes used to encode count (<= 8) values.
     * If you clip a buffer after these many bytes, you can still decode
     * the first "count" values correctly (if the remaining size() -
     * partialSize() bytes are filled with garbage).
     */
    static size_t partialSize(const uint32_t *p, size_t count) {
        assert(count <= kGroupSize);
        size_t s = kHeaderSize + count;
        for (; count; --count, ++p) {
        	s += key(*p);
        }
        return s;
    }

    /**
     * Return the number of values from *p that are valid from an encoded
     * buffer of size bytes.
     */
    static size_t partialCount(const uint8_t *p, size_t size) {
        size_t s = kHeaderSize;
        s += 1 + b0key(p[0]);
        if (s > size) return 0;
        s += 1 + b1key(p[0]);
        if (s > size) return 1;
        s += 1 + b2key(p[0]);
        if (s > size) return 2;
        s += 1 + b3key(p[0]);

        if (s > size) return 3;
        s += 1 + b0key(p[1]);
        if (s > size) return 4;
        s += 1 + b1key(p[1]);
        if (s > size) return 5;
        s += 1 + b2key(p[1]);
        if (s > size) return 6;
        s += 1 + b3key(p[1]);
        if (s > size) return 7;
        return 8;
    }

    /**
     * Given a pointer to the beginning of an VarintGB-encoded block,
     * return the number of bytes used by the encoding.
     */
    static size_t encodedSize(const uint8_t *p) {
        return (kHeaderSize + kGroupSize +
                b0key(p[0]) + b1key(p[0]) + b2key(p[0]) + b3key(p[0]) +
                b0key(p[1]) + b1key(p[1]) + b2key(p[1]) + b3key(p[1]));
    }

    /**
     * Encode eight uint32_t values into the buffer pointed-to by dst, and return
     * the next position in the buffer (that is, one character past the last
     * encoded byte).  dst needs to have at least size()+4 bytes available.
     */
    static uint8_t *encode(uint8_t *dst, uint32_t a, uint32_t b, uint32_t c, uint32_t d,
                                 uint32_t e, uint32_t f, uint32_t g, uint32_t h) {
        uint8_t k00 = key(a);
        uint8_t k01 = key(b);
        uint8_t k02 = key(c);
        uint8_t k03 = key(d);
        *reinterpret_cast<uint8_t *>(dst++) = (k03 << 6) | (k02 << 4) | (k01 << 2) | k00;

        uint8_t k10 = key(e);
        uint8_t k11 = key(f);
        uint8_t k12 = key(g);
        uint8_t k13 = key(h);
        *reinterpret_cast<uint8_t *>(dst++) = (k13 << 6) | (k12 << 4) | (k11 << 2) | k10;

        *reinterpret_cast<uint32_t *>(dst) = a;
        dst += k00 + 1;
        *reinterpret_cast<uint32_t *>(dst) = b;
        dst += k01 + 1;
        *reinterpret_cast<uint32_t *>(dst) = c;
        dst += k02 + 1;
        *reinterpret_cast<uint32_t *>(dst) = d;
        dst += k03 + 1;

        *reinterpret_cast<uint32_t *>(dst) = e;
        dst += k10 + 1;
        *reinterpret_cast<uint32_t *>(dst) = f;
        dst += k11 + 1;
        *reinterpret_cast<uint32_t *>(dst) = g;
        dst += k12 + 1;
        *reinterpret_cast<uint32_t *>(dst) = h;
        dst += k13 + 1;
        return dst;
    }

    /**
     * Encode eight uint32_t values from the array pointed-to by src into the
     * buffer pointed-to by dst, similar to encode(dst,a,b,c,d,e,f,g,h) above.
     */
    static uint8_t *encode(uint8_t *dst, const uint32_t *src) {
        return encode(dst, src[0], src[1], src[2], src[3],
                           src[4], src[5], src[6], src[7]);
    }

    /**
     * Encode into the buffer pointed-to by dst with count (< 8) uint32_t values
     * from the array pointed-to by src and (8 - count) 0s.
     */
    static uint8_t * encode(uint8_t *dst, const uint32_t *src, size_t count) {
    	assert(count < kGroupSize);

        switch (count) {
        case 7: return dst = encode(dst, src[0], src[1], src[2], src[3], src[4], src[5], src[6], 0);

        case 6: return dst = encode(dst, src[0], src[1], src[2], src[3], src[4], src[5], 0, 0);

        case 5: return dst = encode(dst, src[0], src[1], src[2], src[3], src[4], 0, 0, 0);

        case 4: return dst = encode(dst, src[0], src[1], src[2], src[3], 0, 0, 0, 0);

        case 3: return dst = encode(dst, src[0], src[1], src[2], 0, 0, 0, 0, 0);

        case 2: return dst = encode(dst, src[0], src[1], 0, 0, 0, 0, 0, 0);

        case 1: return dst = encode(dst, src[0], 0, 0, 0, 0, 0, 0, 0);

        default: return dst;
        }
    }

    /**
     * Decode eight uint32_t values from a buffer, and return the next position
     * in the buffer (that is, one character past the last encoded byte).
     * The buffer needs to have at least 3 extra bytes available (they
     * may be read but ignored).
     */
    static const uint8_t *decode_simple(const uint8_t *src, uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *d,
    		                                                uint32_t *e, uint32_t *f, uint32_t *g, uint32_t *h) {
//        uint8_t k0  = *reinterpret_cast<const uint8_t*>(src);
//        uint8_t k1 = *reinterpret_cast<const uint8_t*>(src + 1);
        uint16_t k = *reinterpret_cast<const uint16_t*>(src);
        uint8_t k0 = k & 0xff, k1 = k >> 8;
        const uint8_t *end = src + VarintTables::VarintGBInputOffset[k0] + VarintTables::VarintGBInputOffset[k1];
        src += kHeaderSize;

        size_t k00 = b0key(k0);
        *a = *reinterpret_cast<const uint32_t *>(src) & kMask[k00];
        src += k00 + 1;
        size_t k01 = b1key(k0);
        *b = *reinterpret_cast<const uint32_t *>(src) & kMask[k01];
        src += k01 + 1;
        size_t k02 = b2key(k0);
        *c = *reinterpret_cast<const uint32_t *>(src) & kMask[k02];
        src += k02 + 1;
        size_t k03 = b3key(k0);
        *d = *reinterpret_cast<const uint32_t *>(src) & kMask[k03];
        src += k03 + 1;

        size_t k10 = b0key(k1);
        *e = *reinterpret_cast<const uint32_t *>(src) & kMask[k10];
        src += k10 + 1;
        size_t k11 = b1key(k1);
        *f = *reinterpret_cast<const uint32_t *>(src) & kMask[k11];
        src += k11 + 1;
        size_t k12 = b2key(k1);
        *g = *reinterpret_cast<const uint32_t *>(src) & kMask[k12];
        src += k12 + 1;
        size_t k13 = b3key(k1);
        *h = *reinterpret_cast<const uint32_t *>(src) & kMask[k13];
        src += k13 + 1;

        return end;
    }

//    // table lookup is slower
//    static const uint8_t *decode_simple(const uint8_t *src, uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *d,
//    		                                                uint32_t *e, uint32_t *f, uint32_t *g, uint32_t *h) {
////        uint8_t k0  = *reinterpret_cast<const uint8_t*>(src);
////        uint8_t k1 = *reinterpret_cast<const uint8_t*>(src + 1);
//        uint16_t k = *reinterpret_cast<const uint16_t*>(src);
//        uint8_t k0 = k & 0xff, k1 = k >> 8;
//        const uint8_t *end = src + VarintTables::VarintGBInputOffset[k0] + VarintTables::VarintGBInputOffset[k1];
//        src += kHeaderSize;
//
//        size_t k00 = VarintTables::VarintGBKeys[k0][0];
//        *a = *reinterpret_cast<const uint32_t *>(src) & kMask[k00];
//        src += k00 + 1;
//        size_t k01 = VarintTables::VarintGBKeys[k0][1];
//        *b = *reinterpret_cast<const uint32_t *>(src) & kMask[k01];
//        src += k01 + 1;
//        size_t k02 = VarintTables::VarintGBKeys[k0][2];
//        *c = *reinterpret_cast<const uint32_t *>(src) & kMask[k02];
//        src += k02 + 1;
//        size_t k03 = VarintTables::VarintGBKeys[k0][3];
//        *d = *reinterpret_cast<const uint32_t *>(src) & kMask[k03];
//        src += k03 + 1;
//
//        size_t k10 = VarintTables::VarintGBKeys[k1][0];
//        *e = *reinterpret_cast<const uint32_t *>(src) & kMask[k10];
//        src += k10 + 1;
//        size_t k11 = VarintTables::VarintGBKeys[k1][1];
//        *f = *reinterpret_cast<const uint32_t *>(src) & kMask[k11];
//        src += k11 + 1;
//        size_t k12 = VarintTables::VarintGBKeys[k1][2];
//        *g = *reinterpret_cast<const uint32_t *>(src) & kMask[k12];
//        src += k12 + 1;
//        size_t k13 = VarintTables::VarintGBKeys[k1][3];
//        *h = *reinterpret_cast<const uint32_t *>(src) & kMask[k13];
//        src += k13 + 1;
//
//        return end;
//    }

    static const uint8_t *decode_simple(const uint8_t *src, uint32_t *dst) {
    	return decode_simple(src, dst, dst + 1, dst + 2, dst + 3,
                                  dst + 4, dst + 5, dst + 6, dst + 7);
    }

#ifdef __SSSE3__
    static void decode(const uint8_t * &src, uint32_t * &dst) {
//        uint8_t k0 = src[0], k1 = src[1];
        uint16_t k = *reinterpret_cast<const uint16_t *>(src);
        uint8_t k0 = k & 0xff, k1 = k >> 8;

        const __m128i val0 = _mm_lddqu_si128((const __m128i *)(src + 2));
        const __m128i val1 = _mm_lddqu_si128((const __m128i *)(src + 1 + VarintTables::VarintGBInputOffset[k0]));
        const __m128i mask0 = VarintTables::VarintGBSSEMasks[k0];
        const __m128i mask1 = VarintTables::VarintGBSSEMasks[k1];

#ifdef __AVX2__
        // Note: _mm256_loadu2_m128i isn't supported in gcc 4.8.3; use _mm256_inserti128_si256 instead.
        __m256i val = _mm256_inserti128_si256(_mm256_castsi128_si256(val0), val1, 1);
        __m256i mask = _mm256_inserti128_si256(_mm256_castsi128_si256(mask0), mask1, 1);

        __m256i result = _mm256_shuffle_epi8(val, mask);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), result);
#else  /* !__AVX2__*/
        __m128i result0 = _mm_shuffle_epi8(val0, mask0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), result0);

        __m128i result1 = _mm_shuffle_epi8(val1, mask1);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 4), result1);
#endif /* __AVX2__ */

        src += VarintTables::VarintGBInputOffset[k0] + VarintTables::VarintGBInputOffset[k1];
        dst += kGroupSize;
    }

#else  /* !__SSSE3__ */
    static void decode(const uint8_t * &src, uint32_t * &dst) {
        src = decode_simple(src, dst);
        dst += kGroupSize;
    }
#endif  /* __SSSE3 */

    static const uint8_t * decode(const uint8_t *src, uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *d,
    		                                          uint32_t *e, uint32_t *f, uint32_t *g, uint32_t *h) {
        return decode_simple(src, a, b, c, d, e, f, g, h);
    }
};

#endif /* VARINTGB_H_ */
