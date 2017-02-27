/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef ENTROPY_H_
#define ENTROPY_H_

#include "common.h"
#include "util.h"

class EntropyRecorder {
public:
    EntropyRecorder(): counter(), totallength(0) { }

    void clear() {
        counter.clear();
        totallength = 0;
    }

	std::ostream &display(std::ostream &os = std::cout, const std::string &prefix = "") {
		os << prefix << "\t" << computeShannon() << std::endl;
		return os;
	}

    void eat(const uint32_t * in, const size_t length) {
        if (length == 0)
            return;
        totallength += length;
        for (uint32_t k = 0; k < length; ++k, ++in) {
            maptype::iterator i = counter.find(*in);
            if (i != counter.end())
                i->second += 1;
            else
                counter[*in] = 1;
        }
    }

    double computeShannon() {
        double total = 0;
        for (maptype::iterator i = counter.begin(); i != counter.end(); ++i) {
            const double x = static_cast<double>(i->second);
            total += x / static_cast<double>(totallength) * log(static_cast<double>(totallength) / x) / log(2.0);
        }
        return total;
    }

    __attribute__ ((pure))
    double computeDataBits() {
        double total = 0;
        for (maptype::const_iterator i = counter.begin(); i != counter.end(); ++i) {
            total += static_cast<double>(i->second) / static_cast<double>(totallength) * static_cast<double>(gccbits(i->first));
        }
        return total;
    }

    typedef std::unordered_map<uint32_t, size_t>  maptype;
    maptype counter;
    size_t totallength;
};

#endif /* ENTROPY_H_ */
