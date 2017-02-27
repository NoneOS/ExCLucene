/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef DELTAFACTORY_H_
#define DELTAFACTORY_H_

#include "common.h"
#include "Delta.h"

template <typename T>
using DeltaMap = std::map<std::string, std::shared_ptr<Delta<T> > >;

template <typename T>
class DeltaFactory {
public:
    static DeltaMap<T> sdeltamap;

    // hacked for convenience
    static std::vector<std::shared_ptr<Delta<T> > > allSchemes() {
        std::vector<std::shared_ptr<Delta<T> > > ans;
        for (auto i = sdeltamap.begin(); i != sdeltamap.end(); ++i) {
            ans.push_back(i->second);
        }
        return ans;
    }

    static std::vector<std::string> allNames() {
        std::vector<std::string> ans;
        for (auto i = sdeltamap.begin(); i != sdeltamap.end(); ++i) {
            ans.push_back(i->first);
        }
        return ans;
    }

    static std::shared_ptr<Delta<T> >  getFromName(const std::string &name) {
        if (sdeltamap.find(name) == sdeltamap.end()) {
            std::cerr << "name " << name << " does not refer to a Delta." << std::endl;
            std::cerr << "possible choices:" << std::endl;
            for (auto i = sdeltamap.begin(); i != sdeltamap.end(); ++i) {
                std::cerr << static_cast<std::string>(i->first) << std::endl; // useless cast, but just to be clear
            }
            std::cerr << "for now, I'm going to just return 'RegularDelta'" << std::endl;
            return sdeltamap["RegularDelta"];
        }
        return sdeltamap[name];
    }
};

// C++11 allows better than this, but neither Microsoft nor Intel support C++11 fully.
template <typename T>
static inline DeltaMap<T> initializefactory() {
	DeltaMap<T> cmap;

	cmap["RegularDelta"] = std::shared_ptr<Delta<T> >(new RegularDelta<T>);
	cmap["RegularDeltaUnrolled"] = std::shared_ptr<Delta<T> >(new RegularDeltaUnrolled<T>);

	cmap["RegularDeltaSSE"] = std::shared_ptr<Delta<T> >(new RegularDeltaSSE);
	cmap["RegularDeltaAVX"] = std::shared_ptr<Delta<T> >(new RegularDeltaAVX);

	cmap["CoarseDelta2SSE"] = std::shared_ptr<Delta<T> >(new CoarseDelta2SSE);
	cmap["CoarseDelta4SSE"] = std::shared_ptr<Delta<T> >(new CoarseDelta4SSE);
	cmap["CoarseDelta8AVX"] = std::shared_ptr<Delta<T> >(new CoarseDelta8AVX);

    return cmap;
}

template <typename T>
DeltaMap<T> DeltaFactory<T>::sdeltamap = initializefactory<T>();

#endif /* DELTAFACTORY_H_ */
