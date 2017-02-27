/**
* This code is released under the
* Apache License Version 2.0 http://www.apache.org/licenses/.
*
*/

#ifndef LINEARREGRESSIONFACTORY_H_
#define LINEARREGRESSIONFACTORY_H_

#include "common.h"
#include "LinearRegression.h"

using LRVec = std::vector<std::shared_ptr<LR> >;
using LRMap = std::map<std::string, LRVec>; 

class LRFactory {
public:
	static LRMap slrmap;

	static std::vector<LRVec> allSchemes() {
		std::vector<LRVec> ans;
		for (auto i = slrmap.begin(); i != slrmap.end(); ++i) {
			ans.push_back(i->second);
		}
		return ans;
	}

	static std::vector<std::string> allNames() {
		std::vector<std::string> ans;
		for (auto i = slrmap.begin(); i != slrmap.end(); ++i) {
			ans.push_back(i->first);
		}
		return ans;
	}

	static LRVec getFromName(const std::string &name) {
		if (slrmap.find(name) == slrmap.end()) {
			std::cerr << "name " << name << " does not refer to a linear regression algorithm." << std::endl;
			std::cerr << "possible choices:" << std::endl;
			for (auto i = slrmap.begin(); i != slrmap.end(); ++i) {
				std::cerr << static_cast<std::string>(i->first) << std::endl;
			}
			std::cerr << "for now, I'm going to just return 'LR'" << std::endl;
			return slrmap["LR"];
		}
		return slrmap[name];
	}
};

#endif /* LINEARREGRESSIONFACTORY_H_ */
