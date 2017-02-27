/*
 * codecfactory.h
 *
 */

#ifndef SRC_NEW_FUNCTION_CODECFACTORY_H_
#define SRC_NEW_FUNCTION_CODECFACTORY_H_


#include "common.h"
#include "WandQuery.h"
#include "BMWQuery.h"
#include "type.h"


static std::map<string, shared_ptr<QueryBase>> initializefactory() {
    std::map <string, shared_ptr<QueryBase>> schemes;

    schemes["wand"] = shared_ptr<QueryBase>(new WandQuery());
    schemes["bmw"] = shared_ptr<QueryBase>(new BMWQuery());

    return schemes;
}


class Factory{
public:
	static map<string,shared_ptr<QueryBase>> querymap;
	static shared_ptr<QueryBase> defaultptr;

	static vector<shared_ptr<QueryBase>> allSchemes(){
		vector <shared_ptr<QueryBase>> ans;
		for(auto i = querymap.begin();i != querymap.end(); ++i)
			ans.push_back(i->second);
		return ans;
	}

	static vector<string> allNames(){
		vector <string> ans;
		for (auto i = querymap.begin(); i != querymap.end(); ++i)
			ans.push_back(i->first);
		return ans;
	}

	static string getName(QueryBase &v) {
	    for (auto i = querymap.begin(); i != querymap.end() ; ++i) {
	        if (i->second.get() == &v)
	            return i->first;
	        }
	        return "UNKNOWN";
	    }
	static bool valid(string name) {
	    return (querymap.find(name) != querymap.end()) ;
	    }
	static shared_ptr<QueryBase> &getFromName(string name) {
	     if (querymap.find(name) == querymap.end()) {
	            cerr << "name " << name << " does not refer to a CODEC." << endl;
	            cerr << "possible choices:" << endl;
	            for (auto i = querymap.begin(); i != querymap.end(); ++i) {
	                cerr << static_cast<string>(i->first) << endl; // useless cast, but just to be clear
	            }
	            return defaultptr;
	        }
	        return querymap[name];
	    }

};

map<string, shared_ptr<QueryBase>> Factory::querymap = initializefactory();

shared_ptr<QueryBase> Factory::defaultptr = shared_ptr<QueryBase> (nullptr);












#endif /* SRC_NEW_FUNCTION_CODECFACTORY_H_ */
