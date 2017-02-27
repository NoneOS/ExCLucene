/*
 * querybase.h
 *
 */

#ifndef SRC_NEW_FUNCTION_HEADER_QUERYBASE_H_
#define SRC_NEW_FUNCTION_HEADER_QUERYBASE_H_

#include "common.h"
#include "type.h"

using namespace std;

class QueryBase{
public:

	virtual void loadMaxScore(const string &maxScore_path) = 0;

	virtual void loadQuerySet(const string &querySet_path) = 0;

	//virtual void oneQuery_process(queryEntry &qE,priority_queue <result> &topResult) ;


	virtual void querySet_process(const string &resultSet_path) {}

	virtual void print(ostream &os, priority_queue<result> &res) = 0;

//	void print_info(){
//		 cout << "redefine this function, print your needed information." << endl;
//	}

	virtual string query_method() const = 0;

	virtual ~QueryBase() = default;

};




















#endif /* SRC_NEW_FUNCTION_HEADER_QUERYBASE_H_ */
