/*
 * ListPtr.h
 *
 */

#ifndef SRC_NEW_FUNCTION_HEADER_LISTPTR_H_
#define SRC_NEW_FUNCTION_HEADER_LISTPTR_H_

#include "type.h"
#include "common.h"



using namespace std;
using namespace lucene::analysis;
using namespace lucene::index;
using namespace lucene::util;
using namespace lucene::queryParser;
using namespace lucene::document;
using namespace lucene::search;

class ListPtr{
public:
	float_t maxScore;
	const TCHAR* term;
//	Scorer* scoreDocs;
	int32_t curDoc;
	vector<int32_t> posting;
	vector<float_t> score;
	int32_t elem;



	inline ListPtr(Term* term,unordered_map<wstring,float_t> &score_map){
		int32_t count = 0;
		this->term = term->text();
		Query* q = QueryParser::parse(term->text(),_T("content"),&analyzer);
		Weight* w = q->weight(searcher);
		Scorer* s = w->scorer(reader);
		Scorer* temp_s = s;
		temp_s->next();
		this->curDoc = temp_s->doc();
		wstring termText = term->text();
		this->maxScore = score_map[termText];
		while (s->next()){
			this->posting.push_back(s->doc());
			this->score.push_back(s->score());
		}

		this->posting.push_back(DOCNUM + 1);
		this->score.push_back(0);
		this->elem = 0;








//		s->next();
//		this->scoreDocs = s ;
//		this->curDoc = s->doc();
//		printf ("\n\nTime taken: %d\n\n", (int32_t)(Misc::currentTimeMillis() - time2));
	}


	inline void nextGEQ(const unsigned &d){
//		if(d < curDoc)
//			return ;
//		else{
//
//			if(scoreDocs->skipTo(d)){
//				curDoc = scoreDocs->doc();
//				return ;
//			}
//			else{
//				curDoc = MAXDID +1;
//				return ;
//			}
//
//		}

		if(d <=curDoc){
			return ;
		}
		else{
			while(posting[elem] < d){
				elem ++;
			}
			curDoc = posting[elem];

		}
	}




	ListPtr& operator =(const ListPtr &lp){
		this->term = lp.term;
		this->maxScore = lp.maxScore;
//		this->scoreDocs = lp.scoreDocs;
		this->elem = lp.elem;
		this->posting = lp.posting;
		this->score = lp.score;
		this->curDoc = lp.curDoc;
		return *this;
	}

	~ListPtr()
	{
	}

	friend bool operator<(const ListPtr &la, const ListPtr &lb)
				{
				    return la.curDoc < lb.curDoc;
				}

};



#endif /* SRC_NEW_FUNCTION_HEADER_LISTPTR_H_ */
