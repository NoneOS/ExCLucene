/*
 * WandQuery.cpp
 *
 */

#include "WandQuery.h"

void WandQuery::oneQuery_process(queryEntry &qE,priority_queue <result> &topResult){
		float_t threshold = 0;
		qE_iterator pivot;
		const TCHAR* pivotTerm;
		int32_t pivotDoc;
		//	int32_t i = 0;
		//	Term* term = NULL;
		while(true){

			sortByDid<qE_iterator, ListPtr>(qE.begin(), qE.end());
			float_t maxScoreSum = 0.0;
			for(pivot = qE.begin();pivot != qE.end();pivot++){

				maxScoreSum +=(*pivot)->maxScore;
				if(maxScoreSum > threshold){
					break;
				}
			}
			if(pivot == qE.end()){
				break;
			}
			pivotDoc = (*pivot)->curDoc;
			if(pivotDoc > MAXDID){
				break;
			}
			pivotTerm = (*pivot)->term;
			if(qE[0]->curDoc == pivotDoc){
				float_t sum = 0.0f;
				for(qE_iterator it = qE.begin();it != qE.end() && (*it)->curDoc == pivotDoc;)
				{
						int32_t c = (*it)->curDoc;

//						if((*it)->scoreDocs->doc() == c){
//							float_t score = (*it)->scoreDocs->score();
//							sum += score;
//						}
						float_t score = (*it)->score[(*it)->elem];
						sum += score;


					(*it)->nextGEQ(pivotDoc + 1);
					++it;
				}
				if(sum > threshold){
					result res;
					res.did = pivotDoc;
					res.score = sum;
					topResult.push(res);
					if(topResult.size() > topK){
						topResult.pop();
						threshold = topResult.top().score;
					}
				}
			}
			else{
				for(qE_iterator it = qE.begin();(*it)->curDoc < pivotDoc;)
				{
					(*it)->nextGEQ(pivotDoc);
					++it;
				}
			}
		}
	}

