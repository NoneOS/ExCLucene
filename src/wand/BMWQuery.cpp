/*
 * BMWQuery.cpp
 *
 */

#include "BMWQuery.h"


void BMWQuery::oneQuery_process(queryEntry_B &qE,priority_queue <result> &topResult){
		float threshold = 0;

		    qE_iterator_B pivot;
		    qE_iterator_B it;
		    qE_iterator_B pivot_next;


		    int32_t pivotDoc = -1;
		    const TCHAR* pivotTerm;
		    float_t maxposs = 0;
		    int32_t min_nextDoc = MAXDID + 1;

		#ifdef DebugTime
		    timeval begin_time, end_time;
		#endif

		    while(true)
		    {
		#ifdef DebugTime
		        gettimeofday(&begin_time, NULL);
		#endif

		        sortByDid<qE_iterator_B, ListPtr_B>(qE.begin(), qE.end());


		        float_t MaxScoreSum = 0.0;
		        pivotDoc = -1;

		        for(pivot = qE.begin(); pivot != qE.end(); pivot++)
		        {
		            MaxScoreSum += (*pivot)->maxScore;
		            if(MaxScoreSum > threshold)
		            {
//		            	cout << "test...." << endl;
		                pivotDoc = (*pivot)->curDoc;
		                pivotTerm = (*pivot)->term;
		                while((pivot + 1) != qE.end() && (*(pivot + 1))->curDoc == pivotDoc)
		                {
		                    pivot++;
		                }
		                break;
		            }

		        }

		        if(pivot == qE.end())
		            break;

		        if(pivotDoc > MAXDID)
		            break;
//		        cout << pivotDoc << endl;

		        maxposs = 0;

		        for(it = qE.begin(); it <= pivot; ++it)
		        {
		            (*it)->nextShallow(pivotDoc);
		            maxposs += (*it)->getcheckBlockMaxScore();
		        }
		#ifdef DebugTime
		        gettimeofday(&end_time, NULL);
		        calcu_maxposs_time += getTime(begin_time, end_time);

		        gettimeofday(&begin_time, NULL);
		#endif

		        if(maxposs > threshold)
		        {
		            if(qE[0]->curDoc == pivotDoc)
		            {
		                float_t s = 0.0f;
		                for(it = qE.begin(); it <= pivot ;)
		                {
//		                    float_t cur_score = (*it)->idf * BM25((*it)->getFreq(), getDocLen(pivotDoc));
		                	float_t cur_score = (*it)->score[(*it)->posting_ptr + (*it)->elem];
		                    maxposs = maxposs - (*it)->blockMaxScore[(*it) ->currentBlock] +  cur_score;



//		                    cout << "curDoc is " << (*it)->curDoc <<endl;
		                    (*it)->nextGEQ(pivotDoc + 1);
		                    ++it;
//		                    cout <<"curDoc is " << (*it)->curDoc << endl;
//		                    cout << "cur block is " << (*it)->currentBlock << endl;


		                    if(maxposs <= threshold)
		                        break;
		                    s += cur_score;

		                }
		                if(s > threshold)
		                {
		                    result res;
		                    res.did = pivotDoc ;
		                    res.score = s;
		                    topResult.push(res);
		                    if(topResult.size() > topK)
		                    {
		                        topResult.pop();
		                        threshold = topResult.top().score;
		                    }
		                }
		            }
		            else
		            {
		                for(it = qE.begin(); (*it)->curDoc < pivotDoc;) // why not condition: it != qE.end()??
		                {
		                    (*it)->nextGEQ(pivotDoc);
		                    ++it;
		                }
		            }
		#ifdef DebugTime
		            gettimeofday(&end_time, NULL);
		            maxpossMoreThreshold_processTime += getTime(begin_time, end_time);
		#endif
		        }
		        else
		        {
		            min_nextDoc = MAXDID + 1;  // !!!!!! +1
		            pivot_next = pivot + 1;
		            for(it = qE.begin(); it <= pivot; ++it)
		            {
		            	if((*it)->blockMaxDid[(*it)->checkBlock] + 1 < min_nextDoc)
		            		min_nextDoc = (*it)->blockMaxDid[(*it)->checkBlock] + 1;
		            }

		            if(pivot_next!= qE.end() && (*pivot_next) ->curDoc < min_nextDoc)
		                min_nextDoc = (*pivot_next) ->curDoc;

		            for(it = qE.begin(); it <= pivot;)
		            {
		                (*it)->nextGEQ(min_nextDoc);
		                ++it;
		            }
		#ifdef DebugTime
		            gettimeofday(&end_time, NULL);
		            maxpossLessThreshold_processTime += getTime(begin_time, end_time);
		#endif
		        }
		    }

	}



