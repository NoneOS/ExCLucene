/*
 * BMWQuery.h
 *
 */

#ifndef SRC_NEW_FUNCTION_HEADER_BMWQUERY_H_
#define SRC_NEW_FUNCTION_HEADER_BMWQUERY_H_

#include "querybase.h"
#include "type.h"
#include "ListPtr_B.h"

class BMWQuery: public QueryBase{

protected:

	vector<queryEntry_B> vecQuerySet_B;
	unordered_map<wstring,float_t> maxscoreMap;

	double calcu_maxposs_time;
	double maxpossMoreThreshold_processTime;
	double maxpossLessThreshold_processTime;

public:
	BMWQuery(): calcu_maxposs_time(0),maxpossMoreThreshold_processTime(0),maxpossLessThreshold_processTime(0)
	{
	}


	void loadMaxScore(const string &maxScore_path)
	{
		cout << "start loadMaxScore..." << endl;
		ios_base::sync_with_stdio(false);
		locale loc("zh_CN.utf8");
		wcin.imbue(loc);
		wcout.imbue(loc);
		wifstream file(maxScore_path);
		file.imbue(loc);
		wstring wline;
		while(getline(file,wline)){
			wstring termId;
			float_t maxScore;
			wistringstream stream(wline);
			stream >> termId >> maxScore;
			maxscoreMap[termId] = maxScore;
			}

	}

	void loadQuerySet(const string &querySet_path)
	{
		cout << "start loadQuerySet..." << endl;
		wifstream fquery(querySet_path);
		wstring wline;
		wstring termId;
		while(getline(fquery,wline)){
			wistringstream stream(wline);
			queryEntry_B oneQuery;
			while(stream >> termId){
		//			termId = tempId.c_str();
		//			const TCHAR* fld = L"content";
					Term* term = new Term(L"content",termId.c_str());
					ListPtr_B* list = new ListPtr_B(term,maxscoreMap);
					oneQuery.push_back(list);
				}
			vecQuerySet_B.push_back(oneQuery);
		}


	}

	std::string query_method() const
	{
		return "BMW";
	}

	void oneQuery_process(queryEntry_B &qE,priority_queue <result> &topResult);

	/*

	void oneQuery_process(queryEntry_B &qE,priority_queue <result> &topResult){
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
		                	float_t cur_score = (*it)->score[(*it)->curDoc];
		                    maxposs = maxposs - (*it)->blockMaxScore[(*it) ->currentBlock] +  cur_score;

		                    (*it)->nextGEQ(pivotDoc + 1);
		                    ++it;

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

*/




	void querySet_process(const string &resultSet_path)
	{
	    vector<queryEntry>::size_type queryNum = vecQuerySet_B.size();
	    cout << "query num: " << queryNum << endl;
	    //clock_t whole_time = 0, ct; // !!!!!set initial value
	    double whole_time = 0.0;
	    timeval start_time, finish_time;

	    char topkStr[20];
	    sprintf(topkStr , "%d", topK);
	    string path = resultSet_path + query_method() + "top" + topkStr + ".txt";
	    cout << path << endl;
	    ofstream os(path.c_str());
	    query_no = 0;
	    for(vector<queryEntry_B>::iterator it = vecQuerySet_B.begin(); it != vecQuerySet_B.end(); it++)
	    {
	        gettimeofday(&start_time, NULL);

	        priority_queue<result> res;
	        oneQuery_process(*it, res);
//	        cout << "test......" << endl;
	        query_no++;
	        gettimeofday(&finish_time, NULL);
	      //  whole_time += clock() - ct;
	        whole_time += getTime(start_time, finish_time); // ms
	        print(os, res);
	    }
	    cout << "query method: " << query_method() << endl;
	   // cout << "process querySet, whole use time:" << 1.0 *  whole_time / CLOCKS_PER_SEC  << "s" << endl;
	    cout << "process querySet, whole use time:" <<  whole_time / 1000  << "s" << endl;
	    cout << "querys/s: " << 1.0 * queryNum * 1000.0 / whole_time << endl;
	    cout << "ms/per query: " << whole_time / queryNum << endl;
//	    print_info();

	    os.close();
	}


	void print(ostream &os, priority_queue<result> &res)
	{
	    os << res.size() << "\t";
//	    while(res.size() > 0)
//	    {
////	    	docResultFreq[res.top().did]++;
//
//	        os << res.top().did << "(" << res.top().score << ")" << "\t";
//	        res.pop();
//	    }
	    unsigned int  did[20];
	    float score[20];
	    int i = 0;
	    while(res.size() > 0)
	    {
	    //	    	docResultFreq[res.top().did]++;
//	    	os << res.top().did << "(" << res.top().score << ")" << "\t";
	    	did[i] = res.top().did;
	    	score[i] = res.top().score;
	    	i++;
	    	res.pop();
	    }
	    for(int j = i - 1;j >= 0;j--){
	    	os << did[j] << "(" << score[j] << ")" << "\t";
	    }
	    os << endl;
	}

	qE_iterator_B  nextShallowed(queryEntry &qE, qE_iterator it, unsigned PivotDoc, unsigned &min_nextDoc, float &maxposs);

	void print_info()
	{
		cout << "calcu_maxposs_time: " << calcu_maxposs_time / 1000 << "s" << endl;
		cout << "maxpossMoreThreshold_processTime: " << maxpossMoreThreshold_processTime / 1000 << "s" << endl;
        cout << "maxpossLessThreshold_processTime: " << maxpossLessThreshold_processTime / 1000 << "s" << endl;
	}



};



#endif /* SRC_NEW_FUNCTION_HEADER_BMWQUERY_H_ */
