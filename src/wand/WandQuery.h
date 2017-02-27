/*
 * WandQuery.h
 */

#ifndef SRC_NEW_FUNCTION_HEADER_WANDQUERY_H_
#define SRC_NEW_FUNCTION_HEADER_WANDQUERY_H_

#include "querybase.h"
#include "type.h"
#include "ListPtr.h"


class WandQuery: public QueryBase{

protected:
//	vecGlobalInfo &vGI;
	vector<queryEntry> vecQuerySet;
	unordered_map<wstring,float_t> maxscoreMap;

	double findPivotDoc_time;
	double if_processTime;
	double else_processTime;

public:

	WandQuery():findPivotDoc_time(0),if_processTime(0),else_processTime(0)
	{}


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
//		int  i = 0;
		while(getline(fquery,wline)){
//			cout << i++ << endl;
			wistringstream stream(wline);
			queryEntry oneQuery;
			while(!stream.eof()){
					stream >> termId;
//					termId = tempId.c_str();
//					const TCHAR* fld = L"content";
					Term* term = new Term(L"content",termId.c_str());
					ListPtr* list = new ListPtr(term,maxscoreMap);
					oneQuery.push_back(list);
				}
			vecQuerySet.push_back(oneQuery);

		}


	}

	void oneQuery_process(queryEntry &qE,priority_queue <result> &topResult);
/*
	void oneQuery_process(queryEntry &qE,priority_queue <result> &topResult){
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
//						cout  << "curDoc:"<< (*it)->curDoc << "  " << "docId:" << (*it)->termId->doc() << endl;

						if((*it)->scoreDocs->doc() == c){
							float_t score = (*it)->scoreDocs->score();
							sum += score;
						}

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

	*/


	std::string query_method() const
	{
		return "wand";
	}

	void querySet_process(const string &resultSet_path)
	{
	    vector<queryEntry>::size_type queryNum = vecQuerySet.size();
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
	    for(vector<queryEntry>::iterator it = vecQuerySet.begin(); it != vecQuerySet.end(); it++)
	    {
	        gettimeofday(&start_time, NULL);

	        priority_queue<result> res;
	        oneQuery_process(*it, res);
	        //res = process_onequery_withKth(*it, query_no);
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
	    for(int j = i - 1;j >= 0;j--)
	    	{
	    	  	os << did[j] << "(" << score[j] << ")" << "\t";
	    	}

	    os << endl;
	}


	void print_info()
	{
		cout << "findPivotDoc_time: " << findPivotDoc_time / 1000 << "s" << endl;
		cout << "if_processTime: " << if_processTime / 1000 << "s" << endl;
		cout << "else_processTime: " << else_processTime / 1000 << "s" << endl;
	}



};























#endif /* SRC_NEW_FUNCTION_HEADER_WANDQUERY_H_ */
