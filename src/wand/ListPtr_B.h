/*
 * ListPtr_B.h
 *
 */

#ifndef SRC_NEW_FUNCTION_HEADER_LISTPTR_B_H_
#define SRC_NEW_FUNCTION_HEADER_LISTPTR_B_H_


#include "common.h"
#include "type.h"



using namespace std;
using namespace lucene::analysis;
using namespace lucene::index;
using namespace lucene::util;
using namespace lucene::queryParser;
using namespace lucene::document;
using namespace lucene::search;

class ListPtr_B{
public:
	float_t maxScore;
	const TCHAR* term;
	int32_t curDoc;
	vector<int32_t> posting;
	vector<float_t> score;

	unsigned blockNum; // the number of block  // !
	unsigned currentBlock;
	unsigned checkBlock;
	unsigned elem;
	unsigned posting_ptr;

	unsigned last_block_size;
	vector<int32_t> blockMaxDid;
	vector<float_t> blockMaxScore;
	vector<unsigned> blockPostingSize;
	bool fBflag;




	inline ListPtr_B(Term* term,unordered_map<wstring,float_t> &score_map){
		unsigned count = 0;
		unsigned blockNum = 0;
		float_t blockMaxScore = 0;
		this->term = term->text();
		Query* q = QueryParser::parse(term->text(),_T("content"),&analyzer);
		Weight* w = q->weight(searcher);
		Scorer* s = w->scorer(reader);
		Scorer* temp_s = s;
		temp_s->next();
		this->curDoc = temp_s->doc();
		wstring termText = term->text();
		this->maxScore = score_map[termText];
		this->fBflag = false;
		this->elem = 0;
		this->currentBlock = 0;
		this->checkBlock = 0;
		this->posting_ptr = 0;
		while (s->next()){
			this->posting.push_back(s->doc());
			this->score.push_back(s->score());
			count ++;
			if(blockMaxScore < s->score())
				blockMaxScore = s->score();
			if(count == BS){
				this->blockMaxDid.push_back(s->doc());
				this->blockMaxScore.push_back(blockMaxScore);
				this->blockPostingSize.push_back(count);
				blockNum++;
				count = 0;
				blockMaxScore = 0;
			}
		}



		//  is last block?
		if(count != 0){
			this->last_block_size = count;
			this->blockPostingSize.push_back(count);
			blockNum++;
			this->blockNum = blockNum;
			this->blockMaxScore.push_back(blockMaxScore);
		}
		this->posting.push_back(DOCNUM);
		this->score.push_back(0);
		this->blockMaxScore.push_back(0);
		this->blockMaxDid.push_back(DOCNUM);



	}

	inline void nextGEQ(const int32_t &d) // guarantee d <= blockMaxDid_it[blockNum - 1]
	{

	    if(d <= curDoc) return;

	    if(d > blockMaxDid[currentBlock]){
	        while (d > blockMaxDid[currentBlock]) {
	        	posting_ptr += blockPostingSize[currentBlock];
	        	currentBlock++;


	            }

	        if(currentBlock == blockNum  )
	        {
	            curDoc = MAXDID + 1;
	            return ;
	        }
//	        if (currentBlock != blockNum - 1)
//	        {
//	            curDoc = posting[currentBlock*BS];
//	        }
//	        else {
//	        	curDoc = posting[currentBlock*BS];
//
//	        }
//	        if(currentBlock != blockNum -1){
//	        	curDoc = posting[posting_ptr];
//	        }
//	        else{
//	        	curDoc = posting[posting_ptr];
//
//	        }
	        elem = 0;
	        fBflag = false;
//	        curDoc = posting[currentBlock*BS];

	    }

	    while (d > curDoc) {
	        elem++;
	        curDoc = posting[posting_ptr + elem];
	    }
	}



	inline unsigned nextShallow(const int32_t &d)
	{
	    //assert(d <= blockMaxDid_ptr[blockNum - 1]);
	    checkBlock = currentBlock;
//	    cout << "checkBlock is " << checkBlock << endl;
//	    cout << "block is " << blockMaxDid.size() << endl;
	    while(d > blockMaxDid[checkBlock])
	    {
	        checkBlock++;
	    }
	    return blockMaxDid[checkBlock] + 1;
	}

	inline float_t getcheckBlockMaxScore()
	{
		return blockMaxScore[checkBlock];
	};

	inline float_t getcurrentBlockMaxScore()
	{
	    return blockMaxScore[currentBlock];
	};


	friend bool operator<(const ListPtr_B &la, const ListPtr_B &lb)
	{
		return la.curDoc < lb.curDoc;
	}






	ListPtr_B& operator =(const ListPtr_B &lp){
		this->maxScore = lp.maxScore;
		this->term = lp.term;
		this->curDoc = lp.curDoc;
		this->posting = lp.posting;
		this->score = lp.score;
		this->blockNum = lp.blockNum;
		this->currentBlock = lp.currentBlock;
		this->checkBlock = lp.checkBlock;
		this->elem = lp.elem;
		this->blockMaxDid = lp.blockMaxDid;
		this->blockMaxScore = lp.blockMaxScore;
		this->fBflag = lp.fBflag;
		this->posting_ptr = lp.posting_ptr;
		this->last_block_size = lp.last_block_size;
		this->blockPostingSize = lp.blockPostingSize;

		return *this;
	}

	~ListPtr_B()
	{
	}

};


#endif /* SRC_NEW_FUNCTION_HEADER_LISTPTR_B_H_ */
