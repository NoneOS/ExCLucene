
#ifndef INTEGRATIONOFREADERANDPISEQUENTIAL_H_
#define INTEGRATIONOFREADERANDPISEQUENTIAL_H_

//#include "IntegrationOfReaderAndPISEQUENTIAL.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unordered_map>
#include "CLucene/_ApiHeader.h"
#include "CLucene/StdHeader.h"
#include "CLucene/_clucene-config.h"
#include "CLucene/search/Similarity.h"
#include "CLucene/search/BooleanQuery.h"
#include "CLucene.h"
#include "CLucene/config/repl_tchar.h"
#include "CLucene/config/repl_wchar.h"
#include "CLucene/util/Misc.h"
#include "CLucene/util/Equators.h"
#include "CLucene/search/Scorer.h"
#include "CLucene/search/Similarity.h"
#include "CLucene/search/Query.h"
#include "CLucene/search/Hits.h"
#include "ListPtr.hpp"
#include "CLucene/search/TermQuery.h"
#include "CLucene/index/Terms.h"


using namespace std;
using namespace lucene::analysis;
using namespace lucene::index;
using namespace lucene::util;
using namespace lucene::queryParser;
using namespace lucene::document;
using namespace lucene::search;



#include "CLucene/_ApiHeader.h"
#include "../core/CLucene/index/IndexReader.h"
#include "../core/CLucene/index/IndexWriter.h"

#include "IOUtils.hpp"
#include "Utils.hpp"
#include "Grammar.hpp"

using namespace Grammar_Consts;

struct posting_index{
	uint32_t length;
	uint64_t offset;
};

class IntegOfRAP{
      public :

	~IntegOfRAP();

//      void TermConversion(const char* in_filename,const char* out_filename);
//      void queryConversion(const char* in_filename1,const char* in_filename2,const char* out_filename);
//      void IndexConversion(const char* in_filename,const char* first_filename,const char* doc_filename,const char* freq_filename);
      void IndexConversion(const char* in_filename,const char* out_filename);
      inline Production *Insert_new_list();
      inline bool Compare(Production *p, uint32_t *list);
      inline bool NT_find_prefix(Production *p, uint32_t *list, uint32_t &length);
      inline uint32_t getProTlen(const Production* p);
      inline Production *Hash_find_prefix(uint32_t *list, uint32_t len, uint32_t &tag) ;
      inline uint32_t DJBHash(char *str, uint32_t len);
      inline uint32_t Hash_Function(uint32_t x, uint32_t y);
      inline uint32_t Key_find(uint32_t p);
      inline bool Find_aplha_beta(uint32_t x, uint32_t y, Node *r, uint32_t &key, Digram *&q);
      inline void Add_digram_index(Node *fp, Production *q, uint32_t key);
      void Remove_digram_index(uint32_t x, uint32_t y, Node *tp);
      inline uint32_t Insert_new_reduced_production(Node *tp, Digram *dp, uint32_t contain_nt);
      inline void Update(uint32_t sn, Digram *dp);
      inline void Append_production(Production *p, uint32_t beta);
      void grammar_list(uint32_t* list, uint32_t len, bool re_only);
      void Remove_waste();
      void Write_Index();


//      Digram *dp_tbl[digram_index_size + 1];;
//      Production *hash_tbl[hash_tbl_size + 1];
//      Production pp_tbl[pp_max_productions]; // posting patterns
//      Production rp_tbl[rp_max_productions]; // productions
//      uint32_t rp_appear[rp_max_productions];
//      uint32_t pp_num;
//      uint32_t rp_num;
//      bool I;
//
//  	IntegOfRAP(){
//  		//dp_tbl = new Digram* [digram_index_size + 1];
//  		//hash_tbl = new Production* [hash_tbl_size + 1];
//  		//pp_tbl = new Production[pp_max_productions];
//  		//rp_tbl = new Production[rp_max_productions];
//  		//rp_appear = new uint32_t [rp_max_productions];
//  		memset(rp_appear, 0, rp_max_productions * sizeof(uint32_t));
//  		pp_num = rp_num = 0;
//  		I = false;
//  	}
//      uint32_t phrase[2];
//
//      typedef std::map<uint32_t, uint32_t> MapRelation;
//      typedef std::vector<std::vector<uint32_t>> RmvList;
//
//      RmvList rmvdict, rmvpost;
//      std::string glb_inputp, glb_outputp;

};

//IndexReader code
uint64_t OFFSET = 0;
unordered_map <wstring,unsigned> hmap;
//unordered_map <wstring,int>::iterator iter

using namespace Grammar_Consts;

/******************** Global Struct ********************/
static Digram *dp_tbl[digram_index_size + 1]; // bigram hash
static Production *hash_tbl[hash_tbl_size + 1]; // rp_tbl hash
static Production pp_tbl[pp_max_productions]; // posting patterns
static Production rp_tbl[rp_max_productions]; // productions

static uint32_t phrase[2];
static uint32_t rp_appear[rp_max_productions] = { 0 };

static uint32_t pp_num = 0;
static uint32_t rp_num = 0;
uint32_t hbits = 1;
static bool I = false;

typedef std::map<uint32_t, uint32_t> MapRelation;
typedef std::vector<std::vector<uint32_t>> RmvList;

RmvList rmvdict, rmvpost;

std::string glb_inputp, glb_outputp;

#define ArrayIndex 26000000
static uint32_t  outArray[ArrayIndex];

void IntegOfRAP::IndexConversion(const char* in_filename,const char* out_filename)
{
	ios_base::sync_with_stdio(false);
	//locale loc("zh_CN.utf8");
	locale loc("en_US.utf8");
	//wcin.imbue(loc);
	//wcout.imbue(loc);
	locale::global(loc);

//	FILE* outWriter = fopen(out_filename, "wb");
//	FILE* docWriter = fopen(doc_filename, "wb");
//	FILE* freqWriter = fopen(out_filename, "wb");

	unsigned freq;
	unsigned docId;
	int32_t count;
	int32_t length;
	IndexReader* reader = IndexReader::open(in_filename);
	if(reader == NULL)
	{
		perror("Open file recfile !");
		exit(1);
	}
	//int32_t numdoc = reader->numDocs();
//	cout << "numdoc is " << numdoc << endl;
	TermEnum* termEnum = reader->terms();
	//uint32_t i=0;
	while(termEnum->next()){
		count = 0;
		Term* term=termEnum->term();
		TermDocs* termDocs = reader->termDocs(term);
		while(termDocs->next())
		{
			docId = unsigned(termDocs->doc());
			freq = unsigned(termDocs->freq());
			outArray[count] = docId;
//			fwrite(&docId,sizeof(unsigned),1,docWriter);
//			fwrite(&freq,sizeof(unsigned),1,freqWriter);
			count++;
		}
		for(uint32_t i = 1; i < count; i++)
			if(outArray[i] < outArray[i-1])
				cout << "error" << endl;
		grammar_list(outArray, count, false);
		termDocs->close();
	}
	Remove_waste();
	Write_Index();
	//fwrite(outArray,count,1,outWriter);
//	fclose(firstWriter);
//	fclose(docWriter);
//	fclose(freqWriter);
//	fclose(outWriter);
	reader->close();
	termEnum->close();
}


/*****************/
//something useful in grammer_list()
// Sample usage: ./create_grammar_index --input=/media/indexDisk/gov2url/gov2url --output=/media/indexDisk/gov2url_gm



/******************** Global Struct ********************/



inline Production *IntegOfRAP::Insert_new_list()
{
	Production *p = &pp_tbl[pp_num];
	p->head.next = &p->tail;
	p->tail.before = &p->head;
	p->len = 0;
	p->rdc = false;
	p->sn = pp_num;
	return p;
}

inline bool IntegOfRAP::Compare(Production *p, uint32_t *list)
{
	Node *q = p->head.next;
	uint32_t i = 0;
	while (q != &(p->tail))
	{
		if (q->num != list[i++])
			return false;
		else
			q = q->next;
	}
	return true;
}

inline bool IntegOfRAP::NT_find_prefix(Production *p, uint32_t *list, uint32_t &length)
{
	Node *n = &p->head;
	Production *q;
	uint32_t len = 0;
	for (uint32_t i = 0; i < p->len; i++) {
		n = n->next;
		if (n->num & NT) {
			q = &rp_tbl[n->num & PRODUCTION_MASK];
			if (NT_find_prefix(q, &list[len], len))
				continue;
			else
				return false;
		} else {
			if (n->num == list[len]) {
				len++;
				continue;
			} else
				return false;
		}
	}
	length += len;
	return true;
}

inline uint32_t IntegOfRAP::getProTlen(const Production* p)
{
	Node* tp = p->head.next;
	uint32_t rlen = 0;
	for (uint32_t i = 0; i < p->len; i++)
	{
		if (tp->num & NT)
			rlen += getProTlen(&rp_tbl[tp->num & PRODUCTION_MASK]);
		else
			rlen++;
		tp = tp->next;
	}
	return rlen;
}

inline Production *IntegOfRAP::Hash_find_prefix(uint32_t *list, uint32_t len, uint32_t &tag)
{
	Production *p, *q;
	uint32_t key = list[0] % hash_tbl_size + 1;
	uint32_t max_len, pos;
	for (q = nullptr, max_len = 0, p = hash_tbl[key]; p; p = p->next) {
		if (getProTlen(p) > len)
			continue;
		if (p->sn & CONTAIN_NT) {
			pos = 0;
			if (NT_find_prefix(p, list, pos)) {
				if (pos > max_len) {
					q = p;
					max_len = pos;
				}
			} else
				continue;
		} else {
			if ((p->len <= max_len) || (!Compare(p, list)))
				continue;
			q = p;
			max_len = p->len;
		}
	}
	tag = max_len;
	return q;
}


inline uint32_t IntegOfRAP::DJBHash(char *str, uint32_t len)
{
	uint32_t hash = 5381;
	uint32_t i = 0;
	for (i = 0; i < len; str++, i++)
	{
		hash = ((hash << 5) + hash) + (*str);
	}
	return hash;
}

inline uint32_t IntegOfRAP::Hash_Function(uint32_t x, uint32_t y)
{
	phrase[0] = x;
	phrase[1] = y;
	char *tmp = (char *) phrase;
	return (DJBHash(tmp, 8) % digram_index_size);
}

inline uint32_t IntegOfRAP::Key_find(uint32_t p)
{
	if (p & NT)
	{
		Production *q = &rp_tbl[p & PRODUCTION_MASK];
		return Key_find(q->head.next->num);
	}
	else
		return p;
}

inline bool IntegOfRAP::Find_aplha_beta(uint32_t x, uint32_t y, Node *r, uint32_t &key, Digram *&q)
{
	key = Hash_Function(x, y);
	Digram *p;
	for (p = dp_tbl[key]; p; p = p->next)
	{
		if (p->index->num == x && p->index->next->num == y)
		{
			if (r == p->index->next)
				continue;
			q = p;
			return true;
		}
	}
	return false;
}

inline void IntegOfRAP::Add_digram_index(Node *fp, Production *q, uint32_t key)
{
	Digram *p = new Digram;
	p->index = fp;
	p->pro = q;
	p->next = dp_tbl[key];
	dp_tbl[key] = p;
}

void IntegOfRAP::Remove_digram_index(uint32_t x, uint32_t y, Node *tp)
{
	uint32_t key = Hash_Function(x, y);
	Digram *p, *q;
	for (p = dp_tbl[key]; p; p = p->next)
	{
		if (p->index->num == x && p->index->next->num == y && tp == p->index)
		{
			if (p == dp_tbl[key])
				dp_tbl[key] = p->next;
			else
				q->next = p->next;
			delete p;
			return;
		}
		q = p;
	}
}

inline uint32_t IntegOfRAP::Insert_new_reduced_production(Node *tp, Digram *dp, uint32_t contain_nt)
{
	uint32_t x = Key_find(tp->num);
	uint32_t key = x % hash_tbl_size + 1;
	Production *p = &rp_tbl[rp_num];
	Node *q;
	p->rdc = false;
	p->len = 2;
	q = &p->head;
	q->next = new Node;
	q->next->num = tp->num;
	q->next->before = q;
	q = q->next;
	q->next = new Node;
	q->next->num = tp->next->num;
	q->next->before = q;
	q->next->next = &p->tail;
	p->tail.before = q->next;
	dp->index = q;
	dp->pro = p;
	p->sn = rp_num | contain_nt | REDUCED_PRODUCTION;
	rp_num++;
	p->next = hash_tbl[key];
	hash_tbl[key] = p;
	return p->sn;
}

inline void IntegOfRAP::Update(uint32_t sn, Digram *dp)
{
	uint32_t key;
	Node *n = dp->index;
	Node *dq;
	Production *p = dp->pro;
	p->rdc = true;
	if (n->before == &p->head)
	{
		if (n->next->next == &p->tail)
		{
			n->num = sn;
			dq = n->next;
			n->next = n->next->next;
			n->next->before = n;
			delete dq;
		}
		else
		{
			Remove_digram_index(n->next->num, n->next->next->num, n->next);
			n->num = sn;
			dq = n->next;
			n->next = n->next->next;
			n->next->before = n;
			delete dq;
			key = Hash_Function(n->num, n->next->num);
			Add_digram_index(n, p, key);
		}
	}
	else
	{
		if (n->next->next == &p->tail)
		{
			Remove_digram_index(n->before->num, n->num, n->before);
			n->num = sn;
			dq = n->next;
			n->next = n->next->next;
			n->next->before = n;
			delete dq;
			key = Hash_Function(n->before->num, n->num);
			Add_digram_index(n->before, p, key);
		}
		else
		{
			Remove_digram_index(n->before->num, n->num, n->before);
			Remove_digram_index(n->next->num, n->next->next->num, n->next);
			n->num = sn;
			dq = n->next;
			n->next = n->next->next;
			n->next->before = n;
			delete dq;
			key = Hash_Function(n->before->num, n->num);
			Add_digram_index(n->before, p, key);
			key = Hash_Function(n->num, n->next->num);
			Add_digram_index(n, p, key);
		}
	}
	p->len--;
	p->sn |= CONTAIN_NT;
}

inline void IntegOfRAP::Append_production(Production *p, uint32_t beta)
{
	Node *n = p->tail.before;
	n->next = new Node;
	n->next->before = n;
	n->next->next = &(p->tail);
	p->tail.before = n->next;
	n->next->num = beta;
	p->len++;
	if (beta & NT) // obtain max
		p->sn |= NT;
	uint32_t key = Hash_Function(n->num, n->next->num);
	Add_digram_index(n, p, key);
}

/*******************************************************/
//PISequential tar code
void IntegOfRAP::grammar_list(uint32_t* list, uint32_t len, bool re_only)
{
	Digram *fdp, odp;
	Node *tp;
	Production *p, *q;
	uint32_t i, j, pos, sn;
	uint32_t key = 0, tag_length;

	p = Insert_new_list();
	tp = &p->head;
	for (i = 0; i < len;)
	{
		q = Hash_find_prefix(list + i, len - i, tag_length);
		if (q)
		{
			p->sn |= CONTAIN_NT;
			tp->next = new Node;
			tp->next->num = (q->sn | NT);
			tp->next->before = tp;
			tp->next->next = &p->tail;
			p->tail.before = tp->next;
			i += tag_length;
			rp_appear[(q->sn & PRODUCTION_MASK)]++;
		}
		else
		{
			tp->next = new Node;
			tp->next->num = list[i];
			tp->next->before = tp;
			tp->next->next = &p->tail;
			p->tail.before = tp->next;
			i++;
		}
		p->len++;
		if (p->len < 2 || re_only)
		{
			tp = tp->next;
			continue;
		}
		// search "alpha-beta"
		if (I == false)
		{
			// case 1 or 2
			if (!Find_aplha_beta(tp->num, tp->next->num, tp, key, fdp))
			{
				Add_digram_index(tp, p, key);
				tp = tp->next;
				continue;
			}
			odp.index = fdp->index;
			odp.pro = fdp->pro;
			sn = Insert_new_reduced_production(tp, fdp,
			    (tp->num & NT) | (tp->next->num & NT));
			sn |= NT;
			Update(sn, &odp);
			if (tp->before != &p->head)
			{
				Remove_digram_index(tp->before->num, tp->num, tp->before);
				tp->num = sn;
				delete tp->next;
				tp->next = &p->tail;
				p->tail.before = tp;
				key = Hash_Function(tp->before->num, tp->num);
				Add_digram_index(tp->before, p, key);
			}
			else
			{
				tp->num = sn;
				delete tp->next;
				tp->next = &p->tail;
				p->tail.before = tp;
			}
			p->len--;
			p->sn |= CONTAIN_NT;
			I = true;
		}
		else
		{
			// I == 1, case 1 or 3
			if (odp.index->next == &((odp.pro)->tail)
			    || odp.index->next->num != tp->next->num
			    || odp.index->next == tp->next->before)
			{
				// case 1
				I = false;
				key = Hash_Function(tp->num, tp->next->num);
				Add_digram_index(tp, p, key);
				tp = tp->next;
				continue;
			}
			Append_production(&rp_tbl[rp_num - 1], tp->next->num);
			Node *dq;
			dq = odp.index->next;
			if (odp.index->next->next == &((odp.pro)->tail))
				Remove_digram_index(odp.index->num, tp->next->num, odp.index);
			else
			{
				Remove_digram_index(odp.index->num, tp->next->num, odp.index);
				Remove_digram_index(tp->next->num, odp.index->next->next->num,
				    odp.index->next);
				key = Hash_Function(odp.index->num, odp.index->next->next->num);
				Add_digram_index(odp.index, odp.pro, key);
			}
			odp.index->next = odp.index->next->next;
			odp.index->next->before = odp.index;
			delete dq;
			odp.pro->len--;
			delete tp->next;
			tp->next = &p->tail;
			p->tail.before = tp;
			p->len--;
		}
	}
	pp_num++;
}

void IntegOfRAP::Write_Index() // removed grammar
{
	char filename[50];
	uint32_t output;

	sprintf(filename, "%s/refile", glb_outputp.c_str());
	FILE* fgm = fopen(filename, "wb");
	for (auto elem : rmvpost) {
		output = elem.size();
		fwrite(&output, sizeof(uint32_t), 1, fgm);
		for (auto e : elem) {
			output = e;
			fwrite(&output, sizeof(uint32_t), 1, fgm);
		}
	}
	fclose(fgm);

	sprintf(filename, "%s/dictionary", glb_outputp.c_str());
	FILE* fdt = fopen(filename, "wb");
	for (auto elem : rmvdict) {
		output = elem.size();
		fwrite(&output, sizeof(uint32_t), 1, fdt);
		for (auto e : elem) {
			output = e;
			fwrite(&output, sizeof(uint32_t), 1, fdt);
		}
	}
	fclose(fdt);
}

void IntegOfRAP::Remove_waste()
{
	bool *tag = new bool[rp_num];
	uint32_t *map = new uint32_t[rp_num];

	std::vector<uint32_t> list;
	uint32_t _i, _j, newid = 0;
	uint32_t rpid, right1, right2;

	for (_i = 0; _i < rp_num; _i++)
	{
		Production *p = &rp_tbl[_i];
		map[_i] = newid;
		if (!p->rdc && p->len == 2 && (rp_appear[_i] == 0))
			tag[_i] = false;
		else
		{
			tag[_i] = true;
			++newid;
		}
	}

	for (_i = 0; _i < rp_num; _i++)
	{
		if (tag[_i])
		{
			list.clear();
			Production *p = &rp_tbl[_i];
			Node* n = p->head.next;
			for (_j = 0; _j < p->len; _j++, n = n->next)
			{
				if (n->num & NT)
				{
					rpid = n->num & PRODUCTION_MASK;
					if (!tag[rpid])
					{
						right1 = rp_tbl[rpid].head.next->num;
						right2 = rp_tbl[rpid].head.next->next->num;
						if (right1 & NT)
							list.push_back(map[right1 & PRODUCTION_MASK] | NT);
						else
							list.push_back(right1);
						if (right2 & NT)
							list.push_back(map[right2 & PRODUCTION_MASK] | NT);
						else
							list.push_back(right2);
					}
					else
						list.push_back(map[rpid] | NT);
				}
				else
					list.push_back(n->num);
			}
			rmvdict.push_back(list);
		}
	}

	for (_i = 0; _i < pp_num; _i++)
	{
		list.clear();
		Production *p = &pp_tbl[_i];
		Node* n = p->head.next;
		for (_j = 0; _j < p->len; _j++, n = n->next)
		{
			if (n->num & NT)
			{
				rpid = n->num & PRODUCTION_MASK;
				if (!tag[rpid])
				{
					right1 = rp_tbl[rpid].head.next->num;
					right2 = rp_tbl[rpid].head.next->next->num;
					if (right1 & NT)
						list.push_back(map[right1 & PRODUCTION_MASK] | NT);
					else
						list.push_back(right1);
					if (right2 & NT)
						list.push_back(map[right2 & PRODUCTION_MASK] | NT);
					else
						list.push_back(right2);
				}
				else
					list.push_back(map[rpid] | NT);
			}
			else
				list.push_back(n->num);
		}
		rmvpost.push_back(list);
	}
	delete tag;
	delete map;
}

IntegOfRAP::~IntegOfRAP(){}
#endif /* INTEGRATIONOFREADERANDPISEQUENTIAL_H_ */

