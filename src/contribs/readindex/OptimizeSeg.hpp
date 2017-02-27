
#ifndef SRC_CONTRIBS_READINDEX_OPTIMIZESEG_HPP_
#define SRC_CONTRIBS_READINDEX_OPTIMIZESEG_HPP_

#define BUFFER_ALLOC(x) do{ x = new uint32_t [list_upperb]; }while(false);

#include "IOUtils.hpp"
#include "Utils.hpp"
#include "Grammar.hpp"
#include "SortComp.hpp"
#include "string.h"
#include <vector>
#include <string>
#include <algorithm>

#include "FastPFor/headers/deltautil.h"
#include "BlockCodecs.hpp"

using namespace Grammar_Consts;

typedef std::vector<std::list<uint32_t>> VectorList;
typedef std::map<uint32_t, uint32_t> MapRelation;
typedef std::pair<uint32_t, uint32_t> MapPair;

VectorList SegDictionary, TempDictionary;
VectorList SegGrammar;
MapRelation OnMapping;

template <typename BlockCodec>
class OptimizeSeg{
public:

	std::string glb_inputpath;
	std::string glb_outputpath;
	std::vector<uint8_t*> lists;
	uint16_t glb_threshold;
	uint16_t glb_filetype;
	bool glb_enablecheck;
	uint32_t *gap_l, *cmp_l, *chk_l;
	OptimizeSeg(){
		glb_threshold = 8;
		glb_filetype = 2;
		glb_enablecheck = false;
	}
	~OptimizeSeg(){}
	inline void check_symbol(std::list<uint32_t> &res, uint32_t rpn);
	void Expand_pattern();
	void Filter_pattern();
	void Prune_pattern(const uint16_t theshold);

	void Reorder_pattern();
	bool flush_check();
	inline void check_modify(uint64_t& checknum, uint64_t& checksum);
	inline void loop_epilogue();
	void Option_Analy(const int argc, char* const argv[]);
	void Flush_DLists();
	void Flush_GLists();
	void readfiles(const std::string &dictpath, VectorList *ref, VectorList *dict);
	void MergeOfOpSeg();
};

template <typename BlockCodec>
inline void OptimizeSeg<BlockCodec>::check_symbol(std::list<uint32_t> &res, uint32_t rpn) {
    auto rpl = SegDictionary.at(rpn);
    auto it = rpl.begin();
    while (it != rpl.end()) {
        if (*it & NT)
            check_symbol(res, get_rpnum(*it));
        else
            res.push_back(*it);
        it++;
    }
}

template <typename BlockCodec>
void OptimizeSeg<BlockCodec>::Expand_pattern() {
    for (auto& ep : SegDictionary) {
        auto it = ep.begin();
        while (it != ep.end()) {
            if (*it & NT) {
                auto pl = SegDictionary.at(get_rpnum(*it));
                auto size = pl.size();
                it = ep.erase(it);
                ep.insert(it, pl.begin(), pl.end());
                while (size--)
                    it--;
            } else
                it++;
        }
    }
    std::cout << "Expand pattern" << std::endl;
}

template <typename BlockCodec>
void OptimizeSeg<BlockCodec>::Filter_pattern() {
    uint32_t nrp_id = 0;
    for (auto& ep : SegGrammar) {
        for (auto& es : ep)
            if (es & NT) {
                auto it = OnMapping.find(get_rpnum(es));
                if (it == OnMapping.end())
                    OnMapping.insert(MapPair(get_rpnum(es), 0));
            }
    }
    for (auto& em : OnMapping)
        em.second = nrp_id++;

    for (auto& ep : SegGrammar) {
        for (auto& es : ep)
            if (es & NT)
                es = (OnMapping.at(get_rpnum(es)) | NT);
    }
    nrp_id = 0;
    for (auto& ep : SegDictionary) {
        auto itr = OnMapping.find(nrp_id);
        if (itr != OnMapping.end()) {
            assert(itr->second == TempDictionary.size());
            TempDictionary.push_back(ep);
        }
        ++nrp_id;
    }
    SegDictionary.swap(TempDictionary); // Swap
    std::cout << "Filter pattern" << std::endl;
}

template <typename BlockCodec>
void OptimizeSeg<BlockCodec>::Prune_pattern(const uint16_t theshold) {
    uint32_t oid = 0, nid = 0;
    TempDictionary.clear();
    OnMapping.clear();
    for (auto& ep : SegDictionary) {
        if (ep.size() >= theshold) {
            TempDictionary.push_back(ep);
            OnMapping.insert(MapPair(oid, nid));
            ++nid;
        }
        ++oid;
    }
    for (auto& ep : SegGrammar) {
        auto it = ep.begin();
        while (it != ep.end()) {
            if (*it & NT) {
                auto rp = SegDictionary.at(get_rpnum(*it));
                if (rp.size() < theshold) {
                    it = ep.erase(it);
                    ep.insert(it, rp.begin(), rp.end());
                    --it;
                } else
                    *it = (OnMapping.at(get_rpnum(*it)) | NT);
            }
            ++it;
        }
    }
    SegDictionary.swap(TempDictionary); // Swap
    std::cout << "Prune pattern" << std::endl;
}

/*********************************************/
template <typename BlockCodec>
void OptimizeSeg<BlockCodec>::Reorder_pattern() {
    uint32_t rpid = 0;
    OnMapping.clear();

    for (auto& ep : SegDictionary) {
        auto it = ep.begin();
        ep.insert(it, rpid);
        ++rpid;
    }
    std::sort(SegDictionary.begin(), SegDictionary.end(), compareList);
    rpid = 0;
    for (auto& ep : SegDictionary) {
        auto it = ep.begin();
        OnMapping.insert(MapPair(*it, rpid));
        ++rpid;
        ep.erase(it);
    }
    for (auto& ep : SegGrammar) {
        auto it = ep.begin();
        while (it != ep.end()) {
            if (*it & NT)
                *it = (OnMapping.at(get_rpnum(*it)) | NT);
            ++it;
        }
    }
    std::cout << "Reorder pattern" << std::endl;
}

template <typename BlockCodec>
bool OptimizeSeg<BlockCodec>::flush_check(){
	std::stringstream ss;
	ss << glb_inputpath.substr(0, glb_inputpath.find_last_of("/\\")) << "/OptBlk_p"
	   << glb_threshold;
	ss >> glb_outputpath;

	if ((access(glb_outputpath.c_str(), F_OK)) == -1)
	{
		mkdir(glb_outputpath.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
		return false;
	}
	std::cerr << "Output " << glb_outputpath << " is existed, stop working." << std::endl;
	return true;
}


/*********************************************/
template <typename BlockCodec>
inline void OptimizeSeg<BlockCodec>::check_modify(uint64_t& checknum, uint64_t& checksum) {
    std::list<uint32_t> result;
    checknum = 0, checksum = 0;
    for (auto& ep : SegGrammar)
        for (auto& es : ep) {
            if (es & NT) {
                check_symbol(result, get_rpnum(es));
                for (auto& ec : result) {
                    checknum++;
                    checksum += ec;
                }
                result.clear();
            } else {
                checknum++;
                checksum += es;
            }
        }
}

template <typename BlockCodec>
inline void OptimizeSeg<BlockCodec>::loop_epilogue() {
    SegDictionary.clear();
    TempDictionary.clear();
    SegGrammar.clear();
    OnMapping.clear();
}

template <typename BlockCodec>
void OptimizeSeg<BlockCodec>::Option_Analy(const int argc, char* const argv[]) {
	memset(pr_dp, '=', 25);
    int res;
    const char *short_opts = "hi:t:f:c";
    const struct option long_opts[] = {
        { "help", no_argument, NULL, 'h'},
        { "input", required_argument, NULL, 'i'},
        { "prune", required_argument, NULL, 't'},
        { "type", required_argument, NULL, 'f'},
		{ "check", no_argument, NULL, 'c'},
        { 0, 0, 0, 0}
    };
    while ((res = getopt_long(argc, argv, short_opts, long_opts, NULL)) != -1) {
        switch (res) {
            case 'h':
                break;
            case 'i':
                glb_inputpath = optarg;
                break;
            case 't':
                glb_threshold = atoi(optarg);
                break;
            case 'f':
                glb_filetype = atoi(optarg);
                break;
            case 'c':
            	glb_enablecheck = true;
            	break;
            case '?':
            default:
                ;
        }
    }
    if(glb_inputpath.at(glb_inputpath.size() - 1) == '/')
    	glb_inputpath.erase(glb_inputpath.end() - 1);
}

template <typename BlockCodec>
void OptimizeSeg<BlockCodec>::Flush_DLists(){

	std::vector<uint8_t> out;
	std::vector<uint32_t> preptn;
	uint32_t totalsize = 0;

	FastPForLib::IntegerCODEC & s16codec = *(new FastPForLib::Simple16<true>());

	for (uint32_t lid = 0; lid < SegDictionary.size(); ++lid) {

		std::vector<uint32_t> ptn;
		for (auto &e : SegDictionary.at(lid))
			ptn.push_back(e);
		FastPForLib::TightVariableByte::encode_single(ptn.size(), out);	// patten length

		totalsize += 1 + ptn.size() + ceil_div(ptn.size(), BlockCodec::block_size);

		uint32_t _i = 0;
		if (lid > 0) {
			while (_i < preptn.size() && _i < ptn.size()) {
				if (preptn.at(_i) != ptn.at(_i))
					break;
				_i++;
			}
		}
		FastPForLib::TightVariableByte::encode_single(_i, out); // same length

		uint32_t idx = 0;
		for (; _i < ptn.size(); _i++)
			gap_l[idx++] = ptn.at(_i);
		FastPForLib::Delta::delta(gap_l, idx);

		std::vector<uint8_t> buf(2 * 4 * idx);
		size_t out_len = buf.size();
		s16codec.encodeArray(gap_l, idx,
				reinterpret_cast<uint32_t*>(buf.data()), out_len);
		out_len *= 4;
		out.insert(out.end(), buf.data(), buf.data() + out_len);

		std::swap(preptn, ptn);
	}

	if (glb_enablecheck) {
		const uint8_t *dp = out.data(), *dend = out.data() + out.size();
		uint32_t lid = 0, len, slen;
		while(dp < dend){

			dp = FastPForLib::TightVariableByte::decode(dp, &len, 1);
			dp = FastPForLib::TightVariableByte::decode(dp, &slen, 1);
			size_t out_len = len - slen;

			for(uint32_t i = 0; i < slen; i++)
				chk_l[i] = gap_l[i];
			dp = reinterpret_cast<uint8_t const*>
            (s16codec.decodeArray(reinterpret_cast<uint32_t const*>(dp), 0,
            		chk_l + slen, out_len));
			FastPForLib::Delta::inverseDelta(chk_l + slen, out_len);

			assert(SegDictionary.at(lid).size() == len);
			uint32_t _i = 0;
			for(auto &e : SegDictionary.at(lid))
				assert(e == chk_l[_i++]);

			std::swap(chk_l, gap_l);
			++lid;
		}
		std::cerr << "Check dictionary finish" << std::endl;
	}

	char path[path_maxlen];
	sprintf(path, "%s/dictionary", glb_outputpath.c_str());
	FILE* fdt = fopen(path, "wb");
	assert(fdt != nullptr);
	fwrite(&totalsize, sizeof(uint32_t), 1, fdt);
	fwrite(out.data(), sizeof(uint8_t), out.size(), fdt);
	fclose(fdt);
}

template <typename BlockCodec>
void OptimizeSeg<BlockCodec>::Flush_GLists(){
	const uint32_t block_size = BlockCodec::block_size;

	std::vector<uint8_t> out;
	std::vector<uint32_t> docs_buf(block_size);
	std::vector<uint32_t> list_ends;

	for (auto &ep : SegGrammar) {
		uint32_t n = ep.size();
		FastPForLib::TightVariableByte::encode_single(n, out);

		uint32_t blocks = ceil_div(n, block_size);
		uint32_t begin_block_maxs = out.size();
		uint32_t begin_block_endpoints = begin_block_maxs + 4 * blocks;
		uint32_t begin_blocks = begin_block_endpoints + 4 * (blocks - 1);
		out.resize(begin_blocks);

		auto doc_it = ep.cbegin();
		uint32_t last_doc(-1);
		uint32_t block_base(0);

		for (size_t b = 0; b < blocks; ++b) {
			std::vector<uint8_t> pos;
			uint32_t last_nt(-1);
			uint32_t cur_block_size =
					((b + 1) * block_size <= n) ? block_size : (n % block_size);

			for (uint32_t _i = 0; _i < cur_block_size; _i++) {
				uint32_t symbol(*doc_it++);
				if (symbol & NT) {
					docs_buf[_i] = get_rpnum(symbol) - last_nt - 1;
					last_doc = SegDictionary.at(get_rpnum(symbol)).back();
					last_nt = get_rpnum(symbol);
					pos.push_back((uint8_t)_i);
				} else {
					docs_buf[_i] = symbol - last_doc - 1;
					last_doc = symbol;
				}
			}

			*((uint32_t*)&out[begin_block_maxs + 4 * b]) = last_doc;
			out.push_back((uint8_t) pos.size());
			for (auto &e : pos)
				out.push_back(e);

			BlockCodec::encode(docs_buf.data(),	last_doc - block_base - (cur_block_size - 1),
					cur_block_size, out);

			if (b != blocks - 1) {
				*((uint32_t*) &out[begin_block_endpoints + 4 * b]) = out.size() - begin_blocks;
			}
			block_base = last_doc + 1;
		}
		list_ends.push_back((uint32_t)out.size());
	}

	FastPForLib::IntegerCODEC & s16codec = *(new FastPForLib::Simple16<true>());
	FastPForLib::Delta::delta(list_ends.data(), list_ends.size() - 1);
	std::vector<uint8_t> buf(2 * 4 * (list_ends.size() - 1));
	size_t out_len = buf.size();
	s16codec.encodeArray(list_ends.data(), list_ends.size() - 1,
			reinterpret_cast<uint32_t*>(buf.data()), out_len);

	if (glb_enablecheck) {

		const uint32_t listnum = list_ends.size();
		const uint8_t *offset = buf.data();
		uint8_t *list = out.data();
		size_t length = listnum - 1;

		reinterpret_cast<uint8_t const*>(s16codec.decodeArray(
				reinterpret_cast<uint32_t const*>(offset), 0, gap_l,
				length));
		assert(length == listnum - 1);
		FastPForLib::Delta::inverseDelta(gap_l, length);

		lists.push_back(list);
		for(uint32_t _i = 1; _i < listnum; _i++) {
			lists.push_back(list + gap_l[_i - 1]);
		}

		for(uint32_t lid = 0; lid < listnum; lid++){
			uint32_t n;
			const uint8_t * begin_block_maxs =
					FastPForLib::TightVariableByte::decode(lists.at(lid), &n, 1);
			uint32_t blocks = ceil_div(n, block_size);
			const uint8_t* begin_block_endpoints = begin_block_maxs
					+ blocks * 4;
			const uint8_t* begin_blocks = begin_block_endpoints
					+ (blocks - 1) * 4;

			uint32_t idx = 0, idy = 0;
			for (uint32_t b = 0; b < blocks; b++) {
				uint32_t last_nt(-1), last_doc(-1);
				const uint8_t * block_begin = begin_blocks;
				uint32_t cur_block_size = ((b + 1) * block_size <= n) ?
								block_size : (n % block_size);

				if (b > 0) {
					last_doc = *((uint32_t*) (begin_block_maxs + 4 * (b - 1)));
					block_begin = begin_blocks + *((uint32_t*) (begin_block_endpoints
											+ 4 * (b - 1)));
				}

				uint8_t ntnum = *block_begin++;
				BlockCodec::decode(block_begin + ntnum, docs_buf.data(), 0,
						cur_block_size);
				for(uint8_t _i = 0; _i < ntnum; _i++)
					docs_buf.at(*(block_begin+_i)) |= NT;

				for (uint32_t _i = 0; _i < cur_block_size; _i++) {
					if (docs_buf.at(_i) & NT) {
						docs_buf[_i] += last_nt + 1;
						last_doc = SegDictionary.at(get_rpnum(docs_buf[_i])).back();
						last_nt = get_rpnum(docs_buf[_i]);
					} else {
						docs_buf.at(_i) += last_doc + 1;
						last_doc = docs_buf.at(_i);
					}
					gap_l[idx++] = docs_buf.at(_i);
				}
			}

			auto &el = SegGrammar.at(lid);
			for(auto &e : el)
				chk_l[idy++] = e;

			assert(idy == idx);
			for(uint32_t _i = 0; _i < idx; _i++)
				assert(chk_l[_i] == gap_l[_i]);
		}
		std::cerr << "Check postings finish" << std::endl;

	}

	char path[path_maxlen];
	sprintf(path, "%s/postings", glb_outputpath.c_str());
	FILE* fpost = fopen(path, "wb");
	assert(fpost != nullptr);
	uint32_t output = list_ends.size();
	fwrite(&output, sizeof(uint32_t), 1, fpost);
	fwrite(buf.data(), sizeof(uint8_t), out_len * 4, fpost);
	fwrite(out.data(), sizeof(uint8_t), out.size(), fpost);
	fclose(fpost);
	out.clear();
	buf.clear();
}

template <typename BlockCodec>
void OptimizeSeg<BlockCodec>::readfiles(const std::string &dictpath, VectorList *ref, VectorList *dict) {
	char path[path_maxlen];
	std::list<uint32_t> nlist;
	uint32_t len, i;

	uint32_t *readbuf;
	BUFFER_ALLOC(readbuf);

	sprintf(path, "%s/%s", dictpath.c_str(), "refile");
	FILE* fgm = fopen(path, "rb");
	assert(fgm != nullptr);
	while (fread(&len, sizeof(uint32_t), 1, fgm) > 0) {
		fread(readbuf, sizeof(uint32_t), len, fgm);
		for (i = 0; i < len; i++)
			nlist.push_back(readbuf[i]);
		ref->push_back(nlist);
		nlist.clear();
	}
	fclose(fgm);

	sprintf(path, "%s/%s", dictpath.c_str(), "dictionary");
	FILE* fdt = fopen(path, "rb");
	assert(fdt != nullptr);
	while (fread(&len, sizeof(uint32_t), 1, fdt) > 0) {
		fread(readbuf, sizeof(uint32_t), len, fdt);
		for (i = 0; i < len; i++)
			nlist.push_back(readbuf[i]);
		dict->push_back(nlist);
		nlist.clear();
	}
	fclose(fdt);

	delete readbuf;
}

/*********************************************/
template <typename BlockCodec>
void OptimizeSeg<BlockCodec>::MergeOfOpSeg(){
	SegDictionary.reserve(rp_max_productions + 1);
	    SegGrammar.reserve(2 * pp_max_productions + 1); // In Regroup_GList, 2-fold size

	    readfiles(glb_inputpath, &SegGrammar, &SegDictionary);

		if (flush_check())
			throw std::runtime_error("Target dictionary existed.");

		BUFFER_ALLOC(gap_l)
		BUFFER_ALLOC(cmp_l)
		BUFFER_ALLOC(chk_l)

		Expand_pattern();
		Filter_pattern();
		Prune_pattern(glb_threshold);
		Reorder_pattern();

		Flush_DLists();
		Flush_GLists();

		loop_epilogue();

	    delete gap_l;
	    delete cmp_l;
	    delete chk_l;

	    cout<<"complete"<<endl;
	    //return 0;
}

#endif /* SRC_CONTRIBS_READINDEX_OPTIMIZESEG_HPP_ */
