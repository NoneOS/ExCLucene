#ifndef INCLUDE_QUERIES_HPP_
#define INCLUDE_QUERIES_HPP_

#include "IOUtils.hpp"
#include "Grammar.hpp"
#include "deltautil.h"
#include "BlockCodecs.hpp"
#include "Utils.hpp"

#define FREE_S(x) do{ if(x != nullptr) delete []x; x = nullptr; }while(false);

#define   likely(x)      __builtin_expect(!!(x),   1)
#define   unlikely(x)    __builtin_expect(!!(x),   0)

namespace Grammar_Query {
struct GrammarIndex {
	template<typename BlockCodec>
	struct Dictionary {
		uint32_t* _data;
		std::vector<const uint32_t*> _item;

		~Dictionary() {
			FREE_S(_data)
			_item.clear();
		}

		void load(const char* fpath) {
			struct stat buf;
			if (stat(fpath, &buf) == -1)
				throw std::runtime_error("Dictionary not existed.");
			std::vector<uint8_t> in(buf.st_size);

			FILE* fdict = fopen(fpath, "rb");
			fread(in.data(), sizeof(uint8_t), buf.st_size, fdict);
			fclose(fdict);

			uint32_t len, slen;
			uint32_t totalsize = *((uint32_t*) in.data());
			uint32_t *gap_l = nullptr;
			const uint8_t *dp = in.data() + sizeof(uint32_t), *dend = in.data()
					+ in.size();
			_data = new uint32_t[totalsize];
			uint32_t* plist = _data;

			while (dp < dend) {
				_item.push_back(plist);
				dp = FastPForLib::TightVariableByte::decode(dp, &len, 1);
				dp = FastPForLib::TightVariableByte::decode(dp, &slen, 1);
				size_t out_len = len - slen;

				plist[0] = len;
				uint32_t blocks = ceil_div(len, BlockCodec::block_size);
				uint32_t *pstart = plist + blocks + 1;
				for (uint32_t i = 0; i < slen; i++)
					pstart[i] = gap_l[i];
				dp = reinterpret_cast<uint8_t const*>(decoder.decodeArray(
						reinterpret_cast<uint32_t const*>(dp), 0, pstart + slen,
						out_len));

				FastPForLib::Delta::inverseDelta(pstart + slen, out_len);
				for (uint32_t i = 0; i < blocks; i++)
				{
					if(i == blocks - 1)
						plist[i + 1] = pstart[len - 1];
					else
						plist[i + 1] = pstart[(i + 1) * BlockCodec::block_size - 1];
				}
				gap_l = pstart;
				plist = pstart + len;
			}
			assert(plist == _data + totalsize);
			in.clear();
		}
	};

	template<typename BlockCodec>
	struct Posting {
		std::vector<uint8_t> _data;
		std::vector<const uint8_t*> _item;

		~Posting() {
			_data.clear();
			_item.clear();
		}
		void load(const char* fpath) {
			struct stat buf;
			if (stat(fpath, &buf) == -1)
				throw std::runtime_error("Posting not existed.");
			_data.resize(buf.st_size);

			FILE* fpost = fopen(fpath, "rb");
			fread(_data.data(), sizeof(uint8_t), buf.st_size, fpost);
			fclose(fpost);

			uint32_t listnum = *((uint32_t*) _data.data());
			std::vector<uint32_t> decodebuf(listnum);

			const uint8_t *offset = _data.data() + sizeof(uint32_t);
			size_t length = listnum - 1;
			const uint8_t* dp =
					reinterpret_cast<uint8_t const*>(decoder.decodeArray(
							reinterpret_cast<uint32_t const*>(offset), 0,
							decodebuf.data(), length));
			assert(length == listnum - 1);
			FastPForLib::Delta::inverseDelta(decodebuf.data(), length);

			_item.push_back(dp);
			for (uint32_t _i = 1; _i < listnum; _i++)
				_item.push_back(dp + decodebuf[_i - 1]);
		}
	};

	static FastPForLib::Simple16<true> decoder;

	struct Dictionary<FastPForLib::optpfor_block> dict;
	struct Posting<FastPForLib::optpfor_block> post;

	~GrammarIndex() {
	}

	GrammarIndex() {
	}

	GrammarIndex(const char* dpath) {
		if (access(dpath, F_OK) == -1)
			throw std::runtime_error("Directory not existed.");
		char path[path_maxlen];
		sprintf(path, "%s/dictionary", dpath);
		dict.load(path);
		sprintf(path, "%s/postings", dpath);
		post.load(path);
		std::cout << "Load index finish" << std::endl;
	}

	void load(const char* dpath) {
		if (access(dpath, F_OK) == -1)
			throw std::runtime_error("Directory not existed.");
		if (dict._item.empty() && post._item.empty()) {
			char path[path_maxlen];
			sprintf(path, "%s/dictionary", dpath);
			dict.load(path);
			sprintf(path, "%s/postings", dpath);
			post.load(path);
			std::cout << "Load index finish" << std::endl;
		}
	}

};
}



#endif /* INCLUDE_QUERIES_HPP_ */
