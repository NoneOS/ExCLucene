/*
 * ConjuncQuery.hpp
 *
 */

#ifndef CONJUNCQUERY_HPP_
#define CONJUNCQUERY_HPP_

#include "Compress.hpp"
#include "IOUtils.hpp"
#include "Grammar.hpp"

typedef std::vector<uint32_t> QueryTask;

namespace Grammar_Query {

    class GQHashIndex {
    public:
        uint32_t dt_size; // dict size(symbols)
        uint32_t pt_size; // post size(symbols)
        uint32_t dicpatn_num; // patterns num
        uint32_t posting_num; // posts num
        uint32_t gtype;
        uint32_t* dict_data;
        uint32_t* post_data;
        uint32_t* readbuf;
        uint32_t** dicpatn;
        uint32_t** postid;
        uint32_t** ntptr;
        uint32_t** ttptr;

        GQHashIndex() : dt_size(0), pt_size(0), dicpatn_num(0),
        posting_num(0), gtype(0), dict_data(nullptr),
        post_data(nullptr), dicpatn(nullptr), postid(nullptr),
        ntptr(nullptr), ttptr(nullptr) {
            readbuf = new uint32_t[Grammar_Consts::list_upperb];
        }

        virtual ~GQHashIndex() {
            free_s(dict_data);
            free_s(post_data);
            free_s(dicpatn);
            free_s(postid);
            free_s(ntptr);
            free_s(ttptr);
        }

        void free_s(uint32_t* &ptr) {
            if (ptr != nullptr)
                delete []ptr;
        }

        void free_s(uint32_t** &ptr) {
            if (ptr != nullptr)
                delete []ptr;
        }

        void lddict(const char*);
        void lpost(const char*);
        uint32_t binFindpost(const uint32_t) const;
    };
/*
    void GQHashIndex::lddict(const char* dicpath) {
        uint32_t input, pos = 0, len, j = 0;
        FILE* fdict = fopen(dicpath, "rb");
        while (fread(&input, sizeof (uint32_t), 1, fdict) > 0) {
            dt_size += input + 1;
            fread(&input, sizeof (uint32_t), 1, fdict);
            fseek(fdict, sizeof (uint32_t) * input, SEEK_CUR);
            ++dicpatn_num;
        }
        dict_data = new uint32_t[dt_size + 32];			//TODO:Decode Trap
        dicpatn = new uint32_t*[dicpatn_num];
        fseek(fdict, 0, SEEK_SET);
        while (fread(&len, sizeof (uint32_t), 1, fdict) > 0) {
            dicpatn[j++] = &dict_data[pos]; //TODO:Q_MALLOC
            dict_data[pos++] = len;
            fread(&input, sizeof (uint32_t), 1, fdict);
            fread(readbuf, sizeof (uint32_t), input, fdict);
            Simple16Decode(readbuf, input, &dict_data[pos], len);
            for (uint32_t i = 1; i < len; i++)
                dict_data[pos + i] += dict_data[pos + i - 1];
            pos += len;
        }
        fclose(fdict);
        free_s(readbuf);
    }
*/
    void GQHashIndex::lddict(const char* dicpath) {
        uint32_t input, pos = 0, len, j = 0;
        FILE* fdict = fopen(dicpath, "rb");
        while (fread(&input, sizeof (uint32_t), 1, fdict) > 0) {
            dt_size += input + 2;
            fread(&input, sizeof (uint32_t), 1, fdict);
            fseek(fdict, sizeof (uint32_t) * input, SEEK_CUR);
            ++dicpatn_num;
        }
        dict_data = new uint32_t[dt_size + 32];			//TODO:Decode Trap
        dicpatn = new uint32_t*[dicpatn_num];

        fseek(fdict, 0, SEEK_SET);
        while (fread(&len, sizeof (uint32_t), 1, fdict) > 0) {
            dicpatn[j++] = &dict_data[pos]; //TODO:Q_MALLOC
            dict_data[pos++] = len;
            fread(&input, sizeof (uint32_t), 1, fdict);
            fread(readbuf, sizeof (uint32_t), input, fdict);
            Simple16Decode(readbuf, input, &dict_data[pos + 1], len);
            for (uint32_t i = 1; i < len; i++)
                dict_data[pos + i + 1] += dict_data[pos + i];
            dict_data[pos] = dict_data[pos + len];
            pos += len + 1;
        }
        fclose(fdict);
        free_s(readbuf);
    }

    void GQHashIndex::lpost(const char* gpath) {
        char* tp = (char*) strrchr(gpath, '/');
        assert(tp != nullptr);
        gtype = Grammar_Consts::PostingFormat(*(tp - 1) - '0');

        struct stat buf;
        uint32_t lid, pos, *pp;
        FILE* fgm = fopen(gpath, "rb");
        assert(fgm != nullptr);
        stat(gpath, &buf);
        pt_size = buf.st_size / sizeof (uint32_t);
        post_data = new uint32_t[pt_size];
        fread(post_data, sizeof (uint32_t), pt_size, fgm);
        fclose(fgm);

        if (gtype == Grammar_Consts::TwoPart) {
            pp = post_data;
            while (pp < post_data + pt_size) {
                lid = *pp;
                if (contain_NT(lid)) {
                    pp += 2;
                    pp += *pp + 1;
                    if (contain_TT(lid)) {
                        ++pp;
                        pp += *pp + 1;
                    }
                } else {
                    pp += 2;
                    pp += *pp + 1;
                }
                ++posting_num;
            }
            ntptr = new uint32_t*[posting_num];
            ttptr = new uint32_t*[posting_num];
            postid = new uint32_t*[posting_num];
            pp = post_data;
            pos = 0;
            while (pp < post_data + pt_size) {
                postid[pos] = pp;
                lid = *postid[pos];
                *postid[pos] >>= 2;
                if (contain_NT(lid)) {
                    ++pp;
                    ntptr[pos] = pp;
                    ++pp;
                    pp += *pp + 1;
                    if (contain_TT(lid)) {
                        ttptr[pos] = pp;
                        ++pp;
                        pp += *pp + 1;
                    } else
                        ttptr[pos] = nullptr;
                } else {
                    ntptr[pos] = nullptr;
                    ++pp;
                    ttptr[pos] = pp;
                    ++pp;
                    pp += *pp + 1;
                }
                ++pos;
            }
        } else if (gtype == Grammar_Consts::OddEven) {
            pp = post_data;
            while (pp < post_data + pt_size) {
                pp += 2;
                pp += *pp + 1;
                ++posting_num;
            }
            postid = new uint32_t*[posting_num]; // Only use postid
            pp = post_data;
            pos = 0;
            while (pp < post_data + pt_size) {
                postid[pos] = pp;
                pp += 2;
                pp += *pp + 1;
                ++pos;
            }
        } else
            assert(false);
    }

    uint32_t GQHashIndex::binFindpost(const uint32_t key) const {
        uint32_t lt = 0, rt = posting_num - 1, pos;
        while (lt <= rt && rt < posting_num) {
            pos = (lt + rt) / 2;
            if (key == *postid[pos])
                return pos;
            else if (key < *postid[pos])
                rt = pos - 1;
            else
                lt = pos + 1;
        }
        return posting_num;
    }
}



#endif /* CONJUNCQUERY_HPP_ */
