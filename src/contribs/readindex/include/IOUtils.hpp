/*
 * IOUtils.hpp
 *
 */

#ifndef IOUTILS_HPP_
#define IOUTILS_HPP_

#include <cassert>
#include <dirent.h>
#include <getopt.h>
#include <list>
#include <map>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sstream>

#include "Array.hpp"

const uint32_t gov2_maxlen = 0x18FFFFF;	// 26,000,000
//const uint32_t gov2_maxlen = 0x7FFFFFFF; // 26,000,000
const uint32_t path_maxlen = 128;

//#define DEBUG_INFO_IOUTILS

/***********************Filetype**********************************/
namespace File_Type {
    const char* DICT_FILE = ".dt";
    const char* GRAM_FILE = ".gm";
    const char* FREQ_FILE = ".qm";

    enum {
        OTHER = 0, DICT = 1, POST, FREQ
    };
}

int32_t file_filter(const struct dirent* entry) {
    if (strstr(entry->d_name, File_Type::DICT_FILE) != nullptr)
        return File_Type::DICT;
    if (strstr(entry->d_name, File_Type::GRAM_FILE) != nullptr)
        return File_Type::POST;
    if (strstr(entry->d_name, File_Type::FREQ_FILE) != nullptr)
        return File_Type::FREQ;
    return File_Type::OTHER;
}

int32_t file_sort(const struct dirent **d1, const struct dirent **d2) {
    uint32_t dl1 = strlen((*d1)->d_name);
    uint32_t dl2 = strlen((*d2)->d_name);

    if (dl1 != dl2)
        return (dl1 < dl2 ? -1 : 1);
    else
        return alphasort(d1, d2);
}
/***********************Progress**********************************/

char pr_dp[25]; // progress bar

inline void prog_play(const char* prefix, const uint32_t total,
        const uint32_t pr) {
    uint32_t cur = (pr * 100) / total;
    uint32_t ocur = ((pr - 1) * 100) / total;
    if (cur - ocur > 0) {
        printf("%s progress:[%s]%d%%\r", prefix, pr_dp + (100 - cur) / 4, cur);
        fflush(stdout);
    }
    if (pr == total)
        std::cout << std::endl;
}

typedef struct _at_term_ind1 {
    uint32_t m_urlcount; // doc-list length
    uint64_t m_off; // doc-list begin position
} Term_index;

template<class T> inline void chkmin(T &a, T b) {
    if (a > b)
        a = b;
}

template<class T> inline void chkmax(T &a, T b) {
    if (a < b)
        a = b;
}

class IndexBase {
    Array *postings; // posting-lists
    Term_index *ar_ind1; // term-lists

    uint64_t term_count;
    uint64_t read_count;

    uint32_t *next_list;
    Term_index *next_index;
    FILE *fetch_find1;
    FILE *fetch_find2;

public:

    IndexBase() :
    postings(nullptr), ar_ind1(nullptr), term_count(0), read_count(0), next_list(
    nullptr), next_index(nullptr), fetch_find1(nullptr), fetch_find2(
    nullptr) {
    }

    void load_2_index(const char *ind1_file, const char *ind2_file);
    void clean_2_index();
    Array* get_list(uint64_t listno);

    void fetch_prologue(const char *ind1_file, const char *ind2_file);
    void fetch_epilogue();
    uint32_t* fetch_next(uint32_t &len);

    void fetch_reset();
};

/**
 * Load index(l1,l2) to ar_ind1,postings
 * Note: ALLOCATE as needed; l1,l2 files CLOSED after reading
 */
void IndexBase::load_2_index(const char *ind1_file, const char *ind2_file) {
    FILE *_find1 = fopen(ind1_file, "rb"); // Read ind1 file
    if (_find1 != nullptr) {
        struct stat _ibuf;
        stat(ind1_file, &_ibuf);
        term_count = _ibuf.st_size / sizeof (Term_index);
        try {
            ar_ind1 = new Term_index[term_count];
            size_t fres = fread(ar_ind1, sizeof (Term_index), term_count, _find1);
            assert(fres == term_count);
        } catch (const std::bad_alloc &e) {
            std::cout << "Alloc ar_ind1 failed: " << e.what() << std::endl;
            fclose(_find1);
            _Exit(EXIT_FAILURE);
        }
        //std::cout<<"Reading "<<ind1_file<<" finished."<<std::endl;
        fclose(_find1);
    } else {
        perror(ind1_file);
        _Exit(EXIT_FAILURE);
    }

    FILE *_find2 = fopen(ind2_file, "rb"); // Read ind2 file
    if (_find2 != nullptr) {
        uint32_t urlcount = 0; // list length
        uint64_t listoff = 0; // list offset
        try {
            postings = new Array[term_count];
        } catch (const std::bad_alloc &e) {
            std::cout << "Alloc Array[] failed: " << e.what() << std::endl;
            delete ar_ind1;
            fclose(_find2);
            _Exit(EXIT_FAILURE);
        }
        for (uint64_t list_id = 0; list_id < term_count; list_id++) {
            urlcount = (ar_ind1 + list_id)->m_urlcount;
            listoff = (ar_ind1 + list_id)->m_off;
            postings[list_id].assign(urlcount);
            fseek(_find2, listoff, SEEK_SET);
            size_t fres = fread(postings[list_id].pt, sizeof (uint32_t), urlcount,
                    _find2);
            postings[list_id].length = urlcount;
            assert(fres == urlcount);
        }
        //std::cout<<"Reading "<<ind2_file<<" finished."<<std::endl;
        fclose(_find2);
    } else {
        delete ar_ind1;
        perror(ind2_file);
        _Exit(EXIT_FAILURE);
    }
}

/**
 * Get list(Array) by given listno
 * Note: must called after load_2_index
 */
Array* IndexBase::get_list(uint64_t listno) {
    if (listno < this->term_count)
        return &(postings[listno]);

    return nullptr;
}

/**
 * Clean alloc lists
 * Note: should called after load_2_index / get_list
 */
void IndexBase::clean_2_index() {
    if (ar_ind1 != nullptr)
        delete ar_ind1;
    if (postings != nullptr) {
        for (uint32_t lid = 0; lid < term_count; lid++)
            postings[lid].final();
        delete[] postings;
    }
}

/**
 * Fetch list one by one, which saving much more space
 * Must call fetch_prologue first, then call fetch_epilogue last
 */
void IndexBase::fetch_prologue(const char *ind1_file, const char *ind2_file) {
    fetch_find1 = fopen(ind1_file, "rb"); // Open ind1 file
    if (fetch_find1 != nullptr) {
        struct stat _ibuf;
        stat(ind1_file, &_ibuf);
        term_count = _ibuf.st_size / sizeof (Term_index);
        read_count = 0;
        next_index = new Term_index;

        //std::cout<<"Prepare "<<ind1_file<<"."<<std::endl;
    } else {
        perror(ind1_file);
        _Exit(EXIT_FAILURE);
    }

    fetch_find2 = fopen(ind2_file, "rb"); // Open ind2 file
    if (fetch_find2 != nullptr) {
        try {
            next_list = new uint32_t[gov2_maxlen];
        } catch (const std::bad_alloc &e) {
            delete next_index;
            std::cout << "Alloc next_list failed: " << e.what() << std::endl;
            fclose(fetch_find1);
            _Exit(EXIT_FAILURE);
        }
        //std::cout<<"Prepare "<<ind2_file<<"."<<std::endl;
    } else {
        delete next_index;
        perror(ind2_file);
        fclose(fetch_find1);
        _Exit(EXIT_FAILURE);
    }
}

void IndexBase::fetch_epilogue() {
    if (next_index != nullptr)
        delete next_index;
    if (next_list != nullptr)
        delete next_list;
    fclose(fetch_find1);
    fclose(fetch_find2);
}

/**
 * Fetch list as order(termid), var len will save list length
 * Return value is the list,
 */
uint32_t *IndexBase::fetch_next(uint32_t &len) {
    if (read_count < term_count) {
        fread(next_index, sizeof (Term_index), 1, fetch_find1);
        fseek(fetch_find2, next_index->m_off, SEEK_SET);
        fread(next_list, sizeof (uint32_t), next_index->m_urlcount, fetch_find2);
        read_count++;
        len = next_index->m_urlcount;
        return next_list;
    }
    return nullptr;
}

void IndexBase::fetch_reset() {
    read_count = 0;
    fseek(fetch_find1, 0, SEEK_SET);
}

/******************************************************************/
class IndexDirectory {
protected:
    uint32_t maxsegno; // real-exist files
    uint32_t *readbuf;
    std::vector<std::string> dictfiles;
    std::vector<std::string> gramfiles;

public:

    IndexDirectory() :
    maxsegno(0), readbuf(nullptr) {
    }

    virtual ~IndexDirectory() {
        dictfiles.clear();
        gramfiles.clear();
        if (readbuf != nullptr)
            delete readbuf;
    }
    uint32_t get_segnum();
    virtual void read_prelogue(const char*);
    virtual void read_segfile(const uint32_t, std::vector<std::list<uint32_t>>*,
            const int32_t);
    virtual void read_segfiles(const uint32_t, std::vector<std::list<uint32_t>>*,
            std::vector<std::list<uint32_t>>*);
};

uint32_t IndexDirectory::get_segnum() {
    return this->maxsegno;
}

void IndexDirectory::read_prelogue(const char* dpath) {
    char path[path_maxlen];
    struct dirent **namelist;
    uint32_t n = scandir(dpath, &namelist, file_filter, file_sort);
    maxsegno = n / 2;
    for (uint32_t i = 0; i < n; i++) {
        sprintf(path, "%s/%s", dpath, namelist[i]->d_name);
        (i % 2 == 0) ?
                dictfiles.push_back(std::string(path)) :
                gramfiles.push_back(std::string(path));
        delete namelist[i];
    }
    delete namelist;
    try {
        readbuf = new uint32_t[gov2_maxlen];
    } catch (const std::bad_alloc &e) {
        std::cout << "Alloc readbuf failed: " << e.what() << std::endl;
        _Exit(EXIT_FAILURE);
    }
}

void IndexDirectory::read_segfile(const uint32_t sid,
        std::vector<std::list<uint32_t>>*set, const int32_t type) {
    uint32_t llen, i;
    std::list<uint32_t> nlist;
    FILE* ft =
            (type == File_Type::DICT) ?
            fopen(dictfiles.at(sid).c_str(), "rb") :
            fopen(gramfiles.at(sid).c_str(), "rb");
    assert(ft != nullptr);
    while (fread(&llen, sizeof (uint32_t), 1, ft) > 0) {
        fread(readbuf, sizeof (uint32_t), llen, ft);
        for (i = 0; i < llen; i++)
            nlist.push_back(readbuf[i]);
        set->push_back(nlist);
        nlist.clear();
    }
    fclose(ft);
}

void IndexDirectory::read_segfiles(const uint32_t sid,
        std::vector<std::list<uint32_t>>*dict,
        std::vector<std::list<uint32_t>>*post) {
    read_segfile(sid, dict, File_Type::DICT);
    read_segfile(sid, post, File_Type::POST);
}

#endif /* IOUTILS_HPP_ */
