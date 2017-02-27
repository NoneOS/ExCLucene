
#ifndef IOUTILS_HPP_
#define IOUTILS_HPP_

#include <iostream>
#include <cassert>
#include <string.h>
#include <dirent.h>
#include <getopt.h>
#include <list>
#include <string>
#include <vector>
#include <map>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sstream>

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
