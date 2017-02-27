/*
 * Snippet.h
 *
 */

#ifndef SRC_CONTRIBS_LIB_CLUCENE_CACHE_SNIPPET_H_
#define SRC_CONTRIBS_LIB_CLUCENE_CACHE_SNIPPET_H_

#include "CLucene/_ApiHeader.h"
#include <iostream>
#include <string>
using namespace std;

CL_NS_DEF2(search, cache)

class CLUCENE_EXPORT Snippet : LUCENE_BASE {
    tstring title;
    tstring url;
    tstring summarization;

public:

    tstring getTitle() const {
        return title;
    }

    void setTitle(tstring title) {
        this->title = title;
    }

    tstring getUrl() const {
        return url;
    }

    void setUrl(tstring url) {
        this->url = url;
    }

    tstring getSummarization() const {
        return summarization;
    }

    void setSummarization(tstring summarization) {
        this->summarization = summarization;
    }

    friend std::wostream& operator<<(std::wostream& wos, const Snippet& s) {
        wos << L"url: " << s.url << L"title: " << s.title << L"\nsummarization: " << s.summarization;
        return wos;
    }

    tstring toTString() {
        tstring ts = L"url: " + url + L"title: " + title + L"\nsummarization: " + summarization;
        //tstring ts = L"url: title: nsummarization: " ;
        return ts;
        //return L"url: " + url + L"title: " + title + L"\nsummarization: " + summarization;
    }

    Snippet();
    virtual ~Snippet();
};

CL_NS_END2

#endif /* SRC_CONTRIBS_LIB_CLUCENE_CACHE_SNIPPET_H_ */
