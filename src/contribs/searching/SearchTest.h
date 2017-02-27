/*
 * SearchTest.h
 *
 */

#ifndef SRC_CONTRIBS_LIB_CLUCENE_SEARCHING_SEARCHTEST_H_
#define SRC_CONTRIBS_LIB_CLUCENE_SEARCHING_SEARCHTEST_H_

#include "CLucene.h"
#include "CLucene/_clucene-config.h"
#include "CLucene/config/repl_tchar.h"
#include "CLucene/config/repl_wchar.h"
#include "CLucene/debug/_condition.h"
#include "CLucene/util/StringBuffer.h"
#include "CLucene/util/Misc.h"

#include "CLucene/store/Lock.h"
#include "CLucene/index/TermVector.h"
#include "CLucene/queryParser/MultiFieldQueryParser.h"

#include "CLucene/store/RAMDirectory.h"
#include "CLucene/highlighter/QueryTermExtractor.h"
#include "CLucene/highlighter/QueryScorer.h"
#include "CLucene/highlighter/Highlighter.h"
#include "CLucene/highlighter/TokenGroup.h"
#include "CLucene/highlighter/SimpleHTMLFormatter.h"
#include "CLucene/highlighter/SimpleFragmenter.h"

#include <iostream>
#include <locale> // for locale
#include <fstream>
#include <algorithm> // for sort

#include "CLucene/cache/Cache.h"
#include "CLucene/cache/CacheNode.h"
#include "CLucene/cache/StaticCache.h"
#include "CLucene/cache/DynLRU.h"
#include "CLucene/cache/DynQTFDF.h"

#include "InputParameters.h"
// By rui
#include "OutputMeasurement.h"
#include "SortedBySomething.h"

CL_NS_USE(index)
CL_NS_USE(util)
CL_NS_USE(store)
CL_NS_USE(search)
CL_NS_USE(document)
CL_NS_USE(queryParser)
CL_NS_USE(analysis)
CL_NS_USE2(analysis, standard)
CL_NS_USE2(search, highlight)
CL_NS_USE2(search, cache)

//#define debug
typedef HashMap<wstring, CacheNode<wstring, vector<Snippet> > > SearchHashMap;
typedef SearchHashMap::iterator SearchHashMap_It;
typedef SearchHashMap::value_type SearchHashMap_ValueType;

typedef HashMap<wstring, CacheNode<wstring, Snippet> > SnippetHashMap;
typedef SnippetHashMap::iterator SnippetHashMap_It;
typedef SnippetHashMap::value_type SnippetHashMap_ValueType;

typedef HashMap<int32_t, CacheNode<int32_t, Document*> > DocumentHashMap;
typedef DocumentHashMap::iterator DocumentHashMap_It;
typedef DocumentHashMap::value_type DocumentHashMap_ValueType;

class SearchTest {
    uint32_t hitNum;
    uint32_t topK;
    uint32_t topQuery;

    IndexReader* hl_reader;
    IndexSearcher* hl_searcher;
    StandardAnalyzer hl_analyzer;

    Cache<wstring, vector<Snippet> >* resultCache; // Cache is an abstract class, we must define its pointer
    Cache<int32_t, Document*>* documentCache;
    Cache<wstring, Snippet>* snippetCache;

    InputParameters InPara;
    OutputMeasurement Output;

public:

    void settopK(const uint32_t k) {
        this->topK = k;
    }

    void settopQuery(const uint32_t k) {
        this->topQuery = k;
    }

    void initializeCacheInfo(InputParameters &ip) { // BUG1
        hitNum = 0;
        topK = 10;
        topQuery = 500;
        InPara = ip;

        switch (ip.cacheStrategy_QRC) {
            case CacheStrategy::Static_QTF_Cache:
                resultCache = _CLNEW StaticCache<wstring, vector<Snippet> >(ip.memoryLimitMB_QRC);
                break;
            case CacheStrategy::Static_QTFDF_Cache:
                resultCache = _CLNEW StaticCache<wstring, vector<Snippet> >(ip.memoryLimitMB_QRC);
                break;
            case CacheStrategy::Dynamic_LRU_Cache:
                resultCache = _CLNEW DynLRU<wstring, vector<Snippet> >(ip.memoryLimitMB_QRC);
                break;
            case CacheStrategy::Dynamic_LFU_Cache:
                resultCache = _CLNEW DynQTFDF<wstring, vector<Snippet>, SortFrequencyClass<wstring, vector<Snippet> > >
                        (ip.memoryLimitMB_QRC, SortFrequencyClass<wstring, vector<Snippet> >(true));
                break;
            case CacheStrategy::Dynamic_QTFDF_Cache:
                resultCache = _CLNEW DynQTFDF<wstring, vector<Snippet>, SortFrequencySizeClass<wstring, vector<Snippet> > >
                        (ip.memoryLimitMB_QRC, SortFrequencySizeClass<wstring, vector<Snippet> >(true));
                break;
            case CacheStrategy::Dynamic_GDS_Cache:
                resultCache = _CLNEW DynQTFDF<wstring, vector<Snippet>, SortGDSClass<wstring, vector<Snippet> > >
                        (ip.memoryLimitMB_QRC, SortGDSClass<wstring, vector<Snippet> >(true));
                break;
            default:
                _CLTHROWA_DEL(CL_ERR_UnsupportedOperation, "Cache Strategy not supported.");
        }

        switch (ip.cacheStrategy_SC) {
            case CacheStrategy::Static_QTF_Cache:
                snippetCache = _CLNEW StaticCache<wstring, Snippet>(ip.memoryLimitMB_SC);
                break;
            case CacheStrategy::Static_QTFDF_Cache:
                snippetCache = _CLNEW StaticCache<wstring, Snippet>(ip.memoryLimitMB_SC);
                break;
            case CacheStrategy::Dynamic_LRU_Cache:
                snippetCache = _CLNEW DynLRU<wstring, Snippet>(ip.memoryLimitMB_SC);
                break;
            case CacheStrategy::Dynamic_LFU_Cache:
                snippetCache = _CLNEW DynQTFDF<wstring, Snippet, SortFrequencyClass<wstring, Snippet> >
                        (ip.memoryLimitMB_SC, SortFrequencyClass<wstring, Snippet>(true));
                break;
            case CacheStrategy::Dynamic_QTFDF_Cache:
                snippetCache = _CLNEW DynQTFDF<wstring, Snippet, SortFrequencySizeClass<wstring, Snippet> >
                        (ip.memoryLimitMB_SC, SortFrequencySizeClass<wstring, Snippet>(true));
                break;
            case CacheStrategy::Dynamic_GDS_Cache:
                snippetCache = _CLNEW DynQTFDF<wstring, Snippet, SortGDSClass<wstring, Snippet> >
                        (ip.memoryLimitMB_SC, SortGDSClass<wstring, Snippet>(true));
                break;
            default:
                _CLTHROWA_DEL(CL_ERR_UnsupportedOperation, "Cache Strategy not supported.");
        }

        switch (ip.cacheStrategy_DC) {
            case CacheStrategy::Static_QTF_Cache:
                documentCache = _CLNEW StaticCache<int32_t, Document*>(ip.memoryLimitMB_DC);
                break;
            case CacheStrategy::Static_QTFDF_Cache:
                documentCache = _CLNEW StaticCache<int32_t, Document*>(ip.memoryLimitMB_DC);
                break;
            case CacheStrategy::Dynamic_LRU_Cache:
                documentCache = _CLNEW DynLRU<int32_t, Document*>(ip.memoryLimitMB_DC);
                break;
            case CacheStrategy::Dynamic_LFU_Cache:
                documentCache = _CLNEW DynQTFDF<int32_t, Document*, SortFrequencyClass<int32_t, Document*> >
                        (ip.memoryLimitMB_DC, SortFrequencyClass<int32_t, Document*>(true));
                break;
            case CacheStrategy::Dynamic_QTFDF_Cache:
                documentCache = _CLNEW DynQTFDF<int32_t, Document*, SortFrequencySizeClass<int32_t, Document*> >
                        (ip.memoryLimitMB_DC, SortFrequencySizeClass<int32_t, Document*>(true));
                break;
            case CacheStrategy::Dynamic_GDS_Cache:
                documentCache = _CLNEW DynQTFDF<int32_t, Document*, SortGDSClass<int32_t, Document*> >
                        (ip.memoryLimitMB_DC, SortGDSClass<int32_t, Document*>(true));
                break;
            default:
                _CLTHROWA_DEL(CL_ERR_UnsupportedOperation, "Cache Strategy not supported.");
        }

    }

    uint32_t getHitNum() {
        return hitNum;
    }

    string getDCHitNum() {
        return this->documentCache->getHitRatio();
    }

    string getSCHitNum() {
        return this->snippetCache->getHitRatio();
    }

    inline wstring generateQueryBiasedSnippetKey(Query* query, int32_t docID) {
        return query->toString() + StringToWString("-") + ToWString(docID);
    }

    void generateSnippet(Document* document, Query* query, Snippet &retSnippet) {
        //std::cout << "start generateSnippet... " << std::endl;
        assert(document != NULL);
        assert(query != NULL);
        QueryScorer scorer(query);
        Highlighter highlighter(&scorer);
        SimpleFragmenter fragmenter(40);
        highlighter.setTextFragmenter(&fragmenter);

        const TCHAR* Turl = document->get(Configure::ITEM_URL);
        const TCHAR* Title = document->get(Configure::ITEM_TITLE);
        const TCHAR* Tcontent = document->get(Configure::ITEM_CONTENT);


        //  std::wcout << document->get(L"docno") << std::endl;

        retSnippet.setUrl(Turl == NULL ? _T("") : Turl);
        retSnippet.setTitle(Title == NULL ? _T("") : Title);

        if (Tcontent != NULL) {
            StringReader reader(Tcontent);
            TokenStream* tokenStream = hl_analyzer.tokenStream(_T("content"), &reader);
            //TCHAR* tempSummarization = highlighter.getBestFragment(tokenStream, Tcontent);
            TCHAR* tempSummarization = highlighter.getBestFragments(tokenStream, Tcontent, 1, _T("..."));
            retSnippet.setSummarization(tempSummarization);
            _CLDELETE(tempSummarization);
        } else
            retSnippet.setSummarization(_T(""));
        //    std::wcout <<retSnippet.toTString()<< std::endl;
    }

    Document* getDocumentByID(int32_t docID, bool & notDeleteDocument) {
        //   std::cout << "start getDocumentByID..." << std::endl;
        //std::cout << notDeleteDocument << std::endl;
        //   return  hl_searcher->doc(docID);
        // check document cache first
        if (this->InPara.memoryLimitMB_DC > 0 && this->InPara.documentCacheTurnOn) {
            DocumentHashMap_It tempDocumentPtr = documentCache->getCacheEntry(docID);
            if (tempDocumentPtr == documentCache->cache_end()) {
                // cache miss get the document from disk   
                Output.countOfDocumentIO++;
                uint64_t t0 = Misc::currentTimeMillis();
                Document*document = hl_searcher->doc(docID);
                uint64_t t1 = Misc::currentTimeMillis();
                Output.timeOfDocumentIO += (t1 - t0);
                // std::wcout << (*document).get(Configure::ITEM_URL) << std::endl;
                //and update document cache
                if (InPara.documentCacheTurnOn && InPara.memoryLimitMB_DC > 0) {
                    //  Document tempDoc=*document;   
                    uint32_t DSize = RAMEstimator::getNumBytesOfDocument(document);
                    documentCache->addByetsOfLookUps(DSize);
                    Output.lengthOfDocumentIO += DSize;
                    CacheNode<int32_t, Document*> tempNode(docID, document, DSize);
                    if (!InPara.isStaticCache_DC()) {
                        //std::cout <<"Dynamic document cache" << std::endl;
                        if (documentCache->isCacheFullPlusOneNode(tempNode)) {
                            //   std::cout << "Delete document" << std::endl;                     
                            //Document*td = documentCache->getValueOfEvictedNode();
                            vector<Document*> EvictedDocuments;

                            //cout << typeid(td).name() << endl;
                            notDeleteDocument = documentCache->push(tempNode, EvictedDocuments);
                            //std::cout << Output.numOfQueries<<"Delete evicted document from cache" << std::endl;
                            for (int i = 0; i < EvictedDocuments.size(); i++) {
                                _CLDELETE(EvictedDocuments[i]);
                            }
                            //_CLDELETE(td);
                        } else notDeleteDocument = documentCache->push(tempNode);
                    } else notDeleteDocument = documentCache->push(tempNode);
                    //notDeleteDocument=documentCache->push(tempNode);
                }

                // std::wcout << (*document).get(Configure::ITEM_URL) << std::endl;
                //std::cout << notDeleteDocument << std::endl;
                return document;
            } else {
                // cache hit, return the document in cache directly
                //std::cout << Output.numOfQueries<<"Document cache hit" << std::endl;
                notDeleteDocument = true;
                //std::cout << notDeleteDocument << std::endl;
                return tempDocumentPtr->second.value;
            }
        } else {
            notDeleteDocument = false;
            Output.countOfDocumentIO++;
            uint64_t t0 = Misc::currentTimeMillis();
            Document*document = hl_searcher->doc(docID);
            uint64_t t1 = Misc::currentTimeMillis();
            Output.timeOfDocumentIO += (t1 - t0);
            uint32_t DSize = RAMEstimator::getNumBytesOfDocument(document);
            Output.lengthOfDocumentIO += DSize;
            //std::cout << notDeleteDocument << std::endl;
            return document;
        }
    }

    void getSnippetsByDocIDAndQuery(int32_t docID, Query* query, Snippet &snippet) {
        //   std::cout << "start getSnippetsByDocIDAndQuery..." << std::endl;
        wstring key = generateQueryBiasedSnippetKey(query, docID);
        if (this->InPara.memoryLimitMB_SC > 0 && this->InPara.snippetCacheTurnOn) {
            SnippetHashMap_It tempSnippetPtr = snippetCache->getCacheEntry(key);
            if (tempSnippetPtr == snippetCache->cache_end()) {
                // cache miss, generate snippet
                bool notDeleteDocument = true;
                Document* document = getDocumentByID(docID, notDeleteDocument);
                //  Document* document = hl_searcher->doc(docID);
                assert(document != NULL);

                uint64_t t0 = Misc::currentTimeMillis();
                generateSnippet(document, query, snippet);
                uint64_t t1 = Misc::currentTimeMillis();
                Output.timeofSnippetGeneration += (t1 - t0);

                // and update snippet cache
                if (InPara.snippetCacheTurnOn && InPara.memoryLimitMB_SC > 0) {
                    uint32_t SSize = RAMEstimator::getNumBytesOfSnippet(snippet);
                    snippetCache->addByetsOfLookUps(SSize);
                    CacheNode<wstring, Snippet> tempNode(key, snippet, SSize);
                    snippetCache->push(tempNode);
                }

                if (!notDeleteDocument)_CLDELETE(document);
            } else {
                // cache hit, return the snippet in cache directly
                snippet = tempSnippetPtr->second.value;
            }
        } else {// SC if off
            bool notDeleteDocument = false;
            Document* document = getDocumentByID(docID, notDeleteDocument);
            assert(document != NULL);
            uint64_t t0 = Misc::currentTimeMillis();
            generateSnippet(document, query, snippet);
            uint64_t t1 = Misc::currentTimeMillis();
            Output.timeofSnippetGeneration += (t1 - t0);
            if (!notDeleteDocument)_CLDELETE(document);
        }
        //std::wcout << "queryNum:"<<Output.numOfQueries<< std::endl;
        //std::wcout <<"tempSummarization:"<< snippet.getSummarization() << std::endl;
    }

    bool need_processQuery(const wstring queryStr) {
        if (this->InPara.memoryLimitMB_QRC > 0 && this->InPara.resultCacheTurnOn) {
            return (resultCache->getCacheEntry(queryStr) == resultCache->cache_end());
        } else return true;
    }

    void generateTopSnippetVec(const wstring& queryStr, vector<Snippet> &SnippetVec) {
        Query *hl_originalquery = NULL, *hl_rewrittenquery = NULL;
        hl_originalquery = QueryParser::parse(queryStr.c_str(), _T("content"), &hl_analyzer);
        hl_rewrittenquery = hl_originalquery->rewrite(hl_reader);

        uint64_t t0 = Misc::currentTimeMillis();
        TopDocs* tempResults = this->hl_searcher->_search(hl_rewrittenquery, NULL, topK);
        uint64_t t1 = Misc::currentTimeMillis();
        Output.timeofIndexServer += (t1 - t0);

        assert(tempResults != NULL);
        for (uint32_t i = 0; i < tempResults->scoreDocsLength; ++i) {
            Snippet tempSnippet;
            //      std::cout << "start getSnippetsByDocIDAndQuery..." << std::endl;
            getSnippetsByDocIDAndQuery(tempResults->scoreDocs[i].doc, hl_rewrittenquery, tempSnippet);
            //      std::cout << "finish getSnippetsByDocIDAndQuery." << std::endl;
            SnippetVec.push_back(tempSnippet);
        }
        if (hl_originalquery != hl_rewrittenquery)
            _CLDELETE(hl_rewrittenquery);
        _CLDELETE(hl_originalquery);
        _CLDELETE(tempResults);
    }
    //By rui, we can skip the index server for faster experiment

    void generateTopSnippetVec(const wstring& queryStr, vector<Snippet> &SnippetVec, vector<int32_t>& scoreDocIDs) {
        Query *hl_originalquery = NULL, *hl_rewrittenquery = NULL;
        hl_originalquery = QueryParser::parse(queryStr.c_str(), _T("content"), &hl_analyzer);
        hl_rewrittenquery = hl_originalquery->rewrite(hl_reader);

        for (uint32_t i = 0; i < scoreDocIDs.size(); ++i) {
            Snippet tempSnippet;
            //      std::cout << "start getSnippetsByDocIDAndQuery..." << std::endl;
            getSnippetsByDocIDAndQuery(scoreDocIDs[i], hl_rewrittenquery, tempSnippet);
            //      std::cout << "finish getSnippetsByDocIDAndQuery." << std::endl;
            SnippetVec.push_back(tempSnippet);
        }
        if (hl_originalquery != hl_rewrittenquery)
            _CLDELETE(hl_rewrittenquery);
        _CLDELETE(hl_originalquery);
    }

    //    void getTopSnippetVec(const wstring& queryStr, vector<Snippet> &SnippetVec) {
    //        // vector<Snippet>* SnippetVecPtr = resultCache->getVptr(queryStr);
    //        //        if (SnippetVecPtr != NULL) {
    //        //if (resultCache->getVvalue(queryStr, SnippetVec)) {
    //        SearchHashMap_It ptr = resultCache->getCacheEntry(queryStr);
    //        if (ptr != resultCache->cache_end()) {
    //            SnippetVec = ptr->second.value; // FIXME: copy vector
    //#ifdef debug
    //            wofstream wof_hitResult(_PathInfo.get_hitResult().c_str(), ios::app | ios::ate);
    //            wof_hitResult << L"hitNum: " << hitNum << L" query:" << queryStr << std::endl;
    //            for (uint32_t i = 0; i < SnippetVec.size(); i++) {
    //                wof_hitResult << SnippetVec[i] << std::endl;
    //            }
    //            wof_hitResult << L"---------------------------------------------" << std::endl;
    //            wof_hitResult.close();
    //#else
    //#endif
    //            hitNum++;
    //            return;
    //        }
    //        generateTopSnippetVec(queryStr, SnippetVec);
    //        resultCache->addByetsOfLookUps(getBytesOfSnippetVec(SnippetVec));
    //    }
    //
    //    void getTopSnippetVecWithDynamic(const wstring& queryStr, vector<Snippet> &SnippetVec) {
    //        // vector<Snippet>* SnippetVecPtr = resultCache->getVptr(queryStr);
    //        //        if (SnippetVecPtr != NULL) {
    //        //if (resultCache->getVvalue(queryStr, SnippetVec)) {
    //        SearchHashMap_It ptr = resultCache->getCacheEntry(queryStr);
    //        if (ptr != resultCache->cache_end()) {
    //            SnippetVec = ptr->second.value; // FIXME: copy vector      
    //#ifdef debug
    //            wofstream wof_hitResult(_PathInfo.get_hitResult().c_str(), ios::app | ios::ate);
    //            wof_hitResult << L"hitNum: " << hitNum << L" query:" << queryStr << std::endl;
    //            for (uint32_t i = 0; i < SnippetVec.size(); i++) {
    //                wof_hitResult << SnippetVec[i] << std::endl;
    //            }
    //            wof_hitResult << L"---------------------------------------------" << std::endl;
    //            wof_hitResult.close();
    //#else
    //#endif
    //            hitNum++;
    //            return;
    //        }
    //        generateTopSnippetVec(queryStr, SnippetVec);
    //        uint32_t tempNumBytes = getBytesOfSnippetVec(SnippetVec);
    //        resultCache->addByetsOfLookUps(tempNumBytes);
    //        CacheNode<wstring, vector<Snippet> > tempNode(queryStr, SnippetVec, tempNumBytes);
    //        resultCache->push(tempNode);
    //    }

    inline uint32_t getBytesOfSnippetVec(vector<Snippet> &SnippetVec) {
        uint32_t bytes = 0, i = 0;
        for (i = 0; i < SnippetVec.size(); i++) {
            bytes += RAMEstimator::getNumBytesOfSnippet(SnippetVec[i]);
        }
        return bytes;
    }

    bool process_onequery(const wstring& QStr, vector<Snippet>& SVec) {
        generateTopSnippetVec(QStr, SVec);
        if (InPara.resultCacheTurnOn && InPara.memoryLimitMB_QRC > 0) {
            uint32_t SSize = getBytesOfSnippetVec(SVec);
            resultCache->addByetsOfLookUps(SSize);
            CacheNode<wstring, vector<Snippet> > tempNode(QStr, SVec, SSize);
            return resultCache->push(tempNode);
        } else
            return true;
    }
    //By rui

    bool process_onequery(const wstring& QStr, vector<Snippet>& SVec, vector<int32_t>& docIDs) {
        generateTopSnippetVec(QStr, SVec, docIDs);
        if (InPara.resultCacheTurnOn && InPara.memoryLimitMB_QRC > 0) {
            uint32_t SSize = getBytesOfSnippetVec(SVec);
            resultCache->addByetsOfLookUps(SSize);
            CacheNode<wstring, vector<Snippet> > tempNode(QStr, SVec, SSize);
            return resultCache->push(tempNode);
        } else
            return true;
    }

    void runTrainQueriesFromFile() {
        string line;
        wstring currentQueryStr;
        uint32_t queryNum = 0;
        std::vector<Snippet> SnippetVec;

        if (!InPara.SkipIndexServer) {
            cout << "trainQuerySet filename: " << InPara._PathInfo.get_trainQuerySet() << endl;
            ifstream if_query(InPara._PathInfo.get_trainQuerySet().c_str());
            //ofstream of_temp(InPara._PathInfo.get_processingQuery().c_str());
            while (getline(if_query, line) && queryNum < topQuery) {
                assert(line != "");
                queryNum++;
                Output.numOfQueries++;
                //            if(queryNum < 77694)
                //            	continue;
                //of_temp << "queryNum: " << queryNum << " " << line << endl;
                currentQueryStr = StringToWString(line);
                if (need_processQuery(currentQueryStr)) {
                    SnippetVec.clear();
                    if (!process_onequery(currentQueryStr, SnippetVec)) {
                        std::cout << "resultCache is full on query["
                                << queryNum << "]." << std::endl;
                        break;
                    }
                }
                if (queryNum % 100000 == 0)std::cout << queryNum << std::endl;
            }
            cout << "train qeuryNum: " << queryNum << endl;
            if_query.close();
        } else {
            cout << "trainQuerySet filename: " << InPara._PathInfo.get_trainQuerySetWithDocID() << endl;
            ifstream if_query(InPara._PathInfo.get_trainQuerySetWithDocID().c_str());
            while (getline(if_query, line) && queryNum < topQuery) {
                assert(line != "");
                queryNum++;
                Output.numOfQueries++;
                //            if(queryNum < 77694)
                //            	continue;
                //of_temp << "queryNum: " << queryNum << " " << line << endl;
                currentQueryStr = StringToWString(line);
                vector<int32_t> docIDs;
                for (int i = 0; i < 10; i++) {
                    //if_query>>docIDs[i];
                    getline(if_query, line);
                    docIDs.push_back(std::atoi(line.c_str()));
                    //cout<<docIDs[i]<<endl;
                }
                if (need_processQuery(currentQueryStr)) {
                    SnippetVec.clear();
                    if (!process_onequery(currentQueryStr, SnippetVec, docIDs)) {
                        std::cout << "resultCache is full on query["
                                << queryNum << "]." << std::endl;
                        break;
                    }
                }
                if (queryNum % 100000 == 0)std::cout << queryNum << std::endl;
            }
            cout << "train qeuryNum: " << queryNum << endl;
            if_query.close();
        }

        //of_temp.close();
        // if_query.clear();
    }

    void runTestQueriesFromFile() {
        string line;
        wstring currentQueryStr;
        uint32_t queryNum = 0;
        SearchHashMap_It ptr;
        std::vector<Snippet> SnippetVec;

        if (!InPara.SkipIndexServer) {
            cout << "testQuerySet filename: " << InPara._PathInfo.get_testQuerySet() << endl;
            ifstream if_query(InPara._PathInfo.get_testQuerySet().c_str());

            while (getline(if_query, line) && queryNum < topQuery) {
                assert(line != "");
                queryNum++;
                Output.numOfQueries++;
                currentQueryStr = StringToWString(line);
                SnippetVec.clear();
                // 1.search cachenode & if exist to update
                if (this->InPara.memoryLimitMB_QRC > 0 && this->InPara.resultCacheTurnOn) {
                    SearchHashMap_It ptr = resultCache->getCacheEntry(currentQueryStr);
                    if (ptr != resultCache->cache_end()) {
                        SnippetVec = ptr->second.value;
                        hitNum++;
                        continue;
                    }
                }
                // 2.generate new snippet            
                process_onequery(currentQueryStr, SnippetVec);
                // getTopSnippetVec(currentQueryStr, SnippetVec);
                // getTopSnippetVecWithDynamic(currentQueryStr, SnippetVec); // BUG1        
                if (queryNum % 100000 == 0)std::cout << queryNum << std::endl;
            }
            cout << "test qeuryNum: " << queryNum << endl;

            Output.numOfQueries = queryNum;

            if_query.close();
        } else {
            cout << "testQuerySet filename: " << InPara._PathInfo.get_testQuerySetWithDocID() << endl;
            ifstream if_query(InPara._PathInfo.get_testQuerySetWithDocID().c_str());

            while (getline(if_query, line) && queryNum < topQuery) {
                assert(line != "");
                queryNum++;
                Output.numOfQueries++;
                currentQueryStr = StringToWString(line);
                vector<int32_t> docIDs;
                for (int i = 0; i < 10; i++) {
                    //if_query>>docIDs[i];
                    getline(if_query, line);
                    docIDs.push_back(std::atoi(line.c_str()));
                    //cout<<docIDs[i]<<endl;
                }
                SnippetVec.clear();

                // 1.search cachenode & if exist to update
                if (this->InPara.memoryLimitMB_QRC > 0 && this->InPara.resultCacheTurnOn) {
                    SearchHashMap_It ptr = resultCache->getCacheEntry(currentQueryStr);
                    if (ptr != resultCache->cache_end()) {
                        SnippetVec = ptr->second.value;
                        hitNum++;
                        continue;
                    }
                }
                // 2.generate new snippet 
                process_onequery(currentQueryStr, SnippetVec, docIDs);
                // getTopSnippetVec(currentQueryStr, SnippetVec);
                // getTopSnippetVecWithDynamic(currentQueryStr, SnippetVec); // BUG1        
                if (queryNum % 100000 == 0)std::cout << queryNum << std::endl;
            }
            cout << "test qeuryNum: " << queryNum << endl;

            Output.numOfQueries = queryNum;

            if_query.close();
        }




        // By Rui, delete all the document in DC after finish testing
        DocumentHashMap documentCacheContent = documentCache->getCacheContent();
        for (DocumentHashMap_It it = documentCacheContent.begin();
                it != documentCacheContent.end(); ++it) {
            _CLDELETE((it->second).value);
        }
        // if_query.clear();
    }

    void setMemoryLimit(bool isReset) {

        double memoryLimitMB = RAMEstimator::UN_LIMIT_MEMORY;
        if (InPara.isStaticCache_QRC() && InPara.memoryLimitMB_QRC > 0 && InPara.resultCacheTurnOn) {
            if (isReset) memoryLimitMB = InPara.memoryLimitMB_QRC;
            resultCache->setMemoryLimitMB(memoryLimitMB);
        }

        if (InPara.isStaticCache_DC() && InPara.memoryLimitMB_DC > 0 && InPara.documentCacheTurnOn) {
            if (isReset) memoryLimitMB = InPara.memoryLimitMB_DC;
            documentCache->setMemoryLimitMB(memoryLimitMB);
        }

        if (InPara.isStaticCache_SC() && InPara.memoryLimitMB_SC > 0 && InPara.snippetCacheTurnOn) {
            if (isReset) memoryLimitMB = InPara.memoryLimitMB_SC;
            snippetCache->setMemoryLimitMB(memoryLimitMB);
        }
    }

    void fillStaticCache(string cacheType) {
        std::cout << "start fillStaticCache..." << std::endl;
        uint32_t i = 0;

        //fill static QRC
        if (cacheType == Configure::QRC_RESULT_CACHE) {
            SearchHashMap resultCacheContent = resultCache->getCacheContent();
            resultCache->destroy();
            resultCache->setenable(false);

            vector< CacheNode<wstring, vector<Snippet> >* > CacheNode_vec;
            for (SearchHashMap_It it = resultCacheContent.begin();
                    it != resultCacheContent.end(); ++it)
                CacheNode_vec.push_back(&(it->second));
#ifdef debug
            wofstream wof_CacheNodeVec(_PathInfo.get_cacheNodeVec().c_str());
            for (i = 0; i < CacheNode_vec.size(); i++) {
                wof_CacheNodeVec << L"query: " << CacheNode_vec[i]->key << L" frequency: "
                        << CacheNode_vec[i]->frequency << L" Snippets: " << CacheNode_vec[i]->value.size() << std::endl;
                for (uint32_t j = 0; j < CacheNode_vec[i]->value.size(); j++) {
                    wof_CacheNodeVec << CacheNode_vec[i]->value[j] << std::endl;
                }
                wof_CacheNodeVec << L"-----------------------------------------" << std::endl;
            }
            wof_CacheNodeVec.close();
#else
#endif
            if (InPara.cacheStrategy_QRC == Static_QTF_Cache)
                sort(CacheNode_vec.begin(), CacheNode_vec.end(), SortCacheNodeByFrequency<wstring, vector<Snippet> >);
            if (InPara.cacheStrategy_QRC == Static_QTFDF_Cache)
                sort(CacheNode_vec.begin(), CacheNode_vec.end(), SortCacheNodeByFrequencyAndSize<wstring, vector<Snippet> >);
            for (i = 0; i < CacheNode_vec.size(); i++) {
                //std::wcout << L"fill CacheNode: " << CacheNode_vec[i].key << std::endl;
                assert(CacheNode_vec[i]->value.size() <= 10);
                if (!resultCache->push(*(CacheNode_vec[i]))) {
                    std::cout << "resultCache is full. Stop puting." << std::endl;
                    break;
                }
            }
            //            std::cout << "resultCache fill CacheNode num: " << i << std::endl;
            //            std::cout << "resultCache MemoryLimitMB: " << resultCache->getMemoryLimitMB() << std::endl;
            //            std::cout << "resultCache UsedMemoryInCacheMB: " << resultCache->getUsedMemoryInCacheMB() << std::endl;
            //            std::cout << "resultCache size: " << resultCache->getCacheContent().size() << std::endl;

#ifdef debug
            wofstream wof_resultCache(_PathInfo.get_resultCache().c_str());
            SearchHashMap_It hm_itr;
            SearchHashMap tempmap = resultCache->getCacheContent();
            for (hm_itr = tempmap.begin(); hm_itr != tempmap.end(); ++hm_itr) {
                //            std::wcout << hm_itr->first << std::endl;
                //            std::cout << i  << " CacheNode address: " << &(hm_itr->second.value)
                //             << " Snippets: " << hm_itr->second.value.size() << std::endl;
                //            std::wcout << L"query: " << hm_itr->first << std::endl;
                wof_resultCache << L"query: " << hm_itr->first << L" frequency: "
                        << hm_itr->second.frequency << L" Snippets: " << hm_itr->second.value.size() << std::endl;
                for (uint32_t j = 0; j < hm_itr->second.value.size(); j++) {
                    wof_resultCache << hm_itr->second.value[j].toTString() << std::endl;
                }
                wof_resultCache << L"----------------------------------------" << std::endl;
            }
            wof_resultCache.close();
#else
#endif
        }

        //fill static DC
        if (cacheType == Configure::DC_DOCUMENT_CACHE) {
            DocumentHashMap documentCacheContent = documentCache->getCacheContent();
            documentCache->destroy();
            documentCache->setenable(false);

            std::wcout << documentCacheContent.size() << std::endl;

            vector< CacheNode<int32_t, Document* >* > CacheNode_vec;
            for (DocumentHashMap_It it = documentCacheContent.begin();
                    it != documentCacheContent.end(); ++it) {
                //std::wcout <<it->second.key<< std::endl;
                CacheNode_vec.push_back(&(it->second));
                //_CLDELETE(it->second.value);
                //Document* td=it->second.value;
                //std::wcout << L"title:" <<(*td).get(Configure::ITEM_TITLE)<< std::endl;
                //std::wcout << L"url:" <<(*td).get(Configure::ITEM_URL)<< std::endl;
                //std::wcout << L"content:" <<(*td).get(Configure::ITEM_CONTENT)<< std::endl;
            }

            if (InPara.cacheStrategy_DC == Static_QTF_Cache)
                sort(CacheNode_vec.begin(), CacheNode_vec.end(), SortCacheNodeByFrequency<int32_t, Document* >);
            if (InPara.cacheStrategy_DC == Static_QTFDF_Cache)
                sort(CacheNode_vec.begin(), CacheNode_vec.end(), SortCacheNodeByFrequencyAndSize<int32_t, Document* >);
            for (i = 0; i < CacheNode_vec.size(); i++) {
                //std::wcout << L"fill CacheNode: " << CacheNode_vec[i]->key << std::endl;
                //int32_t docID=(*(CacheNode_vec[i])).key;
                //Document* dd=hl_searcher->doc(docID);
                //*(CacheNode_vec[i])->value=dd;
                if (!documentCache->push(*(CacheNode_vec[i]))) {
                    //_CLDELETE(dd);
                    std::cout << "Num of cacheNodes:" << i << ", documentCache is full. Stop puting." << std::endl;
                    break;
                }
            }
            //double count=0;
            for (int j = i; j < CacheNode_vec.size(); j++) {

                //std::cout << "Delete document after filling static cache: " << CacheNode_vec[j]->key<< std::endl; 
                //std::wcout << L"url:" <<(*td).get(Configure::ITEM_URL)<< std::endl;
                //std::wcout << L"url:" <<(*(*(CacheNode_vec[j])).value).get(Configure::ITEM_URL)<< std::endl;
                //std::wcout << L"content:" <<(*td).get(Configure::ITEM_CONTENT)<< std::endl;
                //count+=RAMEstimator::getNumBytesOfDocument((*(CacheNode_vec[j])).value);
                _CLDELETE((*(CacheNode_vec[j])).value);
                //std::wcout << L"url:" <<(*(*(CacheNode_vec[j])).value).get(Configure::ITEM_URL)<< std::endl;
                //Document* td=(*(CacheNode_vec[j])).value;
                //_CLDELETE(td);
                //_CLDELETE((*(CacheNode_vec[j])).value);
                //CacheNode_vec[j]->~CacheNode();
            }
            //std::cout<<count<<std::endl;
            //std::wcout << L"url:" <<(*(*(CacheNode_vec[i])).value).get(Configure::ITEM_URL)<< std::endl;
            documentCacheContent.clear();
            CacheNode_vec.clear();
        }

        //fill static SC
        if (cacheType == Configure::SC_SNIPPET_CACHE) {
            SnippetHashMap snippetCacheContent = snippetCache->getCacheContent();
            snippetCache->destroy();
            snippetCache->setenable(false);

            vector< CacheNode<wstring, Snippet >* > CacheNode_vec;
            for (SnippetHashMap_It it = snippetCacheContent.begin();
                    it != snippetCacheContent.end(); ++it)
                CacheNode_vec.push_back(&(it->second));
            if (InPara.cacheStrategy_SC == Static_QTF_Cache)
                sort(CacheNode_vec.begin(), CacheNode_vec.end(), SortCacheNodeByFrequency<wstring, Snippet >);
            if (InPara.cacheStrategy_SC == Static_QTFDF_Cache)
                sort(CacheNode_vec.begin(), CacheNode_vec.end(), SortCacheNodeByFrequencyAndSize<wstring, Snippet >);
            for (i = 0; i < CacheNode_vec.size(); i++) {
                if (!snippetCache->push(*(CacheNode_vec[i]))) {
                    std::cout << "snippetCache is full. Stop puting." << std::endl;
                    break;
                }
            }
        }
    }

    void trainingPhase() {
        cout << "start trainingPhase..." << endl;
        //trainingPhrase_normal();
        //trainingPhrase_Dynamic(); // BUG1
        hl_reader = IndexReader::open(InPara._PathInfo.get_indexDirectory().c_str());
        hl_searcher = _CLNEW IndexSearcher(hl_reader);

        uint64_t startTime = Misc::currentTimeMillis();

        setMemoryLimit(false);

        runTrainQueriesFromFile();

        setMemoryLimit(true);
        if (InPara.isStaticCache_QRC() && InPara.memoryLimitMB_QRC > 0 && InPara.resultCacheTurnOn)
            fillStaticCache("QRC_RESULT_CACHE");
        if (InPara.isStaticCache_DC() && InPara.memoryLimitMB_DC > 0 && InPara.documentCacheTurnOn)
            fillStaticCache("DC_DOCUMENT_CACHE");
        if (InPara.isStaticCache_SC() && InPara.memoryLimitMB_SC > 0 && InPara.snippetCacheTurnOn)
            fillStaticCache("SC_SNIPPET_CACHE");

        std::cout << "trainingPhase end: " << Misc::currentTimeMillis() - startTime << "ms" << std::endl;
        //std::cout << "resultCache MemoryLimitMB: " << resultCache->getMemoryLimitMB() << std::endl;
        //std::cout << "resultCache UsedMemoryInCacheMB: " << resultCache->getUsedMemoryInCacheMB() << std::endl;
        //std::cout << "resultCache size: " << resultCache->getCacheContent().size() << std::endl;
        cout << endl;
    }

    void toolForTestIO() {
        cout << "testing IO..." << endl;

        hl_reader = IndexReader::open(InPara._PathInfo.get_indexDirectory().c_str());
        hl_searcher = _CLNEW IndexSearcher(hl_reader);

        string line;
        wstring currentQueryStr;
        uint32_t queryNum = 0;

        uint64_t startTime = Misc::currentTimeMillis();

        ifstream if_query(InPara._PathInfo.get_testQuerySetWithDocID().c_str());

        while (getline(if_query, line) && queryNum < topQuery) {
            assert(line != "");
            queryNum++;
            Output.numOfQueries++;
            currentQueryStr = StringToWString(line);
            //vector<int32_t> docIDs;
            for (int i = 0; i < 10; i++) {
                //if_query>>docIDs[i];
                getline(if_query, line);
                //docIDs.push_back(std::atoi(line.c_str()));
                int docID = std::atoi(line.c_str());
                cout<<docID<<endl;
                uint64_t t0 = Misc::currentTimeMillis();
                Document*document = hl_searcher->doc(docID);
                uint64_t t1 = Misc::currentTimeMillis();
                Output.timeOfDocumentIO += (t1 - t0);
                uint32_t DSize = RAMEstimator::getNumBytesOfDocument(document);
                Output.lengthOfDocumentIO += DSize;
            }
        }

        // getTopSnippetVec(currentQueryStr, SnippetVec);
        // getTopSnippetVecWithDynamic(currentQueryStr, SnippetVec); // BUG1        
        if (queryNum % 100 == 0)std::cout << queryNum << std::endl;

        std::cout << "testing IO end: " << Misc::currentTimeMillis() - startTime << "ms" << std::endl;
        //std::cout << "resultCache MemoryLimitMB: " << resultCache->getMemoryLimitMB() << std::endl;
        //std::cout << "resultCache UsedMemoryInCacheMB: " << resultCache->getUsedMemoryInCacheMB() << std::endl;
        //std::cout << "resultCache size: " << resultCache->getCacheContent().size() << std::endl;
        cout << endl;
    }
    void readDocumentFromDisk() {
        cout << "reading documents..." << endl;

        hl_reader = IndexReader::open(InPara._PathInfo.get_indexDirectory().c_str());
        hl_searcher = _CLNEW IndexSearcher(hl_reader);

        string line;
        wstring currentQueryStr;
        uint32_t queryNum = 0;
        
        double timeOfDocumentIO=0;
        int32_t countOfDocumentIO = 0;
        long lengthOfDocumentIO = 0;

        uint64_t startTime = Misc::currentTimeMillis();

        ifstream if_query(InPara._PathInfo.get_testQuerySetWithDocID().c_str());

        while (getline(if_query, line) && queryNum < topQuery) {
            assert(line != "");
            queryNum++;
            currentQueryStr = StringToWString(line);
            //vector<int32_t> docIDs;
            for (int i = 0; i < 10; i++) {
                //if_query>>docIDs[i];
                getline(if_query, line);
                //docIDs.push_back(std::atoi(line.c_str()));
                int docID = std::atoi(line.c_str());
                //cout<<docIDs[i]<<endl;
                std::stringstream docpath;
                docpath<<"/hdd-zr/code_result/documents/"<<docID<<".txt";
                ifstream readDoc(docpath.str());
                std::stringstream tempdoc;
                uint64_t t0 = Misc::currentTimeMillis();
                tempdoc<<readDoc.rdbuf();
                uint64_t t1 = Misc::currentTimeMillis();
                string document=tempdoc.str();
                timeOfDocumentIO += (t1 - t0);
                uint32_t DSize = RAMEstimator::getNumBytesOfString(document);
                lengthOfDocumentIO += DSize;
                //cout<<document<<endl;
            }
        }

        // getTopSnippetVec(currentQueryStr, SnippetVec);
        // getTopSnippetVecWithDynamic(currentQueryStr, SnippetVec); // BUG1        
        if (queryNum % 100000 == 0)std::cout << queryNum << std::endl;

        std::cout << "testing IO end: " << Misc::currentTimeMillis() - startTime << "ms" << std::endl;
        //std::cout << "resultCache MemoryLimitMB: " << resultCache->getMemoryLimitMB() << std::endl;
        //std::cout << "resultCache UsedMemoryInCacheMB: " << resultCache->getUsedMemoryInCacheMB() << std::endl;
        //std::cout << "resultCache size: " << resultCache->getCacheContent().size() << std::endl;
        double averageTimeOfDocumentIO=timeOfDocumentIO/queryNum;
        cout<<"timeOfDocumentIO: "<<timeOfDocumentIO<<endl;
        cout<<"averageTimeOfDocumentIO: "<<averageTimeOfDocumentIO<<endl;
        cout<<"countOfDocumentIO: "<<countOfDocumentIO<<endl;
        cout<<"lengthOfDocumentIO: "<<lengthOfDocumentIO<<endl;
        cout << endl;
    }

    void outPutInfo() {
        //ofstream out(InPara._PathInfo.get_outputInfo().c_str());
        ofstream out(InPara.get_outputInfo().c_str());

        Output.setParameterFinally();

        out << "NumOfQueries: " << Output.numOfQueries << endl;
        out << "TimeOfTest: " << Output.timeOfTest << "ms" << endl;
        out << "AverageTimeOfTest: " << Output.averageTimeOfTest << "ms" << endl;

        out << "TimeOfDocumentIO: " << Output.timeOfDocumentIO << endl;
        out << "AverageTimeOfDocumentIO: " << Output.averageTimeOfDocumentIO << endl;
        out << "TimeofSnippetGeneration: " << Output.timeofSnippetGeneration << endl;
        out << "AverageTimeofSnippetGeneration: " << Output.averageTimeofSnippetGeneration << endl;
        out << "TimeofIndexServer: " << Output.timeofIndexServer << endl;
        out << "AverageTimeofIndexServer: " << Output.averageTimeofIndexServer << endl;
        out << "CountOfDocumentIO: " << Output.countOfDocumentIO << endl;
        out << "LengthOfDocumentIO: " << Output.lengthOfDocumentIO << endl;

        if (InPara.resultCacheTurnOn && InPara.memoryLimitMB_QRC > 0) {
            out << "Result cache hit ratio: " << Output.hitRatioOfQRC << endl;
            out << "NumOfQRCNodes: " << Output.numOfQRCNodes << endl;
            out << "UsedMemoryOfQRC: " << Output.usedMemoryOfQRC << endl;
        }
        if (InPara.documentCacheTurnOn && InPara.memoryLimitMB_DC > 0) {
            out << "Document cache hit ratio: " << Output.hitRatioOfDC << endl;
            out << "NumOfDCNodes: " << Output.numOfDCNodes << endl;
            out << "UsedMemoryOfDC: " << Output.usedMemoryOfDC << endl;
        }
        if (InPara.snippetCacheTurnOn && InPara.memoryLimitMB_SC > 0) {
            out << "Snippet cache hit ratio: " << Output.hitRatioOfSC << endl;
            out << "NumOfSCNodes: " << Output.numOfSCNodes << endl;
            out << "UsedMemoryOfSC: " << Output.usedMemoryOfSC << endl;
        }
    }

    void testingPhase() {
        cout << "start testingPhase..." << endl;
        Output.reset();
        resultCache->reset();
        snippetCache->reset();
        documentCache->reset();
        setMemoryLimit(true);

        uint64_t startTime = Misc::currentTimeMillis();
        runTestQueriesFromFile();
        uint64_t endTime = Misc::currentTimeMillis();

        std::cout << "NumOfQueries: " << Output.numOfQueries << std::endl;
        Output.timeOfTest = endTime - startTime;
        Output.setParameterFinally();
        /*std::cout << "TimeOfTest: " << Output.timeOfTest << "ms" << std::endl;
        std::cout << "AverageTimeOfTest: " << Output.averageTimeOfTest << "ms" << std::endl;
        
        std::cout << "TimeOfDocumentIO: " << Output.timeOfDocumentIO << std::endl;
        std::cout << "AverageTimeOfDocumentIO: " << Output.averageTimeOfDocumentIO << std::endl;
        std::cout << "TimeofSnippetGeneration: " << Output.timeofSnippetGeneration << std::endl;
        std::cout << "AverageTimeofSnippetGeneration: " << Output.averageTimeofSnippetGeneration << std::endl;
        std::cout << "CountOfDocumentIO: " << Output.countOfDocumentIO << std::endl;
        std::cout << "LengthOfDocumentIO: " << Output.lengthOfDocumentIO << std::endl;*/

        if (InPara.resultCacheTurnOn && InPara.memoryLimitMB_QRC > 0) {
            Output.hitRatioOfQRC = resultCache->getHitRatio();
            Output.numOfQRCNodes = resultCache->cacheSize();
            Output.usedMemoryOfQRC = resultCache->getUsedMemoryInCacheMB();

            /*std::cout << "Result cache hit ratio: " << Output.hitRatioOfQRC << std::endl;
            std::cout << "NumOfQRCNodes: " << Output.numOfQRCNodes << std::endl;
            std::cout << "UsedMemoryOfQRC: " << Output.usedMemoryOfQRC << std::endl;*/
        }
        if (InPara.documentCacheTurnOn && InPara.memoryLimitMB_DC > 0) {
            Output.hitRatioOfDC = documentCache->getHitRatio();
            Output.numOfDCNodes = documentCache->cacheSize();
            Output.usedMemoryOfDC = documentCache->getUsedMemoryInCacheMB();

            /*std::cout << "Document cache hit ratio: " << Output.hitRatioOfDC << std::endl;
            std::cout << "NumOfDCNodes: " << Output.numOfDCNodes << std::endl;
            std::cout << "UsedMemoryOfDC: " << Output.usedMemoryOfDC << std::endl;*/
        }
        if (InPara.snippetCacheTurnOn && InPara.memoryLimitMB_SC > 0) {
            Output.hitRatioOfSC = snippetCache->getHitRatio();
            Output.numOfSCNodes = snippetCache->cacheSize();
            Output.usedMemoryOfSC = snippetCache->getUsedMemoryInCacheMB();

            /*std::cout << "Snippet cache hit ratio: " << Output.hitRatioOfSC << std::endl;
            std::cout << "NumOfSCNodes: " << Output.numOfSCNodes << std::endl;
            std::cout << "UsedMemoryOfSC: " << Output.usedMemoryOfSC << std::endl;*/
        }

        //std::cout << "resultCache MemoryLimitMB: " << resultCache->getMemoryLimitMB() << std::endl;
        //std::cout << "resultCache UsedMemoryInCacheMB: " << resultCache->getUsedMemoryInCacheMB() << std::endl;
        //std::cout << "resultCache size: " << resultCache->getCacheContent().size() << std::endl;
        //std::cout << "hitRatio: " << resultCache->getHitRatio() << std::endl;
        std::cout << "testingPhase end: " << Misc::currentTimeMillis() - startTime << "ms" << std::endl;

        outPutInfo();

    }

    void statisticDocumentAndQuery() {

        string line;
        wstring currentQueryStr;
        uint32_t queryNum = 0;

        cout << "trainQuerySet filename: " << InPara._PathInfo.get_trainQuerySetWithDocID() << endl;
        ifstream if_query(InPara._PathInfo.get_trainQuerySetWithDocID().c_str());
        while (getline(if_query, line) && queryNum < topQuery) {
            assert(line != "");
            queryNum++;
            Output.numOfQueries++;
            //            if(queryNum < 77694)
            //            	continue;
            //of_temp << "queryNum: " << queryNum << " " << line << endl;
            currentQueryStr = StringToWString(line);
            vector<int32_t> docIDs;
            for (int i = 0; i < 10; i++) {
                //if_query>>docIDs[i];
                getline(if_query, line);
                docIDs.push_back(std::atoi(line.c_str()));
                //cout<<docIDs[i]<<endl;
            }

            if (queryNum % 100000 == 0)std::cout << queryNum << std::endl;
        }
        cout << "train qeuryNum: " << queryNum << endl;
        if_query.close();

        cout << "testQuerySet filename: " << InPara._PathInfo.get_testQuerySetWithDocID() << endl;
        ifstream if_query1(InPara._PathInfo.get_testQuerySetWithDocID().c_str());

        while (getline(if_query1, line) && queryNum < topQuery) {
            assert(line != "");
            queryNum++;
            Output.numOfQueries++;
            currentQueryStr = StringToWString(line);

            vector<int32_t> docIDs;
            for (int i = 0; i < 10; i++) {
                //if_query>>docIDs[i];
                getline(if_query1, line);
                docIDs.push_back(std::atoi(line.c_str()));
                //cout<<docIDs[i]<<endl;
            }

            if (queryNum % 100000 == 0)std::cout << queryNum << std::endl;
        }
        cout << "test qeuryNum: " << queryNum << endl;

        Output.numOfQueries = queryNum;

        if_query1.close();
    }

public:

    SearchTest();
    SearchTest(InputParameters &);
    virtual ~SearchTest();
};

#endif /* SRC_CONTRIBS_LIB_CLUCENE_SEARCHING_SEARCHTEST_H_ */
