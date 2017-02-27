/*
 * InputParameters.h
 *
 */

#ifndef SRC_CONTRIBS_LIB_CLUCENE_SEARCHING_INPUTPARAMETERS_H_
#define SRC_CONTRIBS_LIB_CLUCENE_SEARCHING_INPUTPARAMETERS_H_

#include <string>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

enum CacheStrategy {
    Static_QTF_Cache,
    Static_QTFDF_Cache,
    Dynamic_LRU_Cache,
    Dynamic_LFU_Cache,
    Dynamic_QTFDF_Cache,
    Dynamic_GDS_Cache,
    Dynamic_GDSFK_Cache
};

class InputParameters {
public:

    class SearchTestPathInfo {
        std::string indexDirectory;
        std::string trainQuerySet_filename;
        std::string testQuerySet_filename;
        std::string debugOutput_filename;
        
        //By rui
        std::string trainQuerySetWithDocID_filename;
        std::string testQuerySetWithDocID_filename;
    public:

        SearchTestPathInfo() {
            indexDirectory = "";
            trainQuerySet_filename = "";
            testQuerySet_filename = "";
            debugOutput_filename = "";
        }

        void set_indexDirectory(const char* path) {
            assert(!access(path, R_OK));
            indexDirectory = path;
        }

        void set_trainQuerySet(const char* path) {
            assert(!access(path, R_OK));
            trainQuerySet_filename = path;
        }

        void set_testQuerySet(const char* path) {
            assert(!access(path, R_OK));
            testQuerySet_filename = path;
        }
        
        //By Rui
        void set_trainQuerySetWithDocID(const char* path) {
            assert(!access(path, R_OK));
            trainQuerySetWithDocID_filename = path;
        }

        void set_testQuerySetWithDocID(const char* path) {
            assert(!access(path, R_OK));
            testQuerySetWithDocID_filename = path;
        }

        void set_debugOutput(const char* path) {
            assert(!access(path, W_OK));
            debugOutput_filename = path;
        }

        std::string get_indexDirectory() {
            return indexDirectory;
        }

        std::string get_trainQuerySet() {
            return trainQuerySet_filename;
        }

        std::string get_testQuerySet() {
            return testQuerySet_filename;
        }
        
        //By rui
        std::string get_trainQuerySetWithDocID() {
            return trainQuerySetWithDocID_filename;
        }

        std::string get_testQuerySetWithDocID() {
            return testQuerySetWithDocID_filename;
        }

        std::string get_debugOutput() {
            return debugOutput_filename;
        }

        std::string get_resultCache() {
            return debugOutput_filename + std::string("/resultCache.txt");
        }

        std::string get_cacheNodeVec() {
            return debugOutput_filename + std::string("/cacheNodeVec.txt");
        }

        std::string get_hitResult() {
            return debugOutput_filename + std::string("/hitResult.txt");
        }

        std::string get_processingQuery() {
            return debugOutput_filename + std::string("/processingQuery.txt");
        }

        std::string get_outputInfo() {
            return debugOutput_filename + std::string("/output.txt");
        }

    } _PathInfo;

    double memoryLimitMB_QRC;
    double memoryLimitMB_PLC;
    double memoryLimitMB_SC;
    double memoryLimitMB_DC;

    bool resultCacheTurnOn; // enable the result cache
    bool postingListCacheTurnOn;
    bool documentCacheTurnOn;
    bool snippetCacheTurnOn;

    CacheStrategy cacheStrategy_QRC;
    CacheStrategy cacheStrategy_PLC;
    CacheStrategy cacheStrategy_DC;
    CacheStrategy cacheStrategy_SC;

    static bool isSkipListSearch;
    static bool isDictionarySearch;
    static bool isPostingListFetching;
    
    bool SkipIndexServer;

    InputParameters() : _PathInfo() {
        memoryLimitMB_QRC = 1.5;
        memoryLimitMB_PLC = 0;
        memoryLimitMB_DC = 0;
        memoryLimitMB_SC = 0;

        resultCacheTurnOn = true;
        postingListCacheTurnOn = false;
        documentCacheTurnOn = false;
        snippetCacheTurnOn = false;

        cacheStrategy_QRC = CacheStrategy::Static_QTF_Cache;
        cacheStrategy_PLC = CacheStrategy::Static_QTF_Cache;
        cacheStrategy_DC = CacheStrategy::Static_QTF_Cache;
        cacheStrategy_SC = CacheStrategy::Static_QTF_Cache;
    }

    static void setSearchPolics(const bool SLS, const bool DS, const bool PLF) {
        isSkipListSearch = SLS;
        isDictionarySearch = DS;
        isPostingListFetching = PLF;
    }

public:

    bool isStaticCache_QRC() {
        if (cacheStrategy_QRC == CacheStrategy::Static_QTF_Cache ||
                cacheStrategy_QRC == CacheStrategy::Static_QTFDF_Cache) {
            return true;
        } else return false;
    }

    bool isStaticCache_DC() {
        if (cacheStrategy_DC == CacheStrategy::Static_QTF_Cache ||
                cacheStrategy_DC == CacheStrategy::Static_QTFDF_Cache) {
            return true;
        } else return false;
    }

    bool isStaticCache_SC() {
        if (cacheStrategy_SC == CacheStrategy::Static_QTF_Cache ||
                cacheStrategy_SC == CacheStrategy::Static_QTFDF_Cache) {
            return true;
        } else return false;
    }

    std::string get_outputInfo() {
        std::string cacheStr = "/";
        if (resultCacheTurnOn) {
            cacheStr.append("QRC_");
            //cacheStr.append(cacheStrategy_QRC);
            switch (cacheStrategy_QRC) {
                case CacheStrategy::Static_QTF_Cache:
                    cacheStr.append("StaQTF");
                    break;
                case CacheStrategy::Static_QTFDF_Cache:
                    cacheStr.append("StaQTFDF");
                    break;
                case CacheStrategy::Dynamic_LRU_Cache:
                    cacheStr.append("LRU");
                    break;
                case CacheStrategy::Dynamic_LFU_Cache:
                    cacheStr.append("LFU");
                    break;
                case CacheStrategy::Dynamic_QTFDF_Cache:
                    cacheStr.append("DynQTFDF");
                    break;
                case CacheStrategy::Dynamic_GDS_Cache:
                    cacheStr.append("GDS");
                    break;
            }
            cacheStr.append("_");
            //cacheStr.append(memoryLimitMB_QRC);
            char buffer[50];
            sprintf(buffer,"%f",memoryLimitMB_QRC);
            cacheStr.append(buffer);
            cacheStr.append("_");
        }
        if (documentCacheTurnOn) {
            cacheStr.append("DC_");
            //cacheStr.append(cacheStrategy_DC);
            switch (cacheStrategy_DC) {
                case CacheStrategy::Static_QTF_Cache:
                    cacheStr.append("StaQTF");
                    break;
                case CacheStrategy::Static_QTFDF_Cache:
                    cacheStr.append("StaQTFDF");
                    break;
                case CacheStrategy::Dynamic_LRU_Cache:
                    cacheStr.append("LRU");
                    break;
                case CacheStrategy::Dynamic_LFU_Cache:
                    cacheStr.append("LFU");
                    break;
                case CacheStrategy::Dynamic_QTFDF_Cache:
                    cacheStr.append("DynQTFDF");
                    break;
                case CacheStrategy::Dynamic_GDS_Cache:
                    cacheStr.append("GDS");
                    break;
            }
            cacheStr.append("_");
            //cacheStr.append(memoryLimitMB_DC);
            char buffer[50];
            sprintf(buffer,"%f",memoryLimitMB_DC);
            cacheStr.append(buffer);
            cacheStr.append("_");
        }
        if (snippetCacheTurnOn) {
            cacheStr.append("SC_");
            //cacheStr.append(cacheStrategy_SC);
            switch (cacheStrategy_SC) {
                case CacheStrategy::Static_QTF_Cache:
                    cacheStr.append("StaQTF");
                    break;
                case CacheStrategy::Static_QTFDF_Cache:
                    cacheStr.append("StaQTFDF");
                    break;
                case CacheStrategy::Dynamic_LRU_Cache:
                    cacheStr.append("LRU");
                    break;
                case CacheStrategy::Dynamic_LFU_Cache:
                    cacheStr.append("LFU");
                    break;
                case CacheStrategy::Dynamic_QTFDF_Cache:
                    cacheStr.append("DynQTFDF");
                    break;
                case CacheStrategy::Dynamic_GDS_Cache:
                    cacheStr.append("GDS");
                    break;
            }            
            cacheStr.append("_");
            //cacheStr.append(memoryLimitMB_SC);
            char buffer[50];
            sprintf(buffer,"%f",memoryLimitMB_SC);
            cacheStr.append(buffer);
            cacheStr.append("_");
        }
        cacheStr.append("SSD.txt");
        return _PathInfo.get_debugOutput() + cacheStr;
    }

    virtual ~InputParameters();
};

#endif /* SRC_CONTRIBS_LIB_CLUCENE_SEARCHING_INPUTPARAMETERS_H_ */
