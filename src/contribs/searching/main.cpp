/*
 * main.cpp
 *
 */

#include "SearchTest.h"

void setIp(string drive, InputParameters& ip) {
    if (drive == "SSD") {
        ip._PathInfo.set_indexDirectory("/clueWeb/clueweb_index");

        ip._PathInfo.set_trainQuerySet("/clueWeb/clueWeb-train1M.txt");
        ip._PathInfo.set_testQuerySet("/clueWeb/clueWeb-test1M.txt");

        ip._PathInfo.set_trainQuerySetWithDocID("/clueWeb/clueWeb-train1M-withDocID.txt");
        ip._PathInfo.set_testQuerySetWithDocID("/clueWeb/clueWeb-test1M-withDocID.txt");

        ip._PathInfo.set_debugOutput("/code_result");
    }
    if (drive == "RAID") {
        ip._PathInfo.set_indexDirectory("/clueWeb/clueweb_index");

        ip._PathInfo.set_trainQuerySet("/clueWeb/clueWeb-train1M.txt");
        ip._PathInfo.set_testQuerySet("/clueWeb/clueWeb-test1M.txt");

        ip._PathInfo.set_trainQuerySetWithDocID("/clueWeb/clueWeb-train1M-withDocID.txt");
        ip._PathInfo.set_testQuerySetWithDocID("/clueWeb/clueWeb-test1M-withDocID.txt");

        ip._PathInfo.set_debugOutput("/code_result");
    }
    if (drive == "HDD") {
        ip._PathInfo.set_indexDirectory("/clueWeb/clueweb_index");

        ip._PathInfo.set_trainQuerySet("/clueWeb/clueWeb-train1M.txt");
        ip._PathInfo.set_testQuerySet("/clueWeb/clueWeb-test1M.txt");

        ip._PathInfo.set_trainQuerySetWithDocID("/clueWeb/clueWeb-train1M-withDocID.txt");
        ip._PathInfo.set_testQuerySetWithDocID("/clueWeb/clueWeb-test1M-withDocID.txt");

        ip._PathInfo.set_debugOutput("/code_result");
    }
}

int main() {
    InputParameters ip;

    //hdd,raid,ssd  
    vector<string> drives;
    drives.push_back("SSD");
    //drives.push_back("HDD");
    //drives.push_back("RAID");
    //drives.push_back("SSD");

    for (int k = 0; k < drives.size(); k++) {
        setIp(drives[k], ip);

        ip.resultCacheTurnOn = true;
        ip.documentCacheTurnOn = true;
        ip.snippetCacheTurnOn = true;

        ip.memoryLimitMB_QRC = 0;
        ip.memoryLimitMB_DC = 0;
        ip.memoryLimitMB_SC = 0;

        vector<CacheStrategy> CacheStrategyVec;
        CacheStrategyVec.push_back(CacheStrategy::Static_QTF_Cache);
        //CacheStrategyVec.push_back(CacheStrategy::Static_QTFDF_Cache);
        //CacheStrategyVec.push_back(CacheStrategy::Dynamic_LRU_Cache);
        //CacheStrategyVec.push_back(CacheStrategy::Dynamic_LFU_Cache);
        //CacheStrategyVec.push_back(CacheStrategy::Dynamic_QTFDF_Cache);

        //ip.SkipIndexServer = true;
        ip.SkipIndexServer = false;

        for (int i = 0; i < CacheStrategyVec.size(); i++) {
            ip.cacheStrategy_QRC = CacheStrategyVec[i];
            ip.cacheStrategy_SC = CacheStrategyVec[i];
            ip.cacheStrategy_DC = CacheStrategyVec[i];
            cout << CacheStrategyVec[i] << endl;

            vector<double> cacheSize;
            cacheSize.push_back(0);
            //cacheSize.push_back(20);
            //cacheSize.push_back(40);
            //cacheSize.push_back(60);
            //cacheSize.push_back(80);
            cacheSize.push_back(5);
            cacheSize.push_back(10);
            cacheSize.push_back(15);           
            cacheSize.push_back(20);

            for (int j = 0; j < cacheSize.size(); j++) {

                double DCSize = 1024.0f - cacheSize[j];

                //ip.memoryLimitMB_QRC = 0;
                ip.memoryLimitMB_QRC = cacheSize[j];
                //ip.memoryLimitMB_DC = DCSize;
                //ip.memoryLimitMB_DC = 445.0f;
                ip.memoryLimitMB_DC = 944.0f;
                //ip.memoryLimitMB_SC = cacheSize[j];
                ip.memoryLimitMB_SC = 80.0f;               

                SearchTest st(ip);
                st.settopQuery(1000000);
                //st.settopQuery(1000);

                st.trainingPhase();
                st.testingPhase();
                //st.toolForTestIO();
                //st.outPutInfo();
                //st.readDocumentFromDisk();
            }
        }
    }

    /*vector<double> cacheSize;
    cacheSize.push_back(0);
    //cacheSize.push_back(5);
    //cacheSize.push_back(10);
    //cacheSize.push_back(15);
    //cacheSize.push_back(20);

    for (int j = 0; j < cacheSize.size(); j++) {

        ip.resultCacheTurnOn = true;
        ip.documentCacheTurnOn = true;
        ip.snippetCacheTurnOn = true;

        //double DCSize = 1024.0f - cacheSize[j];

        ip.memoryLimitMB_QRC = 10;
        //ip.memoryLimitMB_DC = DCSize;
        //ip.memoryLimitMB_SC = cacheSize[j];

        //ip.memoryLimitMB_QRC = cacheSize[j];
        ip.memoryLimitMB_DC = 984.0f;
        ip.memoryLimitMB_SC = 40.0f;

        vector<CacheStrategy> CacheStrategyVec;
        //CacheStrategyVec.push_back(CacheStrategy::Static_QTF_Cache);
        //CacheStrategyVec.push_back(CacheStrategy::Static_QTFDF_Cache);
        CacheStrategyVec.push_back(CacheStrategy::Dynamic_LRU_Cache);
        //CacheStrategyVec.push_back(CacheStrategy::Dynamic_LFU_Cache);
        //CacheStrategyVec.push_back(CacheStrategy::Dynamic_QTFDF_Cache);

        //ip.SkipIndexServer = true;
        ip.SkipIndexServer = false;

        for (int i = 0; i < CacheStrategyVec.size(); i++) {
            ip.cacheStrategy_QRC = CacheStrategyVec[i];
            ip.cacheStrategy_SC = CacheStrategyVec[i];
            ip.cacheStrategy_DC = CacheStrategyVec[i];
            cout << CacheStrategyVec[i] << endl;
            SearchTest st(ip);
            st.settopQuery(1000000);

            st.trainingPhase();
            st.testingPhase();
            //_CLLDELETE(&st);
        }
    }*/


    /*vector<double> cacheSize;
    //cacheSize.push_back(0);
    //cacheSize.push_back(20);
    cacheSize.push_back(40);
    //cacheSize.push_back(60);
    //cacheSize.push_back(80);

    for (int j = 0; j < cacheSize.size(); j++) {
        double DCSize = 1024.0f - cacheSize[j];
        //ip.SkipIndexServer = false;
        ip.SkipIndexServer = true;

        ip.memoryLimitMB_QRC = 0;
        ip.memoryLimitMB_DC = DCSize;
        ip.memoryLimitMB_SC = cacheSize[j];

        ip.resultCacheTurnOn = false;
        ip.documentCacheTurnOn = true;
        ip.snippetCacheTurnOn = true;

        ip.cacheStrategy_QRC = CacheStrategy::Dynamic_LRU_Cache;
        ip.cacheStrategy_SC = CacheStrategy::Dynamic_LRU_Cache;
        ip.cacheStrategy_DC = CacheStrategy::Dynamic_LRU_Cache;

        SearchTest st(ip);
        st.settopQuery(1000000);

        st.trainingPhase();
        st.testingPhase();
    }*/

    return 0;
}
//
