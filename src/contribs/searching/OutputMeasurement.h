/* 
 * File:   OutputMeasurement.h
 *
 */

#ifndef OUTPUTMEASUREMENT_H
#define	OUTPUTMEASUREMENT_H

#include<string>
using namespace std;

class OutputMeasurement {
public:
    OutputMeasurement();
    OutputMeasurement(const OutputMeasurement& orig);
    virtual ~OutputMeasurement();
    
    long numOfQueries = 0;
    
    // information about cache
    // used memory of caches
    double usedMemoryOfQRC = 0;
    double usedMemoryOfDC = 0;
    double usedMemoryOfSC = 0;
    // num of cache nodes in caches
    int32_t numOfQRCNodes = 0;
    int32_t numOfDCNodes = 0;
    int32_t numOfSCNodes = 0;
    // average size of cache nodes
    //double averageSizeOfQRCNodes = 0;
    //double averageSizeOfDCNodes = 0;
    //double averageSizeOfSCNodes = 0;
    // hit ratio informations
    string hitRatioOfQRC = "";
    string hitRatioOfDC = "";
    string hitRatioOfSC = "";
    
    // time meters
    double timeOfDocumentIO = 0;
    double averageTimeOfDocumentIO = 0;
    
    double timeofSnippetGeneration = 0;
    double averageTimeofSnippetGeneration = 0;
    
    double timeofIndexServer = 0;
    double averageTimeofIndexServer = 0;
    
    double timeOfTest = 0;
    double averageTimeOfTest = 0;
    
    // IO meters
    int32_t countOfDocumentIO = 0;
    long lengthOfDocumentIO = 0;
    
    // reset parameters to the default values
    void reset(){
        
        numOfQueries = 0;
    
        hitRatioOfQRC = "";
        hitRatioOfDC = "";
        hitRatioOfSC = "";
    
        timeOfDocumentIO = 0;
        averageTimeOfDocumentIO = 0;
    
        timeofSnippetGeneration = 0;
        averageTimeofSnippetGeneration = 0;
        
        timeofIndexServer=0;
        averageTimeofIndexServer=0;
    
        timeOfTest = 0;
        averageTimeOfTest = 0;
    
        countOfDocumentIO = 0;
        lengthOfDocumentIO = 0;
        
    }
    
    // set the parameters after the running program
    void setParameterFinally(){
        this->averageTimeOfTest = this->timeOfTest / this->numOfQueries;
        this->averageTimeOfDocumentIO = this->timeOfDocumentIO / this->numOfQueries;
        this->averageTimeofSnippetGeneration = this->timeofSnippetGeneration / this->numOfQueries;
        this->averageTimeofIndexServer = this->timeofIndexServer / this->numOfQueries;
    }
    
    string toString(){
        setParameterFinally();
        string output="";
        
        /*string numOfQuery = "Num of queries: " + this->numOfQueries + "\n";
        
        string hitRatio = "";
        string usedMemoryOfCache = "";
        string numOfCacheNodes = "";
        string averageSizeOfCacheNodes = "";
        if(this->hitRatioOfQRC.length() > 0) {
            hitRatio.append("Result cache hit ratio: " + this->hitRatioOfQRC + "\n");
            usedMemoryOfCache.append("UsedMemoryOfQRC: " + this->usedMemoryOfQRC + "\n");
            numOfCacheNodes.append("NumOfQRCNodes: " + this->numOfQRCNodes + "\n");
            averageSizeOfCacheNodes("AverageSizeOfQRCNodes: " + this-> averageSizeOfQRCNodes + "\n");
        }
        if(this->hitRatioOfDC.length() > 0) {
            hitRatio.append("Document cache hit ratio: " + this->hitRatioOfDC + "\n");
            usedMemoryOfCache.append("UsedMemoryOfDC: " + this->usedMemoryOfDC + "\n");
            numOfCacheNodes.append("NumOfDCNodes: " + this->numOfDCNodes + "\n");
            averageSizeOfCacheNodes("AverageSizeOfDCNodes: " + this-> averageSizeOfDCNodes + "\n");
        }
        if(this->hitRatioOfSC.length() > 0) {
            hitRatio.append("Snippet cache hit ratio: " + this->hitRatioOfSC + "\n");
            usedMemoryOfCache.append("UsedMemoryOfSC: " + this->usedMemoryOfSC + "\n");
            numOfCacheNodes.append("NumOfSCNodes: " + this->numOfSCNodes + "\n");
            averageSizeOfCacheNodes("AverageSizeOfSCNodes: " + this-> averageSizeOfSCNodes + "\n");
        }
        
        string TimeOfDocumentIO = "TimeOfDocumentIO: " + this->timeOfDocumentIO + "\n";
        string AverageTimeOfDocumentIO = "AverageTimeOfDocumentIO: " + this->averageTimeOfDocumentIO + "\n";
        string TimeofSnippetGeneration = "TimeofSnippetGeneration: " + this->timeofSnippetGeneration +"\n";
        string AverageTimeofSnippetGeneration = "AverageTimeofSnippetGeneration: " +this->averageTimeofSnippetGeneration + "\n";
        string TimeOfTest = "TimeOfTest: " + this->timeOfTest + "\n";
        string averageTimeOfTest = "averageTimeOfTest: " + this->averageTimeOfTest + "\n";
        
        string CountOfDocumentIO = "CountOfDocumentIO: " + this->countOfDocumentIO + "\n";
        string LengthOfDocumentIO = "LengthOfDocumentIO: " + this->lengthOfDocumentIO + "\n";
        
        output.append(numOfQuery);
        
        output.append(hitRatio);
        output.append(usedMemoryOfCache);
        output.append(numOfCacheNodes);
        output.append(averageSizeOfCacheNodes);
        
        output.append(TimeOfDocumentIO);
        output.append(AverageTimeOfDocumentIO);
        output.append(TimeofSnippetGeneration);
        output.append(AverageTimeofSnippetGeneration);
        output.append(TimeOfTest);
        output.append(averageTimeOfTest);
        
        output.append(CountOfDocumentIO);
        output.append(LengthOfDocumentIO);*/
        
        return output;
    }
    
private:

};

#endif	/* OUTPUTMEASUREMENT_H */

