#include <iostream>
#include "IntegrationOfReaderAndPISEQUENTIAL.hpp"
using namespace std;

int main(){
  const char* index = "clucene_index_path";
  const char* outputIndex = "output_path";
  IntegOfRAP TestIndex;
  TestIndex.IndexConversion(index,outputIndex);
  return 0;
}
