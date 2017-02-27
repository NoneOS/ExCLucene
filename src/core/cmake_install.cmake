# Install script for directory: /home/zr/nblucene/src/core

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/queryParser" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/queryParser/QueryParserConstants.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/queryParser" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/queryParser/MultiFieldQueryParser.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/queryParser/legacy" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/queryParser/legacy/MultiFieldQueryParser.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/queryParser/legacy" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/queryParser/legacy/QueryParser.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/queryParser/legacy" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/queryParser/legacy/QueryToken.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/queryParser" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/queryParser/QueryParserTokenManager.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/queryParser" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/queryParser/QueryParser.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/queryParser" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/queryParser/QueryToken.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/analysis" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/analysis/CachingTokenFilter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/analysis" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/analysis/AnalysisHeader.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/analysis" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/analysis/Analyzers.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/analysis/standard" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/analysis/standard/StandardTokenizer.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/analysis/standard" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/analysis/standard/StandardFilter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/analysis/standard" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/analysis/standard/StandardTokenizerConstants.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/analysis/standard" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/analysis/standard/StandardAnalyzer.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/store" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/store/LockFactory.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/store" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/store/Directory.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/store" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/store/FSDirectory.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/store" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/store/RAMDirectory.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/store" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/store/Lock.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/store" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/store/IndexOutput.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/store" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/store/IndexInput.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/index" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/index/MergeScheduler.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/index" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/index/MultipleTermPositions.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/index" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/index/DirectoryIndexReader.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/index" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/index/Term.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/index" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/index/IndexModifier.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/index" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/index/MultiReader.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/index" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/index/MergePolicy.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/index" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/index/IndexDeletionPolicy.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/index" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/index/TermVector.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/index" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/index/IndexWriter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/index" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/index/IndexReader.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/index" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/index/Terms.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/index" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/index/Payload.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/CLConfig.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/StdHeader.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/debug" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/debug/lucenebase.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/debug" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/debug/mem.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/debug" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/debug/error.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/document" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/document/Document.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/document" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/document/DateField.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/document" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/document/FieldSelector.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/document" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/document/DateTools.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/document" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/document/NumberTools.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/document" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/document/Field.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/util" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/util/VoidMap.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/util" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/util/CLStreams.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/util" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/util/VoidList.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/util" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/util/PriorityQueue.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/util" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/util/Reader.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/util" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/util/Equators.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/util" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/util/Array.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/util" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/util/BitSet.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/IndexSearcher.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/FieldDoc.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/CachingSpanFilter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/Query.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/Explanation.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/RangeQuery.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/SpanFilterResult.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/PrefixQuery.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/MultiSearcher.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/SpanQueryFilter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/FuzzyQuery.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/ScorerDocQueue.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/DateFilter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/ChainedFilter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/Sort.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/BooleanClause.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/ConstantScoreQuery.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/CachingWrapperFilter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/SpanFilter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/QueryFilter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/Scorer.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/TermQuery.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search/spans" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/spans/SpanNotQuery.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search/spans" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/spans/SpanNearQuery.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search/spans" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/spans/SpanScorer.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search/spans" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/spans/SpanWeight.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search/spans" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/spans/SpanOrQuery.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search/spans" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/spans/SpanFirstQuery.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search/spans" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/spans/SpanTermQuery.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search/spans" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/spans/Spans.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search/spans" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/spans/SpanQuery.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/PhraseQuery.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/RangeFilter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/MatchAllDocsQuery.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/Searchable.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/BooleanQuery.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/FilteredTermEnum.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/FilterResultCache.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/SearchHeader.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/WildcardTermEnum.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/Hits.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/Compare.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/FieldCache.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/MultiTermQuery.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/Similarity.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/Filter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/MultiPhraseQuery.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/FieldSortedHitQueue.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/search" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene/search/WildcardQuery.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/zr/nblucene/src/core/CLucene.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene" TYPE FILE FILES "/home/zr/nblucene/src/shared/CLucene/SharedHeader.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene" TYPE FILE FILES "/home/zr/nblucene/src/shared/CLucene/LuceneThreads.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/CLuceneConfig.cmake" TYPE FILE FILES "/home/zr/nblucene/src/core/CLuceneConfig.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/zr/nblucene/src/core/libclucene-core.pc")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene" TYPE FILE FILES "/home/zr/nblucene/src/shared/CLucene/clucene-config.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene" TYPE FILE FILES "/home/zr/nblucene/src/core/CLuceneConfig.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "runtime")
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libclucene-core.so.2.3.3.4"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libclucene-core.so.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libclucene-core.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/zr/nblucene/bin/libclucene-core.so.2.3.3.4"
    "/home/zr/nblucene/bin/libclucene-core.so.1"
    "/home/zr/nblucene/bin/libclucene-core.so"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libclucene-core.so.2.3.3.4"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libclucene-core.so.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libclucene-core.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_REMOVE
           FILE "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

