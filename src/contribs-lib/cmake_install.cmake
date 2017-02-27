# Install script for directory: /home/zr/nblucene/src/contribs-lib

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/analysis" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/analysis/PorterStemmer.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/analysis/de" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/analysis/de/GermanStemFilter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/analysis/de" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/analysis/de/GermanStemmer.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/analysis/de" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/analysis/de/GermanAnalyzer.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/analysis" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/analysis/LanguageBasedAnalyzer.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/analysis/cjk" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/analysis/cjk/CJKAnalyzer.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/highlighter" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/highlighter/Highlighter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/highlighter" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/highlighter/QueryTermExtractor.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/highlighter" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/highlighter/SimpleHTMLEncoder.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/highlighter" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/highlighter/QueryScorer.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/highlighter" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/highlighter/Scorer.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/highlighter" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/highlighter/SimpleHTMLFormatter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/highlighter" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/highlighter/Formatter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/highlighter" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/highlighter/TokenGroup.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/highlighter" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/highlighter/Fragmenter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/highlighter" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/highlighter/Encoder.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/highlighter" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/highlighter/SimpleFragmenter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/highlighter" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/highlighter/TokenSources.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/highlighter" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/highlighter/WeightedTerm.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/highlighter" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/highlighter/HighlightScorer.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/highlighter" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/highlighter/TextFragment.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/cache" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/cache/Configure.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/cache" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/cache/StaticCache.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/cache" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/cache/Snippet.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/cache" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/cache/util.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/cache" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/cache/RAMEstimator.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/cache" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/cache/CacheStrategy.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/cache" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/cache/DynLRU.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/cache" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/cache/CacheNode.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/cache" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/cache/DynQTFDF.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/cache" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/cache/DynamicCache.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/cache" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/cache/Cache.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/SnowballFilter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/SnowballAnalyzer.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_ISO_8859_1_german.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_UTF_8_porter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_ISO_8859_1_danish.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_UTF_8_swedish.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_UTF_8_english.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_ISO_8859_1_dutch.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_KOI8_R_russian.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_UTF_8_norwegian.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_ISO_8859_1_finnish.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_ISO_8859_1_english.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_ISO_8859_1_french.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_UTF_8_finnish.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_UTF_8_russian.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_UTF_8_danish.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_UTF_8_german.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_ISO_8859_1_italian.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_UTF_8_dutch.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_ISO_8859_1_porter.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_ISO_8859_1_norwegian.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_UTF_8_french.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_ISO_8859_1_spanish.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_ISO_8859_1_portuguese.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_UTF_8_italian.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_UTF_8_portuguese.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_UTF_8_spanish.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/src_c" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/src_c/stem_ISO_8859_1_swedish.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/include" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/include/libstemmer.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/libstemmer" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/libstemmer/modules.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/runtime" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/runtime/header.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball/runtime" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/runtime/api.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/snowball" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/snowball/libstemmer.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/util" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/util/streamarray.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/util" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/util/gzipinputstream.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/util" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/util/gzipcompressstream.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/util" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/util/arrayinputstream.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/util" TYPE FILE FILES "/home/zr/nblucene/src/contribs-lib/CLucene/util/byteinputstream.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "runtime")
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libclucene-contribs-lib.so.2.3.3.4"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libclucene-contribs-lib.so.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libclucene-contribs-lib.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/zr/nblucene/bin/libclucene-contribs-lib.so.2.3.3.4"
    "/home/zr/nblucene/bin/libclucene-contribs-lib.so.1"
    "/home/zr/nblucene/bin/libclucene-contribs-lib.so"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libclucene-contribs-lib.so.2.3.3.4"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libclucene-contribs-lib.so.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libclucene-contribs-lib.so"
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

