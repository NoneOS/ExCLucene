PROJECT(readindex)

INCLUDE (DefineOptions)
DEFINE_OPTIONS(EXTRA_OPTIONS EXTRA_LIBS)
message(${EXTRA_OPTIONS})
ADD_DEFINITIONS(${EXTRA_OPTIONS})
ADD_DEFINITIONS("-mcmodel=medium -msse4.1")

message(${PROJECT_SOURCE_DIR})
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR} )
INCLUDE_DIRECTORIES( ${clucene_SOURCE_DIR}/src/contribs-lib )

file(GLOB_RECURSE readindex_HEADERS ${PROJECT_SOURCE_DIR}/*.hpp)

#SET(readindex_files
# ./IntegrationOfReaderAndPISEQUENTIAL.cpp
#  ./main.cpp
# ./OptimizeSeg.cpp
#   ${readindex_HEADERS}
#)

IF ( USE_SHARED_OBJECT_FILES )                                                                                      
    GET_SHARED_FILES(clucene_shared_Files)
ENDIF ( USE_SHARED_OBJECT_FILES )

#MESSAGE( "readindex make option is ${CMAKE_CXX_FLAGS}" )

ADD_EXECUTABLE(cl_readindex main.cpp)
TARGET_LINK_LIBRARIES(cl_readindex clucene-core clucene-shared clucene-contribs-lib ${EXTRA_LIBS})
