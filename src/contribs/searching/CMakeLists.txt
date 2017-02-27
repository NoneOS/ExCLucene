PROJECT(searching)

INCLUDE (DefineOptions)
DEFINE_OPTIONS(EXTRA_OPTIONS EXTRA_LIBS)
ADD_DEFINITIONS(${EXTRA_OPTIONS})

INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR} )
INCLUDE_DIRECTORIES( ${clucene_SOURCE_DIR}/src/contribs-lib )

file(GLOB_RECURSE searching_HEADERS ${PROJECT_SOURCE_DIR}/*.h)

SET(searching_files
  ./SearchTest.cpp
  ./InputParameters.cpp
  ./OutputMeasurement.cpp
  ./main.cpp
  ${searching_HEADERS}
)

IF ( USE_SHARED_OBJECT_FILES )                                                                                      
    GET_SHARED_FILES(clucene_shared_Files)
ENDIF ( USE_SHARED_OBJECT_FILES )

MESSAGE( "searching make option is ${CMAKE_CXX_FLAGS}" )

ADD_EXECUTABLE(cl_searching ${searching_files} )
TARGET_LINK_LIBRARIES(cl_searching clucene-core clucene-shared clucene-contribs-lib ${EXTRA_LIBS})
