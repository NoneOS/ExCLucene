PROJECT(cl_wand)

INCLUDE (DefineOptions)
DEFINE_OPTIONS(EXTRA_OPTIONS EXTRA_LIBS)
ADD_DEFINITIONS(${EXTRA_OPTIONS})

INCLUDE_DIRECTORIES( ${clucene-wand_SOURCE_DIR} )
INCLUDE_DIRECTORIES( ${clucene_SOURCE_DIR}/src/core )

file(GLOB_RECURSE wand_HEADERS ${CMAKE_SOURCE_DIR}/*.h)

ADD_EXECUTABLE(wand_query
./WandQuery.cpp
./BMWQuery.cpp
./example.cpp
${wand_HEADERS}
)

ADD_EXECUTABLE(create_data
./CreateWandData.cpp
${wand_HEADERS}
)


TARGET_LINK_LIBRARIES(wand_query clucene-core clucene-shared ${EXTRA_LIBS})
TARGET_LINK_LIBRARIES(create_data clucene-core clucene-shared ${EXTRA_LIBS})
