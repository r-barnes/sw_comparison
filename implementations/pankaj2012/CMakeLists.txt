cmake_minimum_required(VERSION 3.13)

project(swift LANGUAGES CUDA CXX)

if(DEFINED TMP_DIR)
  add_compile_definitions(TMP_DIR=${TMP_DIR})
else()
  add_compile_definitions(TMP_DIR="/tmp/")
endif()

if(DEFINED GPU_TO_USE)
  add_compile_definitions(SELECTED_DEVICE_INDEX=${GPU_TO_USE})
else()
  add_compile_definitions(SELECTED_DEVICE_INDEX=0)
endif()

set(GENERAL_SOURCES
  src/preprocess.cu
  src/input.cu
  src/search.cu
  src/search2.cu
  src/query.cu
  src/reference.cu
  src/refPosList.cu
  src/hitList.cu
  src/align.cu
  src/memory.cu
  src/output.cu
  src/array.cu
  src/smithWaterman.cu
  src/refPosMap.cu
  src/refNameMap.cu
  src/refMap.cu
  src/lookupTable.cu
  src/lookupTable2.cu
  src/lookupTable3.cu
  src/lookupTable4.cu
  src/lookupTable5.cu
  src/lookupTable6.cu
  src/lookupTable7.cu
  src/lookupTable8.cu
  src/lookupTable9.cu
  src/lookupTable10.cu
  src/lookupTable11.cu
  src/lookupTable13.cu
  src/mapHits.cu
  src/mapHits2.cu
  src/mapHits3.cu
  src/mapHits4.cu
  src/mapHits5.cu
  src/mapHits6.cu
)

set(TEST_SOURCES
  test/testHitList.cu
  test/testRefPosList.cu
  test/testReference.cu
  test/testArray.cu
  test/testInput.cu
  test/testQuery.cu
  test/testSearch.cu
  test/testAlign.cu
  test/testSmithWaterman.cu
  test/testRefPosMap.cu
  test/testRefNameMap.cu
  test/testRefMap.cu
  test/testLookupTable.cu
  test/testLookupTable5.cu
  test/testLookupTable6.cu
  test/testLookupTable7.cu
  test/testMapHits.cu
  test/testMapHits2.cu
  test/testMapHits3.cu
  test/testMapHits4.cu
  test/testMapHits5.cu
  test/testMapHits6.cu
  test/testMain.cu
)

add_executable(swift ${GENERAL_SOURCES} src/main.cu)
# add_executable(swift_test ${GENERAL_SOURCES} ${TEST_SOURCES})
add_executable(swift_output_test test/testAlignOutput.cu)

target_include_directories(swift PUBLIC src)
# target_include_directories(swift_test PRIVATE test)
target_include_directories(swift_output_test PRIVATE test)

#TODO: Add fast_math
if(DEFINED GPU_SM_ARCH)
  target_compile_options(swift             PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
  # target_compile_options(swift_test        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
  target_compile_options(swift_output_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
endif()