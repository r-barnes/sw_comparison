cmake_minimum_required(VERSION 3.13)

project(gasal2 LANGUAGES CUDA CXX)

find_package(OpenMP REQUIRED)
find_package(CUDA)

add_library(gasal2
  src/args_parser.cpp
  src/host_batch.cpp
  src/ctors.cpp
  src/interfaces.cpp
  src/res.cpp
  src/gasal_align.cu
)
target_link_libraries(gasal2 PUBLIC ${CUDA_LIBRARIES})
target_include_directories(gasal2 PUBLIC ${CUDA_INCLUDE_DIRS})

option(DEBUG "Whether to print debugging messages")
set(MAX_QUERY_LEN "1024" CACHE STRING "Maximum query length")
set(N_CODE "0x4E" CACHE STRING "Code for non-based value N")

message(STATUS "ahmed2019: MAX_QUERY_LEN=${MAX_QUERY_LEN}")
message(STATUS "ahmed2019: N_CODE=${N_CODE}")

if(DEBUG)
  add_compile_definitions(DEBUG)
endif()

add_compile_definitions(MAX_QUERY_LEN=${MAX_QUERY_LEN})
add_compile_definitions(N_CODE=${N_CODE})
if(DEFINED N_PENALTY)
  add_compile_definitions(N_PENALTY=${N_PENALTY})
endif()

if(DEFINED GPU_SM_ARCH)
  target_compile_options(gasal2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
endif()

if(DEFINED GPU_COMPUTE_ARCH)
  target_compile_options(gasal2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_COMPUTE_ARCH}>)
endif()

set_target_properties(gasal2 PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
)

target_include_directories(gasal2 PUBLIC src)



add_executable(test_prog test_prog/test_prog.cpp)

target_link_libraries(test_prog PRIVATE OpenMP::OpenMP_CXX gasal2 ${CUDA_LIBRARIES})

file(COPY test_prog/run_tests.sh DESTINATION ${CMAKE_BINARY_DIR} FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_SOURCE_DIR}/test_prog/query_batch.fasta.tar.gz
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_SOURCE_DIR}/test_prog/target_batch.fasta.tar.gz
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/
)