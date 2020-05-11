cmake_minimum_required(VERSION 3.13)

project(swimm2 LANGUAGES C)

find_package(OpenMP REQUIRED)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  message(FATAL_ERROR "Intel compiler required!")
endif()

set(SRC_COMMON arguments.c sequences.c swimm.c utils.c)

add_executable(swimm2_avx2 ${SRC_COMMON} charSubmat.c AVX2search.c)
target_compile_definitions(swimm2_avx2 PRIVATE MEMALIGN=32 VECTOR_LENGTH=32 DB_SEQ_LEN_MULT=3 QUERY_SEQ_LEN_MULT=4)
target_link_libraries(swimm2_avx2 PRIVATE OpenMP::OpenMP_C)

add_executable(swimm2_sse41 ${SRC_COMMON} charSubmat.c SSE41search.c)
target_compile_definitions(swimm2_sse41 PRIVATE MEMALIGN=16 VECTOR_LENGTH=16 DB_SEQ_LEN_MULT=3 QUERY_SEQ_LEN_MULT=4)
target_link_libraries(swimm2_sse41 PRIVATE OpenMP::OpenMP_C)

add_executable(swimm2_avx512f ${SRC_COMMON} intSubmat.c AVX-512Fsearch.c)
target_compile_definitions(swimm2_avx512f PRIVATE  MEMALIGN=64 VECTOR_LENGTH=16 DB_SEQ_LEN_MULT=8 QUERY_SEQ_LEN_MULT=2 )
target_link_libraries(swimm2_avx512f PRIVATE OpenMP::OpenMP_C)

add_executable(swimm2_avx512bw ${SRC_COMMON} charSubmat.c AVX-512BWsearch.c)
target_compile_definitions(swimm2_avx512bw PRIVATE  MEMALIGN=64 VECTOR_LENGTH=64 DB_SEQ_LEN_MULT=3 QUERY_SEQ_LEN_MULT=4)
target_link_libraries(swimm2_avx512bw PRIVATE OpenMP::OpenMP_C)

add_executable(swimm2_knl_avx2 ${SRC_COMMON} charSubmat.c AVX2search.c)
target_compile_definitions(swimm2_avx2 PRIVATE KNL  MEMALIGN=64 VECTOR_LENGTH=32 DB_SEQ_LEN_MULT=3 QUERY_SEQ_LEN_MULT=4)
target_link_libraries(swimm2_knl_avx2 PRIVATE OpenMP::OpenMP_C)

add_executable(swimm2_knl_avx512f ${SRC_COMMON} intSubmat.c AVX-512Fsearch.c)
target_compile_definitions(swimm2_avx512f PRIVATE KNL MEMALIGN=64 VECTOR_LENGTH=16 DB_SEQ_LEN_MULT=8 QUERY_SEQ_LEN_MULT=2 )
target_link_libraries(swimm2_knl_avx512f PRIVATE OpenMP::OpenMP_C)
