cmake_minimum_required(VERSION 3.9)

project(swimm2 LANGUAGES C)

find_package(OpenMP REQUIRED)

if(NOT CMAKE_C_COMPILER_ID STREQUAL "Intel")
  message(FATAL_ERROR "Intel compiler required, found CXX=${CMAKE_CXX_COMPILER_ID} and CC=${CMAKE_C_COMPILER_ID}!")
endif()

add_executable(swimm2_avx2 AVX2search.c charSubmat.c   arguments.c sequences.c swimm.c utils.c)
target_compile_definitions(swimm2_avx2 PRIVATE AVX2 MEMALIGN=32 VECTOR_LENGTH=32 DB_SEQ_LEN_MULT=3 QUERY_SEQ_LEN_MULT=4)
target_link_libraries(swimm2_avx2 PRIVATE OpenMP::OpenMP_C)

add_executable(swimm2_sse41 SSE41search.c charSubmat.c   arguments.c sequences.c swimm.c utils.c)
target_compile_definitions(swimm2_sse41 PRIVATE SSE41 MEMALIGN=16 VECTOR_LENGTH=16 DB_SEQ_LEN_MULT=3 QUERY_SEQ_LEN_MULT=4)
target_link_libraries(swimm2_sse41 PRIVATE OpenMP::OpenMP_C)

add_executable(swimm2_avx512 AVX-512Fsearch.c intSubmat.c   arguments.c sequences.c swimm.c utils.c)
target_compile_definitions(swimm2_avx512 PRIVATE AVX512F MEMALIGN=64 VECTOR_LENGTH=16 DB_SEQ_LEN_MULT=8 QUERY_SEQ_LEN_MULT=2)
target_link_libraries(swimm2_avx512 PRIVATE OpenMP::OpenMP_C)

add_executable(swimm2_avx512b AVX-512BWsearch.c charSubmat.c   arguments.c sequences.c swimm.c utils.c)
target_compile_definitions(swimm2_avx512b PRIVATE AVX512BW MEMALIGN=64 VECTOR_LENGTH=64 DB_SEQ_LEN_MULT=3 QUERY_SEQ_LEN_MULT=4)
target_link_libraries(swimm2_avx512b PRIVATE OpenMP::OpenMP_C)

add_executable(swimm2_knl_avx2 AVX2search.c charSubmat.c   arguments.c sequences.c swimm.c utils.c)
target_compile_definitions(swimm2_knl_avx2 PRIVATE AVX2 KNL MEMALIGN=64 VECTOR_LENGTH=32 DB_SEQ_LEN_MULT=3 QUERY_SEQ_LEN_MULT=4)
target_link_libraries(swimm2_knl_avx2 PRIVATE OpenMP::OpenMP_C)

add_executable(swimm2_knl_avx512 AVX-512Fsearch.c intSubmat.c   arguments.c sequences.c swimm.c utils.c)
target_compile_definitions(swimm2_knl_avx512 PRIVATE AVX512F KNL MEMALIGN=64 VECTOR_LENGTH=16 DB_SEQ_LEN_MULT=8 QUERY_SEQ_LEN_MULT=2)
target_link_libraries(swimm2_knl_avx512 PRIVATE OpenMP::OpenMP_C)
