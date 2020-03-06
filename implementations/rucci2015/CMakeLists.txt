cmake_minimum_required(VERSION 3.13)

project(swimm LANGUAGES C)

find_package(OpenMP REQUIRED)

if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
  message(WARNING "Cannot compile Rucci2015 for MIC offloading without Intel compiler!")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
else()
  message(SEND_ERROR "Cannot compile Rucci2015 without clang or Intel compiler!")
endif()

add_executable(swimm
  arguments.c
  CPUsearch.c
  HETsearch.c
  MICsearch.c
  sequences.c
  submat.c
  swimm.c
  utils.c
)

target_include_directories(swimm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
  target_compile_options(swimm PRIVATE -fdeclspec -mavx512f)
endif()

target_link_libraries(swimm PRIVATE OpenMP::OpenMP_C)