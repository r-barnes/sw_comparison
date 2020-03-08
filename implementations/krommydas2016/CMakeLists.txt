cmake_minimum_required(VERSION 3.13)

project(open_dwarves LANGUAGES CXX C)

find_package(OpenCL REQUIRED)

add_executable(open_dwarves
  alignments.cpp
  param.cpp
  prints.cpp
  sequences.cpp
  swat.cpp
  timeRec.cpp
  opts/opts.c
  common_args.c
  rdtsc.c
)
target_include_directories(open_dwarves PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(open_dwarves PRIVATE OpenCL::OpenCL)