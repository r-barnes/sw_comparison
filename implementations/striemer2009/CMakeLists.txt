cmake_minimum_required(VERSION 3.13)

project(striemer LANGUAGES CUDA CXX)

add_executable(striemer
  SmithWaterman/inc/cutil.cpp
  SmithWaterman/inc/cmd_arg_reader.cpp
  SmithWaterman/inc/stopwatch.cpp
  SmithWaterman/inc/bank_checker.cpp
  SmithWaterman/inc/stopwatch_linux.cpp
  SmithWaterman/SmithWaterman_gold.cpp
  SmithWaterman/SmithWaterman_kernel.cu
  SmithWaterman/SmithWaterman.cu
)

target_include_directories(striemer PRIVATE SmithWaterman)
target_include_directories(striemer PRIVATE SmithWaterman/inc)

#TODO: Add fast_math
if(DEFINED GPU_SM_ARCH)
  target_compile_options(striemer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
endif()