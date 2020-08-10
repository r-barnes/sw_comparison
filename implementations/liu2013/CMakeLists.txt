cmake_minimum_required(VERSION 3.13)

project(cudasw2 LANGUAGES CUDA CXX)

option(BENCHMARKING "Whether or not we are benchmarking")

add_executable(cudasw3
  src/CFastaFile.cpp
  src/CParams.cpp
  src/Aligner.cpp
  src/GenericFunction.cu
  src/CFastaSW.cu
  src/CFastaSWScalar.cu
  src/CSearch.cu
  src/CSearchScalar.cu
  src/CSearchMGPUScalar.cu
  src/main.cu
)

if(DEFINED MAX_QUERY_LEN)
  add_compile_definitions(MAX_SEQ_LENGTH_THRESHOLD=${MAX_QUERY_LEN})
else()
  add_compile_definitions(MAX_SEQ_LENGTH_THRESHOLD=3072)
endif()

#TODO: Add fast_math
if(DEFINED GPU_SM_ARCH)
  target_compile_options(cudasw3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
endif()

if(BENCHMARKING)
  target_compile_definitions(cudasw3 PRIVATE BENCHMARKING)
endif()

if(DEFINED DISABLE_CPU_THREADS)
  add_compile_definitions(DISABLE_CPU_THREADS)
endif()

if(DEFINED RT_DEBUG)
  add_compile_definitions(RT_DEBUG)
endif()

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    # add_compile_definitions(HAVE_SSSE3)
endif()
