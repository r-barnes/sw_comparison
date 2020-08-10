cmake_minimum_required(VERSION 3.13)

project(cudasw2 LANGUAGES CUDA CXX)

option(BENCHMARKING "Whether or not we are benchmarking")

add_executable(cudasw2
  CFastaSW.cu
  CFastaSWScalar.cu
  CFastaSWVec.cu
  CSearch.cu
  CSearchMGPUScalar.cu
  CSearchMGPUVec.cu
  CSearchScalar.cu
  CSearchVec.cu
  GenericFunction.cu
  main.cu
  CFastaFile.cpp
  CParams.cpp
)

#TODO: Add fast_math
if(DEFINED GPU_SM_ARCH)
  target_compile_options(cudasw2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
endif()

if(BENCHMARKING)
  target_compile_definitions(cudasw2 PRIVATE BENCHMARKING)
endif()