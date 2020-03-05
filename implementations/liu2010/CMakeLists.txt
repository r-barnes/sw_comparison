cmake_minimum_required(VERSION 3.13)

project(cudasw2 LANGUAGES CUDA CXX)

if(DEFINED GPU_SM_ARCH)
  target_compile_options(gasal2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
endif()

if(DEFINED GPU_COMPUTE_ARCH)
  target_compile_options(gasal2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_COMPUTE_ARCH}>)
endif()

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
