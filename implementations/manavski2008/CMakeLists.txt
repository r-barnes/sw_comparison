cmake_minimum_required(VERSION 3.13)

project(swcuda LANGUAGES CUDA CXX)

find_package(Qt4 REQUIRED COMPONENTS QTCORE)
find_package(CUDA)


add_executable(swcuda
  swcuda/src/bioconfig.cpp
  swcuda/src/hardwarearchitecturecpu.cpp
  swcuda/src/hardwarearchitecturecpuprof.cpp
  swcuda/src/hardwarearchitecturecpusse2.cpp
  swcuda/src/hardwarearchitecturecuda.cpp
  swcuda/src/hardwarearchitecturecudaprof.cu
  # swcuda/src/hardwarearchitecturecudatxt.cpp
  swcuda/src/hardwarearchitecturenet.cpp
  swcuda/src/inout.cpp
  swcuda/src/jobdirector.cpp
  swcuda/src/main.cpp
  swcuda/src/smithwaterman.cpp
  swcuda/src/sw_cpu.cpp
  cudakernels/e2g_hndlscan.cu
  cudakernels/handler_solexa.cu
  cudakernels/smithwaterman.cu
  cudakernels/smithwaterman_kernel.cu
  cudakernels/swhandler4.cu
  cudakernels/swhandler4global.cu
  cudakernels/swhandler5.cu
  cudakernels/swhandler6_global.cu
  cudakernels/swhandler7_global.cu
  cudakernels/swhandlerprof.cu
  cudakernels/swhandlertxt.cu
  cudakernels/sw_kernel2.cu
  cudakernels/sw_kernel3.cu
  gpubiolib/alignmentpair.cpp
  gpubiolib/biosequence.cpp
  gpubiolib/ConfigFile.cpp
  gpubiolib/e2gengine.cpp
  gpubiolib/handler_solexa.cpp
  gpubiolib/shortsequencescache.cpp
)

target_include_directories(swcuda PRIVATE swcuda/src ${CMAKE_CURRENT_SOURCE_DIR} cudakernels)
target_link_libraries(swcuda PRIVATE gpubiolib Qt4::QtCore ${CUDA_cudadevrt_LIBRARY})

#TODO: Add fast_math
# if(DEFINED GPU_SM_ARCH)
  # target_compile_options(swcuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
# endif()