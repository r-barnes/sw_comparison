cmake_minimum_required(VERSION 3.13)

project(soap3 LANGUAGES CUDA C CXX)

find_package(ZLIB REQUIRED)
find_package(CUDA REQUIRED)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX )

add_compile_options($<$<COMPILE_LANGUAGE:C>:-mpopcnt>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mpopcnt>)



add_library(2bwt-lib
  2bwt-lib/BWT.c
  2bwt-lib/dictionary.c
  2bwt-lib/DNACount.c
  2bwt-lib/HSP.c
  2bwt-lib/HSPstatistic.c
  2bwt-lib/iniparser.c
  2bwt-lib/inistrlib.c
  2bwt-lib/karlin.c
  2bwt-lib/MemManager.c
  2bwt-lib/MiscUtilities.c
  2bwt-lib/QSufSort.c
  2bwt-lib/r250.c
  2bwt-lib/Socket.c
  2bwt-lib/TextConverter.c
  2bwt-lib/Timing.c
)
target_include_directories(2bwt-lib PUBLIC 2bwt-lib)
target_link_libraries(2bwt-lib PRIVATE m)



add_library(samlib
  samtools-0.1.18/bam.c
  samtools-0.1.18/bam_aux.c
  samtools-0.1.18/bam_import.c
  samtools-0.1.18/bam_pileup.c
  samtools-0.1.18/bgzf.c
  samtools-0.1.18/faidx.c
  samtools-0.1.18/kstring.c
  samtools-0.1.18/razf.c
  samtools-0.1.18/sam.c
  samtools-0.1.18/sam_header.c
)
target_include_directories(samlib PUBLIC samtools-0.1.18 ZLIB::ZLIB)



add_library(2bwt-flex
  2bwt-flex/HOCC.c
  2bwt-flex/HOCCConstruct.c
  2bwt-flex/LT.c
  2bwt-flex/LTConstruct.c
  2bwt-flex/SRA2BWTCheckAndExtend.c
  2bwt-flex/SRA2BWTMdl.c
)
target_include_directories(2bwt-flex PRIVATE 2bwt-flex)
target_link_libraries(2bwt-flex PUBLIC 2bwt-lib)



add_executable(soap3-dp
  SOAP3-DP.cu
  BGS-IO.cpp
  BGS-HostAlgnmtAlgo2.cpp
  BGS-HostAlgnmtAlgoSingle.cpp
  DV-Kernel.cu
  PE.cpp
  SAList.cpp
  CPUfunctions.cpp
  alignment.cu
  PEAlgnmt.cpp
  SAM.cpp
  DV-SemiDP.cu
  OutputDPResult.cpp
  AlgnResult.cpp
  DV-DPfunctions.cu
  DV-DPForBothUnalign.cu
  DV-DPForSingleReads.cu
  global_arrays.cpp
  aio_thread.cpp
  ssse3_popcount.cpp
  IniParam.cpp
  UsageInterface.cpp
  FileUtilities.cpp
  IndexHandler.cpp
  QueryParser.cpp
)
target_link_libraries(soap3-dp PRIVATE samlib 2bwt-lib 2bwt-flex ZLIB::ZLIB cuda cudart)
if(DEFINED GPU_SM_ARCH)
  target_compile_options(soap3-dp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
endif()



add_executable(sample
  sample.cu
  BGS-IO.cpp
  BGS-HostAlgnmtAlgo2.cpp
  BGS-HostAlgnmtAlgoSingle.cpp
  DV-Kernel.cu
  PE.cpp
  SAList.cpp
  CPUfunctions.cpp
  alignment.cu
  PEAlgnmt.cpp
  SAM.cpp
  DV-SemiDP.cu
  OutputDPResult.cpp
  AlgnResult.cpp
  DV-DPfunctions.cu
  DV-DPForBothUnalign.cu
  DV-DPForSingleReads.cu
  soap3-dp-module.cu
  global_arrays.cpp
  ssse3_popcount.cpp
  IniParam.cpp
  UsageInterface.cpp
  FileUtilities.cpp
  IndexHandler.cpp
  QueryParser.cpp
)
target_link_libraries(sample PRIVATE samlib 2bwt-flex 2bwt-lib ZLIB::ZLIB cuda cudart)
if(DEFINED GPU_SM_ARCH)
  target_compile_options(sample PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
endif()



add_executable(BGS-View BGS-View.cpp)
target_link_libraries(BGS-View PRIVATE 2bwt-lib)

add_executable(BGS-View-PE BGS-View-PE.cpp)
target_link_libraries(BGS-View-PE PRIVATE 2bwt-lib)

add_executable(soap3-dp-builder
  2bwt-flex/2BWT-Builder.c 
  2bwt-lib/BWTConstruct.c
)
target_link_libraries(soap3-dp-builder PRIVATE 2bwt-flex)

add_executable(BGS-Build BGS-Build.cpp)
target_link_libraries(BGS-Build PRIVATE 2bwt-lib)
