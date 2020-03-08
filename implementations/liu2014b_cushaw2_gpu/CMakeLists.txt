cmake_minimum_required(VERSION 3.13)

project(cushaw2-gpu LANGUAGES CUDA CXX C)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ZLIB REQUIRED)



add_library(bamreader
  bamreader/bam_aux.c
  bamreader/bam.c
  bamreader/bam_import.c
  bamreader/bgzf.c
  bamreader/faidx.c
  bamreader/kstring.c
  bamreader/razf.c
  bamreader/sam.c
  bamreader/sam_header.c
)
target_include_directories(bamreader PUBLIC bamreader)
target_link_libraries(bamreader PUBLIC ZLIB::ZLIB)



add_executable(cushaw2-gpu
  Aligner.cpp  
  CigarAlign.cpp  
  Genome.cpp  
  MemEngine.cpp  
  SAM.cpp  
  Seed.cpp  
  BWT.cpp  
  SeqFileParser.cpp  
  Sequence.cpp  
  Utils.cpp  
  Options.cu  
  GPUMacros.cu  
  GPUMemEngine.cu  
  SingleEnd.cu  
  GPUUtils.cu  
  GPUSA.cu  
  GPUSeeds.cu  
  GPUSW.cu  
  GPUBWT.cu  
  GPUVariables.cu  
  SuffixArray.cu  
  main.cu  
  PairedEnd.cu
)
target_include_directories(cushaw2-gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cushaw2-gpu PRIVATE bamreader Threads::Threads OpenMP::OpenMP_CXX)
target_compile_definitions(cushaw2-gpu PRIVATE
  HAVE_SM_30
  COMPRESSED_INPUT
  HAVE_TWICE_RESCUE
  FREE_UNUSED_GENOME_MEMORY
  MAX_USER_READ_LENGTH=320
  HAVE_SSSE3
)
target_compile_options(cushaw2-gpu PUBLIC 
  $<$<COMPILE_LANGUAGE:CXX>:-msse4>
  $<$<COMPILE_LANGUAGE:CUDA>:-rdc true>
)
set_target_properties(cushaw2-gpu PROPERTIES CXX_STANDARD 98)

if(DEFINED GPU_SM_ARCH)
  target_compile_options(cushaw2-gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
endif()



add_executable(cushaw2-gpu-wrapper
  Sequence.cpp
  SeqFileParser.cpp
  Utils.cpp
  Options.cu
  GPUMacros.cu
  MGPUwrapper.cu
)
target_include_directories(cushaw2-gpu-wrapper PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cushaw2-gpu-wrapper PRIVATE bamreader Threads::Threads OpenMP::OpenMP_CXX)
target_compile_definitions(cushaw2-gpu-wrapper PRIVATE
  HAVE_SM_30
  COMPRESSED_INPUT
  HAVE_TWICE_RESCUE
  FREE_UNUSED_GENOME_MEMORY
  MAX_USER_READ_LENGTH=320
  HAVE_SSSE3
)
target_compile_options(cushaw2-gpu-wrapper PUBLIC 
  $<$<COMPILE_LANGUAGE:CXX>:-msse4>
  $<$<COMPILE_LANGUAGE:CUDA>:-rdc true>
)
set_target_properties(cushaw2-gpu-wrapper PROPERTIES CXX_STANDARD 98)

if(DEFINED GPU_SM_ARCH)
  target_compile_options(cushaw2-gpu-wrapper PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
endif()

