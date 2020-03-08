cmake_minimum_required(VERSION 3.13)

project(paswas LANGUAGES CUDA C)

find_package(ZLIB)

add_executable(paswas
  gpuAlign.cu
  smithwaterman.cu
)
target_include_directories(paswas PRIVATE onGPU)

if(DEFINED GPU_SM_ARCH)
  target_compile_options(paswas PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
endif()



add_executable(no_partitioner partitioner/no_partitioner.c)
target_include_directories(no_partitioner PRIVATE partitioner)
target_link_libraries(no_partitioner PRIVATE ZLIB::ZLIB m)

add_executable(partitioner_alg_threshold partitioner/partitioner_alg_threshold.c)
target_include_directories(partitioner_alg_threshold PRIVATE partitioner)
target_link_libraries(partitioner_alg_threshold PRIVATE ZLIB::ZLIB m)
  
add_executable(partitioner_alg2_threshold partitioner/partitioner_alg2_threshold.c)
target_include_directories(partitioner_alg2_threshold PRIVATE partitioner)
target_link_libraries(partitioner_alg2_threshold PRIVATE ZLIB::ZLIB m)
