cmake_minimum_required(VERSION 3.13)

project(swsharp LANGUAGES CUDA C)

find_package(MPI)

add_library(swsharp
  swsharp/src/align.c
  swsharp/src/alignment.c
  swsharp/src/chain.c
  swsharp/src/constants.c
  swsharp/src/cpu_module.c
  swsharp/src/database.c
  swsharp/src/db_alignment.c
  swsharp/src/post_proc.c
  swsharp/src/pre_proc.c
  swsharp/src/reconstruct.c
  swsharp/src/scorer.c
  swsharp/src/sse_module.c
  swsharp/src/thread.c
  swsharp/src/threadpool.c
  swsharp/src/utils.c
  swsharp/src/ssw/ssw.c
  swsharp/src/cuda_utils.cu
  swsharp/src/evalue.cu
  swsharp/src/gpu_module.cu
  swsharp/src/hw_end_data_gpu.cu
  swsharp/src/nw_find_score_gpu.cu
  swsharp/src/nw_linear_data_gpu.cu
  swsharp/src/ov_end_data_gpu.cu
  swsharp/src/ov_find_score_gpu.cu
  swsharp/src/score_database_gpu.cu
  swsharp/src/score_database_gpu_long.cu
  swsharp/src/score_database_gpu_short.cu
  swsharp/src/sw_end_data_gpu.cu
)

target_include_directories(swsharp PUBLIC swsharp/api)

if(DEFINED GPU_SM_ARCH)
  target_compile_options(swsharp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
endif()



add_executable(swsharpdb swsharpdb/src/main.c)
target_link_libraries(swsharpdb PRIVATE swsharp)

if(MPI_FOUND)
  add_executable(swsharpdbmpi swsharpdbmpi/src/main.c swsharpdbmpi/src/mpi_module.c)
  target_link_libraries(swsharpdbmpi PRIVATE swsharp)
  target_include_directories(swsharpdbmpi PRIVATE swsharpdbmpi/src ${MPI_C_INCLUDE_DIRS})
  target_link_libraries(swsharpdbmpi PRIVATE MPI::MPI_C)
endif()

add_executable(swsharpn swsharpn/src/main.c)
target_link_libraries(swsharpn PRIVATE swsharp)

add_executable(swsharpnc swsharpnc/src/main.c)
target_link_libraries(swsharpnc PRIVATE swsharp)

add_executable(swsharpout swsharpout/src/main.c)
target_link_libraries(swsharpout PRIVATE swsharp)

add_executable(swsharpp swsharpp/src/main.c)
target_link_libraries(swsharpp PRIVATE swsharp)