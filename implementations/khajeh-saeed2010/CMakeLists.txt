cmake_minimum_required(VERSION 3.13)

project(khajeh2010 LANGUAGES CUDA CXX)

find_package(MPI REQUIRED)

add_subdirectory(cutil)

add_executable(khajeh2010
  Main.cu
)

if(DEFINED GPU_SM_ARCH)
  target_compile_options(khajeh2010 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
endif()

target_link_libraries(khajeh2010 PRIVATE MPI::MPI_CXX cutil)
target_include_directories(khajeh2010 PRIVATE ${MPI_CXX_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
