cmake_minimum_required(VERSION 3.9)

project(ssw LANGUAGES CUDA CXX C)

find_package(ZLIB REQUIRED)


add_library(ssw
  src/ssw.c
)
target_include_directories(ssw PUBLIC src/)


add_library(ssw_cpp
  src/ssw_cpp.cpp
)
target_include_directories(ssw_cpp PUBLIC src/)
target_link_libraries(ssw_cpp PUBLIC ssw)


add_executable(ssw_test
  src/main.c
)
target_include_directories(ssw_test PUBLIC src/)
target_link_libraries(ssw_test PUBLIC ssw ZLIB::ZLIB m)


add_executable(example
  src/example.c
)
target_include_directories(example PUBLIC src/)
target_link_libraries(example PUBLIC ssw)


add_executable(example_cpp
  src/example.cpp
)
target_include_directories(example_cpp PUBLIC src/)
target_link_libraries(example_cpp PUBLIC ssw_cpp)
