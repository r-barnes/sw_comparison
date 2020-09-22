cmake_minimum_required(VERSION 3.9)

project(logan LANGUAGES CXX)

find_package(OpenMP)
find_package(GTest)

add_library(libxavier
  src/score.cpp
  src/seed.cpp
  src/aligner.cpp
  src/trace.cpp
  src/vectors.cpp
  src/xavier.cpp
)
target_compile_options(libxavier PUBLIC -mavx2 -Wall -Wextra -pedantic -Wno-write-strings)
target_compile_features(libxavier PUBLIC cxx_std_17)
target_include_directories(libxavier PUBLIC include)

add_executable(run_xavier_tests test/test.cpp)
target_link_libraries(run_xavier_tests PRIVATE libxavier GTest::GTest)
target_compile_options(run_xavier_tests PRIVATE -Wall -Wextra -pedantic)

add_executable(xavier_demo examples/demo.cpp)
target_link_libraries(xavier_demo PRIVATE libxavier)
target_compile_options(xavier_demo PRIVATE -Wall -Wextra -pedantic)
