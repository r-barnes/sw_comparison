cmake_minimum_required(VERSION 3.13)

project(swipe LANGUAGES CXX)

find_package(Threads)

add_executable(swipe
  align.cc
  asnparse.cc
  database.cc
  hits.cc
  matrices.cc
  query.cc
  search16.cc
  search16s.cc
  search63.cc
  search7.cc
  stats.cc
  swipe.cc
)

target_include_directories(swipe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(swipe PRIVATE SWIPE_SSSE3)
target_compile_options(swipe PRIVATE -march=native)
target_link_libraries(swipe PRIVATE Threads::Threads)