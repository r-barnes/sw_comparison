cmake_minimum_required(VERSION 3.13)

project(gasal2 LANGUAGES CXX)

find_package(OpenMP REQUIRED)
find_package(ZLIB REQUIRED)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX )


if (CMAKE_C_COMPILER_ID STREQUAL "Intel")
  add_compile_options(-Kc++ -wd383,981,1572 -axP -ipo -openmp -static-intel)
else()
  add_compile_options(-mmmx -msse -msse2 -Wall -Wno-deprecated)
endif()

add_executable(gmapper
  common/anchors.c
  common/gen-st.c
  common/input.c
  common/my-alloc.c
  common/sw-full-cs.c
  common/sw-gapless.c
  common/bitmap.c
  common/fasta.c
  common/output.c
  common/read_hit_heap.c
  common/sw-full-ls.c
  common/sw-post.c
  common/sw-vector.c
  common/util.c
  gmapper/genome.c
  gmapper/gmapper.c
  gmapper/mapping.c
  gmapper/output.c
  gmapper/seeds.c
)

#NOTE: Intentionally omitted the following since it is disabled in Makefile
# add_executable(lineindex
#   mergesam/lineindex.c
#   mergesam/lineindex_lib.c
#   mergesam/file_buffer.c
# )

add_executable(fasta2fastq
  mergesam/file_buffer.c
  mergesam/fasta_reader.c
  mergesam/fasta2fastq.c
  mergesam/lineindex_lib.c
)

add_executable(mergesam
  mergesam/file_buffer.c
  mergesam/sam2pretty_lib.c
  mergesam/mergesam_heap.c
  mergesam/mergesam.c
  mergesam/fastx_readnames.c
  mergesam/sam_reader.c
  mergesam/render.c
)

add_executable(probcalc
  probcalc/probcalc.c
  common/fasta.c
  common/dynhash.c
  common/input.c
  common/output.c
  common/util.c
)

add_executable(probcalc_mp
  probcalc_mp/probcalc_mp.c
)

add_executable(shrimp_var
  shrimp_var/shrimp_var.c
)

add_executable(prettyprint
  prettyprint/prettyprint.c
  common/fasta.c
  common/dynhash.c
  common/sw-full-cs.c
  common/sw-full-ls.c
  common/input.c
  common/output.c
  common/util.c
  common/anchors.c
)

add_executable(shrimp2sam
  shrimp2sam/shrimp2sam.c
  common/fasta.c
  common/dynhash.c
  common/input.c
  common/output.c
  common/util.c
  common/anchors.c
)

add_executable(split-contigs
  utils/split-contigs.c
  common/fasta.c
  common/util.c
)

add_executable(temp-sink
  utils/temp-sink.c
)



target_link_libraries(gmapper       OpenMP::OpenMP_CXX ZLIB::ZLIB rt)
target_link_libraries(fasta2fastq   OpenMP::OpenMP_CXX ZLIB::ZLIB)
target_link_libraries(mergesam      OpenMP::OpenMP_CXX ZLIB::ZLIB)
target_link_libraries(probcalc      OpenMP::OpenMP_CXX ZLIB::ZLIB)
target_link_libraries(probcalc_mp   OpenMP::OpenMP_CXX ZLIB::ZLIB)
target_link_libraries(shrimp_var    OpenMP::OpenMP_CXX ZLIB::ZLIB)
target_link_libraries(prettyprint   OpenMP::OpenMP_CXX ZLIB::ZLIB)
target_link_libraries(shrimp2sam    OpenMP::OpenMP_CXX ZLIB::ZLIB)
target_link_libraries(split-contigs OpenMP::OpenMP_CXX ZLIB::ZLIB)
target_link_libraries(temp-sink     OpenMP::OpenMP_CXX ZLIB::ZLIB)