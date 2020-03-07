cmake_minimum_required(VERSION 3.13)

project(barracuda LANGUAGES CUDA C)

find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

add_executable(barracuda
  barracuda.cu

  bamlite.c
  bntseq.c
  bwape.c
  bwase.c
  bwaseqio.c
  bwt.c
  bwt_gen.c
  bwt_lite.c
  bwtaln.c
  bwtgap.c
  bwtindex.c
  bwtio.c
  bwtmisc.c
  bwtsw2_aux.c
  bwtsw2_chain.c
  bwtsw2_core.c
  bwtsw2_main.c
  bwtsw2_pair.c
  cs2nt.c
  fastmap.c
  is.c
  kstring.c
  ksw.c
  main.c
  QSufSort.c
  simple_dp.c
  stdaln.c
  utils.c
)


target_compile_definitions(barracuda PRIVATE HAVE_PTHREAD UNIX)
target_link_libraries(barracuda PRIVATE ZLIB::ZLIB Threads::Threads)

#TODO: Add fast_math
if(DEFINED GPU_SM_ARCH)
  target_compile_options(barracuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
endif()



# g++ -fPIC -o linux/release/barracuda  linux/release/objs/bwaseqio.c.o linux/release/objs/main.c.o linux/release/objs/bntseq.c.o linux/release/objs/bamlite.c.o linux/release/objs/bwt.c.o linux/release/objs/bwape.c.o linux/release/objs/bwtgap.c.o linux/release/objs/bwt_lite.c.o linux/release/objs/kstring.c.o linux/release/objs/bwase.c.o linux/release/objs/bwtsw2_aux.c.o linux/release/objs/stdaln.c.o linux/release/objs/bwtindex.c.o linux/release/objs/bwtsw2_chain.c.o linux/release/objs/is.c.o linux/release/objs/cs2nt.c.o linux/release/objs/ksw.c.o linux/release/objs/bwt_gen.c.o linux/release/objs/bwtaln.c.o linux/release/objs/utils.c.o linux/release/objs/bwtmisc.c.o linux/release/objs/bwtio.c.o linux/release/objs/bwtsw2_core.c.o linux/release/objs/bwtsw2_pair.c.o linux/release/objs/bwtsw2_main.c.o linux/release/objs/simple_dp.c.o linux/release/objs/fastmap.c.o linux/release/objs/QSufSort.c.o linux/release/objs/barracuda.cu.o  -L/opt/nvidia/cudatoolkit9.1/9.1.85_3.10-1.0502.df1cc54.3.1/lib64   -lcudart_static -ldl -lrt  -lz  -lpthread
# echo "Done!"
