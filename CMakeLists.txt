cmake_minimum_required(VERSION 3.13)

project(gpu_sequencing LANGUAGES C CXX CUDA)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  message(STATUS "Using ccache")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache) # Less useful to do it for linking, see edit2
endif(CCACHE_FOUND)

option(BENCHMARKING "Whether or not we are benchmarking")

add_subdirectory(submodules/alignment_boilerplate)

# add_subdirectory(implementations/awan2020)
add_subdirectory(implementations/ahmed2019)
# add_subdirectory(implementations/bowtie2)
add_subdirectory(implementations/david2011)
add_subdirectory(implementations/khajeh-saeed2010)
add_subdirectory(implementations/klus2012)
add_subdirectory(implementations/krommydas2016)
add_subdirectory(implementations/liu2010)
add_subdirectory(implementations/liu2013)
add_subdirectory(implementations/liu2014b_cushaw2_gpu)
add_subdirectory(implementations/luo2013)
add_subdirectory(implementations/nvbio)
add_subdirectory(implementations/okada2015)
add_subdirectory(implementations/pankaj2012)
add_subdirectory(implementations/rognes2011)
add_subdirectory(implementations/sjolund2016)
add_subdirectory(implementations/striemer2009)
add_subdirectory(implementations/szalkowski2008)
add_subdirectory(implementations/warris2015/PaSWAS)


#add_subdirectory(implementations/manavski2008)
# add_subdirectory(implementations/muller2020)
# add_subdirectory(implementations/rucci2015)  #Requires clang
# add_subdirectory(implementations/rucci2018b) #Requires clang
# add_subdirectory(implementations/ugene)
