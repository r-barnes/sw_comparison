CXXFLAGS=-O3 -std=c++11 -g
LIBS=-lpthread -Lokada2015/lib -lswsharp
CU=nvcc

all:
	$(CU) $(CXXFLAGS) compare_gpu_cpu.cpp ssw/src/ssw.c -Iokada2015/include -Issw/src/ $(LIBS)
