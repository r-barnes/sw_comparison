CXXFLAGS=-O3 -std=c++11 -Wall
LIBS=-lpthread -Lokada2015/lib -lswsharp

all:
	$(CXX) $(CXXFLAGS) compare_gpu_cpu.cpp ssw/src/ssw.c -Iokada2015/include -Issw/src/ $(LIBS)