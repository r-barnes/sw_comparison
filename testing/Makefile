CXXFLAGS=-O3 -std=c++11 -Wall
LIBS=-lpthread -Lokada2015/lib -lswsharp

all:
	$(CXX) $(CXXFLAGS) compare_gpu_cpu.cpp ssw/ssw.c -Iokada2015/include -I../ssw/ $(LIBS)