CXXFLAGS=-O3 -std=c++11 -Wall
LIBS=-lpthread

all:
	$(CXX) $(CXXFLAGS) compare_gpu_cpu.cpp swsharp/*.c swsharp/*.cu ssw/ssw.c -I. -I../ssw/ $(LIBS)