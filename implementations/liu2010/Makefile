ARCH = sm_30	#use sm_30 and sm_35 for Kepler, sm_20 for Fermi, sm_12 for 1.2 and sm_13 for 1.3
CXX = nvcc
NVCC = nvcc

NVCCOPTIONS = -use_fast_math -arch $(ARCH) -maxrregcount 85 --ptxas-options=-v 
NVCCFLAGS = -O3 $(NVCCOPTIONS) -I .
CXXFLAGS = $(NVCCFLAGS)
NVCCLIBS = -lm

CXXOBJS = CFastaFile.o CParams.o
NVCCOBJS = GenericFunction.o CFastaSW.o CFastaSWVec.o CFastaSWScalar.o \
			CSearch.o CSearchVec.o CSearchScalar.o CSearchMGPUVec.o \
			CSearchMGPUScalar.o main.o

EXEC = cudasw

all: $(CXXOBJS) $(NVCCOBJS)
	$(NVCC) $(NVCCFLAGS) -o $(EXEC) $(CXXOBJS) $(NVCCOBJS) $(NVCCLIBS)
	strip $(EXEC)

main.o:
	$(NVCC) $(NVCCFLAGS) -c main.cu
GenericFunction.o:
	$(NVCC) $(NVCCFLAGS) -c GenericFunction.cu
CSearch.o:
	$(NVCC) $(NVCCFLAGS) -c CSearch.cu
CSearchVec.o:
	$(NVCC) $(NVCCFLAGS) -c CSearchVec.cu
CSearchScalar.o:
	$(NVCC) $(NVCCFLAGS) -c CSearchScalar.cu
CSearchMGPUScalar.o:
	$(NVCC) $(NVCCFLAGS) -c CSearchMGPUScalar.cu
CSearchMGPUVec.o:
	$(NVCC) $(NVCCFLAGS) -c CSearchMGPUVec.cu
CFastaSW.o:
	$(NVCC) $(NVCCFLAGS) -c CFastaSW.cu
CFastaSWScalar.o:
	$(NVCC) $(NVCCFLAGS) -c CFastaSWScalar.cu
CFastaSWVec.o:
	$(NVCC) $(NVCCFLAGS) -c CFastaSWVec.cu
clean: 
	rm -rf $(EXEC)  *.o

