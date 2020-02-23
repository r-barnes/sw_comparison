sources=Aligner.cpp CigarAlign.cpp Genome.cpp \
	MemEngine.cpp SAM.cpp Seed.cpp BWT.cpp SeqFileParser.cpp Sequence.cpp Utils.cpp

cuda_sources = Options.cu GPUMacros.cu GPUMemEngine.cu SingleEnd.cu GPUUtils.cu \
	GPUSA.cu GPUSeeds.cu GPUSW.cu GPUBWT.cu GPUVariables.cu SuffixArray.cu main.cu \
	PairedEnd.cu

src_dir = .
objs_dir = objs

#CUSHAW2-GPU
objs += $(patsubst %.cpp, $(objs_dir)/%.cpp.o, $(sources))
objs += $(patsubst %.cu, $(objs_dir)/%.cu.o, $(cuda_sources))

#CUSHAW2-GPU-WRAPPER
wrapper_sources = Sequence.cpp SeqFileParser.cpp Utils.cpp
cuda_wrapper_sources = Options.cu GPUMacros.cu MGPUwrapper.cu
objs_wrapper += $(patsubst %.cpp, $(objs_dir)/%.wrapper.cpp.o, $(wrapper_sources))
objs_wrapper += $(patsubst %.cu, $(objs_dir)/%.wrapper.cu.o, $(cuda_wrapper_sources))

mylibs = -lpthread -lm -lz -lgomp
EXEC = cushaw2-gpu
EXEC_WRAPPER = cushaw2-gpu-wrapper

MACROS = -DCOMPRESSED_INPUT -DHAVE_TWICE_RESCUE -DFREE_UNUSED_GENOME_MEMORY

#specify maximum allowable read length. MUST be multiple of 16
MACROS += -DMAX_USER_READ_LENGTH=320

#have the support for Intel SSSE3?
have_ssse3 = 1
ifneq ($(have_ssse3), 0)
	MACROS += -DHAVE_SSSE3
	SSE_ARCH = -Xcompiler -msse4
else
	MACROS +=
	SSE_ARCH = -Xcompiler -msse2
endif

#GPU architecture
GPU_ARCH = sm_30
GPU_ARCH_FLAGS = -arch $(GPU_ARCH)
ifeq ($(GPU_ARCH), sm_35)
	GPU_ARCH_FLAGS += -DHAVE_SM_35
endif

ifeq ($(GPU_ARCH), sm_30)
	GPU_ARCH_FLAGS += -DHAVE_SM_30
endif

#compile optionsK
NVCCFLAGS += -O3 $(GPU_ARCH_FLAGS) -rdc true -Xcompiler -fopenmp -Xcompiler -funroll-loops $(MACROS) $(SSE_ARCH)
NVCC = nvcc


#bam reader
BAMREADER = bamreader
mylibs += -L $(BAMREADER) -lbamreader
NVCCFLAGS += -I $(BAMREADER)

all: dir bamr $(objs) $(objs_wrapper)
	$(NVCC) $(NVCCFLAGS) -o $(EXEC) $(objs) $(mylibs)
	strip $(EXEC)

	$(NVCC) $(NVCCFLAGS) -o $(EXEC_WRAPPER) $(objs_wrapper) $(mylibs)
	strip $(EXEC_WRAPPER)

dir:
	mkdir -p $(objs_dir)

bamr:
	make -C $(BAMREADER)

clean:
	-rm -rf $(objs_dir) $(EXEC) $(EXEC_WRAPPER)
	make -C $(BAMREADER) clean

$(objs_dir)/%.cpp.o: $(src_dir)/%.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

$(objs_dir)/%.cu.o: $(src_dir)/%.cu
	$(NVCC) $(NVCCFLAGS) --ptxas-options=-v -o $@ -c $<

$(objs_dir)/%.wrapper.cpp.o: $(src_dir)/%.cpp
	$(NVCC) $(NVCCFLAGS) -DHAVE_WRAPPER -o $@ -c $<

$(objs_dir)/%.wrapper.cu.o: $(src_dir)/%.cu
	$(NVCC) $(NVCCFLAGS) -DHAVE_WRAPPER -o $@ -c $<

