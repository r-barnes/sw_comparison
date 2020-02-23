################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Aligner.cpp \
../src/BWT.cpp \
../src/CigarAlign.cpp \
../src/Genome.cpp \
../src/MemEngine.cpp \
../src/Options.cpp \
../src/PairedEnd.cpp \
../src/SAM.cpp \
../src/Seed.cpp \
../src/SeqFileParser.cpp \
../src/Sequence.cpp \
../src/SingleEnd.cpp \
../src/SuffixArray.cpp \
../src/Utils.cpp \
../src/main.cpp 

CU_SRCS += \
../src/GPUMacros.cu \
../src/GPUMemEngine.cu \
../src/Options.cu \
../src/SingleEnd.cu 

CU_DEPS += \
./src/GPUMacros.d \
./src/GPUMemEngine.d \
./src/Options.d \
./src/SingleEnd.d 

OBJS += \
./src/Aligner.o \
./src/BWT.o \
./src/CigarAlign.o \
./src/GPUMacros.o \
./src/GPUMemEngine.o \
./src/Genome.o \
./src/MemEngine.o \
./src/Options.o \
./src/PairedEnd.o \
./src/SAM.o \
./src/Seed.o \
./src/SeqFileParser.o \
./src/Sequence.o \
./src/SingleEnd.o \
./src/SuffixArray.o \
./src/Utils.o \
./src/main.o 

CPP_DEPS += \
./src/Aligner.d \
./src/BWT.d \
./src/CigarAlign.d \
./src/Genome.d \
./src/MemEngine.d \
./src/Options.d \
./src/PairedEnd.d \
./src/SAM.d \
./src/Seed.d \
./src/SeqFileParser.d \
./src/Sequence.d \
./src/SingleEnd.d \
./src/SuffixArray.d \
./src/Utils.d \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I"/home/yongchao/cuda-workspace/cushaw2-gpu/bamreader" -I"/home/yongchao/cuda-workspace/cushaw2-gpu/src" -G -g -O0 -Xcompiler -msse2 -gencode arch=compute_35,code=sm_35 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I"/home/yongchao/cuda-workspace/cushaw2-gpu/bamreader" -I"/home/yongchao/cuda-workspace/cushaw2-gpu/src" -G -g -O0 -Xcompiler -msse2 --compile  -x c++ -o  "$@" "$<" -Xcompiler -fopenmp -Xcompiler -msse4
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I"/home/yongchao/cuda-workspace/cushaw2-gpu/bamreader" -I"/home/yongchao/cuda-workspace/cushaw2-gpu/src" -G -g -O0 -Xcompiler -msse2 -gencode arch=compute_35,code=sm_35 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -I"/home/yongchao/cuda-workspace/cushaw2-gpu/bamreader" -I"/home/yongchao/cuda-workspace/cushaw2-gpu/src" -O0 -Xcompiler -msse2 -g -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<" -Xcompiler -fopenmp -Xcompiler -msse4
	@echo 'Finished building: $<'
	@echo ' '


