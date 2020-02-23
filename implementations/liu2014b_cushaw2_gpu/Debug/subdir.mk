################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Aligner.cpp \
../BWT.cpp \
../CigarAlign.cpp \
../Genome.cpp \
../MemEngine.cpp \
../SAM.cpp \
../Seed.cpp \
../SeqFileParser.cpp \
../Sequence.cpp \
../Utils.cpp 

CU_SRCS += \
../GPUBWT.cu \
../GPUMacros.cu \
../GPUMemEngine.cu \
../GPUSA.cu \
../GPUSW.cu \
../GPUSeeds.cu \
../GPUUtils.cu \
../GPUVariables.cu \
../Options.cu \
../PairedEnd.cu \
../SingleEnd.cu \
../SuffixArray.cu \
../cushaw2-gpu-wrapper.cu \
../main.cu 

CU_DEPS += \
./GPUBWT.d \
./GPUMacros.d \
./GPUMemEngine.d \
./GPUSA.d \
./GPUSW.d \
./GPUSeeds.d \
./GPUUtils.d \
./GPUVariables.d \
./Options.d \
./PairedEnd.d \
./SingleEnd.d \
./SuffixArray.d \
./cushaw2-gpu-wrapper.d \
./main.d 

OBJS += \
./Aligner.o \
./BWT.o \
./CigarAlign.o \
./GPUBWT.o \
./GPUMacros.o \
./GPUMemEngine.o \
./GPUSA.o \
./GPUSW.o \
./GPUSeeds.o \
./GPUUtils.o \
./GPUVariables.o \
./Genome.o \
./MemEngine.o \
./Options.o \
./PairedEnd.o \
./SAM.o \
./Seed.o \
./SeqFileParser.o \
./Sequence.o \
./SingleEnd.o \
./SuffixArray.o \
./Utils.o \
./cushaw2-gpu-wrapper.o \
./main.o 

CPP_DEPS += \
./Aligner.d \
./BWT.d \
./CigarAlign.d \
./Genome.d \
./MemEngine.d \
./SAM.d \
./Seed.d \
./SeqFileParser.d \
./Sequence.d \
./Utils.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I"/home/yongchao/cuda-workspace/cushaw2-gpu/bamreader" -G -g -O0 -gencode arch=compute_35,code=sm_35 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I"/home/yongchao/cuda-workspace/cushaw2-gpu/bamreader" -G -g -O0 --compile  -x c++ -o  "$@" "$<" -Xcompiler -fopenmp -Xcompiler -msse4
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I"/home/yongchao/cuda-workspace/cushaw2-gpu/bamreader" -G -g -O0 -gencode arch=compute_35,code=sm_35 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --device-c -G -I"/home/yongchao/cuda-workspace/cushaw2-gpu/bamreader" -O0 -g -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<" -Xcompiler -fopenmp -Xcompiler -msse4
	@echo 'Finished building: $<'
	@echo ' '


