################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../bamreader/bam.c \
../bamreader/bam_aux.c \
../bamreader/bam_import.c \
../bamreader/bgzf.c \
../bamreader/faidx.c \
../bamreader/kstring.c \
../bamreader/razf.c \
../bamreader/sam.c \
../bamreader/sam_header.c 

OBJS += \
./bamreader/bam.o \
./bamreader/bam_aux.o \
./bamreader/bam_import.o \
./bamreader/bgzf.o \
./bamreader/faidx.o \
./bamreader/kstring.o \
./bamreader/razf.o \
./bamreader/sam.o \
./bamreader/sam_header.o 

C_DEPS += \
./bamreader/bam.d \
./bamreader/bam_aux.d \
./bamreader/bam_import.d \
./bamreader/bgzf.d \
./bamreader/faidx.d \
./bamreader/kstring.d \
./bamreader/razf.d \
./bamreader/sam.d \
./bamreader/sam_header.d 


# Each subdirectory must supply rules for building sources it contributes
bamreader/%.o: ../bamreader/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I"/home/yongchao/cuda-workspace/cushaw2-gpu/bamreader" -G -g -O0 -gencode arch=compute_35,code=sm_35 -odir "bamreader" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I"/home/yongchao/cuda-workspace/cushaw2-gpu/bamreader" -G -g -O0 --compile  -x c -o  "$@" "$<" -Xcompiler -fopenmp -Xcompiler -msse4
	@echo 'Finished building: $<'
	@echo ' '


